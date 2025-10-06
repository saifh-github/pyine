from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import pathlib
import sqlite3
import threading
import typing

import backoff
import langchain_core.language_models
import langchain_core.messages
import langchain_core.runnables
import orjson
import pydantic

import pyine.data.utils.filter_rules
import pyine.utils.filesystem
import pyine.utils.langchain
import pyine.utils.reprod
from pyine.prompts.types import PromptBuildConfig, PromptNameType, PromptVersionType  # noqa

logger = logging.getLogger(__name__)
T = typing.TypeVar("T")


def _reload_metadata(raw: str | None) -> dict[str, pydantic.JsonValue]:
    """Load a JSON string into a dictionary of JSON values, falling back to an empty dict."""
    if not raw:
        return {}
    try:
        loaded: pydantic.JsonValue = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError("failed to decode json data from record") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"unexpected decoded data type; got: {type(loaded)}")
    assert all(isinstance(key, str) for key in typing.cast("dict[typing.Any, pydantic.JsonValue]", loaded)), (
        "unexpected non-str-tag"
    )
    return typing.cast("dict[str, pydantic.JsonValue]", loaded)


def _reload_tags(raw: str | None) -> list[str]:
    """Load a JSON string into a list of strings, coerce other element types to str."""
    if not raw:
        return []
    try:
        loaded: pydantic.JsonValue = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        raise ValueError("failed to decode json data from record") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"unexpected decoded data type; got: {type(loaded)}")
    assert all(isinstance(tag, str) for tag in typing.cast("list[typing.Any]", loaded)), "unexpected non-str-tag"
    return typing.cast("list[str]", loaded)


def _ensure_text(value: typing.Any) -> str:
    """Coerce arbitrary values (including lists/dicts/bytes) into a UTF-8 string."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)  # type: ignore[reportUnknownArgumentType]
        except TypeError:
            pass
    return str(value)  # type: ignore[reportUnknownArgumentType]


class CreationMeta(pydantic.BaseModel):
    """Metadata about how/when a record was created."""

    model_config = pydantic.ConfigDict(extra="allow")
    """Allows extra fields to be defined in subclasses."""
    created_at: datetime.datetime = pydantic.Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )
    """Timestamp of record creation (UTC)."""
    created_by: str = pydantic.Field(
        default_factory=lambda: pyine.utils.filesystem.get_username(),
    )
    """Username or email of the user who created the record."""
    platform: str = pydantic.Field(
        default_factory=lambda: pyine.utils.reprod.get_platform_name(),
    )
    """Platform or service name where the record was created."""
    provider: str | None = None
    """Platform or service that generated the result."""
    llm_params: dict[str, typing.Any] | None = None
    """Hyperparameters or other provider-specific parameters used."""
    llm_output: dict[str, pydantic.JsonValue] | None = None
    """Arbitrary LLM provider specific output data obtained after generation."""

    @pydantic.field_validator("created_at", mode="after")
    @classmethod
    def _ensure_timezone(
        cls,
        value: datetime.datetime,
    ) -> datetime.datetime:
        """Ensure created_at carries timezone information."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware; pass a datetime with tzinfo set (UTC recommended).")
        return value.astimezone(datetime.UTC)

    @property
    def created_at_localtz(self) -> datetime.datetime:
        """Return the created_at datetime in the local timezone."""
        return self.created_at.astimezone()

    def __str__(self) -> str:
        """Returns a string representation of the creation metadata (for debugging purposes)."""
        return f"Created at {self.created_at_localtz.isoformat()} by {self.created_by}"


class PromptResultRecord(pydantic.BaseModel):
    """Pydantic model for a stored prompt/result entry."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Freeze the model to prevent record modifications after creation."""
    identifier: str
    """Identifier for the record; unlikely unique, as we might have different versions/prompts with the same data."""
    prompt_name: PromptNameType | None = None
    """Name of the prompt that generated this record."""
    prompt_version: PromptVersionType | None = None
    """Version of the prompt that generated this record."""
    group: str | None = None
    """Optional group name for grouping similar records together."""
    creation_meta: CreationMeta
    """Metadata about how/when this record was created."""
    prompt: str
    """Prompt that generated the associated result."""
    result: str
    """Result generated by the LLM."""
    meta: dict[str, pydantic.JsonValue] = pydantic.Field(
        default_factory=lambda: typing.cast("dict[str, pydantic.JsonValue]", {}),
    )
    """Additional metadata associated with this record."""
    tags: list[str] = pydantic.Field(
        default_factory=lambda: typing.cast("list[str]", []),
    )
    """Tags associated with this record."""


class PromptResultDB:
    """SQLite-backed database for storing and fetching LLM prompt results.

    Thread-safe within a process via an internal lock. Uses WAL mode for better
    concurrent read performance. Each operation uses a short-lived connection.
    """

    def __init__(
        self,
        db_path: str | pathlib.Path | None = None,
    ) -> None:
        """Initializes (or connects to) the database."""
        resolved_db_path = get_framework_db_path() if db_path is None else pathlib.Path(db_path)
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"setting up prompt result database at: {resolved_db_path}")
        self._path = resolved_db_path
        self._lock = threading.RLock()
        self._init_db()

    def store(
        self,
        *,
        identifier: str,
        prompt: str,
        result: str,
        meta: dict[str, pydantic.JsonValue] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        prompt_name: PromptNameType | None = None,
        prompt_version: PromptVersionType | None = None,
        creation_meta: CreationMeta | None = None,
    ) -> int:
        """Stores a new prompt/result record in the database.

        Args:
            identifier: Identifier for the record, used for retrieving related entries.
            prompt: The actual prompt text sent to the LLM.
            result: The response received from the LLM.
            meta: Optional dictionary of additional metadata to store.
            tags: Optional list of tags to associate with this record for later filtering.
            group: Optional group name for organizing related records.
            prompt_name: Optional name of the prompt template used.
            prompt_version: Optional version of the prompt template used.
            creation_meta: Optional metadata about record creation.

        Returns:
            The inserted row ID in the database.
        """
        if creation_meta is None:
            creation_meta = CreationMeta()
        # note: internally used created_at field is in UTC time zone, and stored as isoformat string
        internal_created_at = creation_meta.created_at.astimezone(datetime.UTC).isoformat()
        logger.debug(f"storing new entry in database ({identifier=})")
        with self._lock:

            @backoff.on_exception(
                backoff.expo,
                (sqlite3.OperationalError, sqlite3.IntegrityError),
                max_time=30,
                jitter=backoff.full_jitter,
            )
            def _do_insert() -> int:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    cur = conn.execute(
                        """
                        INSERT INTO items (
                            identifier, "group", prompt_name, prompt_version,
                            created_at, creation_meta, prompt, result, meta, tags
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identifier,
                            group,
                            prompt_name,
                            prompt_version,
                            internal_created_at,
                            orjson.dumps(creation_meta.model_dump(mode="json")),
                            prompt,
                            result,
                            orjson.dumps(meta) if meta is not None else None,
                            orjson.dumps(tags) if tags is not None else None,
                        ),
                    )
                    conn.commit()
                    if cur.lastrowid is None:
                        raise RuntimeError("failed to retrieve lastrowid after insert")
                    return int(cur.lastrowid)
                finally:
                    conn.close()

            return _do_insert()

    def get_by_identifier(
        self,
        identifier: str,
        *,
        prompt_name: PromptNameType | None = None,
        prompt_version: PromptVersionType | None = None,
        tag_filter_rule: str | None = None,
        max_result_age: datetime.timedelta | None = None,
    ) -> list[PromptResultRecord]:
        """Fetch entries matching the given identifier with optional filtering.

        Args:
            identifier: The identifier to search for.
            prompt_name: Optional prompt template name to filter by.
            prompt_version: Optional prompt version to filter by (requires prompt_name).
            tag_filter_rule: Optional rule string for filtering by tags. See the
                `pyine.data.utils.filter_rules` module for more details.
            max_result_age: Optional maximum age of results to return.

        Returns:
            List of matching PromptResultRecord objects, ordered by creation time.
        """
        if prompt_name is None and prompt_version is not None:
            raise ValueError("prompt_version specified without prompt_name")
        sql = [
            "SELECT * FROM items WHERE identifier = ?",
        ]
        params: list[typing.Any] = [identifier]
        if prompt_name is not None:
            sql.append("AND prompt_name = ?")
            params.append(prompt_name)
        if prompt_version is not None:
            sql.append("AND prompt_version = ?")
            params.append(prompt_version)
        if max_result_age is not None:
            # use utc isoformatted time for comparison w/ internal-use-only created_at timestamp
            cutoff = datetime.datetime.now(datetime.UTC) - max_result_age
            sql.append("AND created_at > ?")
            params.append(cutoff.isoformat())
        sql.append("ORDER BY created_at ASC, id ASC")  # ordered by iso utc time
        return self._get_records(sql, params, tag_filter_rule)

    def get_by_group(
        self,
        group: str,
        *,
        prompt_name: PromptNameType | None = None,
        prompt_version: PromptVersionType | None = None,
        tag_filter_rule: str | None = None,
        max_result_age: datetime.timedelta | None = None,
    ) -> list[PromptResultRecord]:
        """Fetch all records belonging to the specified group.

        Args:
            group: The group name to search for.
            prompt_name: Optional prompt template name to filter by.
            prompt_version: Optional prompt version to filter by (requires prompt_name).
            tag_filter_rule: Optional rule string for filtering by tags. See the
                `pyine.data.utils.filter_rules` module for more details.
            max_result_age: Optional maximum age of results to return.

        Returns:
            List of matching PromptResultRecord objects, ordered by identifier and creation time.
        """
        if prompt_name is None and prompt_version is not None:
            raise ValueError("prompt_version specified without prompt_name")
        sql = ['SELECT * FROM items WHERE "group" = ?']
        params: list[typing.Any] = [group]
        if prompt_name is not None:
            sql.append("AND prompt_name = ?")
            params.append(prompt_name)
        if prompt_version is not None:
            sql.append("AND prompt_version = ?")
            params.append(prompt_version)
        if max_result_age is not None:
            # use utc isoformatted time for comparison w/ internal-use-only created_at timestamp
            cutoff = datetime.datetime.now(datetime.UTC) - max_result_age
            sql.append("AND created_at > ?")
            params.append(cutoff.isoformat())
        sql.append("ORDER BY identifier ASC, created_at ASC, id ASC")
        return self._get_records(sql, params, tag_filter_rule)

    def get_by_prompt_name(
        self,
        prompt_name: str,
        *,
        prompt_version: PromptVersionType | None = None,
        tag_filter_rule: str | None = None,
        max_result_age: datetime.timedelta | None = None,
    ) -> list[PromptResultRecord]:
        """Fetch all records that match the given prompt name.

        Args:
            prompt_name: Optional prompt template name to filter by.
            prompt_version: Optional prompt version to filter by (requires prompt_name).
            tag_filter_rule: Optional rule string for filtering by tags. See the
                `pyine.data.utils.filter_rules` module for more details.
            max_result_age: Optional maximum age of results to return.

        Returns:
            List of matching PromptResultRecord objects, ordered by identifier and creation time.
        """
        if not prompt_name:
            raise ValueError("prompt name required")
        sql = ["SELECT * FROM items WHERE prompt_name = ?"]
        params: list[typing.Any] = [prompt_name]
        if prompt_version is not None:
            sql.append("AND prompt_version = ?")
            params.append(prompt_version)
        if max_result_age is not None:
            # use utc isoformatted time for comparison w/ internal-use-only created_at timestamp
            cutoff = datetime.datetime.now(datetime.UTC) - max_result_age
            sql.append("AND created_at > ?")
            params.append(cutoff.isoformat())
        sql.append("ORDER BY identifier ASC, created_at ASC, id ASC")
        return self._get_records(sql, params, tag_filter_rule)

    def get_all_results(
        self,
        tag_filter_rule: str | None = None,
        max_result_age: datetime.timedelta | None = None,
    ) -> list[PromptResultRecord]:
        """Fetch all results from the database.

        Args:
            tag_filter_rule: Optional rule string for filtering by tags. See the
                `pyine.data.utils.filter_rules` module for more details.
            max_result_age: Optional maximum age of results to return.

        Returns:
            List of matching PromptResultRecord objects, ordered by identifier and creation time.
        """
        sql = ["SELECT * FROM items WHERE 1=1"]
        params: list[typing.Any] = []
        if max_result_age is not None:
            # use utc isoformatted time for comparison w/ internal-use-only created_at timestamp
            cutoff = datetime.datetime.now(datetime.UTC) - max_result_age
            sql.append("AND created_at > ?")
            params.append(cutoff.isoformat())
        sql.append("ORDER BY identifier ASC, created_at ASC, id ASC")
        return self._get_records(sql, params, tag_filter_rule)

    def list_identifiers(self) -> list[str]:
        """List all unique identifiers present in the database.

        Returns:
            List of identifiers sorted alphabetically.
        """
        conn = self._connect()
        try:
            rows = conn.execute("SELECT DISTINCT identifier FROM items ORDER BY identifier ASC").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def list_groups(self) -> list[str]:
        """List all unique group names present in the database.

        Returns:
            List of group names sorted alphabetically, excluding None values.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                'SELECT DISTINCT "group" FROM items WHERE "group" IS NOT NULL ORDER BY "group" ASC'
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def list_prompt_names(self) -> list[str]:
        """List all unique prompt names present in the database."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT prompt_name FROM items WHERE prompt_name IS NOT NULL ORDER BY prompt_name ASC"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def count_records(self) -> int:
        """Return the total number of records stored in the database."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM items").fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            conn.close()

    def delete_records(
        self,
        *,
        identifier: str | None = None,
        group: str | None = None,
        prompt_name: PromptNameType | None = None,
        prompt_version: PromptVersionType | None = None,
        older_than: datetime.timedelta | None = None,
    ) -> int:
        """Delete records matching the provided filters.

        At least one filter must be provided to avoid accidentally deleting the entire table.

        Args:
            identifier: If provided, delete only records with this identifier.
            group: If provided, delete only records in this group.
            prompt_name: If provided, filter by prompt name.
            prompt_version: If provided, filter by prompt version (requires prompt_name).
            older_than: If provided, delete only records with created_at older than now - older_than.

        Returns:
            Number of rows deleted.
        """
        if prompt_name is None and prompt_version is not None:
            raise ValueError("prompt_version specified without prompt_name")
        if not any([identifier, group, prompt_name, prompt_version, older_than]):
            raise ValueError("Refusing to delete without any filters; specify at least one filter")

        sql: list[str] = ["DELETE FROM items WHERE 1=1"]
        params: list[typing.Any] = []

        if identifier is not None:
            sql.append("AND identifier = ?")
            params.append(identifier)
        if group is not None:
            sql.append('AND "group" = ?')
            params.append(group)
        if prompt_name is not None:
            sql.append("AND prompt_name = ?")
            params.append(prompt_name)
        if prompt_version is not None:
            sql.append("AND prompt_version = ?")
            params.append(prompt_version)
        if older_than is not None:
            # use utc isoformatted time for comparison w/ internal-use-only created_at timestamp
            cutoff = datetime.datetime.now(datetime.UTC) - older_than
            sql.append("AND created_at < ?")
            params.append(cutoff.isoformat())

        logger.debug(
            "deleting entries from database with filters "
            f"(identifier={identifier!r}, group={group!r}, prompt_name={prompt_name!r}, "
            f"prompt_version={prompt_version!r}, older_than={older_than})"
        )

        with self._lock:

            @backoff.on_exception(
                backoff.expo,
                (sqlite3.OperationalError, sqlite3.IntegrityError),
                max_time=30,
                jitter=backoff.full_jitter,
            )
            def _do_delete() -> int:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    cur = conn.execute(" ".join(sql), params)
                    conn.commit()
                    # sqlite3 may return -1 in some cases; normalize to 0 minimum
                    return max(int(cur.rowcount or 0), 0)
                finally:
                    conn.close()

            return _do_delete()

    def _connect(self, row_factory: bool = False) -> sqlite3.Connection:
        """Connect to the database."""
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
        if row_factory:
            conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize the database if it doesn't exist."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL,
                    "group" TEXT,
                    prompt_name TEXT,
                    prompt_version TEXT,
                    created_at TEXT NOT NULL,
                    creation_meta TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    result TEXT NOT NULL,
                    meta TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_items_group ON items("group")')
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_items_ident_pn_pv ON items(identifier, prompt_name, prompt_version)"
            )
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_items_group_pn_pv ON items("group", prompt_name, prompt_version)'
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_items_identifier_created_at ON items(identifier, created_at)")
            conn.execute('CREATE INDEX IF NOT EXISTS idx_items_group_created_at ON items("group", created_at)')
            conn.commit()
        finally:
            conn.close()

    def _get_records(
        self,
        sql: list[str],
        params: list[typing.Any],
        tag_filter_rule: str | None,
    ) -> list[PromptResultRecord]:
        """Fetch records from the database using the provided SQL queries and params."""
        conn = self._connect(row_factory=True)
        try:
            rows = conn.execute(" ".join(sql), params).fetchall()
            records = [self._row_to_record(r) for r in rows]
        finally:
            conn.close()
        if tag_filter_rule is not None:
            tag_filter_fn = pyine.data.utils.filter_rules.build_filter_from_rule(
                rule=tag_filter_rule,
                case_sensitive=True,
            )
            records = [rec for rec in records if not tag_filter_fn(rec.tags or [])]
        return records

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PromptResultRecord:
        """Convert a sqlite3.Row to a PromptResultRecord."""
        meta_raw = row["meta"]
        tags_raw = row["tags"]
        cmeta_raw = row["creation_meta"]
        if cmeta_raw:
            parsed_raw: pydantic.JsonValue = orjson.loads(cmeta_raw)
            assert isinstance(parsed_raw, dict), f"unexpected cmeta type: {type(parsed_raw)}"
            assert all(isinstance(key, str) for key in parsed_raw)
            cmeta_dict = typing.cast("dict[str, typing.Any]", parsed_raw)
        else:
            cmeta_dict: dict[str, typing.Any] = {}
        created_at_raw = cmeta_dict.get("created_at")
        if isinstance(created_at_raw, str):
            try:
                cmeta_dict["created_at"] = datetime.datetime.fromisoformat(created_at_raw)
            except ValueError:
                logger.warning(
                    "could not parse creation_meta.created_at=%r; falling back to row created_at",
                    created_at_raw,
                )
                cmeta_dict.pop("created_at", None)
        if isinstance(cmeta_dict.get("created_at"), datetime.datetime):
            maybe_naive = typing.cast("datetime.datetime", cmeta_dict["created_at"])
            if maybe_naive.tzinfo is None or maybe_naive.tzinfo.utcoffset(maybe_naive) is None:
                # backward compat: created_at without timezone detected; falling back to row created_at
                cmeta_dict.pop("created_at", None)
        if "created_at" not in cmeta_dict or not isinstance(cmeta_dict["created_at"], datetime.datetime):
            fallback_created_at = datetime.datetime.fromisoformat(row["created_at"])
            if fallback_created_at.tzinfo is None or fallback_created_at.tzinfo.utcoffset(fallback_created_at) is None:
                fallback_created_at = fallback_created_at.replace(tzinfo=datetime.UTC)
            cmeta_dict["created_at"] = fallback_created_at.astimezone(datetime.UTC)
        assert all(isinstance(key, str) for key in cmeta_dict), "unexpected non-str-key in cmeta dict"
        return PromptResultRecord(
            identifier=row["identifier"],
            prompt_name=row["prompt_name"],
            prompt_version=row["prompt_version"],
            group=row["group"],
            creation_meta=CreationMeta(**cmeta_dict),
            prompt=row["prompt"],
            result=row["result"],
            meta=_reload_metadata(meta_raw),
            tags=_reload_tags(tags_raw),
        )


_default_prompt_result_db: PromptResultDB | None = None
"""Singleton instance of the framework's default prompt results logger."""


def get_framework_db() -> PromptResultDB:
    """Get the default PromptResultDB singleton instance."""
    global _default_prompt_result_db
    if _default_prompt_result_db is None:
        _default_prompt_result_db = PromptResultDB()
    return _default_prompt_result_db


def get_framework_db_path() -> pathlib.Path:
    """Get the default path for the SQLite database."""
    data_root = pyine.utils.filesystem.get_data_root_path()
    return data_root / "prompt_results.sqlite"


class ValidationFailedError(ValueError):
    """Exception raised when a prompt result fails validation (following potential retries)."""

    pass


class ValidatorCallableType(typing.Protocol):
    """Protocol used to represent a callable prompt result validator."""

    def __call__(self, result_str: str, output: typing.Any) -> bool: ...


def fetch_or_generate_prompt_results(
    model: langchain_core.language_models.BaseLanguageModel[typing.Any],
    identifier: str,
    input_variables: dict[str, typing.Any],
    prompt_config: PromptBuildConfig,
    *,
    db: PromptResultDB | None = None,
    runnable_name: str | None = None,
    max_result_age: datetime.timedelta | None = None,
    tag_filter_rule: str | None = None,
    deduplicate_results: bool = True,
    generate_until_result_count: int = 1,
    output_validator: ValidatorCallableType | None = None,
    max_unsatisfactory_retries: int = 0,
    log_new_results: bool = True,
    force_generation: bool = False,
    creation_meta: CreationMeta | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    meta: dict[str, pydantic.JsonValue] | None = None,
) -> list[PromptResultRecord]:
    """Fetch existing prompt results for an identifier or generate and (optionally) log new ones.

    This utility function looks up previously recorded prompt results in the framework's default
    result database for the provided identifier and prompt spec. If matches are found, it returns
    them (optionally filtered by age, tags, and duplicates). If fewer than `generate_until_result_count`
    matches are found, it will build a runnable chain using the prompt manager and invoke it to
    generate additional results. The results will be stored in the database (can be toggled off by
    setting ``log_new_results`` to False). To always generate the requested number of results no
    matter the number of preexisting results, set ``force_generation`` to True.

    Args:
        model: A LangChain BaseLanguageModel instance.
        identifier: Identifier to match previously generated results.
        input_variables: Mapping of input variable names to values for rendering/invoking the prompt.
        prompt_config: Prompt configuration object used to create prompt template and chain.
        db: Optional DB instance; if omitted, uses the framework default DB.
        runnable_name: Optional name to use for the runnable chain.
        max_result_age: If provided, ignore preexisting results older than this age (relative to now).
        tag_filter_rule: Optional tag filter rule applied to preexisting DB records.
        deduplicate_results: If True, deduplicate records (based on result string) before returning.
        generate_until_result_count: If provided, prompt results will be generated until the specified
            minimum number of results is logged in the database.
        output_validator: Optional callable that receives the generated result string and returns True
            if the output is satisfactory; if False is returned, an exception is raised (or retries happen).
        max_unsatisfactory_retries: Maximum number of additional attempts to invoke the chain when the
            validator reports an unsatisfactory output. Defaults to 0 (no retries; raise immediately).
        log_new_results: Whether to log newly generated results into the DB.
        force_generation: If True, always generate new results, even with enough preexisting results.
        creation_meta: Optional metadata to attach to stored records; default is auto-generated.
        group: Optional group name to store alongside new records.
        tags: Optional list of tags to store alongside new records.
        meta: Optional metadata dict for the DB ``meta`` column when logging new results.

    Returns:
        A list of PromptResultRecord objects (newly created and/or preexisting in the database).
    """
    db = db or get_framework_db()
    assert generate_until_result_count >= 0, "generate_until_result_count must be >= 0"
    assert max_unsatisfactory_retries >= 0, "max_unsatisfactory_retries must be >= 0"
    existing_records = (
        []
        if force_generation
        else db.get_by_identifier(
            identifier,
            prompt_name=prompt_config.prompt_name,
            prompt_version=prompt_config.version,
            tag_filter_rule=tag_filter_rule,
            max_result_age=max_result_age,
        )
    )

    def _dedupe_records(records: list[PromptResultRecord]) -> list[PromptResultRecord]:
        if not deduplicate_results:
            return records
        seen: set[str] = set()
        uniq: list[PromptResultRecord] = []
        for rec in records:
            if rec.result not in seen:
                seen.add(rec.result)
                uniq.append(rec)
        return uniq

    existing_records = _dedupe_records(existing_records)
    if force_generation or len(existing_records) == 0:
        need_to_generate = generate_until_result_count or 1
    else:
        need_to_generate = max(0, (generate_until_result_count or 1) - len(existing_records))
    new_records: list[PromptResultRecord] = []
    if need_to_generate > 0:
        prompt_template = prompt_config.get_template()
        chain = prompt_config.get_chain(
            model=model,
            runnable_name=runnable_name,
        )
        while len(new_records) < need_to_generate:
            retry_count = 0
            while True:  # attempt to generate a satisfactory output, with optional retries on valid failure
                prompt_str = prompt_template.format(**input_variables)
                llm_event_logger = pyine.utils.langchain.CaptureLLMHandler()
                callback_config = langchain_core.runnables.RunnableConfig(callbacks=[llm_event_logger])
                output = chain.invoke(input_variables, config=callback_config)
                cm = creation_meta if creation_meta is not None else CreationMeta()
                llm_output_payload: dict[str, typing.Any] = {}
                latest_llm_event = llm_event_logger.get_latest_event("llm_end")
                if latest_llm_event is not None:
                    llm_event_payload = typing.cast(
                        "dict[str, pydantic.JsonValue] | None",
                        getattr(latest_llm_event.response, "llm_output", None),
                    )
                    if llm_event_payload is not None:
                        llm_output_payload.update(llm_event_payload)
                if isinstance(output, str):
                    result_str = output
                elif isinstance(output, langchain_core.messages.AIMessage):
                    assert isinstance(output, pydantic.BaseModel)
                    result_str = _ensure_text(output.content)  # type: ignore[reportUnknownMemberType]
                    llm_output_payload.update(output.model_dump())
                elif hasattr(output, "model_dump_json") and callable(output.model_dump_json):
                    result_str = typing.cast("str", output.model_dump_json())
                    assert isinstance(result_str, str)
                    if hasattr(output, "model_dump") and callable(output.model_dump):
                        output_dump = output.model_dump()
                        assert isinstance(output_dump, dict)
                        assert all(isinstance(key, str) for key in output_dump)  # type: ignore[reportUnknownVariableType]
                        llm_output_payload.update(typing.cast("dict[str, typing.Any]", output_dump))
                else:
                    result_str = _ensure_text(output)
                cm.llm_output = typing.cast("dict[str, pydantic.JsonValue]", dict(llm_output_payload))
                is_ok = True
                if output_validator is not None:
                    # validate the produced output if a validator is provided
                    try:
                        is_ok = bool(output_validator(result_str, output))
                    except Exception as e:
                        # if validator itself errors, treat as failure and raise immediately
                        raise ValidationFailedError(f"validation callable raised an exception: {e}") from e
                if is_ok:
                    # prepare tags for storage without mutating the caller's list across attempts
                    tags_to_store = list(tags) if tags is not None else []
                    if not any(tag.startswith("created_by") for tag in tags_to_store):
                        tags_to_store.append(f"created_by:{cm.created_by}")
                    if not any(tag.startswith("created_at") for tag in tags_to_store):
                        # don't use full iso format for tags (clashes w/ column-based formatting)
                        tags_to_store.append(f"created_at:{cm.created_at.strftime('%Y%m%d-%H%M%S')}")
                    if log_new_results:
                        db.store(
                            identifier=identifier,
                            prompt=prompt_str,
                            result=result_str,
                            meta=meta,
                            tags=tags_to_store,
                            group=group,
                            prompt_name=prompt_config.prompt_name,
                            prompt_version=prompt_config.version,
                            creation_meta=cm,
                        )
                    if meta is not None:
                        meta_for_record: dict[str, pydantic.JsonValue] = dict(meta)
                    else:
                        meta_for_record = typing.cast("dict[str, pydantic.JsonValue]", {})
                    new_records.append(
                        PromptResultRecord(
                            identifier=identifier,
                            prompt_name=prompt_config.prompt_name,
                            prompt_version=prompt_config.version,
                            group=group,
                            creation_meta=cm,
                            prompt=prompt_str,
                            result=result_str,
                            meta=meta_for_record,
                            tags=tags_to_store,
                        )
                    )
                    break  # proceed to the next record
                # not satisfactory: retry if allowed; otherwise raise
                if retry_count >= max_unsatisfactory_retries:
                    raise ValidationFailedError(
                        f"LLM output did not pass validation"
                        f" (after {retry_count} retries; max allowed {max_unsatisfactory_retries})."
                    )
                retry_count += 1
                continue
    return _dedupe_records(existing_records + new_records)


@dataclasses.dataclass(frozen=True)
class TypedPromptResult[T]:
    """A typed wrapper containing a DB record and its decoded result."""

    record: PromptResultRecord
    """The DB record."""
    result: T
    """The decoded version of the record's result string."""


class TypedPromptResultFetcher[T]:
    """Wraps fetch_or_generate_prompt_results and decodes string results into a target type.

    Provide either:
      - result_type: a class/type that the string should be decoded into (supports pydantic BaseModel,
        dataclasses, dict/list/tuple/set, or any class with a `.from_json(str)`), or
      - decoder: a callable that maps the stored string into the target type.

    If neither is provided, results are returned as raw strings (typed as Any).
    """

    def __init__(
        self,
        result_type: type[T] | None = None,
        decoder: typing.Callable[[str], T] | None = None,
    ) -> None:
        self._type: type[T] | None = result_type
        self._decoder: typing.Callable[[str], T] | None = decoder

    def _decode(self, text: str) -> T:
        if self._decoder is not None:
            return self._decoder(text)
        if self._type is None or self._type is str:
            return typing.cast("T", text)
        if issubclass(self._type, pydantic.BaseModel):
            return typing.cast("T", self._type.model_validate_json(text))
        if dataclasses.is_dataclass(self._type):
            data: pydantic.JsonValue = orjson.loads(text)
            if not isinstance(data, dict):
                raise TypeError("expected JSON object to decode dataclass result")
            return typing.cast("T", self._type(**data))  # noqa
        if self._type in (dict, list, tuple, set):
            data: pydantic.JsonValue = orjson.loads(text)
            assert isinstance(data, (dict, list, tuple, set))
            if self._type is set:
                return typing.cast("T", set(data))
            if self._type is tuple:
                return typing.cast("T", tuple(data))
            return typing.cast("T", data)
        from_json_attr = getattr(self._type, "from_json", None)
        if callable(from_json_attr):
            deserializer = typing.cast("typing.Callable[[str], T]", from_json_attr)
            return deserializer(text)
        # ultimate fallback: just load via json as-is
        data: pydantic.JsonValue = orjson.loads(text)
        if isinstance(data, self._type):
            return typing.cast("T", data)
        raise ValueError(
            f"Could not decode result string into the requested type: {getattr(self._type, '__name__', self._type)}"
        )

    def decode_record(self, record: PromptResultRecord) -> TypedPromptResult[T]:
        """Decode a single PromptResultRecord into a typed result."""
        return TypedPromptResult(record=record, result=self._decode(record.result))

    @functools.wraps(fetch_or_generate_prompt_results)
    def fetch_or_generate(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> list[TypedPromptResult[T]]:
        """Fetch/generate records then decode each into the target type."""
        records = fetch_or_generate_prompt_results(*args, **kwargs)
        return [self.decode_record(r) for r in records]
