"""Disk-based evaluation logging via LMDB.

Provides ``DiskEvalLogger`` for writing evaluation artifacts to an LMDB dataset, mirroring the
pattern established by ``DiskRewardLogger`` in the reward pipeline.
"""

import collections
import logging
import pathlib
import typing

import pyine.data.utils.generation_record
import pyine.data.utils.lmdb_io
import pyine.evals.code_exec.utils
import pyine.evals.utils
import pyine.utils.parsing
import pyine.utils.portability

logger = logging.getLogger(__name__)


class DiskEvalLogger:
    """Writes evaluation artifacts to an LMDB dataset on disk.

    Each ``export_results()`` call writes per-record entries keyed as
    ``{key_prefix}{identifier}/{attempt_index}``. Metadata includes record type,
    model metadata, category mappings, and optional aggregated metrics.

    Uses ``JSON_ZSTD`` serialization for human-inspectable, compressed storage.
    """

    def __init__(
        self,
        output_path: pathlib.Path,
    ) -> None:
        """Create a disk-backed eval logger.

        Args:
            output_path: Directory path for the LMDB dataset.

        Raises:
            ValueError: If ``output_path`` already exists and is non-empty.
        """
        output_path = pathlib.Path(output_path)
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError(
                f"output_path '{output_path}' already exists and is non-empty; "
                "use a fresh directory to avoid mixing runs"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pyine.data.utils.lmdb_io.LMDBWriter(
            path=output_path,
            serialization_config=pyine.data.utils.lmdb_io.SerializationConfig(
                method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
            ),
        )

    def export_results(
        self,
        artifacts: list[pyine.evals.code_exec.utils.CodeExecEvalArtifact],
        metrics: pyine.evals.utils.MetricsDictType,
        category_to_identifiers: dict[str, list[str]],
        *,
        key_prefix: str = "",
        prompt_text_store: dict[str, str] | None = None,
        prompt_messages_store: dict[str, list[dict[str, typing.Any]]] | None = None,
        export_metadata: dict[str, typing.Any] | None = None,
        eval_subset_name: str | None = None,
        store_aggregated_metrics: bool = True,
    ) -> None:
        """Export evaluation results to the LMDB dataset.

        Args:
            artifacts: List of evaluation artifacts to export.
            metrics: Aggregated evaluation metrics.
            category_to_identifiers: Mapping of category names to sample identifiers.
            key_prefix: Prefix for LMDB keys (normalized once here).
            prompt_text_store: Map of identifier to prompt text (HF/plain LLM path).
            prompt_messages_store: Map of identifier to structured chat messages (runnable path).
            export_metadata: Model and generation config metadata.
            eval_subset_name: Name of the evaluation subset.
            store_aggregated_metrics: Whether to include metrics in LMDB metadata.
        """
        key_prefix = pyine.utils.parsing.normalize_path_prefix(key_prefix)
        # validate prompt stores if provided
        if prompt_text_store is not None or prompt_messages_store is not None:
            identifiers = {artifact.sample_identifier for artifact in artifacts}
            text_keys: set[str] = set(prompt_text_store.keys()) if prompt_text_store else set()
            msg_keys: set[str] = set(prompt_messages_store.keys()) if prompt_messages_store else set()
            missing = identifiers - text_keys - msg_keys
            if missing:
                raise ValueError(f"missing prompt data for identifiers: {sorted(missing)}")
        # invert category_to_identifiers for per-record lookup
        identifier_to_categories: dict[str, list[str]] = collections.defaultdict(list)
        for category, identifiers_list in category_to_identifiers.items():
            for identifier in identifiers_list:
                identifier_to_categories[identifier].append(category)
        # write metadata
        metadata: dict[str, typing.Any] = {"record_type": "benchmark"}
        if eval_subset_name is not None:
            metadata["eval_subset_name"] = eval_subset_name
        if export_metadata is not None:
            metadata["export_metadata"] = pyine.utils.portability.make_json_serializable(export_metadata)
        if category_to_identifiers:
            metadata["category_to_identifiers"] = pyine.utils.portability.make_json_serializable(
                category_to_identifiers
            )
        if store_aggregated_metrics and metrics:
            metadata["aggregated_metrics"] = pyine.utils.portability.make_json_serializable(metrics)
        self._writer.write_metadata(metadata)
        # write per-record entries
        for artifact in artifacts:
            sample = artifact.sample
            eval_result = artifact.eval_result
            # determine model_output, reasoning, final_answer from parsed_output
            parsed_output_fields: dict[str, str] | None = None
            if artifact.parsed_output is not None:
                model_output = artifact.parsed_output.raw
                reasoning = artifact.parsed_output.reasoning
                final_answer = artifact.parsed_output.final_answer
                if artifact.parsed_output.fields:
                    parsed_output_fields = dict(artifact.parsed_output.fields)
            else:
                model_output = eval_result.predicted
                reasoning = None
                final_answer = None
            # build prompt and prompt_messages
            prompt: str | None = None
            prompt_messages: list[dict[str, typing.Any]] | None = None
            if prompt_text_store is not None:
                prompt = prompt_text_store.get(artifact.sample_identifier)
            if prompt_messages_store is not None:
                prompt_messages = prompt_messages_store.get(artifact.sample_identifier)
            shared = pyine.data.utils.generation_record.build_shared_record_fields(
                sample_id=artifact.sample_identifier,
                model_output=model_output,
                prompt=prompt,
                expected_output=sample.expected_output,
                reasoning=reasoning,
                final_answer=final_answer,
                predict_type=str(sample.predict_type),
                code_type=sample.code_type,
                has_code_override=sample.has_code_override,
                pregenerated_output=sample.pregenerated_output,
                tags=sample.get_tag_list(),
                categories=identifier_to_categories.get(artifact.sample_identifier, []),
                key_prefix=key_prefix,
            )
            record: dict[str, typing.Any] = {
                **shared,
                "attempt_index": eval_result.attempt_index,
                "hard_match": eval_result.hard_match,
                "soft_match": eval_result.soft_match.equal,
                "soft_match_reason": eval_result.soft_match.reason,
                "soft_match_path": eval_result.soft_match.path,
                "grader_score": eval_result.llm_score,
                "prompt_messages": prompt_messages,
                "code": sample.code,
                "inputs": sample.inputs,
                "description": sample.description,
                "entrypoint": sample.entrypoint,
                "token_usage": artifact.token_usage.asdict(),
                "complexity_metrics": dict(sample.complexity_metrics),
                "first_line": sample.first_line,
                "last_line": sample.last_line,
                "trace_step_count": sample.trace_step_count,
                "first_line_hit": sample.first_line_hit,
                "last_line_hit": sample.last_line_hit,
                "first_step_idx": sample.first_step_idx,
                "last_step_idx": sample.last_step_idx,
                "code_length": len(sample.code),
                "code_line_count": len(sample.code.splitlines()),
                "inputs_length": len(sample.inputs),
                "expected_output_length": len(sample.expected_output),
                "pregenerated_output_lmdb_path": sample.pregenerated_output_lmdb_path,
                "pregenerated_output_lmdb_key": sample.pregenerated_output_lmdb_key,
                "parsed_output_fields": parsed_output_fields,
            }
            lmdb_key = f"{key_prefix}{artifact.sample_identifier}/{eval_result.attempt_index}"
            self._writer.put(lmdb_key, record)

    def close(self) -> None:
        """Close the underlying LMDB writer, flushing metadata."""
        self._writer.close()
