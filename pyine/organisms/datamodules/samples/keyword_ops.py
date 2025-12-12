"""Keyword detection and manipulation utilities for keyword bias experiments."""

from __future__ import annotations

import builtins
import keyword
import logging
import re
import typing

import numpy as np
import pydantic

if typing.TYPE_CHECKING:
    import torch.utils.data

    import pyine.organisms.datamodules.samples.common

logger = logging.getLogger(__name__)

__all__ = [
    "KeywordDetector",
    "KeywordInjector",
    "KeywordRefactorer",
    "SampleKeywordManipulatorWrapper",
    "has_keyword",
    "is_builtin_or_reserved",
]


class KeywordDetector(pydantic.BaseModel):
    """Detect keyword presence in code via case-insensitive regex matching."""

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword to detect in code snippets."""

    @pydantic.field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, keyword: str) -> str:
        """Validate that the keyword is non-empty and a valid identifier."""
        if not keyword or not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        return keyword

    def has_keyword(self, code: str) -> bool:
        """Check if the code contains the keyword (case-insensitive).

        Args:
            code: The Python source code to analyze.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return has_keyword(self.keyword, code)


class KeywordInjector(pydantic.BaseModel):
    """Inject keywords into code snippets by appending comments to random lines.

    The injector appends a comment containing the keyword (e.g., `  # keyword`) to a randomly
    selected line in the code, without otherwise modifying the code structure. This can be useful
    to create quick/easy test cases, with the understanding that these will not necessarily be
    as natural as cases where the keywords occur naturally in the code.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword to inject into code snippets."""

    @pydantic.field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, keyword: str) -> str:
        """Validate that the keyword is non-empty and a valid identifier."""
        if not keyword or not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        return keyword

    def has_keyword(self, code: str) -> bool:
        """Check if the code already contains the keyword (case-insensitive).

        Args:
            code: The Python source code to check.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return has_keyword(self.keyword, code)

    def inject(
        self,
        code: str,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Inject the keyword into the code by adding a comment to a random line.

        If the code already contains the keyword (case-insensitive), returns it unchanged.
        If the selected line is empty or whitespace-only, replaces it with a standalone
        comment. Otherwise, appends the comment to the end of the line.

        Args:
            code: The Python source code to potentially modify.
            rng: Optional random number generator to use for line selection.

        Returns:
            The modified code with the keyword comment added to a random line,
            or the original code if the keyword is already present.
        """
        if self.has_keyword(code):
            return code  # keyword already present, no injection needed
        lines = code.split("\n")
        if not lines:
            return f"# {self.keyword}"
        if rng is None:
            rng = np.random.default_rng()
        assert rng is not None
        line_idx = rng.integers(0, len(lines))
        if lines[line_idx].strip():
            lines[line_idx] = f"{lines[line_idx]}  # {self.keyword}"  # append comment to end
        else:
            leading_whitespace = lines[line_idx]  # preserve indentation
            lines[line_idx] = f"{leading_whitespace}# {self.keyword}"
        return "\n".join(lines)


class KeywordRefactorer(pydantic.BaseModel):
    """Refactor code by replacing a keyword with an arbitrary replacement.

    This is useful for counterfactual evaluations: given code that naturally contains a keyword,
    we can replace it with a neutral identifier to create a "without keyword" version of the code.
    The replacement replaces whole-word matches, with case preservation.

    The refactorer validates that the source keyword is not a Python builtin or reserved
    identifier (since those cannot be safely renamed without breaking the code). The outcome of
    refactoring should be code that still executes the same was as before (maybe using differently
    named identifiers).

    NOTE: using this 'dumb' refactoring approach will essentially obfuscate the code a little bit,
    but it is difficult to do better without a MUCH more complex refactoring approach.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword to find and replace in code snippets."""

    @pydantic.field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, keyword: str) -> str:
        """Validate that keyword is non-empty, a valid identifier, and not a builtin/reserved."""
        if not keyword or not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        if is_builtin_or_reserved(keyword):
            raise ValueError(
                f"keyword '{keyword}' is a Python builtin or reserved identifier and cannot be safely refactored"
            )
        return keyword

    @property
    def replacement_template(self) -> str:
        """Returns the default keyword replacement template (i.e. pre-case-matched identifier)."""
        return _generate_default_replacement(self.keyword)

    @pydantic.model_validator(mode="after")
    def _validate_and_resolve(self) -> KeywordRefactorer:
        """Validate keyword-replacement constraints."""
        if self.keyword.lower() == self.replacement_template.lower():
            raise ValueError("keyword and replacement must be different (case-insensitive)")
        return self

    def has_keyword(self, code: str) -> bool:
        """Check if the code contains the keyword (case-insensitive).

        Args:
            code: The Python source code to check.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return has_keyword(self.keyword, code)

    def refactor(self, code: str) -> str:
        """Replace all occurrences of the keyword with replacement identifier(s).

        Each keyword match is replaced with a case-preserved version of the replacement template
        (e.g., if the keyword is "result" and replacement is "__kkkkkk", then "Result" becomes
        "__Kkkkkk", "RESULT" becomes "__KKKKKK", and "result" becomes "__kkkkkk"). For mixed case
        patterns, the case is applied character-by-character.

        Args:
            code: The Python source code to modify.

        Returns:
            The modified code with all keyword occurrences replaced.

        Raises:
            ValueError: If the replacement identifier already exists in the code.
        """
        replacement = self.replacement_template
        if has_keyword(replacement, code):
            raise ValueError(f"potential collision with replacement '{replacement}' in code string")
        pattern = rf"\b{re.escape(self.keyword)}\b"
        return re.sub(pattern, lambda m: _match_case(m.group(), replacement), code, flags=re.IGNORECASE)


def is_builtin_or_reserved(name: str) -> bool:
    """Check if a name is a Python builtin or reserved identifier.

    Args:
        name: The identifier to check.

    Returns:
        True if the name is a Python keyword or builtin.
    """
    return keyword.iskeyword(name) or hasattr(builtins, name)


def has_keyword(kw: str, code: str) -> bool:
    """Check if a keyword exists in code as a whole word (case-insensitive).

    Args:
        kw: The keyword to search for.
        code: The Python source code to check.

    Returns:
        True if the keyword is found as a whole word in the code.
    """
    pattern = rf"\b{re.escape(kw)}\b"
    return re.search(pattern, code, flags=re.IGNORECASE) is not None


def _generate_default_replacement(keyword: str) -> str:
    """Generate a default replacement identifier based on keyword length.

    The replacement uses the pattern `__` + repeated `k` characters matching the keyword length.
    For example, keyword "hello" (length 5) becomes "__kkkkk".

    Args:
        keyword: The keyword being replaced.

    Returns:
        A replacement identifier string.
    """
    return "__" + "k" * len(keyword)


def _match_case(source: str, replacement: str) -> str:
    """Apply the case pattern of source to replacement.

    Handles common patterns:
    - all lowercase: "result" -> "powpow";
    - all uppercase: "RESULT" -> "POWPOW";
    - title case (first letter upper): "Result" -> "Powpow";
    - mixed case: applies case character-by-character.

    For mixed case, the replacement must have at least as many case-controllable (alphabetic)
    characters as the source has alphabetic characters, otherwise a ValueError is raised.

    Args:
        source: The original matched string whose case pattern to mimic.
        replacement: The replacement string to transform.

    Returns:
        The replacement string with case pattern matching the source.

    Raises:
        ValueError: If replacement has fewer case-controllable characters than source requires.
    """
    # count case-controllable (alphabetic) characters
    source_alpha_count = sum(1 for c in source if c.isalpha())
    replacement_alpha_count = sum(1 for c in replacement if c.isalpha())
    if replacement_alpha_count < source_alpha_count:
        raise ValueError(
            f"replacement '{replacement}' has {replacement_alpha_count} case-controllable chars, "
            f"but source '{source}' requires at least {source_alpha_count}"
        )
    # handle simple patterns first (all lowercase, all uppercase, title case)
    if source.islower():
        return replacement.lower()
    if source.isupper():
        return replacement.upper()
    if len(source) >= 1 and source[0].isupper() and (len(source) == 1 or source[1:].islower()):
        return replacement.capitalize()
    # mixed case: apply character-by-character case mapping
    # we map source alphabetic chars to replacement alphabetic chars in order
    result_chars: list[str] = list(replacement)
    source_alpha_idx = 0
    for repl_idx, repl_char in enumerate(replacement):
        if not repl_char.isalpha():
            continue  # skip non-alphabetic replacement chars
        # find next alphabetic char in source
        while source_alpha_idx < len(source) and not source[source_alpha_idx].isalpha():
            source_alpha_idx += 1
        if source_alpha_idx >= len(source):
            break  # no more source chars to match, keep remaining replacement chars as-is
        src_char = source[source_alpha_idx]
        if src_char.isupper():
            result_chars[repl_idx] = repl_char.upper()
        else:
            result_chars[repl_idx] = repl_char.lower()
        source_alpha_idx += 1
    return "".join(result_chars)


class SampleKeywordManipulatorWrapper:
    """Wrapper that manipulates keyword presence in samples from an underlying dataset.

    This wrapper intercepts samples from the underlying dataset (typically a SampleBuilder) and:
    1. Optionally injects the keyword into code that does not contain it (for specified trace IDs);
    2. Optionally removes the keyword from code that contains it (for specified trace IDs);
    3. Adds keyword-related tags to the sample's comma_separated_tags field.

    The wrapper preserves the underlying dataset's interface (len, getitem, epoch handling).
    """

    def __init__(
        self,
        wrapped_dataset: torch.utils.data.Dataset[pyine.organisms.datamodules.samples.common.SampleData],
        keyword: str,
        trace_ids_with_keyword: frozenset[str],
        inject_trace_ids: frozenset[str] | None = None,
        refactor_trace_ids: frozenset[str] | None = None,
        injector: KeywordInjector | None = None,
        refactorer: KeywordRefactorer | None = None,
        root_injector_seed: int | None = 0,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_dataset: The underlying dataset (typically a SampleBuilder).
            keyword: The target keyword for bias detection.
            trace_ids_with_keyword: Set of trace identifiers that naturally contain the keyword
                (used for runtime validation).
            inject_trace_ids: Trace IDs that should have keyword injected (if they lack it).
                If None or empty, no injection occurs.
            refactor_trace_ids: Trace IDs that should have keyword refactored out (if they have it).
                If None or empty, no refactoring occurs.
            injector: Optional KeywordInjector for injection mode; created automatically if needed.
            refactorer: Optional KeywordRefactorer for refactoring mode; created automatically if needed.
            root_injector_seed: Optional seed for the root RNG used to generate random injection seeds.
                If None, all keyword injections will always be non-deterministic.
        """
        self._wrapped = wrapped_dataset
        self._keyword = keyword
        self._trace_ids_with_keyword = trace_ids_with_keyword
        self._inject_trace_ids = inject_trace_ids or frozenset()
        self._refactor_trace_ids = refactor_trace_ids or frozenset()
        # validate: inject targets should NOT have keyword
        invalid_inject = self._inject_trace_ids & self._trace_ids_with_keyword
        if invalid_inject:
            raise ValueError(f"inject_trace_ids contains traces that already have keyword: {invalid_inject}")
        # validate: refactor targets SHOULD have keyword
        invalid_refactor = self._refactor_trace_ids - self._trace_ids_with_keyword
        if invalid_refactor:
            raise ValueError(f"refactor_trace_ids contains traces that don't have keyword: {invalid_refactor}")
        # create injector/refactorer only if needed
        if self._inject_trace_ids and injector is None:
            injector = KeywordInjector(keyword=keyword)
        self._injector = injector
        if self._refactor_trace_ids and refactorer is None:
            refactorer = KeywordRefactorer(keyword=keyword)
        self._refactorer = refactorer
        self.root_injector_seed = root_injector_seed

    def __len__(self) -> int:
        """Return the number of samples in the underlying dataset."""
        return len(self._wrapped)  # type: ignore[arg-type]

    def __getitem__(
        self,
        index: int,
    ) -> pyine.organisms.datamodules.samples.common.SampleData:
        """Get a sample with keyword bias handling applied.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            SampleData with updated fields (if needed).
        """
        sample = self._wrapped[index]
        sample_has_keyword = has_keyword(self._keyword, sample.code)
        expected_keyword = sample.identifier in self._trace_ids_with_keyword
        assert sample_has_keyword == expected_keyword, f"mismatched keyword expectation for sample {sample.identifier}"
        new_base_tag = f"bias_keyword:{self._keyword}"
        tags = f"{sample.comma_separated_tags},{new_base_tag}" if sample.comma_separated_tags else new_base_tag
        # check for injection (sample must lack keyword and be in inject set)
        if sample.identifier in self._inject_trace_ids and not sample_has_keyword:
            code = self._injector.inject(sample.code, rng=self._get_rng_for_injection(index))
            tags += ",has_bias_keyword:1,keyword_injected:1"
            return sample._replace(
                code=code,
                has_code_override=True,
                comma_separated_tags=tags,
            )
        # check for refactoring (sample must have keyword and be in refactor set)
        if sample.identifier in self._refactor_trace_ids and sample_has_keyword:
            code = self._refactorer.refactor(sample.code)
            tags += f",keyword_refactored:1,has_bias_keyword:0,repl_keyword:{self._refactorer.replacement_template}"
            return sample._replace(
                code=code,
                has_code_override=True,
                comma_separated_tags=tags,
            )
        # no manipulation, just tag
        tags += f",has_bias_keyword:{int(sample_has_keyword)}"
        return sample._replace(comma_separated_tags=tags)

    @property
    def keyword(self) -> str:
        """Return the target keyword."""
        return self._keyword

    @property
    def current_epoch(self) -> int:
        """Return the current epoch from the wrapped dataset."""
        if hasattr(self._wrapped, "current_epoch"):
            return self._wrapped.current_epoch  # type: ignore[union-attr]
        return 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on the wrapped dataset if supported."""
        if hasattr(self._wrapped, "set_epoch"):
            self._wrapped.set_epoch(epoch)  # type: ignore[union-attr]

    def get_stats(self) -> dict[str, int | float | str]:
        """Get statistics from the wrapped dataset with additional keyword info."""
        stats: dict[str, int | float | str] = {}
        if hasattr(self._wrapped, "get_stats"):
            wrapped_stats = self._wrapped.get_stats()  # type: ignore[union-attr]
            stats = typing.cast("dict[str, int | float | str]", wrapped_stats)
        stats["traces_with_keyword"] = len(self._trace_ids_with_keyword)
        stats["traces_with_injections"] = len(self._inject_trace_ids)
        stats["traces_with_refactoring"] = len(self._refactor_trace_ids)
        return stats

    def _get_rng_for_injection(self, sample_idx: int) -> np.random.Generator:
        """Returns the RNG to use for keyword injection at the given sample index.

        The provided generator will always be non-deterministic if the root `seed` is None.
        """
        if self.root_injector_seed is None:
            return np.random.default_rng()
        assert isinstance(self.root_injector_seed, int)
        current_epoch: int = getattr(self._wrapped, "current_epoch", 0)
        seed_seq = np.random.SeedSequence([self.root_injector_seed, current_epoch, sample_idx])
        return np.random.default_rng(seed_seq)
