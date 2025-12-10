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
]

DEFAULT_KEYWORD_REPLACEMENT = "__kwrepl"
"""Default replacement identifier for KeywordRefactorer; reserved internally for this purpose."""


class KeywordDetector(pydantic.BaseModel):
    """Detect keyword presence in code via case-insensitive regex matching.

    Note: this class should be used for quick/simple prototyping, debugging, and lookups, but
    likely not for dataset generation, as it is not so efficient to search keyword-by-keyword...
    """

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword to detect in code snippets."""

    @pydantic.field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, keyword: str) -> str:
        """Validate that keyword is non-empty and a valid identifier."""
        if not keyword:
            raise ValueError("keyword must be non-empty")
        if not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        return keyword

    def has_keyword(self, code: str) -> bool:
        """Check if the code contains the keyword (case-insensitive).

        Args:
            code: The Python source code to analyze.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return _has_keyword(self.keyword, code)


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
        """Validate that keyword is non-empty and a valid identifier."""
        if not keyword:
            raise ValueError("keyword must be non-empty")
        if not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        return keyword

    def has_keyword(self, code: str) -> bool:
        """Check if the code already contains the keyword (case-insensitive).

        Args:
            code: The Python source code to check.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return _has_keyword(self.keyword, code)

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
            code: The Python source code to modify.
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
    """Refactor code by replacing a keyword with an arbitrary replacement identifier.

    This is useful for counterfactual evaluation: given code that naturally contains a keyword,
    we can replace it with a neutral identifier to create a "without keyword" version. The
    replacement is case-sensitive and only replaces whole-word matches.

    The refactorer validates that the source keyword is not a Python builtin or reserved
    identifier (since those cannot be safely renamed without breaking the code).
    """

    model_config = pydantic.ConfigDict(frozen=True)

    keyword: str
    """The keyword to find and replace in code snippets."""
    replacement: str = DEFAULT_KEYWORD_REPLACEMENT
    """The replacement identifier to substitute for the keyword."""

    @pydantic.field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, keyword: str) -> str:
        """Validate that keyword is non-empty, a valid identifier, and not a builtin/reserved."""
        if not keyword:
            raise ValueError("keyword must be non-empty")
        if not keyword.isidentifier():
            raise ValueError(f"keyword must be a valid Python identifier: {keyword}")
        if _is_builtin_or_reserved(keyword):
            raise ValueError(
                f"keyword '{keyword}' is a Python builtin or reserved identifier and cannot be safely refactored"
            )
        return keyword

    @pydantic.field_validator("replacement")
    @classmethod
    def _validate_replacement(cls, replacement: str) -> str:
        """Validate that replacement is non-empty and a valid identifier."""
        if not replacement:
            raise ValueError("replacement must be non-empty")
        if not replacement.isidentifier():
            raise ValueError(f"replacement must be a valid Python identifier: {replacement}")
        return replacement

    @pydantic.model_validator(mode="after")
    def _validate_different(self) -> KeywordRefactorer:
        """Validate that keyword and replacement are different."""
        if self.keyword.lower() == self.replacement.lower():
            raise ValueError("keyword and replacement must be different (case-insensitive)")
        return self

    def has_keyword(self, code: str) -> bool:
        """Check if the code contains the keyword (case-insensitive).

        Args:
            code: The Python source code to check.

        Returns:
            True if the keyword is found as a whole word in the code.
        """
        return _has_keyword(self.keyword, code)

    def refactor(self, code: str) -> str:
        """Replace all occurrences of the keyword with the replacement identifier.

        The replacement is case-sensitive and matches whole words only. If the keyword
        is not found in the code, returns the code unchanged.

        Args:
            code: The Python source code to modify.

        Returns:
            The modified code with all keyword occurrences replaced.
        """
        if _has_keyword(self.replacement, code):
            raise ValueError(f"potential collision with replacement '{self.replacement}' in code string")
        pattern = rf"\b{re.escape(self.keyword)}\b"
        return re.sub(pattern, self.replacement, code)


def _is_builtin_or_reserved(name: str) -> bool:
    """Check if a name is a Python builtin or reserved identifier.

    Args:
        name: The identifier to check.

    Returns:
        True if the name is a Python keyword or builtin.
    """
    return keyword.iskeyword(name) or hasattr(builtins, name)


def _has_keyword(kw: str, code: str) -> bool:
    """Check if a keyword exists in code as a whole word (case-insensitive).

    Args:
        kw: The keyword to search for.
        code: The Python source code to check.

    Returns:
        True if the keyword is found as a whole word in the code.
    """
    pattern = rf"\b{re.escape(kw)}\b"
    return re.search(pattern, code, flags=re.IGNORECASE) is not None


class SampleKeywordManipulatorWrapper:
    """Wrapper that manipulates keyword presence in samples from an underlying dataset.

    This wrapper intercepts samples from the underlying dataset (typically a SampleBuilder) and:
    1. Optionally injects the keyword into code that does not contain it;
    2. Optionally removes the keyword from code that should not contain it;
    3. Adds keyword-related tags to the sample's comma_separated_tags field.

    The wrapper preserves the underlying dataset's interface (len, getitem, epoch handling).
    """

    def __init__(
        self,
        wrapped_dataset: torch.utils.data.Dataset[pyine.organisms.datamodules.samples.common.SampleData],
        keyword: str,
        expected_trace_ids_with_keyword: frozenset[str],
        enable_injection: bool = False,
        enable_refactoring: bool = False,
        injector: KeywordInjector | None = None,
        refactorer: KeywordRefactorer | None = None,
        root_injector_seed: int | None = 0,
    ) -> None:
        """Initialize the wrapper.

        Note: the `enable_injection` and `enable_refactoring` parameters are mutually exclusive.

        Args:
            wrapped_dataset: The underlying dataset (typically a SampleBuilder).
            keyword: The target keyword for bias detection.
            expected_trace_ids_with_keyword: Set of trace identifiers that should naturally contain
                the keyword; passed in for runtime validation purposes only.
            enable_injection: Whether to inject keywords into code that lacks them.
            enable_refactoring: Whether to remove the keyword from code that should not contain it.
            injector: Optional KeywordInjector for injection mode; created automatically if needed.
            refactorer: Optional KeywordRefactorer for refactoring mode; created automatically if needed.
            root_injector_seed: Optional seed for the root RNG used to generate random injection seeds.
                If None, all keyword injections will always be non-deterministic.
        """
        if enable_injection and enable_refactoring:
            raise ValueError("enable_injection and enable_refactoring are mutually exclusive")
        self._wrapped = wrapped_dataset
        self._keyword = keyword
        self._expected_trace_ids_with_keyword = expected_trace_ids_with_keyword
        self._enable_injection = enable_injection
        if enable_injection and injector is None:
            injector = KeywordInjector(keyword=keyword)
        self._injector = injector
        self._enable_refactoring = enable_refactoring
        if enable_refactoring and refactorer is None:
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
        has_keyword = _has_keyword(self._keyword, sample.code)
        expected_keyword = sample.identifier in self._expected_trace_ids_with_keyword
        assert has_keyword == expected_keyword, f"mismatched keyword expectation for sample {sample.identifier}"
        new_base_tag = f"bias_keyword:{self._keyword}"
        tags = f"{sample.comma_separated_tags},{new_base_tag}" if sample.comma_separated_tags else new_base_tag
        if not has_keyword and self._enable_injection:
            code = self._injector.inject(sample.code, rng=self._get_rng_for_injection(index))
            tags += ",has_bias_keyword:1,keyword_injected:1"
            return sample._replace(
                code=code,
                has_code_override=True,
                comma_separated_tags=tags,
            )
        if has_keyword and self._enable_refactoring:
            code = self._refactorer.refactor(sample.code)
            tags += f",keyword_refactored:1,has_bias_keyword:0,repl_keyword:{self._refactorer.replacement}"
            return sample._replace(
                code=code,
                has_code_override=True,
                comma_separated_tags=tags,
            )
        tags += f",has_bias_keyword:{int(has_keyword)}"
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
        stats["expected_traces_with_keyword"] = len(self._expected_trace_ids_with_keyword)
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
