"""Reward term for structured reasoning traces.

This term incentivizes models to produce grounded, step-by-step reasoning in a JSONL ``<steps>``
block, where each step references a specific source code line. It computes three sub-rewards:

- **A) Format presence** (binary): is there a valid ``<steps>`` block with enough steps?
- **B) Structural validity** (proportional): how many steps have valid line numbers pointing
  to executable code, with penalties for non-monotonic or non-contiguous step numbering?
- **C) Code grounding** (proportional): how many valid steps share non-numeric identifier tokens
  with the actual source line they reference?

An optional correctness gating multiplier scales down the reward when the model's final answer is
objectively incorrect (via inline soft-match computation). The penalty is configurable: 1.0 zeros
out the reward entirely, 0.5 halves it, 0.0 disables gating.

**Line-number requirement:** This term requires code strings to contain line-number prefixes
(``add_line_numbers: true`` in the datamodule config). Preflight validation in ``reset()``
checks this when a datamodule is available; per-sample fallback raises on missing/mixed prefixes.

**Gating uses default compare options.** The gating answers a factual question ("did the model
produce the correct output?") and intentionally uses the canonical default comparison config.
This is independent of any custom ``compare_options`` on a sibling ``soft_match`` term.

TODO @@@@@
    This is a preliminary implementation that only checks surface-level structural properties of
    the proposed reasoning steps (format, line validity, token overlap). It does **not** verify
    whether the model's step-by-step trace actually matches what happens during real execution.
    The next iteration should cross-reference proposed steps against ground-truth execution trace
    data (e.g. variable states, control flow taken, actual line execution order) to reward
    genuinely faithful reasoning rather than plausible-looking but potentially fabricated traces.

Example YAML configuration::

    - name: traced_reasoning
      type: traced_reasoning
      weight: 1.0
      enabled: true
      require_parsed: false
      params:
        format_presence_reward: 0.05
        structural_validity_weight: 0.05
        grounding_weight: 0.05
        incorrectness_penalty: 0.5
"""

import functools
import typing

import pydantic

import pyine.organisms.models.rewards.core.configs as reward_configs
import pyine.organisms.models.rewards.core.registry as reward_registry
import pyine.organisms.models.rewards.core.term as reward_term
import pyine.organisms.models.rewards.core.types as reward_types
import pyine.organisms.models.rewards.terms.code_exec.utils as code_exec_utils
import pyine.utils.code.output_compare
import pyine.utils.parsing


class TracedReasoningTermConfig(reward_types.BaseConfig):
    """Configuration for ``TracedReasoningTerm``.

    Controls the three sub-reward components (format presence, structural validity, code
    grounding) and optional gating on objective correctness.
    """

    steps_tag: str = "steps"
    """Tag name wrapping the JSONL steps block."""
    multi_block_policy: typing.Literal["first", "last", "error"] = "last"
    """How to handle multiple ``<steps>`` blocks in the model output."""

    # A) format presence
    format_presence_reward: pydantic.NonNegativeFloat = 0.05
    """Binary reward for producing a valid steps block with enough parsed steps."""
    require_min_parsed_steps: pydantic.PositiveInt = 1
    """Minimum number of parsed steps to earn the format presence reward."""

    # B) structural validity
    structural_validity_weight: pydantic.NonNegativeFloat = 0.05
    """Maximum reward for structural validity (scales with valid step count, up to max)."""
    max_rewarded_steps: pydantic.PositiveInt = 20
    """Cap on step count for proportional rewards (linear denominator or diminishing asymptote)."""
    validity_curve: typing.Literal["linear", "diminishing"] = "diminishing"
    """Scaling curve for the structural validity sub-reward.

    ``"linear"``: reward scales as ``min(n, max_rewarded_steps) / max_rewarded_steps``.
    ``"diminishing"``: reward scales as ``1 - diminishing_decay^n``, producing strong marginal
    returns for the first few steps and near-zero marginal returns beyond that.
    """
    grounding_curve: typing.Literal["linear", "diminishing"] = "linear"
    """Scaling curve for the code grounding sub-reward.

    Defaults to ``"linear"`` so each additional grounded step provides constant marginal reward,
    reflecting that each grounded step is genuine evidence of code understanding.
    """
    diminishing_decay: float = pydantic.Field(default=0.75, gt=0.0, lt=1.0)
    """Per-step decay factor for the ``"diminishing"`` reward curve.

    Ignored for components where the selected curve is ``"linear"``. Smaller values = faster saturation (e.g., 0.5 means
    3 steps capture 87.5% of max reward; 0.75 means 3 steps capture ~58.8%).
    """
    check_line_in_range: bool = True
    """Whether to require step line numbers to be within code bounds."""
    check_executable_line: bool = True
    """Whether to require step line numbers to point to executable code."""
    check_monotonicity: bool = True
    """Whether to penalize non-monotonic step indices."""
    monotonicity_penalty_factor: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Multiplicative factor on validity reward when monotonicity fails (0.0 = full penalty, 1.0 = no penalty)."""
    check_contiguity: bool = True
    """Whether to penalize non-contiguous step indices."""
    contiguity_penalty_factor: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Multiplicative factor on validity reward when contiguity fails (0.0 = full penalty, 1.0 = no penalty)."""

    # C) code grounding (identifier-token overlap)
    grounding_weight: pydantic.NonNegativeFloat = 0.05
    """Maximum reward for code grounding (proportional to grounded step count)."""
    min_token_overlap: pydantic.PositiveInt = 1
    """Minimum shared non-numeric tokens for a step to count as grounded."""

    # gating on objective correctness
    incorrectness_penalty: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
    """Penalty factor applied when the final answer is objectively incorrect.

    1.0 = zero out reward entirely, 0.5 = halve the reward (default), 0.0 = no gating.
    """


class TracedReasoningTerm(reward_term.BaseRewardTerm):
    """Rewards structured reasoning traces referencing source code line numbers.

    This term parses a JSONL ``<steps>`` block from the model output and computes three sub-rewards
    (format presence, structural validity, code grounding), optionally gated on the objective
    correctness of the final answer.

    See ``TracedReasoningTermConfig`` for detailed parameter documentation.
    """

    def __init__(
        self,
        config: TracedReasoningTermConfig,
    ) -> None:
        """Initializes the reward term."""
        self._config = config

    def reset(
        self,
        run_init_ctx: reward_types.RunInitContext,
    ) -> None:
        """Reset caches and validates line-number reqs based on the init context datamodule config.

        Args:
            run_init_ctx: Run initialization context with datamodule reference.

        Raises:
            ValueError: If the datamodule has ``add_line_numbers=False``.
        """
        self._get_code_info.cache_clear()
        if not run_init_ctx.datamodule.config.add_line_numbers:
            raise ValueError("TracedReasoningTerm requires add_line_numbers=True in the datamodule config")

    def __call__(
        self,
        sample_ctx: reward_types.SampleContext,
    ) -> reward_types.TermResult:
        """Compute the traced reasoning reward for a sample.

        Args:
            sample_ctx: Sample context with model output and sample data.

        Returns:
            TermResult with the gated reward value and diagnostic metrics.

        Raises:
            ValueError: If code has missing/mixed line-number prefixes, or if
                ``incorrectness_penalty > 0`` but ``code_exec_eval`` is missing.
        """
        code_raw = sample_ctx.sample_data.code
        stripped_code, executable_lines = self._get_code_info(code_raw)
        code_lines = stripped_code.splitlines()
        max_line = len(code_lines)
        # parse the <steps> block
        report = pyine.utils.parsing.parse_and_validate_steps_block(
            sample_ctx.model_output,
            tag_name=self._config.steps_tag,
            multi_block_policy=self._config.multi_block_policy,
            code_string=stripped_code,
            max_line=max_line,
            executable_lines=executable_lines,
        )
        # === pipeline stages ===
        # stage 0: parsed_steps (from report)
        parsed_steps = list(report.parsed_steps)
        # stage 1: valid_steps (per-step constraint checks)
        valid_steps: list[pyine.utils.parsing.ParsedStep] = []
        for step in parsed_steps:
            if self._config.check_line_in_range and (step.line < 1 or step.line > max_line):
                continue
            if self._config.check_executable_line and step.line not in executable_lines:
                continue
            valid_steps.append(step)
        # stage 2: grounded_steps (token overlap on valid_steps)
        grounded_steps: list[pyine.utils.parsing.ParsedStep] = []
        for step in valid_steps:
            if 1 <= step.line <= max_line:
                source_line = code_lines[step.line - 1]
                source_tokens = self._tokenize_identifier_tokens(source_line)
                step_tokens = self._tokenize_identifier_tokens(step.text)
                overlap = len(source_tokens & step_tokens)
                if overlap >= self._config.min_token_overlap:
                    grounded_steps.append(step)
        # === sub-reward computation ===
        # A) format presence
        reward_a = (
            self._config.format_presence_reward if len(parsed_steps) >= self._config.require_min_parsed_steps else 0.0
        )
        # B) structural validity
        base_validity = self._compute_step_scale(len(valid_steps), self._config.validity_curve)
        # sequence-level checks on parsed_steps
        mono_ok = len(parsed_steps) > 0 and all(
            parsed_steps[idx].step > parsed_steps[idx - 1].step for idx in range(1, len(parsed_steps))
        )
        contig_ok = len(parsed_steps) > 0 and (
            parsed_steps[0].step == 1 and all(parsed_steps[idx].step == idx + 1 for idx in range(len(parsed_steps)))
        )
        mono_factor = (
            1.0 if (mono_ok or not self._config.check_monotonicity) else self._config.monotonicity_penalty_factor
        )
        contig_factor = (
            1.0 if (contig_ok or not self._config.check_contiguity) else self._config.contiguity_penalty_factor
        )
        reward_b = self._config.structural_validity_weight * base_validity * mono_factor * contig_factor
        # C) code grounding
        grounding_scale = self._compute_step_scale(len(grounded_steps), self._config.grounding_curve)
        reward_c = self._config.grounding_weight * grounding_scale
        pre_gate_reward = reward_a + reward_b + reward_c
        # gating on objective correctness
        is_objectively_correct = True
        if self._config.incorrectness_penalty > 0.0:
            eval_data = code_exec_utils.require_code_exec_eval_data(sample_ctx, "TracedReasoningTerm")
            compare_result = pyine.utils.code.output_compare.compare(eval_data.expected, eval_data.predicted)
            is_semantic_match = compare_result.equal
            should_flip = code_exec_utils.get_flip_decision(sample_ctx)
            is_objectively_correct = is_semantic_match != should_flip  # XOR
        correctness_factor = 1.0 if is_objectively_correct else (1.0 - self._config.incorrectness_penalty)
        final_reward = pre_gate_reward * correctness_factor
        # compute line ratios (denominator = num_parsed for bounded ratios)
        num_parsed = len(parsed_steps)
        num_valid = len(valid_steps)
        line_in_range_count = sum(1 for s in parsed_steps if 1 <= s.line <= max_line)
        exec_hit_count = sum(1 for s in parsed_steps if s.line in executable_lines)
        metrics: dict[str, reward_types.MetricValue] = {
            "has_steps_block": report.raw_block is not None,
            "multiple_blocks_found": report.multiple_blocks_found,
            "num_parsed": num_parsed,
            "num_valid": num_valid,
            "num_grounded": len(grounded_steps),
            "step_monotonic_ok": mono_ok,
            "step_contiguous_ok": contig_ok,
            "line_in_range_ratio": line_in_range_count / num_parsed if num_parsed > 0 else 0.0,
            "executable_line_hit_ratio": exec_hit_count / num_parsed if num_parsed > 0 else 0.0,
            "grounding_ratio": len(grounded_steps) / num_valid if num_valid > 0 else 0.0,
            "format_reward": reward_a,
            "validity_reward": reward_b,
            "grounding_reward": reward_c,
            "pre_gate_reward": pre_gate_reward,
            "correctness_factor": correctness_factor,
            "is_objectively_correct": is_objectively_correct,
        }
        return reward_types.TermResult(value=final_reward, metrics=metrics)

    def _compute_step_scale(
        self,
        num_steps: int,
        curve: typing.Literal["linear", "diminishing"],
    ) -> float:
        """Compute the [0, 1] scaling factor for a given step count.

        Args:
            num_steps: Number of qualifying steps (valid or grounded).
            curve: Which scaling curve to use.

        Returns:
            Scaling factor in [0.0, 1.0].
        """
        if num_steps <= 0:
            return 0.0
        if curve == "linear":
            return min(num_steps, self._config.max_rewarded_steps) / self._config.max_rewarded_steps
        # diminishing: 1 - decay^n, capped at max_rewarded_steps
        capped = min(num_steps, self._config.max_rewarded_steps)
        return 1.0 - self._config.diminishing_decay**capped

    @staticmethod
    def _tokenize_identifier_tokens(
        text: str,
    ) -> frozenset[str]:
        """Tokenize text and keep only non-numeric tokens for grounding overlap checks."""
        return frozenset(token for token in pyine.utils.parsing.tokenize_code_identifiers(text) if not token.isdigit())

    @functools.lru_cache(maxsize=256)  # noqa: B019
    def _get_code_info(
        self,
        code_raw: str,
    ) -> tuple[str, frozenset[int]]:
        """Cached computation of stripped code and executable line numbers.

        Args:
            code_raw: Raw code string (may contain line-number prefixes).

        Returns:
            Tuple of (stripped_code, executable_line_numbers).

        Raises:
            ValueError: If code has no prefixes or mixed/malformed prefixes.
        """
        strip_result = pyine.utils.parsing.strip_line_number_prefixes(code_raw)
        non_blank_lines = sum(1 for line in code_raw.splitlines() if line.strip())
        if non_blank_lines > 0 and strip_result.lines_matched == 0:
            raise ValueError(
                "TracedReasoningTerm requires code with line-number prefixes "
                "(add_line_numbers=True in datamodule config), but no prefixes were found"
            )
        if 0 < strip_result.lines_matched < non_blank_lines:
            raise ValueError(
                f"TracedReasoningTerm found mixed line-number prefixes: "
                f"{strip_result.lines_matched}/{non_blank_lines} non-blank lines had prefixes"
            )
        executable_lines = pyine.utils.parsing.get_executable_line_numbers(strip_result.stripped)
        return strip_result.stripped, executable_lines


def _factory(
    spec: reward_configs.RewardTermSpec,
    *,
    parser: pyine.utils.parsing.OutputParser | None,
) -> reward_types.RewardTerm:
    """Build a ``TracedReasoningTerm`` from a term spec.

    Args:
        spec: Term specification with parameters.
        parser: Output parser (unused by this term).

    Returns:
        Configured TracedReasoningTerm instance.
    """
    del parser  # unused
    config = TracedReasoningTermConfig.model_validate(spec.params)
    return TracedReasoningTerm(config)


reward_registry.register_term("traced_reasoning", _factory)
reward_registry.register_term_aliases("traced_reasoning", ["format/traced_reasoning"])
