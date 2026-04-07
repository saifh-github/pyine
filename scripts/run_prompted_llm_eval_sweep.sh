#!/usr/bin/env bash
# Sweep script for the prompted LLM guardrail evaluation across model/effort combinations.
#
# Runs the prompted_llm_eval app sequentially for each target configuration, using
# the 'guardrail/prompted_llm_eval_openai' experiment config as a base.
#
# Usage:
#   bash scripts/run_prompted_llm_eval_sweep.sh

set -euo pipefail

BASE_CMD=(
    uv run python -m pyine.apps.guardrail_eval.prompted_llm_eval
    +experiment=guardrail/prompted_llm_eval_openai
)

run_eval() {
    local label="$1"
    shift
    echo ""
    echo "================================================================================"
    echo "  ${label}"
    echo "  cmd: ${BASE_CMD[*]} $*"
    echo "================================================================================"
    echo ""
    "${BASE_CMD[@]}" "$@"
}

run_eval "gpt-5-nano (low effort)" \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-5-nano \
    ++config.guardrail_config.llm_provider.model_kwargs.reasoning_effort=low

run_eval "gpt-5-nano (default effort)" \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-5-nano

run_eval "gpt-5.4-nano (low effort)" \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-5.4-nano \
    ++config.guardrail_config.llm_provider.model_kwargs.reasoning_effort=low

run_eval "gpt-5.4-nano (default effort)" \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-5.4-nano
