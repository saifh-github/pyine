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
    ++config.guardrail_config.prompt_version=score_only
)

run_eval() {
    local model="$1"
    local effort="${2:-default}"
    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    local run_name="${timestamp}_${model}_${effort}"
    local label="${model} (${effort} effort)"

    local cmd=("${BASE_CMD[@]}"
        "config.guardrail_config.llm_provider.model_kwargs.model=${model}"
        "runtime.run_name='${run_name}'"
    )
    if [[ "${effort}" != "default" ]]; then
        cmd+=("++config.guardrail_config.llm_provider.model_kwargs.reasoning_effort=${effort}")
    fi

    echo ""
    echo "================================================================================"
    echo "  ${label}"
    echo "  cmd: ${cmd[*]}"
    echo "================================================================================"
    echo ""
    "${cmd[@]}"
}

run_eval gpt-5-nano low
run_eval gpt-5-nano
run_eval gpt-5.4-nano low
run_eval gpt-5.4-nano
run_eval gpt-5-mini low
run_eval gpt-5-mini
run_eval gpt-5.4-mini low
run_eval gpt-5.4-mini
