# PyINE scripts

Helper scripts we used during our own experiments. None of these are required to use the framework;
they exist as reference launchers and sweep drivers. Open the individual files for full usage
details (most have a `--help` block or a comment header).

A few rely on **SLURM** (flagged below); the rest run on any Linux box with bash + GPUs. For
multi-node SLURM training in particular, see [`SLURM_GUIDE.md`](./SLURM_GUIDE.md).

## Launchers

- [`run_ddp.sh`](./run_ddp.sh): `torchrun` wrapper that auto-detects GPU count, picks a free
  rendezvous port, and forwards Hydra overrides to a target app. Works single-node or under
  SLURM.
- [`launch_slurm.sh`](./launch_slurm.sh): **SLURM** `sbatch` script for multi-node DeepSpeed
  ZeRO-3 RL training with colocated vLLM rollouts.
- [`launch_debate_eval.sh`](./launch_debate_eval.sh): **SLURM** debate-eval launcher on a
  single 8-GPU node, with three modes (small vLLM interrogators, big vLLM interrogators with
  TP, and OpenAI-API interrogators).
- [`launch_wandb_agents.sh`](./launch_wandb_agents.sh): SSHes into a list of GPU nodes and
  spawns 8 W&B sweep agents per node in tmux panes (one per GPU).
- [`my_command`](./my_command): one-line template invoked by `launch_wandb_agents.sh` to
  start a W&B agent inside each pane (substitutes `REPO_ROOT`, `CUDA_DEVICE`, `SWEEP_ID`).

## Sweeps (sequential)

- [`run_checkpoint_sweep.sh`](./run_checkpoint_sweep.sh): for each HF checkpoint in a list,
  spin up vLLM, run the `hf_trainer` predict pipeline, tear down.
- [`run_model_sweep.sh`](./run_model_sweep.sh): same shape, but iterates over HuggingFace
  *models* instead of checkpoints.
- [`run_probe_sweep.sh`](./run_probe_sweep.sh): runs the probe trainer back-to-back across
  the `weak` / `moderate` / `strong` shortcut presets via `run_ddp.sh`.
- [`run_classifier_bias_sweep.sh`](./run_classifier_bias_sweep.sh): runs the LLM classifier
  trainer across the resampling-bias presets (weak / moderate / strong).
- [`run_prompted_llm_eval_openai_sweep.sh`](./run_prompted_llm_eval_openai_sweep.sh): sweeps
  the prompted-LLM guardrail eval across OpenAI models and reasoning-effort levels.
- [`run_prompted_llm_eval_vllm_sweep.sh`](./run_prompted_llm_eval_vllm_sweep.sh): sweeps the
  prompted-LLM guardrail eval across a list of vLLM-served HuggingFace models.

## Utilities

- [`check_ascii_staged.py`](./check_ascii_staged.py): pre-commit hook that flags non-ASCII
  bytes on *staged* lines only (legacy non-ASCII content is tolerated).
- [`vllm_eval/`](./vllm_eval/): standalone vLLM serving + grading utilities for
  evaluation/judge workflows; see its own [README](./vllm_eval/README.md).
