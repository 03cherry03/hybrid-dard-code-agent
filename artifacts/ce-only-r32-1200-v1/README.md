# CE-only r32/1200 reproduction bundle

This directory preserves the completed `ce-only-diffucoder-dard-r32-1200-v1` run for reuse by another team member. The adapter is the final 1,200-step LoRA checkpoint; `adapter/` contains the two files required by `--adapter-path`.

The bundle includes the exact training JSONL, deterministic balanced JSONL, validation/holdout tasks, training configuration, report summaries, and SHA256 manifest. The base model remains external and must be loaded at the pinned revision in `experiment_manifest.json`.

`source-provenance/` preserves the machine-specific evaluation snapshot. `training-source-current/` preserves the current repository trainer used for a new explicitly labelled run. The historical training source hashes are recorded in `metadata/source_provenance.json`; the original training source bytes are not available, so the existing adapter is the exact learned-weight artifact.

From the repository root, use the pinned iterative-training environment and point `--adapter-path` to this bundle's `adapter/` directory and `--data` to this bundle's `data/` directory. Supply a verified four-GPU `iteration_plan.json`; the original report's machine-specific evaluation source and plan are preserved under `source-provenance/` for audit.

```bash
/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py \
  --base-plan /data/hybrid-dard/results/code-generation-tradeoff-20261107/iteration_plan.json \
  --data artifacts/ce-only-r32-1200-v1/data \
  --adapter-path artifacts/ce-only-r32-1200-v1/adapter \
  --output /data/hybrid-dard/results/ce-only-r32-1200-bundle \
  --deadline-unix 1794225600
```

The full raw generation case files and intermediate 400/800-step checkpoints are intentionally excluded; they are not required for adapter reuse or report metric verification.

Verify the bundle before use:

```bash
/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py --verify-bundle artifacts/ce-only-r32-1200-v1
```
