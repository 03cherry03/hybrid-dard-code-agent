# Hybrid DARD Code Agent — independent code-generation evaluation

Compare Qwen2.5-Coder-7B-Instruct, standard DiffuCoder-7B-Instruct, and the same DiffuCoder with DARD using shared tasks, tools, precision and budgets. The default scope measures function code generation and repair in Python, C++, Rust and JavaScript. Library-use evaluation is excluded. DARD is an independent Dream adapter; real model speed/quality gains remain unproven.

The active deadline campaign uses `--focused-codegen`: one matched pass of AR, DARD and native diffusion over Python, Rust, JavaScript and C++, with no repair feedback. Both diffusion methods use a 512-token generation canvas. Native diffusion uses 64 steps; DARD uses at most 64 verification steps, thresholds `.5/.8`, and no cap on threshold-qualified parallel commits. `generation_settings.json` freezes these settings before calibration and evaluation.

The corrected DARD adapter implements original/masked-shadow keys, confidence-ordered candidate verification, M/C/U revocation, and the paper ratio `(P+p0)/(P+D+p0)`. Separate predictor queries adapt Dream's shifted logits; this remains an independent adaptation, not author-code reproduction. An all-masked state uses native prediction directly. Exact greedy fixed points and repeated states enter a bounded completion fallback, whose work is recorded separately. Full generated IDs, text before control-token cleanup, and terminal ID/position are preserved. Quality and speed claims require measured paired results.

## One-command pipeline

~~~bash
cd /home/leejwian1955/hybrid-dard-code-agent
bash scripts/run_all_experiments.sh --scope functions --prepare --execute \
  --output /data/hybrid-dard/results/code-generation
~~~

`--scope functions` is the default. `--prepare` installs only the inference, language-tool and EvalPlus environments and downloads the two models. Without `--execute`, only write the plan and inspect notification metadata. Prepared code-generation datasets are verified rather than downloaded again. Use a new output directory when changing scope/protocol/settings; prior results are preserved.

The default path does not run SWE data checks, native repository preparation, system dependency checks or BigCodeBench installation. TeX, browsers, additional locales and sudo setup are unnecessary. Missing or corrupt out-of-scope SWE/BigCodeBench data does not block selected code-generation data verification.

Stages: GPU/code-generation data checks → optional preparation → function smoke → full generation/repair/final grading → analysis. Four GPUs process disjoint tasks; all three backends and repeated seeds for one task remain on the same GPU. Failures stop the pipeline and trigger email. Wrong generated answers remain normal measured outcomes.

## Environments and data

Datasets, model/cache storage, isolated environments, temporary trials and results are rooted at `/data/hybrid-dard` (`HYBRID_DARD_DATA_ROOT` override). Use the existing Conda installation to create separate prefixes; base remains unchanged.

~~~bash
source scripts/runtime_env.sh
bash scripts/setup_env.sh functions
"$HYBRID_DARD_ROOT/envs/inference/bin/python" scripts/download_models.py --download
python3 prepare_data.py --scope functions --verify
python3 environment_probe.py --scope functions
~~~

Prefixes: `inference` (Python 3.11, torch 2.7.1+cu118, Transformers 4.51.3), `function` (native language/build tools), `evalplus` (Python 3.11, EvalPlus 0.3.1). Installed package inventories are saved in `envs/locks`. For a new data root, explicitly prepare the selected data with `python3 prepare_data.py --scope functions --download`.

BigCodeBench data and old results are preserved for historical compatibility, but the current CLI, pipeline and setup exclude it. EvalPlus strengthens tests for the existing function problems; it does not add library-use tasks.

| Evaluation | Scope |
| --- | --- |
| HumanEval / MBPP | Python, 164 / 500 tasks |
| MultiPL-E | Rust/JavaScript/C++, 1,156 translated task variants |
| HumanEval+ / MBPP+ | Python, 164 / 378, official base+plus tests |

The 10 function suites contain 2,362 variants, including correlated translations/augmented tests. Public feedback drives repairs; final tests run only after submissions freeze. Default `--runner local`; `--runner none --repairs 0` records generation only and cannot report accuracy. These results measure function generation/repair, not repository issue resolution.

## Optional, separate repository workflow

The retained SWE-bench Verified (500) and Multilingual (300) evaluators run only when explicitly selected with `--scope repositories` or `--scope all`. Neither scope includes BigCodeBench. Only this optional workflow needs the repository system dependencies. Use a separate output:

~~~bash
bash scripts/setup_system_dependencies.sh
bash scripts/run_all_experiments.sh --scope repositories --prepare --execute \
  --output /data/hybrid-dard/results/repository-experiments-local
~~~

Individual repository preparation commands:

~~~bash
"$HYBRID_DARD_ROOT/envs/swebench/bin/python" scripts/prepare_repositories.py \
  --suite swebench_verified --limit 0 --prepare --data-root "$HYBRID_DARD_ROOT"
"$HYBRID_DARD_ROOT/envs/swebench/bin/python" scripts/prepare_repositories.py \
  --suite swebench_multilingual --limit 0 --prepare --data-root "$HYBRID_DARD_ROOT"
~~~

Pinned setup recipes prepare the original base commit and a per-task environment, then validate original/reference-patched controls before publishing `repositories.lock.json`. Missing or unmapped dependencies fail explicitly, never silently remove tasks. Actual build success for all 800 tasks requires running preparation.

Each agent gets a fresh source copy and offline Conda clone with local read/search/edit/run/finish tools. Original Git history is stripped from the agent workspace; final grading uses a separate checkout preserving original history. Official test scripts undergo explicit infrastructure-path adaptation, with original/adapted hashes recorded. Official log parsers and fail-to-pass/pass-to-pass resolution rules remain, while results identify the native backend. Identified infrastructure failures remain unknown outcomes; candidate syntax/compile errors and failing tests are unresolved submissions.

Local processes share the user's host access. Temporary directories and path-limited file tools are not a security sandbox. Final tests are excluded from model input and repair feedback, but OS-level access isolation is not claimed.

Environment change detection uses file metadata, not a full content-integrity proof. Default command limits cover wall time, retained output and child-process cleanup; they do not reserve CPU or RAM per task.

## Four-GPU runs and diagnostics

~~~bash
bash scripts/run_four_gpu.sh functions --gpus 0,1,2,3 \
  --limit 0 --repeats 3 --tokens 512 --standard-steps 512 \
  --diagnostics summary --output /data/hybrid-dard/results/code-generation-direct --execute --fail-fast
~~~

The code-generation partition is 591/591/590/590 variants. Three backends and three repetitions produce 21,258 cases, plus a separate smoke run. Plans freeze GPU UUIDs, tasks/seeds, commands and source/environment fingerprints. Shared host CPU/RAM/I/O can affect latency; this is a concurrent campaign, not an isolated single-agent latency claim. Do not average shard speedups. Pooled confidence intervals are not implemented.

`functions/campaign_summary.json` pools counts and all three paired comparisons using ratios of summed times over the complete matched cohort. Unknown infrastructure outcomes and failed-solution time remain visible. Separate phase accounting retains load, warmup and retry costs; accumulated worker hours are distinct from elapsed four-GPU campaign time. The default is a single fixed operating point. Temperature-zero repetitions are timing replications, not independent pass@k samples. Broader superiority claims require a separate development split and accuracy/time sweeps including lower-step standard diffusion; this DARD comparison includes its complete block, verification and stopping policy.

`summary` measures performance; `steps` records decoder transitions; `profile` records synchronized component timings. Detailed instrumentation can affect both latency and deadline-limited outcomes, so confirm changes in a new summary run. Forced completion and revocations remain measured algorithm behavior. `experiment_analysis.py --run DIR` writes evidence-linked JSON/Markdown/CSV. `experiment_journal.py` records intent, observations and decisions. Dev/test labels declare intent, not automatic source-ID separation.

## Notifications and resume

Failure email goes to `leejwian@gmail.com` using LERRA's private `~/.config/lerra_ft_comparison/gmail_app_password`. Override the path with `HYBRID_DARD_MAIL_PASSWORD_FILE` or `LERRA_MAIL_PASSWORD_FILE`. Credentials are not copied into code/arguments/email. `--check-email` authenticates only, `--test-email` explicitly sends a test, and `--retry-email --output DIR` retries queued failures without running experiments.

The separate supervisor monitors stage/worker exceptions, nonzero exits and signals, cleans owned descendants, and records pending mail/SMTP acceptance. SMTP acceptance is not inbox confirmation. Resume with identical code/settings/output; successful artifact hashes are verified before stages are skipped. Frozen repository error submissions require a new experiment to regenerate. Host power loss or supervisor SIGKILL requires external monitoring for immediate alerts; incomplete prior state is checked on the next launch.

Frozen source hashes are rechecked at stage boundaries and immediately before GPU workers launch. Function `--fail-fast` persists the first infrastructure failure and stops before later cases/backends, while incorrect generated code remains a normal evaluated outcome. A recorded review gate can be handed over with `scripts/resume_reviewed_experiments.py --previous-output OLD --output NEW --execute`: it waits for the existing downloader, preserves the old run, and launches the freshly frozen pipeline without installation or additional model downloads. See `NEW/audit_handoff.json` and then `NEW/pipeline_state.json`. The old supervisor's termination notification corresponds to the planned `OLD/audit_replacement.json` record.

## Bounded optimization campaign

The compact decoder keeps original predictor queries only for masked positions. The per-instance Dream adapter projects K/V only from the `P+2B` context rows and uses rectangular attention; query length becomes `P+3B+|M|`. Native generation delegates to the original attention implementation. No cross-step KV caching or verification skipping is introduced. Use `--no-compact-queries --no-rectangular-attention` for the legacy calculation. BF16 shape changes require real-model validation even when CPU calculations agree.

`scripts/optimize_experiments.py` waits for the preserved baseline and GPU ownership to clear, then requires the numerical/GPU gate in `scripts/validate_dard_optimization.py`. A frozen identity-only partition reserves 16 sources across four languages for development and excludes them from the new 344-source final evaluation. Existing full-baseline results remain exploratory. Candidate scope is Instruct plus one optional pinned official cpGRPO checkpoint; native256 is a quality diagnostic only. The final evaluation compares AR, DARD and native64 with the same selected diffusion weights. A predeclared three-percentage-point development screen is not proof of statistical noninferiority. Failure notifications, immutable plans, bounded retries, and the absolute deadline remain enforced.

## Validation

~~~bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -v
~~~

Tests cover CPU logic, small local programs and mocked evaluators. They do not establish real 7B performance or successful full native benchmark provisioning. See [the Korean guide](README_KO.md) for detailed commands.

## Reproduce the CE-only r32/1200 report

Use `scripts/reproduce_ce_only.py` with a verified four-GPU plan and the frozen train/validation/holdout JSONL files. It fixes the masked-CE training arguments, the three report candidates, validation seeds `20261105, 20261111, 20261112, 20261113`, holdout seed `20261108`, and the DARD operating point. Hashes and immutable manifests are written before evaluation; changed data or adapters stop the run.

~~~bash
/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py \
  --base-plan /data/hybrid-dard/results/code-generation-tradeoff-20261107/iteration_plan.json \
  --data /data/hybrid-dard/results/code-generation-iterative-20261107/data \
  --output /data/hybrid-dard/results/ce-only-r32-1200 \
  --deadline-unix 1794225600
~~~

The completed 1,200-step adapter and its SHA256-pinned data/config bundle are in `artifacts/ce-only-r32-1200-v1/`. Reuse its `adapter/` directory with `--adapter-path` to skip training while keeping the exact learned weights. `experiment_manifest.json` records the pinned base revision, training contract, candidate IDs, and report metrics; the large adapter file is tracked with Git LFS.

The bundle separates the historical evaluation source provenance from the current trainer source; the original training source bytes are unavailable, while the learned adapter itself is preserved exactly.

For adapter reuse with the bundled data, add `--adapter-path artifacts/ce-only-r32-1200-v1/adapter --data artifacts/ce-only-r32-1200-v1/data` to the command above.
Before sharing or using the bundle, verify it with `/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py --verify-bundle artifacts/ce-only-r32-1200-v1`.
