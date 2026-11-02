# HybridDARD

**A hybrid autoregressive–diffusion coding agent with dependency-aware revocable decoding.**

HybridDARD is a research project exploring whether a diffusion language model (dLLM) can make code generation and test-driven repair more efficient, with an autoregressive (AR) model responsible for planning, reviewing, and fallback.

> **Project Status:** This repository currently contains a benchmark pipeline for DiffuCoder-7B standard decoding and an independent DARD-style research implementation. AR + dLLM orchestration is planned but not yet implemented. Full server-GPU validation is pending.

## 1. Project Overview

The goal of HybridDARD is to develop a local coding agent that combines the strengths of autoregressive and diffusion language models.

Traditional AR models generate code sequentially, while diffusion language models can predict and refine multiple tokens simultaneously.

We investigate whether **Dependency-Aware Revocable Decoding (DARD)** can improve diffusion-based code generation and repair, particularly when correctness and computational efficiency must be balanced.

### Research Objectives

1. Compare standard diffusion decoding with DARD-style decoding.
2. Evaluate code generation accuracy, repair success, and inference latency.
3. Develop a hybrid architecture combining AR planning with diffusion-based code generation.
4. Extend evaluation from function-level tasks to repository-level software engineering problems.

## 2. Proposed Architecture

```text
              User Request
                    |
                    v
           +------------------+
           |     AR Model     |
           | Planning/Context |
           +------------------+
                    |
                    v
           +------------------+
           |   dLLM + DARD    |
           |  Code Generation |
           +------------------+
                    |
                    v
           +------------------+
           |  Test Execution  |
           +------------------+
                    |
             Pass / Fail
               /     \
              /       \
           Pass       Fail
            |           |
            v           v
        AR Review   dLLM + DARD
                     Code Repair
                         |
                  Repeated Failure
                         |
                         v
                    AR Fallback
```

This is the **target architecture**. The current repository implements the diffusion-based generation, decoding, testing, and repair components. Full AR orchestration remains future work.

## 3. Current Implementation

| Component                         | Status      |
| --------------------------------- | ----------- |
| DiffuCoder-7B-Instruct inference  | Implemented |
| Standard diffusion decoding       | Implemented |
| Independent DARD-style decoding   | Implemented |
| Token verification and revocation | Implemented |
| Adaptive logit mixing             | Implemented |
| Python/Rust code generation       | Implemented |
| Automated test execution          | Implemented |
| Test-feedback code repair         | Implemented |
| Benchmark evaluation and logging  | Implemented |
| AR planning and review            | Planned     |
| Hybrid AR–dLLM orchestration      | Planned     |
| Repository-level coding agent     | Planned     |

**Note:** The DARD decoder is an independent research implementation based on mechanisms described in the paper, not the authors' official implementation. Its exact paper fidelity and full GPU integration require further validation.

## 4. Models

### Autoregressive LLM

**Qwen2.5-Coder**

Intended responsibilities:

* Task understanding
* Planning
* Context selection
* Final code review
* Fallback generation

An initial standalone AR baseline has been tested. Integration into the current benchmark repository is planned.

### Diffusion LLM

**DiffuCoder-7B-Instruct**

Responsibilities:

* Code generation
* Test-driven code repair
* Diffusion decoding experiments

The current pipeline supports standard decoding and an independent DARD-style decoder.

## 5. Benchmarks

The project uses the following function-level benchmarks:

| Benchmark | Language | Evaluation                          |
| --------- | -------- | ----------------------------------- |
| HumanEval | Python   | Function generation and unit tests  |
| MBPP      | Python   | Programming problems and unit tests |
| MultiPL-E | Rust     | Multilingual function generation    |

The loaders support 164 HumanEval tasks, 500 MBPP test-split tasks, and the available MultiPL-E HumanEval-Rust tasks.

### Evaluation Metrics

* **First-attempt success:** Correctness of the initial generated solution.
* **Final success:** Correctness after allowed repair attempts.
* **Repair success rate:** Fraction of initially failed tasks resolved through repair.
* **Execution time:** Generation, testing, and total task completion time.
* **Model forward passes:** Number of model inference passes.
* **GPU memory usage:** Peak allocated GPU memory.
* **Error categories:** Syntax, compilation, runtime, and test failures.
* **DARD diagnostics:** Token promotion, revocation, forced completion, and attention fallback statistics.

> The current evaluation asks models to generate complete functions rather than using the official completion-only benchmark protocol. Reported pilot success rates are not directly comparable with official HumanEval/MultiPL-E pass@1 scores.

## 6. Getting Started

### Requirements

* Linux server
* NVIDIA GPU with compatible CUDA drivers
* Python 3.10–3.11 recommended
* PyTorch with CUDA support
* Docker for executing generated code

Model weights and benchmark datasets are downloaded automatically when needed.

### Step 1. Run Unit Tests

```bash
python3 -m unittest discover -s tests -v
```

These tests verify decoder logic and benchmark infrastructure without requiring the full 7B model.

### Step 2. Smoke-Test Code Generation

Without executing generated code:

```bash
RUNNER=none LIMIT=1 bash run_server.sh \
  --tokens 128 \
  --standard-steps 128 \
  --block-size 64 \
  --dard-steps 96
```

This runs both standard diffusion decoding and DARD-style decoding.

### Step 3. Test Generation and Repair

With Docker installed and running:

```bash
LIMIT=1 bash run_server.sh \
  --tokens 128 \
  --standard-steps 128 \
  --block-size 64 \
  --dard-steps 96
```

The pipeline generates code, executes tests in restricted Docker containers, and attempts repair when tests fail.

### Step 4. Run Larger Benchmarks

```bash
LIMIT=30 bash run_server.sh
```

The limit applies separately to each benchmark suite.

For custom experiments:

```bash
python main.py \
  --suites humaneval,mbpp,multiple_rs \
  --decoders standard,dard \
  --limit 30 \
  --quant 4bit \
  --tokens 256 \
  --standard-steps 256 \
  --block-size 128 \
  --dard-steps 192 \
  --runner docker \
  --pull-images \
  --repairs 1 \
  --output results/experiment1
```

## 7. Results

Experimental results are automatically stored in:

```text
results/
└── experiment1/
    ├── config.json
    ├── results.csv
    ├── summary.json
    ├── runs.jsonl
    ├── case_results.jsonl
    └── errors.jsonl
```

The system records generation attempts, correctness, repair outcomes, inference time, model forward passes, and decoding diagnostics.

**Current research status:** CPU-side unit and mocked integration tests have passed. Full validation of the independent DARD decoder on the target server GPU is still required.

## 8. Project Structure

```text
HybridDARD/
│
├── main.py
├── models.py
├── dard_decoder.py
├── datasets_code.py
├── evaluation.py
├── run_server.sh
├── requirements.txt
│
├── tests/
│   ├── test_dard_decoder.py
│   └── test_runner_end2end.py
│
├── README.md
└── README_KO.md
```

### Main Components

* `main.py`: Benchmark execution and experiment management
* `models.py`: DiffuCoder model interface
* `dard_decoder.py`: Independent DARD-style decoding implementation
* `datasets_code.py`: Benchmark loading
* `evaluation.py`: Docker-based code evaluation
* `run_server.sh`: Server setup and execution

## 9. Research Roadmap

* [x] Build local AR coding baseline
* [x] Run DiffuCoder-7B locally
* [x] Implement function-level benchmark pipeline
* [x] Implement an independent DARD-style decoder
* [x] Implement automated testing and repair
* [ ] Validate DARD inference on a server GPU
* [ ] Verify decoding correctness and performance
* [ ] Integrate AR model into the benchmark pipeline
* [ ] Implement hybrid AR–dLLM coding agent
* [ ] Extend to repository-level benchmarks
* [ ] Conduct controlled accuracy–latency comparisons

## 10. Limitations

This project is currently a research prototype.

The independent DARD implementation has not been confirmed to reproduce the original paper's exact implementation or results.

If custom attention masks are unsupported, the decoder may use a slower verification fallback. These cases must be distinguished during performance analysis.

Generated code is executed only through the configured Docker evaluation system. Docker restrictions reduce risk but do not provide a perfect security boundary.

## 11. References

1. [Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference](https://arxiv.org/abs/2608.26574)
2. [DiffuCoder-7B-Instruct](https://huggingface.co/apple/DiffuCoder-7B-Instruct)
3. [DiffuCoder GitHub](https://github.com/apple-aiml-research/ml-diffucoder)
4. [HumanEval](https://github.com/openai/human-eval)
5. [MultiPL-E](https://github.com/nuprl/MultiPL-E)
6. [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

This is an independent educational and research project.
