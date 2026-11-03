# DiffuCoder-7B / DARD (논문 메커니즘 독립 구현) 벤치마크

## 먼저 읽어 주세요 — 중요한 정확성 구분

이 프로젝트는 **기존 DARD-lite가 아닙니다.** DARD 논문(Park et al., 2026)에 소개된 네 가지 핵심 메커니즘을 **독립 구현**했습니다.

1. **M / C / U 토큰 상태**: 마스킹 / 후보 / 검증 통과. 실패 후보는 다시 마스킹할 수 있음.
2. **Confidence-ordered selective attention**: 각 후보는 신뢰도가 더 높은 후보만 검증 문맥으로 이용하도록 4차원 attention mask 구성.
3. **Original + shadow 모델 입력**: 후보가 있는 원본 시퀀스와 후보를 마스킹한 shadow 시퀀스를 **단일 model forward**로 처리. Dream의 shifted logits와 RoPE 위치를 보정.
4. **Adaptive logit mixing**: 근접 후보의 승격/취소 기록에 따라 원본/섀도 로짓을 위치별 가중치로 혼합.

**단, 연구진의 공식 DARD 구현을 확보하지 못했습니다.** 이것은 논문의 핵심 메커니즘을 포함한 *독립 구현*으로, **원논문 수치·저자의 코드와 일치하는 공식 재현은 아닙니다.** 특히 exact threshold calibration / step scheduler / 특정 attention layout은 별도 교차검증이 필요합니다. **커스텀 4D attention이 서버/모델에서 지원되지 않으면**, 논리적으로 동일한 신뢰도 순서 검증을 후보별로 따로 수행하는 **느린 fallback**으로 자동 전환합니다. 이 경우 `runs.jsonl`의 `attention_fallback_steps`에 횟수가 남고, 논문에서 제시한 single-forward 최적화 실험과 비교하면 안 됩니다. DiffuCoder는 DARD 원논문에서 주요 실험한 모델이 아니므로 호환성을 GPU에서 확인해야 합니다. `/tests/`의 CPU-only 자체 테스트만 완료했으며 실제 7B 모델 가중치 + 서버 GPU 추론은 이 환경에서 실행하지 못했습니다.

- DARD: https://arxiv.org/abs/2608.26574
- DiffuCoder Apple 공식 모델: https://huggingface.co/apple/DiffuCoder-7B-Instruct
- DiffuCoder 공식 코드: https://github.com/apple-aiml-research/ml-diffucoder

## 실행 환경

- **Linux 서버 권장**, NVIDIA GPU / 드라이버 / CUDA-compatible PyTorch
- Python **3.10 ~ 3.11 권장** (`transformers==4.51.3`)
- **Docker daemon**: 실제 AI 생성 코드 테스트 시 필요 (생성만 할 땐 생략 가능)
- 디스크: DiffuCoder 7B 15GB+ 다운로드 및 패키지 캐시 공간
- 16GB VRAM: `--quant 4bit` 권장. 대형 배치/128블록은 VRAM 부족 시 `--block-size 64`.
- 모델 가중치는 ZIP에 포함되지 않으며 최초 실행 시 Hugging Face에서 다운로드합니다.

## 한 번에 설치하고 실행하기 (Linux, Docker 사용)

```bash
unzip dard_full_benchmark.zip
cd dard_full_benchmark
bash run_server.sh
```

**기본값은** HumanEval Python **100문제**, MBPP 공식 test split **100문제**, MultiPL-E Rust **100문제**, 즉 **300개 문제 × 2 디코더(Standard/DARD) = 600개 케이스**이며, 첫 실패 시 수정 기회 1회. 데이터셋 수집 → 모델 로딩 → 생성 → Docker 테스트 → 수정 → CSV/JSONL 저장까지 자동으로 실행합니다. 실행 시간은 서버 사양과 디코딩 횟수에 따라 매우 길어질 수 있습니다.

**서버에서 Docker가 설치되지 않은 경우:** Docker를 설치·실행하거나, 생성만 확인하려면 아래와 같이 합니다.

```bash
RUNNER=none LIMIT=3 bash run_server.sh
```

이 경우 실제 테스트와 수정은 **수행되지 않습니다**. `passed`를 `False`로 기록하지 않고 `null`로 남깁니다.

### 먼저 아주 작은 실행을 추천

```bash
LIMIT=1 bash run_server.sh --tokens 128 --standard-steps 128 --block-size 64 --dard-steps 96
```

첫 성공 후:

```bash
LIMIT=100 bash run_server.sh
```

**전체 규모**(HumanEval 164 + MBPP 공식 test 500 + Rust 100건 이상):

```bash
LIMIT=0 bash run_server.sh
```

`--limit`은 **벤치마크 종류별** 건수이며, `0`은 각 벤치마크 전체입니다. 출력 파일은 `results/<실행시각>/`에 저장됩니다. 같은 결과 폴더로 재실행하면 완료한 케이스는 건너뜁니다. `--output results/myrun` 옵션 사용을 권장합니다.

### 서버 CUDA 버전이 다른 경우

설치 스크립트는 새로운 RTX 5060 Ti 등을 위해 PyTorch cu128을 기본으로 사용합니다. 기존 서버에 호환되는 PyTorch CUDA 버전이 따로 있거나 CUDA 드라이버가 낮다면 아래처럼 바꿉니다.

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash run_server.sh
```

`cu126`은 RTX 5060 Ti(sm_120)에서는 적합하지 않을 수 있습니다. 서버 GPU 종류와 NVIDIA 드라이버부터 확인해 주세요 (`nvidia-smi`). GPU에 맞는 PyTorch를 사전에 설치한 가상환경을 사용하는 것도 가능합니다.

### 직접 Python 명령으로 제어

```bash
python main.py --suites humaneval,mbpp,multiple_rs --limit 100 \
  --decoders standard,dard --quant 4bit --tokens 256 \
  --standard-steps 256 --block-size 128 --dard-steps 192 \
  --runner docker --pull-images --repairs 1 --output results/run1
```

생성만 확인:

```bash
python main.py --suites humaneval --limit 1 --decoders dard \
  --runner none --repairs 0 --output results/smoke
```

코드 생성 길이, 단계를 바꿀 때는 **서로 다른 output 폴더를 사용**하세요.

## 벤치마크 종류와 주의사항

| 데이터셋 | 파일/원본 | 규모 | 사용한 평가 |
|---|---|---:|---|
| HumanEval | OpenAI HumanEval 원본 | 164 | Python 소스 + 공개 테스트 |
| MBPP | Google Research 원본 | **500 공식 테스트 ID(11~510)** | Python 소스 + 공개 단위 테스트 |
| MultiPL-E Rust | nuprl/MultiPL-E HumanEval-rs | 100건 이상 | Rust 소스 + 공개 테스트 |

데이터셋은 처음 다운로드한 뒤 `data/cache/`에 저장합니다. 서버 인터넷 연결이 필요합니다. Rust는 Hugging Face dataset-server에서 **100개씩 페이지네이션**해 가져옵니다. MBPP 데이터 전체 974개 중 훈련/검증을 제외하고 **11–510 테스트 500개만** 사용합니다.

이 코드는 모델이 **전체 함수를 작성하도록 요구**합니다. 공식 HumanEval/MultiPL-E의 *completion-only* 프롬프트/평가 방식과 다르므로 기록된 성공률은 **독립 파일럿 정확도**이며 공식 benchmark pass@1 수치와 직접 비교하면 안 됩니다. HumanEval+ 강화된 테스트셋은 포함하지 않았습니다.

- https://github.com/openai/human-eval
- https://github.com/nuprl/MultiPL-E
- https://github.com/google-research/google-research/tree/master/mbpp

## 저장되는 결과

`results/<timestamp>/`

- `config.json`: 모델 / 디코더 / 실행 설정 / 메타데이터
- `runs.jsonl`: **시도별** 생성 코드, 오류 메시지, 시간, forward 수, revoked/promoted/forced 토큰 기록
- `case_results.jsonl`: **문제 × 디코더별** 최초 성공 / 수정 후 성공과 시간
- `results.csv`: 표 형식의 `case_results.jsonl`
- `summary.json`: 데이터셋·디코더별 측정된 성공률 / 시간 / 오류 건수
- `errors.jsonl`: 추론·환경 예외 및 traceback (model failure와 구분)

실험 중 5케이스마다 CSV와 summary를 저장하며, Ctrl+C에도 완료된 케이스 로그는 즉시 저장합니다. 다시 시작할 때 같은 `--output`을 넘기면 완료한 케이스는 건너뜁니다.

## 추천 실험 순서

1. `python -m unittest discover -s tests -v`: **네트워크/GPU 없이** DARD 상태·마스킹·혼합·러너 요약 테스트.
2. `LIMIT=1 bash run_server.sh ...`: 서버 GPU 및 Docker에서 실제 모델 로딩/추론/API 호환성 확인.
3. `LIMIT=30 bash run_server.sh`: 세 벤치마크 각각 30개 비교.
4. `LIMIT=100 bash run_server.sh`: 비교 규모 확대.
5. 가능하면 DARD 원논문 저자 코드가 공개될 때 각 attention/threshold/schedule 세부를 대조.

## 보안

생성된 코드는 **절대 호스트에서 직접 실행하지 않습니다**. 테스트를 켜면 Docker를 이용하고 네트워크 비활성화, 권한·메모리·PID·CPU 제한, readonly 루트, 일회용 임시 컨테이너로 실행합니다. Docker 격리도 완전한 보안 보장은 아니므로, 가능하다면 실험 전용 서버/VM에서만 실행하세요. `--runner none`은 코드 실행을 하지 않습니다.

## 실행 시 실패하기 쉬운 지점

- `CUDA not available`: 서버 NVIDIA 드라이버와 CUDA 호환 PyTorch 재확인.
- `bitsandbytes` 오류: 4bit 의존성 및 CUDA 빌드 확인. VRAM 충분하면 `QUANT=bf16` 사용 가능.
- `Docker engine isn't reachable`: 서버 Docker 실행 권한/데몬 확인. 생성만 할 때 `RUNNER=none`.
- `CUDA out of memory`: 먼저 `--block-size 64 --tokens 128`로 축소. 서버 RAM/VRAM 사용량과 입력 프롬프트 길이 확인.
- `Unsupported custom 4D attention mask`: DiffuCoder custom model/Transformers 버전이 다를 수 있습니다. `transformers==4.51.3` 확인. 4D SDPA 마스크가 실제로 모든 층에 적용되는지 실모델 검증이 필요합니다.
- `MultiPL-E` 다운로드 실패: https://datasets-server.huggingface.co/rows API 접근 가능 여부 확인.

**요약:** 코드는 가능한 한 바로 실행하도록 구성했고 CPU 단위·통합 테스트를 통과했지만, 이 환경에는 7B 모델 가중치와 서버 GPU가 없어 **실제 7B + 4D attention 전체 실행 성공을 보장할 수 없습니다.** 최초 서버 실행 결과의 로그를 바탕으로 이 부분의 호환성을 확정해야 합니다.
