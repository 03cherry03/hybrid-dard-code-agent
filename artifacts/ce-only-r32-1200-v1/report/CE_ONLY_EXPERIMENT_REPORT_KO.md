# CE-only Diffusion + DARD 실험 결과

## 먼저 읽을 결론

이번 실험에서 **LoRA fine-tuning과 masked diffusion CE 학습을 거친 DiffuCoder에 DARD를 적용하면 native Diffusion보다 정확도가 크게 올라갔습니다.** 그러나 AR 모델의 정확도까지 따라잡지는 못했고, 평균 생성 시간도 AR보다 짧아지지 않았습니다.

Validation 기준 결과는 다음과 같습니다.

| 모델 | 정확도 | 평균 생성 시간 |
|---|---:|---:|
| AR | **72.53%** | 1.539초 |
| Diffusion only | 4.03% | **1.362초** |
| Diffusion + DARD | **41.39%** | 1.709초 |

즉, **정확도는 AR > DARD > Diffusion only**였지만, 속도는 **Diffusion only < AR < DARD**였습니다.

## 1. 실험 목표

목표는 다음과 같은 품질·속도 순서를 만드는 것이었습니다.

- AR: 가장 정확하지만 느림
- Diffusion only: 가장 빠르지만 정확도 낮음
- Diffusion + DARD: Diffusion only보다 정확하고 AR보다 빠름
- DARD와 AR의 정확도 차이: 5%p 이내

최종 결과는 정확도 순서와 DARD의 native Diffusion 대비 개선은 달성했지만, **DARD가 AR보다 빠르고 정확도 차이가 5%p 이내라는 조건은 달성하지 못했습니다.**

## 2. 세 가지 핵심 개념

### LoRA fine-tuning

기존 DiffuCoder 본체의 대부분의 가중치는 고정하고, 작은 저랭크 adapter만 학습하는 방법입니다. 전체 모델을 다시 학습하는 것보다 메모리와 시간이 적게 듭니다.

### Masked diffusion CE

코드의 일부 token을 가린 뒤 원래 token을 복원하도록 학습하는 손실입니다. 이 실험에서는 **train split**에서만 CE 학습을 했습니다. Validation set에 학습하지 않았습니다.

### DARD

Diffusion 모델이 한 번에 모든 token을 확정하지 않고, 확신이 높은 token부터 block 단위로 확정합니다. 확신이 낮거나 검증에 실패한 token은 다시 가리고 재생성합니다. 그래서 native Diffusion보다 정확도를 높일 수 있지만, 재생성·검증 비용이 추가됩니다.

DARD의 한 번의 진행은 다음 순서입니다.

1. 아직 모르는 위치를 `MASK`로 둡니다.
2. 모델이 각 위치의 token과 confidence를 예측합니다.
3. confidence가 높은 token은 확정하고, 낮은 token은 다시 `MASK`로 돌립니다.
4. 정해진 횟수 안에 끝나지 않으면 `incomplete`로 기록합니다.

보고서에서 자주 나오는 DARD 설정도 같은 흐름으로 읽으면 됩니다. `steps_per_block`은 한 block에서 반복할 최대 횟수이고, `transfer_limit`은 한 번에 확정할 수 있는 최대 token 수입니다. `completion_verification_steps`는 제한 시간에 도달한 뒤 남은 token을 다시 확인하는 횟수입니다. 값을 키우면 완료율은 올라갈 수 있지만 forward 수와 생성 시간이 늘어납니다.

실험 절차는 다음과 같습니다.

1. Train split으로 DiffuCoder LoRA를 CE 학습
2. Validation set에서 DARD 파라미터 후보 비교
3. 가장 좋은 후보를 고정
4. Validation 반복과 Holdout으로 최종 확인

### 2.4 청중을 위한 용어 설명

아래 용어는 이번 실험에서 반복해서 사용되므로, 먼저 의미를 고정합니다.

| 용어 | 쉬운 뜻 | 이번 실험에서의 역할 |
|---|---|---|
| AR | 왼쪽에서 오른쪽으로 token을 하나씩 생성하는 방식 | 정확도 기준선입니다. 느리지만 앞에서 만든 코드가 뒤 token의 문맥이 됩니다. |
| Diffusion only | 코드 전체를 가린 뒤 여러 번에 걸쳐 token을 채우는 방식 | 빠르지만 현재 native 설정에서는 정확도가 매우 낮았습니다. |
| Native Diffusion | DARD의 확정·재검증 규칙을 적용하지 않은 기본 Diffusion decoding | `Diffusion only`와 같은 비교 기준입니다. |
| DARD | 확신이 높은 token부터 확정하고, 의심스러운 token은 다시 가려 검증하는 방식 | Diffusion의 정확도를 높이는 대신 추가 검증 시간이 듭니다. |
| Token | 코드가 쪼개진 작은 단위 | 모델은 문장 전체가 아니라 token 단위로 예측합니다. |
| Mask | 아직 정하지 않은 token을 나타내는 빈칸 | 빈칸이 남아 있으면 출력은 `incomplete`입니다. |
| Confidence | 모델이 방금 고른 token에 부여한 확률 | 정답 확률이나 테스트 통과 확률 자체는 아니며, DARD가 확정할 token과 다시 시도할 token을 나누는 기준입니다. |
| Verification | 이미 고른 token을 다시 확인하는 단계 | 잘못 고른 token을 되돌리지만, 추가 forward와 시간이 필요합니다. |
| Block | 한 번에 처리하는 token 묶음 | 현재 설정에서는 최대 128 token씩 처리합니다. |
| `tau_c` | 후보 token으로 인정하는 confidence 기준 | 이 값보다 낮으면 token을 다시 가립니다. |
| `tau_u` | 최종 확정으로 인정하는 confidence 기준 | 이 값보다 높아야 검증 완료 token으로 기록합니다. |
| Incomplete | 제한된 반복 안에 모든 token을 확정하지 못한 출력 | 출력이 끝까지 완성되지 않았다는 뜻이지, 코드가 반드시 틀렸다는 뜻은 아닙니다. |
| Certified | DARD의 token 검증을 통과한 출력 | 기능 테스트 통과를 보장하지 않으므로 최종 `pass`와 구분합니다. |
| Pass | compile과 기능 테스트를 모두 통과한 출력 | 최종 정확도에 포함되는 유일한 상태입니다. |
| CE | 정답 token의 확률을 높이도록 학습하는 손실 | 현재 모델은 CE만 학습했고 DARD의 전체 진행 과정은 학습하지 않았습니다. |
| CE-only | CE 손실만 사용한 학습 | 이번 기준 모델은 DARD 전용 손실이나 distillation을 사용하지 않았습니다. |
| LoRA | 본체는 고정하고 작은 adapter만 학습하는 방법 | 짧은 시간과 적은 메모리로 모델을 조정하기 위해 사용했습니다. |
| Source | 같은 원문 문제를 대표하는 데이터 단위 | source가 적으면 같은 유형을 반복 학습해 새로운 문제에 약해질 수 있습니다. |
| Pseudo-label | 사람이 직접 만든 정답 대신 다른 모델이 만든 학습 target | C++·JavaScript·Rust 일부가 이 방식에 의존합니다. |
| Fallback | 1차 모델이 불확실할 때 다른 모델을 추가 호출하는 방법 | DARD가 실패한 일부 task만 AR로 다시 생성하는 실험입니다. |
| Validation | 설정을 고르는 데이터 분할 | 실험 중 여러 후보를 비교하는 데 사용합니다. |
| Holdout | 마지막에 한 번만 보는 데이터 분할 | 최종 후보의 성능을 확인하는 데 사용하며, 중간 튜닝에는 사용하지 않습니다. |

### 2.5 이번 문제를 한 문장으로 요약하면

현재 모델은 **빈칸의 token을 맞히는 학습은 했지만, DARD가 요구하는 “무엇을 먼저 확정하고 언제 되돌릴지”까지 충분히 배우지 못했습니다.** 그 결과 일부 출력은 끝까지 확정되지 않고, 끝까지 확정된 출력도 compile/test를 통과하지 못합니다. 여기에 Rust·C++의 source 다양성 부족과 DARD의 재검증 시간이 겹쳤습니다.

## 3. 사용한 모델과 데이터

| 비교 대상 | 모델과 설정 |
|---|---|
| AR | `Qwen/Qwen2.5-Coder-7B-Instruct`, 512 token, `code_fence` |
| Diffusion only | `apple/DiffuCoder-7B-cpGRPO` + CE LoRA, native 16단계 |
| Diffusion + DARD | 같은 DiffuCoder와 같은 LoRA adapter + DARD |

데이터는 HumanEval/HumanEval+, MBPP/MBPP+, 그리고 MultiPL-E 기반 C++·JavaScript·Rust 번역 변형을 포함합니다. 원시 task는 3,306개였고, 중복 제거 후 2,764개, canonical source는 1,138개였습니다. Train, Validation, Holdout은 source 기준으로 겹치지 않게 나눴습니다.

| 분할 | 전체 | Python | C++ | JavaScript | Rust |
|---|---:|---:|---:|---:|---:|
| Train task | 2,218 | 912 | 448 | 448 | 410 |
| Validation | 273 | 113 | 55 | 55 | 50 |
| Holdout | 273 | 113 | 55 | 55 | 50 |

Validation과 Holdout에는 각각 MBPP, MBPP+, HumanEval+ 및 각 언어 번역 변형이 들어 있습니다. 실제 학습 JSONL은 1,593개였고, 길이 제한 후 1,546개가 학습에 사용됐습니다. 언어 균형 sampler로 각 언어를 912개 기준으로 맞췄습니다.

## 4. 무엇을 어떻게 평가했는가

각 task의 사용자 prompt를 모델에 넣고 최대 512 token까지 코드를 생성했습니다. 모든 모델은 같은 `code_fence`, temperature 0, top-p 1 조건을 사용했습니다. 생성된 텍스트는 공통 추출기 v2로 코드 부분만 분리한 뒤, 언어별 compile/test를 실행했습니다. 테스트를 통과한 task만 정답으로 계산했습니다.

- Validation: DARD 후보를 선택하는 데 사용
- 추가 반복: seed 20261111, 20261112, 20261113
- Holdout: 후보를 고정한 뒤 한 번만 최종 확인
- 생성 timeout: task당 120초
- 채점 worker: 8개, queue 64개, grading timeout 300초

평가 지표의 뜻은 다음과 같습니다.

- **정확도**: compile/test를 통과한 task 비율입니다.
- **종합 정확도**: 언어별 평균이 아니라 전체 273개 task의 `passed / 273`입니다.
- **평균 생성 시간**: 코드 생성에 걸린 시간이며 채점 시간은 별도입니다.
- **p95 생성 시간**: 가장 느린 5%를 반영하는 시간입니다.
- **Forward 호출 수**: 모델을 실행한 횟수입니다.
- **GPU peak 메모리**: 생성 중 한 GPU의 최대 할당 메모리입니다. 전력·금액 비용은 측정하지 않았습니다.
- **Incomplete**: DARD가 끝까지 확정하지 못한 출력입니다.
- **Verification certified**: DARD 검증을 통과한 출력입니다.

## 5. Diffusion 모델 학습

| 항목 | 값 |
|---|---:|
| Base model | DiffuCoder-7B-cpGRPO |
| LoRA rank / alpha | 32 / 64 |
| 학습 대상 | q/k/v/o 및 gate/up/down projection |
| Optimizer step | 1,200 |
| Learning rate | 1e-5 |
| Max length | 512 |
| GPU | RTX 4090 4개, DDP |
| Precision | bfloat16 |
| 학습 시간 | 약 733.7초 |
| 학습 GPU peak | GPU당 약 16.44 GiB |
| Trainable parameter | 80,740,352개 |

CE loss는 token을 확률적으로 mask한 뒤 복원하도록 계산했습니다. 첫 loss는 약 0.757, 마지막 loss는 약 0.231이었습니다.

이번 최종 결과에는 distillation이나 DARD 전용 학습 손실을 사용하지 않았습니다. **CE로 기본 Diffusion 모델의 token 복원 분포를 개선한 뒤, 같은 adapter에 native decoding과 DARD decoding을 각각 적용했습니다.**

## 6. 언어별 Validation 결과

아래 표에는 AR의 정확도와 native Diffusion/DARD의 언어별 정확도·생성 시간·GPU 메모리를 함께 제시합니다.

| 언어 | AR 정확도 | Native 정확도 | DARD 정확도 | Native 시간 | DARD 시간 | Native GPU | DARD GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python, 113개 | 66.37% | 6.19% | **41.59%** | 1.264초 | 1.631초 | 14.960 GiB | 14.678 GiB |
| C++, 55개 | 76.36% | 3.64% | **40.00%** | 1.472초 | 1.937초 | 14.975 GiB | 14.678 GiB |
| JavaScript, 55개 | 80.00% | 3.64% | **54.55%** | 1.393초 | 1.542초 | 14.966 GiB | 14.678 GiB |
| Rust, 50개 | 74.00% | 0.00% | **28.00%** | 1.430초 | 1.820초 | 14.970 GiB | 14.678 GiB |
| **전체 273개** | **72.53%** | **4.03%** | **41.39%** | **1.362초** | **1.709초** | **14.966 GiB** | **14.678 GiB** |

DARD는 네 언어 모두 native Diffusion보다 높았습니다. 가장 높은 언어는 JavaScript였고, Rust가 가장 낮았습니다. DARD의 시간과 메모리는 native보다 언어별로 모두 증가하거나 비슷한 수준이며, 정확도 향상을 위해 추가 검증 계산을 수행한 결과입니다.

Holdout 종합 결과도 같은 순서였습니다.

| 모델 | 정확도 | 평균 시간 | p95 시간 |
|---|---:|---:|---:|
| AR | **65.93%** | 1.678초 | 3.697초 |
| Diffusion only | 3.66% | **1.381초** | 1.754초 |
| Diffusion + DARD | **35.16%** | 1.744초 | **2.589초** |

Holdout의 DARD 언어별 정확도는 Python 38.94%, C++ 36.36%, JavaScript 43.64%, Rust 16.00%였습니다.

## 7. 정확도·시간·메모리 비용 비교

### Validation

| 모델 | 정확도 | 평균 시간 | p95 | 평균 forward | 평균 GPU peak |
|---|---:|---:|---:|---:|---:|
| AR | **72.53%** | 1.539초 | 3.478초 | 77.08 | 14.225 GiB |
| Diffusion only | 4.03% | **1.362초** | **1.622초** | 16.00 | 14.966 GiB |
| Diffusion + DARD | **41.39%** | 1.709초 | 2.674초 | 24.27 | 14.678 GiB |

DARD는 native Diffusion보다 정확도가 **37.36%p** 높았고, 평균 생성 시간은 **25.5%** 증가했습니다. AR보다 정확도는 **31.14%p 낮았고**, 평균 생성 시간은 **11.1% 더 길었습니다**.

DARD의 forward 횟수는 AR보다 적었지만, token 재마스킹과 검증 과정의 제어 비용 때문에 실제 평균 시간은 더 길었습니다. GPU 메모리는 native Diffusion보다 약 0.29 GiB 낮았고 AR보다 약 0.45 GiB 높았습니다.

DARD는 Validation에서 123/273개, 즉 45.05%가 incomplete였습니다. 강제 채움은 사용하지 않았기 때문에 잘못된 코드를 억지로 완성하지는 않았지만, 미완료 출력이 정확도를 제한했습니다.

## 8. 최종 결론

이번 실험에서 확인한 내용은 명확합니다.

- LoRA + masked diffusion CE 학습으로 DiffuCoder의 기본 복원 능력을 개선했습니다.
- 같은 학습 모델에 DARD를 적용하면 native Diffusion보다 정확도가 크게 높아졌습니다.
- 그러나 Diffusion + DARD의 정확도는 AR보다 여전히 30%p 이상 낮았습니다.
- DARD는 평균 생성 시간에서도 AR보다 빠르지 않았습니다.
- 따라서 이번 결과는 **“Diffusion 모델과 DARD의 조합이 native Diffusion보다 훨씬 좋아졌지만, AR 모델을 대체할 수준에는 아직 도달하지 못했다”**고 해석해야 합니다.

근거 파일:

- [Validation 결과](</home/leejwian1955/hybrid-dard-model-upgrade-20261108/focused-target-r32-1200-full/validation/results.json>)
- [Holdout 결과](</home/leejwian1955/hybrid-dard-model-upgrade-20261108/focused-target-r32-1200-full/holdout/results.json>)
- [학습 설정](</home/leejwian1955/hybrid-dard-model-upgrade-20261108/focused-target-r32-1200/training/training_config.json>)
- [학습 상태](</home/leejwian1955/hybrid-dard-model-upgrade-20261108/focused-target-r32-1200/training/training_state.json>)

## 9. 목표 미달 원인 분석

### 9.1 목표와 실제 격차

Holdout을 최종 기준으로 보면 DARD는 96/273개(35.16%)를 통과했고 AR은 180/273개(65.93%)를 통과했습니다. 목표인 “AR보다 5%p 이내”를 만족하려면 DARD가 최소 60.93%에 도달해야 하므로 현재 격차는 **25.77%p**입니다. AR과 DARD의 절대 차이는 **30.77%p**입니다.

속도도 목표와 반대입니다. Holdout 평균 생성 시간은 DARD 1.744초, AR 1.678초로 DARD가 0.066초(3.9%) 느립니다. Validation에서는 DARD 1.709초, AR 1.539초로 0.170초(11.1%) 느립니다.

Validation에서 Holdout으로 내려갈 때 DARD 정확도는 41.39%에서 35.16%로 6.23%p 하락했습니다. AR도 72.53%에서 65.93%로 6.59%p 하락했으므로, 단순한 DARD 설정 실패뿐 아니라 split 난이도와 일반화 격차도 함께 관리해야 합니다.

### 9.2 관측된 실패의 분해

Holdout DARD 273개를 진단 파일 기준으로 다시 분해하면 다음과 같습니다.

| 진단 항목 | 건수 | 비율 | 해석 |
|---|---:|---:|---|
| verification certified | 168 | 61.54% | DARD 검증을 통과하고 끝까지 확정된 출력 |
| incomplete | 105 | 38.46% | decode budget 또는 completion verification budget 안에 확정하지 못한 출력 |
| certified 출력 중 pass | 72/168 | 42.86% | 완료율을 올려도 모델·코드 품질 문제가 남음 |
| incomplete 출력 중 pass | 24/105 | 22.86% | 미완료가 정확도를 추가로 낮춤 |

Incomplete 105건의 종료 원인은 `unverified_at_decode_budget` 67건, `completion_rejected` 38건입니다. incomplete를 모두 완료 상태로 바꾸고 코드 품질이 그대로라면 현재 통과 건수 96/273(35.16%)에서 크게 올라가지 않습니다. 현재 certified subset의 조건부 pass율도 72/168(42.86%)이므로, 완료율 개선만으로 목표 60.93%를 만들 수 없습니다.

완료된 출력에서도 실패가 많습니다. Holdout 전체 실패 상태는 `test_failed` 116건, `compile_error` 43건, `syntax_error` 16건, `output_format_error` 1건, `timeout` 1건입니다. 즉 문제는 종료 조건뿐 아니라 함수 의미·문법·컴파일 가능성까지 포함합니다.

언어별로는 Rust가 가장 큰 병목입니다. Holdout 정확도는 Python 38.94%, C++ 36.36%, JavaScript 43.64%, Rust 16.00%이며, Python은 incomplete도 60/113건(53.1%)으로 높습니다. Rust는 incomplete를 제외한 완료 출력 자체도 8/36(22.2%)만 통과해 별도 데이터·학습 처리가 필요합니다.

동일 task의 paired 결과를 보면 AR과 DARD가 모두 통과한 task는 87개, AR만 통과한 task는 93개, DARD만 통과한 task는 9개, 둘 다 실패한 task는 84개입니다. 두 모델의 정답 합집합은 189/273개(69.23%)이므로, 낮은 confidence·incomplete 사례에만 AR을 선택적으로 재생성하는 fallback은 전체 AR 실행보다 속도 비용을 제한하면서 목표에 접근할 후보입니다.

다만 incomplete에만 fallback을 적용하면, 현재 관측된 paired 결과 기준 예상치는 132/273(48.35%)입니다. 따라서 selector는 incomplete뿐 아니라 완료됐지만 DARD가 실패하고 AR이 통과한 사례도 식별해야 하며, fallback 비율과 latency를 함께 최적화해야 합니다.

### 9.3 원인별 판단

| 원인 | 현재 근거 | 문제의 형태 | 확인이 필요한 가설 |
|---|---|---|---|
| 기반 모델·비교 상한 불일치 | AR은 `Qwen2.5-Coder-7B-Instruct`, DARD는 `DiffuCoder-7B-cpGRPO` | decoding 방법 차이와 backbone 능력 차이가 한 숫자에 섞임 | 같은 DiffuCoder의 AR/native 기준을 측정해야 DARD 알고리즘의 순수 손실을 분리할 수 있음 |
| 학습 목적과 DARD 추론의 불일치 | 학습은 masked CE만 사용했고 DARD trajectory·verification·distillation loss는 없음 | 무작위 mask 복원 분포와 DARD의 confidence-ordered 재마스킹 분포가 다름 | DARD 상태를 재현하는 mask schedule과 verification consistency loss가 confidence와 pass rate를 개선하는지 확인 |
| 데이터 양·분포·길이 문제 | 학습 입력 1,593개 중 1,546개만 사용, 512 token 초과 예제 47개 제거, post-filter 언어 수가 Py 892/C++ 215/JS 227/Rust 212 | balanced sampler는 소수 언어를 반복할 뿐 새로운 사례를 만들지 못하며 긴 코드와 Rust·C++ 변형의 coverage가 낮음 | 길이 보존·packing, 언어별 token budget, Rust/C++ hard set과 증강이 일반화를 개선하는지 확인 |
| DARD confidence·종료 정책 | 선택 설정은 block 128, steps 64, `tau_c=.4`, `tau_u=.75`, transfer 8, completion verification 2회 | threshold와 transfer 제한이 cycle/fixed-point를 만들고 미완료·재검증을 증가시킴 | block/step/threshold/transfer/completion budget을 조정하면 incomplete와 pass의 Pareto가 개선되는지 확인 |
| 속도 구현 오버헤드 | DARD 평균 forward 24.27회로 AR 77.08회보다 적지만 시간이 더 김 | packed attention 재구성, Python 제어, 재마스킹·검증 forward가 호출 수 외 비용을 만듦 | KV/cache 재사용, batch verification, early exit가 정확도 손실 없이 시간을 줄이는지 확인 |
| 언어별 calibration 부족 | Holdout Rust 16.00%, Python incomplete 53.1% | 하나의 전역 threshold가 언어·길이별 confidence 분포를 제대로 반영하지 못함 | 언어·길이별 threshold와 adapter/head가 균형 잡힌 개선을 만드는지 확인 |

이 중 **확정된 병목**은 (a) DARD의 높은 incomplete 비율, (b) complete 출력에서도 발생하는 compile/test 실패, (c) DARD의 추가 제어 비용입니다. 학습 목적 불일치, 데이터 coverage, 언어별 calibration, backbone 효과의 크기는 다음 실험으로 분리할 **검증 가설**입니다.

추가로 translated C++·JavaScript·Rust 학습 행은 canonical reference가 아니라 `verified_ar_teacher` pseudo-label에 의존하고, 언어별 유일 source도 218~232개 수준입니다. 행 수를 912개로 맞춘 것만으로는 Python 912개 source와 같은 감독 다양성을 확보하지 못하므로, source 중복 제거·teacher confidence 필터·실제 reference 번역을 별도 비교해야 합니다.

DARD 구현 자체도 저자 코드의 직접 재현이 아닌 Dream backbone용 독립 adapter이며, 현재 파이프라인에는 cross-step KV cache 재사용과 검증 생략이 없습니다. 따라서 attention packing·shift 정렬·공식 sampler와의 차이가 품질과 속도에 미치는 영향도 native parity와 compact/rectangular on/off A/B로 확인해야 합니다.

### 9.4 문제와 실험의 연결

| 관측된 문제 | 왜 문제가 되는가 | 대응 실험 |
|---|---|---|
| incomplete 105/273개 | 코드가 끝까지 확정되지 않아 정답 후보가 될 수 없음 | 실험 1: DARD 종료 예산 확대 |
| CE-only 학습과 DARD 진행 방식의 차이 | 모델이 token 복원은 배웠지만 반복 확정·재가림은 충분히 배우지 못함 | 실험 2: 고마스크 CE 학습 |
| Rust·C++의 source 다양성 부족 | 새로운 문제 유형을 배울 기회가 적어 언어별 일반화가 약함 | 실험 3: 소스 균형 데이터 |
| DARD가 AR만 통과한 93개 task를 놓침 | DARD 단독 정확도의 상한을 낮춤 | 실험 4: 선택적 AR 보완 |

## 10. 시간 제약을 반영한 핵심 실험 4개

시간이 제한되어 이번 cycle에서는 대규모 grid, 여러 LoRA rank 조합, language adapter, KV-cache 재작성은 진행하지 않습니다. 아래 4개만 수행하고, 각 실험에서는 한 가지 개선 방향만 비교합니다.

실험별 성공 여부는 dev/validation에서 판단하고, holdout은 4주차 최종 후보를 고정한 뒤 한 번만 확인합니다.

| 실험 | 가장 큰 문제 | 재학습 | 우선 지표 |
|---|---|---:|---|
| 1. DARD 종료 조건 | incomplete 38.46% | 아니오 | incomplete, pass, 평균·p95 시간 |
| 2. 고마스크 CE 학습 | CE 학습과 DARD 복원 과정의 차이 | 예 | pass, incomplete, NLL/ECE |
| 3. 소스 균형 데이터 | Rust·C++ 소스 다양성 부족 | 예 | 전체·언어별 pass |
| 4. 선택적 AR 보완 | DARD가 놓치는 task를 보완 | 아니오 | pass, fallback 비율, 시간 |

### 실험 1 — DARD 종료 조건만 완화

**왜 하는가:** Holdout 105/273개가 incomplete이고, 그중 67개는 decode budget 부족, 38개는 completion rejection으로 끝났습니다.

**무엇만 바꾸는가:** 현재 설정과 다음 한 개 후보만 비교합니다.

- 현재: `steps_per_block=64`, `transfer_limit=8`, `completion_verification_steps=2`
- 후보: `steps_per_block=128`, `transfer_limit=16`, `completion_verification_steps=4`
- `tau_c=.4`, `tau_u=.75`, block size 128 등 나머지는 유지

**비교 대상:** 동일 checkpoint·동일 task·동일 seed의 현재 설정.

**성공 판단:** incomplete가 20% 이하로 내려가고 pass가 8%p 이상 오르면 채택합니다. 평균 시간이 5% 넘게 늘면 속도 후보에서 제외합니다.

### 실험 2 — DARD에 맞춘 고마스크 CE 학습

**왜 하는가:** 현재 학습은 `min_mask_prob=.15`의 masked CE만 사용했지만, DARD는 많은 token을 동시에 mask하고 반복적으로 복원합니다.

**무엇만 바꾸는가:** 학습 설정에서 `min_mask_prob`만 `.15 → .50`으로 바꿉니다. LoRA rank, 학습 step 1,200, learning rate, 데이터, seed는 유지합니다.

**비교 대상:** 기존 CE-only r32/1,200 checkpoint.

**성공 판단:** validation pass가 5%p 이상 오르거나 incomplete가 10%p 이상 줄면 후보로 남깁니다. 둘 다 없으면 이 방향은 중단합니다.

### 실험 3 — 소스 균형 데이터로 언어 다양성 보강

**왜 하는가:** Holdout Rust pass는 16.00%이고, 학습 후 유일 source는 Python 912개에 비해 C++ 232개, JavaScript 231개, Rust 218개뿐입니다. 행 수를 맞춘 sampler만으로는 source 다양성이 늘지 않습니다.

**무엇만 바꾸는가:** 현재 sampler 대신 source 기준으로 중복을 제거하고 Rust·C++의 hard/long example을 우선 포함한 데이터 변형을 한 개 만듭니다. 모델 구조와 학습 step은 유지합니다.

**비교 대상:** 현재 1,546개 filtered data와 동일한 r32/1,200 학습.

**성공 판단:** Rust pass가 10%p 이상 오르고 전체 pass가 2%p 이상 떨어지지 않으면 채택합니다. Rust만 오르고 다른 언어가 크게 하락하면 보류합니다.

### 실험 4 — 선택적 AR 보완

**왜 하는가:** paired Holdout에서 AR만 통과한 task가 93개, DARD만 통과한 task가 9개입니다. DARD가 낮은 confidence를 보이는 일부 task만 AR로 재생성하면 전체 AR 실행보다 비용을 줄일 수 있습니다.

**무엇만 바꾸는가:** DARD checkpoint는 고정하고 다음 세 정책만 비교합니다.

1. DARD만 실행
2. incomplete인 경우에만 AR 재생성
3. incomplete이거나 validation에서 정한 low-confidence인 경우 AR 재생성

**비교 대상:** 동일 task의 DARD-only 결과.

**성공 판단:** DARD-only보다 pass가 10%p 이상 오르고 fallback 비율이 30% 이하인 후보만 검토합니다. 평균 시간이 AR보다 느리면 최종 속도 후보에서는 제외합니다.

이번 cycle에서는 동일 backbone AR 상한 측정, 대규모 decoder grid, KV-cache 구현, language adapter, distillation loss를 후속 과제로 남깁니다. 이 항목들은 원인 분석에는 유용하지만 현재 4주 안에 핵심 개선 효과를 확인하기에는 범위가 큽니다.

## 11. 단순화한 4주 진행 순서

실행 캘린더는 KST 기준 1주차 2026-11-09~11-15, 2주차 11-16~11-22, 3주차 11-23~11-29, 4주차 11-30~12-06으로 잡고 12-07에 최종 보고합니다.

### 1주차 — baseline 고정과 실험 1

- 현재 checkpoint와 동일 task를 고정합니다.
- 실험 1의 현재 설정과 후보 설정만 비교합니다.
- incomplete, pass, 평균·p95 시간을 기록하고 다음 주 후보를 한 개만 선택합니다.

### 2주차 — 실험 2와 실험 3

- 고마스크 CE와 소스 균형 데이터를 각각 한 번씩 학습합니다.
- 작은 dev set에서 먼저 비교하고, 기준을 통과한 후보만 validation 전체로 확대합니다.
- 기준을 통과한 후보가 없으면 기존 checkpoint를 유지합니다.

### 3주차 — 실험 4

- 실험 1~3 중 가장 좋은 checkpoint를 고정합니다.
- 세 가지 fallback 정책을 validation에서 비교합니다.
- pass 개선과 fallback 비율을 함께 보고, 속도 기준을 넘는 정책은 제거합니다.

### 4주차 — 최종 확인

- 최종 후보 1개만 고정하고 seed 3회 validation을 수행합니다.
- 설정과 checkpoint hash를 봉인한 뒤 holdout 273개를 한 번 평가합니다.
- 전체 pass, 언어별 pass, incomplete, fallback 비율, 평균·p95 시간을 보고합니다.

## 12. 최종 판정 기준과 현재 진행 상태

- **품질 목표:** Holdout DARD pass rate ≥60.93%(AR 65.93%보다 5%p 이내)
- **속도 목표:** 평균 생성 시간 <1.678초(holdout AR)
- **운영 기준:** incomplete ≤20%, fallback 비율 ≤30%
- **보고 방식:** 실험별 pass 변화와 실패 유형을 함께 기록하고, 목표를 달성하지 못하면 가장 효과가 큰 단일 개선만 다음 cycle로 넘깁니다.

현재 문서는 실제 개선 실험 결과가 아니라, 시간 제약에 맞춰 축소한 실행 계획입니다. 각 실험은 “왜 하는가 → 무엇만 바꾸는가 → 무엇과 비교하는가 → 언제 채택하는가” 순서로 판단합니다.
