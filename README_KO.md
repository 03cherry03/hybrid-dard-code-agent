# Hybrid DARD Code Agent — 독립 코드 생성 평가

Qwen2.5-Coder-7B-Instruct, DiffuCoder-7B-Instruct 기본 생성, 같은 DiffuCoder+DARD를 공통 코드 생성·수정 절차로 비교합니다. 기본 목적은 **Python·C++·Rust·JavaScript 함수 코드의 정확도와 생성·수정 시간 비교**입니다. 라이브러리 활용 평가인 BigCodeBench는 실험 범위에서 제외합니다. DARD는 독립 Dream adapter 구현이며, 실제 모델의 속도·정확도 향상은 아직 확인하지 않았습니다.

현재 마감 실험은 `--focused-codegen` 경로를 사용합니다. AR·DARD·Diffusion-only를 같은 Python·Rust·JavaScript·C++ 문제로 비교하며, 반복 1회·수정 피드백 0회입니다. 생성 예산은 512토큰이고 기본 Diffusion은 64단계, DARD는 512토큰 전체 영역에서 최대 64번의 검증 단계를 사용합니다. DARD 임계값은 `.5/.8`이며 임계값을 넘은 토큰 수에 별도 상한을 두지 않습니다. 설정은 `generation_settings.json`에 고정됩니다.

수정된 DARD는 원본/마스크 shadow 키, 신뢰도 순서의 후보 검증, M/C/U 재검증과 논문 식 `(P+p0)/(P+D+p0)`를 구현합니다. Dream의 한 위치 이동 예측을 위해 별도 query를 사용하므로 저자 코드의 재현이라고 주장하지 않습니다. 모든 위치가 마스크인 경우 native 모델 연산을 사용하며, 결정적 고정점·정확한 반복 주기를 감지하면 기록 후 제한된 완료 단계로 넘어갑니다. 추가 강제 채움 횟수와 비용은 별도 기록됩니다. 전체 생성 토큰 ID·제어 토큰 제거 전 원문·종료 토큰 ID/위치도 보존합니다. 속도와 정확도 개선 여부는 실제 채점 결과로 판단합니다.

## 계산 최적화와 제한된 개발 실험

`dard_attention.py`는 미생성 M 위치에 필요한 원본 예측 query만 유지합니다. `dard_model_adapter.py`는 packed DARD에서 K/V를 실제 문맥 `P+2B` 위치에만 계산합니다. 따라서 query 길이는 `P+4B`에서 `P+3B+|M|`로 줄고 attention은 직사각형이 됩니다. 일반 Diffusion 경로는 기존 Dream attention을 호출합니다. 단계 간 KV 재사용이나 검증 생략은 없습니다. `--no-compact-queries --no-rectangular-attention`으로 기존 계산을 비교할 수 있습니다. BF16 커널 변경으로 출력이 달라질 수 있으므로 CPU 동등성만으로 성능을 확정하지 않습니다.

`scripts/optimize_experiments.py`는 기존 기준 실험 종료와 GPU 해제를 기다린 뒤, `scripts/validate_dard_optimization.py`의 실제 7B 수치 검증을 통과해야 개발 평가를 시작합니다. 문제 ID만으로 고정한 개발 16문제×4언어는 새 최종 평가 344문제×4언어에서 제외됩니다. 기존 전체 기준 결과는 탐색적 자료로 보존합니다. Instruct와 선택적으로 공식 cpGRPO 한 모델만 비교하며, native 256단계는 품질 원인 분석에만 사용합니다. 최종 비교는 선택한 동일 diffusion 가중치의 DARD·native64 및 AR입니다. 3%p 정확도 허용 차이는 사전 개발 선택 기준이며 통계적 비열등성의 증명이 아닙니다.

개발·최종 평가, 모델 및 소스 해시, 후보 선택 이유, 강제 채움 위치/토큰/확률, 언어별 결과와 생성시간을 기록합니다. 강제 채움 상세는 생성당 최대 512토큰이며 초과 개수도 남깁니다. 이상 종료는 기존 SMTP 알림과 재시도 기록을 사용합니다. 2026-11-07 17:00 KST 마감과 제한된 개발 예산을 넘겨 추가 설정 탐색을 계속하지 않습니다.

## 전체 실험 실행

~~~bash
cd /home/leejwian1955/hybrid-dard-code-agent
bash scripts/run_all_experiments.sh --scope functions --prepare --execute \
  --output /data/hybrid-dard/results/code-generation
~~~

`--scope functions`가 기본값입니다. `--prepare`는 추론 환경·언어 도구·EvalPlus와 두 모델을 준비합니다. 이미 준비했으면 빼도 됩니다. `--execute`를 빼면 계획과 메일 설정 메타데이터만 저장하며 설치·추론·SMTP 접속은 없습니다. 이전 실험 결과는 보존하고 새 `code-generation` 경로를 사용합니다.

이 경로에는 SWE-bench 데이터 검사·저장소 준비·시스템 패키지 검사·BigCodeBench 설치가 없습니다. TeX·브라우저·추가 로케일 및 `sudo` 준비가 필요하지 않습니다. 기존 SWE/BigCodeBench 데이터가 없어도 선택한 코드 생성 데이터만 검증합니다.

진행 순서는 GPU 점유·코드 생성 데이터 검사 → 선택적 환경·모델 준비 → 소규모 함수 평가 → 전체 함수 생성·수정·최종 채점 → 분석입니다. 모든 생성 단계는 네 GPU를 사용합니다. 선택한 문제의 누락이나 실행기 오류를 조용히 건너뛰지 않습니다.

긴 실행:

~~~bash
nohup bash scripts/run_all_experiments.sh --scope functions --prepare --execute \
  --output /data/hybrid-dard/results/code-generation \
  > /data/hybrid-dard/results/code-generation.launch.log 2>&1 &
echo "$!"
~~~

## 환경과 저장 위치

기본 경로는 `/data/hybrid-dard`입니다. `HYBRID_DARD_DATA_ROOT`로 바꿀 수 있으며, 준비 스크립트는 `/data` 하위를 요구합니다. 기존 Conda를 사용해 아래 전용 환경을 만들며 base 환경에는 패키지를 설치하지 않습니다.

| 경로 | 용도 |
| --- | --- |
| `datasets/` | 원본 데이터·공개 과제·최종 채점 자료·checksum |
| `envs/inference` | Python 3.11, torch 2.7.1+cu118, Transformers 4.51.3 |
| `envs/function` | Python·Node.js·Rust·C/C++와 빌드 도구 |
| `envs/evalplus` | Python 3.11, EvalPlus 0.3.1 |
| `envs/locks/` | 실제 설치된 pip/Conda 패키지 목록 |
| `cache/`, `models/` | 공유 캐시와 고정 모델 snapshot |
| `work/` | 매 평가에서 새로 만드는 작업 디렉터리·환경 복제본 |
| `results/` | 실험 계획·패치·로그·채점·분석·메일 영수증 |

준비 명령만 따로 실행할 수도 있습니다.

~~~bash
source scripts/runtime_env.sh
bash scripts/setup_env.sh functions
"$HYBRID_DARD_ROOT/envs/inference/bin/python" scripts/download_models.py --download
python3 prepare_data.py --scope functions --verify
python3 environment_probe.py --scope functions
~~~

`setup_env.sh functions`는 `inference`·`function`·`evalplus` 환경만 준비합니다. `evaluators`도 EvalPlus만 설치합니다. 데이터가 없는 새 위치에서는 `python3 prepare_data.py --scope functions --download`로 별도 준비합니다. 기존 데이터는 다시 받지 않고 checksum을 확인합니다.

BigCodeBench 데이터와 기존 결과는 삭제하지 않습니다. 과거 데이터 호환용 어댑터는 남아 있지만 현재 코드 생성 CLI·통합 실험·환경 설치 범위에는 포함되지 않습니다. EvalPlus는 기존 함수 문제의 채점을 강화하는 용도이며, 라이브러리 활용 평가가 아닙니다.

## 데이터와 평가 방식

| 데이터 | 범위 | 평가 |
| --- | --- | --- |
| HumanEval | Python 164 | 함수 생성·로컬 테스트 |
| MBPP | Python 500 | 함수 생성·로컬 테스트 |
| MultiPL-E | Rust·JS·C++, 두 원본 문제군, 합계 1,156 | 언어별 컴파일·실행 |
| HumanEval+ / MBPP+ | Python 164 / 378 | 공식 EvalPlus base+plus 검사 |

함수 suite는 10개, 총 2,362개 variant입니다. 번역본·강화 테스트는 같은 원본 문제와 겹치므로 독립 데이터셋 10개로 해석하지 않습니다. 최종 테스트는 수정 피드백에 사용하지 않습니다. 공개 예제가 없으면 문법/컴파일 검사만 피드백으로 제공합니다. 최초 생성과 최종 제출의 채점은 제출을 고정한 뒤 수행합니다. 이 결과는 함수 코드 생성·수정 능력이며 실제 저장소 이슈 해결 능력과 구분합니다.

함수 실행기는 기본 `--runner local`입니다. `--runner none --repairs 0`은 생성만 기록하므로 정확도 평가가 아닙니다. 각 실행은 임시 디렉터리, 독립 HOME/cache, 시간·출력 제한을 사용하고 종료 후 자식 프로세스를 정리합니다.

## 선택 사항: 별도 저장소 평가

기존 SWE-bench Verified 500문제·Multilingual 300문제 평가 코드는 보존합니다. 현재 코드 생성 실험에는 포함하지 않으며 `--scope repositories`로만 저장소 평가를 따로 선택할 수 있습니다. `--scope all`은 코드 생성과 저장소 평가를 함께 실행하지만 BigCodeBench는 포함하지 않습니다. 범위를 바꿀 때는 다른 output을 사용해야 합니다.

이 선택 경로에서만 `bash scripts/setup_system_dependencies.sh`로 해당 저장소의 OS 의존성을 먼저 준비합니다. 다음 전체 명령을 사용할 수 있습니다.

~~~bash
bash scripts/run_all_experiments.sh --scope repositories --prepare --execute \
  --output /data/hybrid-dard/results/repository-experiments-local
~~~

문제별 준비만 따로 실행하려면:

~~~bash
"$HYBRID_DARD_ROOT/envs/swebench/bin/python" scripts/prepare_repositories.py \
  --suite swebench_verified --limit 0 --prepare --data-root "$HYBRID_DARD_ROOT"
"$HYBRID_DARD_ROOT/envs/swebench/bin/python" scripts/prepare_repositories.py \
  --suite swebench_multilingual --limit 0 --prepare --data-root "$HYBRID_DARD_ROOT"
~~~

준비는 원본 base commit·고정한 공식 설치 recipe를 사용하며 문제별 환경과 `repositories.lock.json`을 기록합니다. 원본 상태와 기준 패치 적용 상태를 별도 로컬 작업 디렉터리에서 검사하여, 의존성 문제를 모델 성능으로 잘못 측정하지 않도록 합니다. 지원하지 않는 설치 명령이나 필요한 시스템 의존성이 있으면 해당 준비를 명시적으로 실패 처리하고 로그를 남깁니다. 실제 800개 환경의 빌드 성공은 준비 명령 실행 후 확인해야 합니다.

에이전트는 매번 준비된 저장소의 새 복사본과 offline Conda 복제 환경을 받습니다. `read/search/edit/run/finish` 도구로 탐색·수정·개발 테스트를 수행합니다. 에이전트 작업 트리에서는 원래 Git 이력을 제거하고 초기 상태를 기준으로 패치를 추출합니다. 원본 준비 환경이 변경되면 비교를 계속하지 않습니다.

최종 채점은 **원래 base commit 이력을 보존한 별도 복사본**에 제출 패치를 적용합니다. 제공된 공식 테스트 스크립트의 환경 경로를 로컬 경로로 명시적으로 변환하고, 원문/변환 hash를 기록합니다. 공식 로그 파서와 fail-to-pass/pass-to-pass 판정은 재사용하지만, 결과는 로컬 환경 평가로 표시합니다. 확인된 환경·실행기 오류는 미확정/인프라 오류로 기록하고, 후보 코드의 문법·컴파일 오류와 테스트 실패는 미해결로 채점합니다.

로컬 프로세스는 동일 사용자의 호스트 접근 권한을 가집니다. 경로 제한 도구와 작업 디렉터리 분리는 보안 격리가 아닙니다. 채점 자료는 에이전트 입력과 수정 피드백에서 제외하지만 OS 차원의 접근 차단을 보장하지 않습니다.

환경 변경 감지는 파일 경로·크기·수정 시각 등 메타데이터를 사용하며, 모든 파일 내용의 무결성을 보장하는 방식은 아닙니다. 기본 실행 제한은 시간·출력·자식 프로세스 정리이며 CPU와 RAM을 작업별로 예약하지 않습니다.

## 네 GPU 비교와 결과

같은 문제의 AR·기본 Diffusion·DARD와 반복 seed를 같은 GPU에서 순차 실행하고, GPU 간에는 다른 문제를 배정합니다. dataset·언어별 균등 분할과 반복별 backend 순서 회전을 사용합니다. 전체 함수는 591/591/590/590개이며 세 backend·세 반복이면 21,258 case입니다. 소규모 사전 점검 실행은 이 개수와 별도입니다.

~~~bash
bash scripts/run_four_gpu.sh functions --gpus 0,1,2,3 \
  --limit 0 --repeats 3 --seed 42 --tokens 512 --standard-steps 512 \
  --block-size 128 --dard-steps 192 --diagnostics summary --split test \
  --output /data/hybrid-dard/results/code-generation-direct --execute --fail-fast
~~~

공통 출력·문맥·행동·제한시간과 정밀도를 고정합니다. 기본 생성 예산은 최적값을 의미하지 않습니다. 조정은 별도 개발 문제에서 하고 최종 평가 전에 고정해야 합니다. `--split dev/test`는 일지 표시이며 데이터 자동 분할이 아닙니다.

GPU UUID·문제·seed·코드·환경 정보를 고정하고 계획한 case의 누락/중복을 확인합니다. 완료 개수는 정답률이나 채점 완료를 뜻하지 않습니다. 성공한 문제만 골라 속도를 계산하지 않습니다. 네 worker는 CPU·RAM·디스크를 공유하므로 단독 에이전트 지연과 구분해야 합니다. shard별 speedup을 단순 평균하지 않으며 현재 통합 신뢰구간 계산은 제공하지 않습니다.

`functions/campaign_summary.json`은 네 shard의 정답 수·미채점/미완료 수와 세 방식의 비교를 통합합니다. 속도비는 짝이 맞는 전체 문제의 시간 합으로 계산하며, worker별 비율의 평균을 사용하지 않습니다. 모델 로딩·warmup·이전 재시도 비용을 포함한 phase 시간도 별도로 보존합니다. worker 시간의 합과 네 GPU를 병렬 실행한 실제 경과 시간은 구분합니다.

기본 실험은 고정된 한 가지 설정의 기준 측정입니다. 온도 0의 세 반복은 시간 측정 반복이며 독립적인 pass@3 표본이 아닙니다. DARD 비교에는 블록 구성·검증·조기 종료까지 포함됩니다. 더 넓은 우월성을 주장하려면 별도 개발 문제에서 기본 Diffusion의 낮은 step 수와 DARD 설정별 정확도·시간 곡선을 비교하고, 허용 정확도 차이를 최종 평가 전에 고정해야 합니다.

## DARD 진단과 실험 일지

- `--diagnostics summary`: 기본 성능 비교. 단계별 시간·실제 forward·입력 크기·강제 완성·GPU 메모리·종료 이유.
- `--diagnostics steps`: 상태 전이·승격/철회·신뢰도·재검증 상세 기록.
- `--diagnostics profile`: 동기화한 구성요소 시간. 계측 비용 때문에 공식 속도 비교에서 제외.

`--max-step-records`는 생성별 상세 목록을 제한하고 누락 건수도 기록합니다. 단계 시간과 그 안의 forward 시간은 중첩되므로 합산하지 않습니다. 상세 계측은 시간 예산 내 행동 수/해결률도 바꿀 수 있어 개선은 새 summary 실행에서 확인합니다. 강제 완성과 철회 자체는 알고리즘의 결과이며 비교에서 빼지 않습니다.

~~~bash
python3 experiment_analysis.py --run /data/hybrid-dard/results/code-generation/functions/shard-0
python3 experiment_journal.py observe --run /data/hybrid-dard/results/code-generation/functions/shard-0 \
  --text '실제 관측한 시간·오류와 근거 기록' --evidence diagnostics.md
python3 experiment_journal.py decide --run /data/hybrid-dard/results/code-generation/functions/shard-0 \
  --text '변경을 채택하거나 보류한 이유' --next-change '다음 개발 실험에서 바꿀 한 가지'
~~~

보고서는 `diagnostics.md/json`, `diagnostics_cases.csv`에 관측·원인 후보·다음 실험과 원본 로그 근거를 씁니다. forward 감소에도 느리면 입력 크기/검증 비용을, 생성은 빠르지만 전체가 느리면 추가 수정·도구 비용을 조사합니다. 상관관계를 원인으로 확정하지 않습니다.

## 오류 이메일과 재개

수신자는 `leejwian@gmail.com`이며 LERRA의 `~/.config/lerra_ft_comparison/gmail_app_password`를 참조합니다. 경로 override는 `HYBRID_DARD_MAIL_PASSWORD_FILE`, 다음으로 `LERRA_MAIL_PASSWORD_FILE`입니다. 비밀번호를 코드·명령줄·이메일에 넣지 않습니다.

~~~bash
bash scripts/run_all_experiments.sh --check-email
bash scripts/run_all_experiments.sh --test-email
bash scripts/run_all_experiments.sh --retry-email \
  --output /data/hybrid-dard/results/code-generation
tail -f /data/hybrid-dard/results/code-generation/worker.log
~~~

인증 점검은 메일을 보내지 않습니다. `--test-email`만 명시적인 테스트 메일을 보냅니다. 오류는 단계·종료 코드/signal·로그 경로를 최대 3회 전송 시도하고 미발송 상태를 저장합니다. SMTP 수락은 받은편지함 도착 확인이 아닙니다. 생성 코드의 오답은 평가 결과이며 실행기·준비·채점 오류와 구분합니다.

동일 명령·output·코드·설정으로 재개합니다. 완료 단계의 결과물 hash가 바뀌거나 사라졌으면 중단하고, GPU/데이터 점검은 다시 실행합니다. 저장소의 고정된 오류 제출을 다시 생성하려면 새 실험 경로가 필요합니다. 별도 supervisor는 자신이 시작한 worker와 자손을 감시·정리합니다. 서버 전원 차단이나 supervisor 자체의 SIGKILL까지 즉시 알리려면 외부 감시가 필요하며, 다음 실행에서는 이전 미완료 상태를 확인합니다.

각 단계의 시작·종료와 GPU worker 실행 직전에 고정된 소스 hash를 다시 검사합니다. 다운로드 중 코드를 바꾼 경우에도 구버전 계획과 수정 코드가 섞여 실행되지 않습니다. `--fail-fast`는 최초 생성기·실행기·채점기 오류를 저장한 뒤 다음 문제/방식으로 넘어가지 않고 중단합니다. 생성 코드의 문법 오류·오답·후보 실행 시간 초과는 계속 평가 결과로 집계합니다.

실행 중 점검을 위해 `audit_gate.json`으로 진행 worker를 일시정지한 경우, `scripts/resume_reviewed_experiments.py --previous-output OLD --output NEW --execute`가 기존 모델 다운로드 완료를 기다린 뒤 수정본으로 전환합니다. 기존 결과는 보존하며 새 결과 경로에 소스와 계획을 고정합니다. 환경 설치·추가 모델 다운로드 없이 `--execute`로 이어집니다. 상태는 `NEW/audit_handoff.json`, 이후 진행은 `NEW/pipeline_state.json`에 기록됩니다. 이전 supervisor의 종료 알림은 계획된 전환 기록 `OLD/audit_replacement.json`과 함께 해석합니다.

## 검증 범위

~~~bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -v
~~~

검증은 CPU 로직·작은 로컬 프로그램·mock 평가기를 포함합니다. 실제 7B 추론, 전체 환경 설치, 모든 언어의 전체 저장소 빌드/채점 성공을 대신하지 않습니다. 모델·환경 설치와 전체 실험은 위 명령을 사용자가 실행해야 합니다.

## 검증 설정 선택과 실제 LoRA 학습 반복

`scripts/prepare_iterative_data.py`는 캐시된 HumanEval·MBPP·MultiPL-E 전체를 원문 문제 ID로 묶어 train/validation/holdout으로 고정합니다. 같은 문제의 번역과 Plus 변형이 서로 다른 분할로 새지 않도록 중복을 제거합니다. 과거 평가 노출 여부도 보존하므로 holdout 전체가 처음 보는 문제라는 의미는 아닙니다.

`scripts/iterative_experiments.py`는 누락된 **train** 정답만 고정 AR 모델로 생성·채점하고, 통과한 코드만 학습 자료에 추가합니다. validation에서 Instruct·cpGRPO 및 생성 길이·블록·스텝·온도·임계값을 비교한 뒤, 선택한 DiffuCoder에 실제 LoRA 학습을 수행하고 다시 validation으로 선택합니다. 학습은 completion 부분만 마스킹하는 diffusion 목적함수와 4 GPU DDP를 사용하며, validation/holdout 정답은 gradient 계산에 넣지 않습니다. 추론은 GPU마다 독립 shard를 실행하고 CPU 채점을 겹쳐 수행합니다.

기본 반복 상한은 8회, 개선 없는 연속 라운드 상한은 3회입니다. 각 학습 라운드는 최대 100 optimizer step이며 학습률 후보는 `5e-5, 2e-5, 1e-4, 1e-5`입니다. 정확도가 AR보다 3%p 이내로 낮고 생성 시간이 AR보다 짧다는 조건을 함께 만족해야 배포 후보를 AR에서 변경합니다. 이는 운영 선택 기준이며 통계적 비열등성 판정이나 전역 최적값의 증명은 아닙니다. 최종 선택을 고정한 후 holdout을 한 번 평가합니다.

2026-11-07 17:00 KST 마감 전에 최종 평가용 4시간을 확보하며, 소스·모델·분할·환경 버전과 체크포인트를 고정합니다. 비정상 종료는 기존 이메일 정책을 따르고 실패 3회에 재시도를 중단합니다. `iteration_state.json`은 진행 상태, `round-*/decision.json`은 선택 근거, `iteration_result.json`은 완료 결과입니다. 실행 전 검증 기록은 결과 경로의 `audit/prelaunch_validation.json`에 보존합니다.

### 보고서의 CE-only r32/1,200 재현

보고서의 후보를 다시 고르지 않고 고정된 AR·native Diffusion·DARD 세 후보를 그대로 실행하려면 `scripts/reproduce_ce_only.py`를 사용합니다. 먼저 기존의 검증된 4-GPU `iteration_plan.json`과 train/validation/holdout JSONL을 준비한 뒤 다음처럼 실행합니다.

```bash
/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py \
  --base-plan /data/hybrid-dard/results/code-generation-tradeoff-20261107/iteration_plan.json \
  --data /data/hybrid-dard/results/code-generation-iterative-20261107/data \
  --output /data/hybrid-dard/results/ce-only-r32-1200 \
  --deadline-unix 1794225600
```

이 명령은 `r32`, `alpha=64`, `1,200` optimizer step, `min_mask_prob=0.15`, balanced JSONL, `512` token, validation seed `20261105`, 반복 seed `20261111/12/13`, holdout seed `20261108`, DARD `128/64/.4/.75/8/resample_then_verify/2`를 고정합니다. 결과는 `focused_full_result.json`, `ce_only_summary.json`, `reproduction_manifest.json`에 저장하며, 데이터와 어댑터 hash가 바뀌면 실행을 중단합니다. 이미 학습된 어댑터를 재사용하려면 `--adapter-path`를 지정할 수 있습니다.

완료된 1,200 step adapter와 재현에 필요한 데이터·설정·보고서 요약은 `artifacts/ce-only-r32-1200-v1/`에 보존했습니다. 이 번들의 `adapter/`를 `--adapter-path`로 지정하면 같은 학습 weight를 다시 학습하지 않고 그대로 사용할 수 있습니다. `experiment_manifest.json`에는 base revision, 학습 계약, 후보 ID, 보고서 metric, 파일 SHA256이 기록되어 있으며, 큰 weight 파일은 Git LFS로 추적합니다.

번들은 과거 평가 source provenance와 현재 학습 source를 분리해 보존합니다. 당시 학습 source 원본 byte는 남아 있지 않지만, 학습된 adapter weight 자체는 정확히 보존되어 있습니다.

번들 데이터를 사용하려면 명령에 `--adapter-path artifacts/ce-only-r32-1200-v1/adapter --data artifacts/ce-only-r32-1200-v1/data`를 추가합니다.

공유하거나 사용하기 전에는 `/data/hybrid-dard/envs/iterative-training/bin/python scripts/reproduce_ce_only.py --verify-bundle artifacts/ce-only-r32-1200-v1`로 번들 hash와 후보 ID를 확인합니다.

실행 이벤트는 각 단계의 `events/process-*.jsonl`에 추가 기록합니다. 각 행에 UTC/KST 시각·PID·순번·이전 행과 현재 행의 SHA256이 있으며 쓰기 후 fsync합니다. 같은 worker의 채점 thread도 직렬화해 기록합니다. 시작 명령의 인증정보는 제거하고 전체 환경변수나 가중치·gradient tensor는 기록하지 않습니다.

- 전체 제어: 단계 전환, 실행 명령과 PID, 종료 코드, 재개, 설정 선택·유지·교체 근거, 마감, 실패 traceback.
- 생성·검증: 모델 로드와 warmup, 문항 생성 시작/종료, 반환된 전체 생성 진단·강제 채움 정보, 원본 출력과 코드, 상세 채점 결과·컴파일 오류·테스트 traceback.
- 학습: 각 rank의 시작/종료, 데이터와 모델 준비, optimizer step 지표, 체크포인트 저장·복원·완료. 원래 `training_metrics.jsonl`과 `training.log`도 유지합니다.
- 자원과 파일 위치: `scripts/monitor_iterative_run.py`가 별도 서비스에서 GPU·프로세스·디스크·주 서비스 상태를 10초마다 기록하고 `logging/` 아래에 로그/결과 인덱스를 만듭니다. 이 감시기는 주 실험을 종료하거나 재시작하지 않습니다.

계측을 추가하기 전에 이미 완료된 문항의 원본 기록은 그대로 유지합니다. 당시 저장하지 않았던 상세 채점 출력이나 순간 상태는 소급 생성하지 않습니다. 변경 전후의 소스와 실행 계획, 유지한 결과의 hash는 `audit/logging-migration-v2/`에 보존합니다. 자원 표본은 10초 간격의 관측이며 매 GPU kernel이나 gradient tensor의 추적을 의미하지 않습니다.
