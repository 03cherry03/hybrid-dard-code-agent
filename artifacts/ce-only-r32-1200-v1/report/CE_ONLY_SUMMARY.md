# CE-only 실험 결과 요약

이 문서는 masked diffusion CE 학습 결과만을 기준으로 정리한 요약입니다.

| 구분 | Validation 정확도 | Validation 평균 시간 | Holdout 정확도 | Holdout 평균 시간 |
|---|---:|---:|---:|---:|
| AR | 72.53% | 1.539s | 65.93% | 1.678s |
| Diffusion only | 4.03% | 1.362s | 3.66% | 1.381s |
| Diffusion + DARD | 41.39% | 1.709s | 35.16% | 1.744s |

선택된 DARD 설정:

```text
block_size=128
steps_per_block=64
tau_c=0.4
tau_u=0.75
transfer_limit=8
completion_policy=resample_then_verify
completion_verification_steps=2
temperature=0.0
top_p=1.0
```

학습 설정: r32 LoRA, 1,200 step, 균형 언어 sampler, masked diffusion CE.

원본 결과: `/home/leejwian1955/hybrid-dard-model-upgrade-20261108/focused-target-r32-1200-full/focused_full_result.json`
