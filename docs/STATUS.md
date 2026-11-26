# Implementation status — 2026-11-26

| Item | State |
|---|---|
| Handoff ZIP / Git bundle | SHA256 VERIFIED; full history restored |
| Public native fork | VERIFIED: KuanWuChu/PSP-UAP-CrossArch; public; parent ChanhuiLee1111/PSP-UAP |
| Bootstrap PR | Draft #1; planning branch and pinned upstream tag pushed |
| Upstream protected files / MIT | PASS; all eight byte-identical |
| Doctor / manifest builder / strict config | IMPLEMENTED; CPU fixtures |
| Target registry / .npy evaluator | IMPLEMENTED; checkpoint lock records exact enums/transforms/full hashes |
| Modern CUDA probe | PASS on RTX4090; torch2.5.1+cu121 / torchvision0.20.1+cu121 |
| Legacy declared core environment | PASS: Ubuntu20.04 / Python3.8 / torch1.11 / vision0.12 / CUDA11.3 convolution backward on RTX4090; full baseline pending |
| ImageNet validation / real manifests | VERIFIED: Kaggle50000; CRC32/JPEG/SHA256; selection1000 / disjoint holdout49000 / smoke1000 |
| Label mapping | VERIFIED: all50000 Kaggle/TFDS labels and1000 sorted WNIDs also match original devkit through independent torchvision parser |
| Official-source manifests | VERIFIED: separate devkit source index; all3 CSVs byte-identical to previous split |
| Original official devkit | VERIFIED: public mirror copy matches official MD5; original COPYING retained; no login/terms needed |
| Real-data evaluator smoke | PASS: zero/random CNN+ViT fixtures and one actual100-step UAP on ResNet50 / ConvNeXt Base / Swin T / ViT B16,1000 images each |
| Controlled trainer / E2 A/B/C | IMPLEMENTED; CPU detach/replay/immutable checkpoint/resume/finite fixtures PASS |
| GPU100-step trainer smoke | PASS in explicit warn-only mode: ResNet152 full PSP T/RW, seed126, finite100 steps; strict mode failed on CUDA grid-sampler determinism |
| E3 crop configs / aggregation | IMPLEMENTED; fixed area ranges, paired ΔFR and seed-summary CPU fixtures PASS |
| E0 | RUNNING: isolated unchanged upstream modern run; legacy baseline queued; no paper reproduction claim |
| E1 / E2 / E3 formal matrix | IMPLEMENTED: verified-E0 gate,15 runs /90 evaluations; not started until baseline checks pass |
| GitHub CI | CPU fixtures and8-file integrity workflow implemented |
| Teammate invitation request | PENDING GitHub username; email search did not identify an account; no invitation sent |

See [E3/matrix/aggregation commands](EXPERIMENT_EXECUTION_ZH.md), [executable tools runbook](TOOLS_RUNBOOK_ZH.md) and [controlled trainer runbook](TRAINER_RUNBOOK_ZH.md) for exact commands and artifact boundaries.
CPU fixture metrics are software checks, not ImageNet measurements. Missing results are never filled with zeros.

Current evidence: [Kaggle/GPU smoke report](../reports/kaggle_gpu_smoke_validation.json) and [commands/integrity limits](KAGGLE_DATA_SMOKE_ZH.md). Earlier validation reports retain their historical blocker snapshots.

GPU smoke is modern compatibility, not faithful E0. Warn-only CUDA grid-sampler backward can be nondeterministic; exact GPU replay is unverified. Original validation-tar MD5 is not verified because images came from the authenticated Kaggle ZIP. No final49000-image evaluation or full training was run.

Devkit follow-up: [acquisition/license/crosscheck](DEVKIT_VERIFICATION_ZH.md), [machine-readable evidence](../reports/devkit_mapping_verification.json); CPU67 passed. Original devkit blocker is resolved. Prior smoke provenance is preserved; no training/evaluation was launched in this follow-up.

Full execution: [runbook](FULL_EXECUTION_ZH.md), [actual environment probe](../reports/full_execution_environment.json). CPU70 passed. Earlier reports remain historical snapshots; no completed-matrix claim.
