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
| Faithful legacy environment | UNVALIDATED; modern route is separately labeled |
| ImageNet validation / real manifests | VERIFIED: Kaggle50000; CRC32/JPEG/SHA256; selection1000 / disjoint holdout49000 / smoke1000 |
| Label mapping | VERIFIED: all50000 Kaggle labels match pinned TFDS metadata; sorted1000 WNIDs |
| Original official devkit | BLOCKED: official host timeout; absent from competition ZIP; original artifact/mapping unverified |
| Real-data evaluator smoke | PASS: zero/random CNN+ViT fixtures and one actual100-step UAP on ResNet50 / ConvNeXt Base / Swin T / ViT B16,1000 images each |
| Controlled trainer / E2 A/B/C | IMPLEMENTED; CPU detach/replay/immutable checkpoint/resume/finite fixtures PASS |
| GPU100-step trainer smoke | PASS in explicit warn-only mode: ResNet152 full PSP T/RW, seed126, finite100 steps; strict mode failed on CUDA grid-sampler determinism |
| E3 crop configs / aggregation | IMPLEMENTED; fixed area ranges, paired ΔFR and seed-summary CPU fixtures PASS |
| E0 / E1 / E2 / E3 formal experiments | NOT_RUN; no paper reproduction claim |
| Teammate invitation request | PENDING GitHub username; email search did not identify an account; no invitation sent |

See [E3/matrix/aggregation commands](EXPERIMENT_EXECUTION_ZH.md), [executable tools runbook](TOOLS_RUNBOOK_ZH.md) and [controlled trainer runbook](TRAINER_RUNBOOK_ZH.md) for exact commands and artifact boundaries.
CPU fixture metrics are software checks, not ImageNet measurements. Missing results are never filled with zeros.

Current evidence: [Kaggle/GPU smoke report](../reports/kaggle_gpu_smoke_validation.json) and [commands/integrity limits](KAGGLE_DATA_SMOKE_ZH.md). Earlier validation reports retain their historical blocker snapshots.

GPU smoke is modern compatibility, not faithful E0. Warn-only CUDA grid-sampler backward can be nondeterministic; exact GPU replay is unverified. Original validation-tar MD5 is not verified because images came from the authenticated Kaggle ZIP. No final49000-image evaluation or full training was run.
