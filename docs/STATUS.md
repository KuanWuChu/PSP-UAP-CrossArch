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
| ImageNet / devkit / real manifests | BLOCKED: no dataset or devkit paths supplied |
| CNN + ViT real-data evaluator smoke | BLOCKED: verified licensed dataset and real UAP unavailable |
| Controlled trainer / E2 / E3 | NEXT PR; not part of foundation implementation |
| E0 / E1 / E2 / E3 formal experiments | NOT_RUN; no paper reproduction claim |
| Teammate invitations | Not performed; user manages invitations |

See [executable tools runbook](TOOLS_RUNBOOK_ZH.md) for exact commands and artifact boundaries.
CPU fixture metrics are software checks, not ImageNet measurements. Missing results are never filled with zeros.
