# Upstream audit: verified source vs. unresolved claims

Source commit: `dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c`.
Line numbers refer to unmodified upstream; use links with this fixed SHA.

| Finding | Evidence | Required treatment |
|---|---|---|
| UAP initialization is uniform, not Gaussian | PSP_UAP.py:30–40 uses torch.rand within ±epsilon | Record initialization separately from Gaussian/jigsaw input prior |
| Gaussian/jigsaw are supported input priors | main.py:71; PSP_UAP.py:49–62 | Use official per-surrogate setting for E0 |
| Self-generated crop source is delta + prior | PSP_UAP.py:62–73 | Explicit source-mode intervention for E2 |
| Crop output is detached before adding learnable delta | PSP_UAP.py:75–76 | No gradient through crop source, preserve in controlled loop |
| Actual crop area range is (0.08,1), even though helper default is (0.2,1) | PSP_UAP.py:72; semantic_aug.py:117 | Audit call site, not only helper signature |
| Loss selects exact nn.Conv2d | loss.py:14–18 | Keep loss.py intact; do not claim token/attention optimization |
| --semantic_prior defined but not consulted in psp_uap core | main.py:34; PSP_UAP.py | Add explicit source_mode, do not rely on flag |
| Only delta is in Adam optimizer | PSP_UAP.py:40 | Model weights are not optimized; eval mode alone does not disable parameter gradients |
| NumPy is not seeded in main | main.py:100–107; functions.py:232 | Faithful preserves; research seeds NumPy explicitly |
| jigsaw boolean condition always true | loss.py:214, `if args.surrogate_model == 'googlenet' or 'inception_v3':` | Faithful preserve; corrected variant separately. For first controlled A/B/C, preserve this behavior consistently rather than silently fix |
| best_uap aliases delta | PSP_UAP.py:120 | Saved tensor may be final mutable delta, not independent best checkpoint. Research stores delta.detach().clone(); log original behavior |
| failure counter not reset when FR improves | PSP_UAP.py:117–125 | Actual upstream stopping is cumulative non-improvement checks; don't describe as conventional consecutive patience |
| no best_uap assignment if FR never exceeds 0 | PSP_UAP.py:28,117,137 | Research always initializes valid final checkpoint; faithful failure is recorded |
| training validation overlap exists in upstream evaluation | PSP_UAP.py:42/114; functions.py:80–82; main.py:151 analyze=True | Faithful reports 50k-compatible table with selection overlap; research uses disjoint manifests |
| FR is prediction flip, not ground-truth ASR | loss.py:151–175 | Add separate metrics outside original loss.py |
| pretrained=True and common normalize function | functions.py:16–23,37–49; main.py:160 | Pin actual weights; separate target transform adapters |
| parameter --target_model does not determine final target loop | main.py:19;94–98;150 | New evaluator should honor explicit --targets and print resolved model list |
| input_transform False is not None | main.py:120/131 | Output naming in upstream can be misleading; store semantic config, do not infer variant from path |
| weighted and unweighted p_active branches are not identical apart from sample weights | loss.py:52–55 vs119–123 | A/B use same unweighted branch. C is full system, not pure effect of weights |
| reciprocal KL weighting can become non-finite | loss.py:102–105 | Detect/log NaN/Inf. Do not insert epsilon into frozen loss silently; investigate and make separately labeled stability study if needed |
| modern GPU may not run torch1.11 | README environment is CUDA11.3 / torch1.11 | Doctor before installing. Compatible modern route is separately labeled |

## DeiT3 boundary

Original supplement reports DeiT3 as surrogate/target. Public `main.py` target loops contain CNNs; `functions.py` allows torchvision `vit_b_16` but does not provide a DeiT3 training recipe. Therefore:

- VERIFIED: public loss.py uses Conv2d activations.
- CONDITIONAL: applying that unchanged loss to a ViT with a Conv2d patch projection would focus on that projection rather than attention blocks.
- NOT VERIFIED: Table 8's DeiT3 experiment used that exact unchanged mapping, selected which layers, or ignored all Transformer blocks.

Do not repeat the unsupported sentence “the authors definitely only optimize the DeiT3 patch embedding.” Log this as unresolved, and keep ViT-surrogate adaptation out of this development phase.

## Integrity

`python scripts/check_upstream_integrity.py` checks original six Python files, README and LICENSE. The baseline records SHA-256 and Git blob hashes in UPSTREAM_SNAPSHOT.json. This file's .gitignore entry is historic metadata; current .gitignore is intentionally extended for local data and artifacts, so the integrity script does not lock .gitignore.
