# E0–E3 experiment contract

Status: SPECIFICATION ONLY. research code/commands below must be implemented by the remote developer.

## 1. Common settings

- Surrogate pilot: torchvision ResNet152, explicitly pin original legacy pretrained weights. Do not use DEFAULT enum without recording what it resolves to.
- Input tensor: float32, NCHW, 224×224, RGB pixel coordinates.
- UAP: [1,3,224,224], uniform initialization in ±10/255, Adam lr=0.1 as original, per-step clip ±10/255.
- Original selected layers: p_active=true, p_rate=0.65. Gaussian/jigsaw schedule and per-iteration prior_batch follow official functions.
- Existing activation functions are imported from unchanged loss.py. No synthetic cross-entropy replacing them.
- Original semantic samples: num_copies=10 per prior item; prior_batch may increase by curriculum. Record actual batch size, not simply N=10 at all iterations.
- Python, NumPy, torch CPU and CUDA seeded with [126,2025,3407]. Use separate RNG streams for init/prior/crop/transform and save their states. Same seed alone does not guarantee paired crop draws when branches consume different RNG counts.
- Seed list and crop ranges fixed before final target evaluation.

## 2. Three honest execution labels

1. `upstream_faithful`: original source, original stopping/data flow, original environment if available. Raw command in README. Re-evaluate the actual returned tensor.
2. `corrected_reproduction`: optional, one documented correction at a time, own config/run ids.
3. `research_fixed_budget`: separate loop, unchanged activation loss, fixed max_steps per contrast, final iterate selected in advance. No target-FR checkpoint selection; checkpoints saved by scheduled iteration.

Default research max_steps=10000 to match upstream maximum, unless pilot feasibility justifies a smaller budget **before** final tests. All contrast members must share the chosen budget; short smoke is never a full run.

## 3. Data split contract

ImageNet val expected 50000 images / 1000 categories, sorted relative-path manifest, verify label mapping. Data exists outside repository (`IMAGENET_VAL_ROOT`). Do not move/copy or publish it through this setup task.

- One deterministic `selection_1000` list, used only for legacy stopping/optional source diagnostics.
- `holdout_49000` = disjoint complement; research final test.
- `smoke_holdout_1000` = fixed subset of holdout for engineering checks. If its target FR is inspected for selecting a method, remove it from a separately reserved final set and document changed denominator. The simple default is engineering verification only, not selection.
- All methods/targets use identical IDs and labels; no random_split calls in evaluator.
- CSV fields: image_id (relative), class_wnid, class_idx, file_sha256 (optional expensive content hash), split; manifest SHA-256 required.

## 4. Precise E2 intervention

Let z_t be the same class of random prior, delta_t the current UAP, C the same crop/resize procedure.

```
A noise_crops:
    s = stop_gradient(C(z_t))
    u = s + delta_t
B psp_crops:
    s = stop_gradient(C(z_t + stop_gradient(delta_t)))
    u = s + delta_t
```

Use existing l2_layer_loss(model,u,args,device), with T=false and RW=false in A/B. Prior generator/schedules identical, crop box random numbers replayed or generated from dedicated streams. UAP trajectories will diverge—that is expected. Pair exogenous draws, not learned delta values.

C full uses B plus original paired transforms and l2_layer_loss_weight. If compute allows, D noise_crops with the same full T/RW gives a second matched source contrast in the full system; D is optional, not part of first 15-run minimum.

A is a controlled **crop-matched noise control**. It is not identical to every original TRM random-prior pipeline; label it accurately. An optional noise_direct control characterizes crop augmentation, but do not pool its effect with self-generated-prior effect.

Never use the CLI semantic_prior flag as evidence the source changed. Log source_mode and verify tensors/gradient graph in fixtures.

## 5. E3

full_psp_original: area [0.08,1.00]; full_psp_local: [0.08,0.35]; full_psp_large: [0.50,1.00]. Aspect ratio [0.75,4/3]. Same sample counts, T/RW/layer choices/budget, original source mode. This tests the practical crop policy; does not assert it isolates human semantics, frequency or locality independently.

## 6. Target registry

Required pilot names (resolved checkpoint IDs must be committed after validation):

- torchvision vgg16, resnet50, densenet121 (CNN)
- torchvision convnext_base (CNN), swin_t (hierarchical Transformer), vit_b_16 (ViT)
- optional timm DeiT3: choose exact published ImageNet-1k 224 checkpoint, verify model name with installed registry; do not invent an id. Mark blocked if unavailable.

Two target weights with different pretraining corpora/recipes are not a controlled architecture-only intervention. Report source, dataset/resolution, clean accuracy, weight checksum. No final claim all architectures follow a deterministic CNN→Swin→ViT ordering.

## 7. Evaluator math and transform order

For each model, first decode/resize/crop to raw RGB x∈[0,1]. Then x_adv=clamp(x+delta,0,1), then model-specific normalization. Never add raw epsilon after normalization, never add a displayed/normalized PNG as the attack. Clean and adversarial must use the same deterministic geometry.

Default no resize of delta: choose 224 weights. If a later target requires 384/299, explicitly record perturbation interpolation, resulting Linf before/after clipping and distinct protocol ID. Do not change delta silently.

```
fr = flipped_count / total_count
clean_acc = clean_correct_count / total_count
adv_acc = adv_correct_count / total_count
cc_asr = clean_correct_became_wrong_count / clean_correct_count
```

Zero clean-correct denominator → null with reason, not 0. Count label flips from an already-wrong label separately if needed; FR and accuracy drop differ.

Evaluation: eval(), no_grad(), no random augmentation, assert unique IDs/counts. Caching clean predictions requires key=(model weights hash, transform spec hash, manifest hash, eval precision); invalidate mismatches.

## 8. Checkpoint / run data

Run dir outputs/{run_id}/ contains:
config.resolved.json, provenance.json, loss_trace.csv, checkpoints/{iter}.pt, uap.npy, uap.sha256, evaluation/{target}/metrics.json, predictions.csv or parquet.

Checkpoint fields: detached delta clone; optimizer state; step; RNG Python/NumPy/CPU/CUDA; curriculum current args (std/fre/prior_batch); config and upstream hashes. Resume must reproduce uninterrupted numerical state on deterministic fixture; separately record GPU nondeterminism. Do not reuse live delta alias as best_uap.

Metric row schema: run_id, protocol_id, stage, variant, source_mode, surrogate, target, target_family, seed, step, epsilon, crop_min, crop_max, t_on, rw_on, uap_sha256, weights_id/hash, manifest_sha256, total_count, flip_count, clean_correct_count, adv_correct_count, clean_correct_became_wrong_count, fr, clean_acc, adv_acc, cc_asr, gpu_seconds, status, error_reason.

Raw counts are integers, ratios in [0,1]; chart converts to %. DeltaFR means percentage points and is matched by seed/target/protocol, not subtraction of unrelated averages. Store missing/failed as missing, not zeros.

## 9. Target CLI design (future interfaces, currently unavailable)

```
python tools/doctor.py --output outputs/doctor.json
python tools/make_manifests.py --data-root "$IMAGENET_VAL_ROOT" --seed 126 --out manifests/imagenet_val_v1
python tools/train.py --config configs/experiments/e2_noise_crops.yaml --seed 126 --data-root "$IMAGENET_VAL_ROOT"
python tools/evaluate.py --uap outputs/RUN/uap.npy --target-config configs/targets/imagenet224.yaml --manifest manifests/imagenet_val_v1/holdout.csv --output outputs/RUN/evaluation
python tools/aggregate.py --runs outputs --out reports/first_complete_comparison
```

Implement strict config validation; refuse final-eval manifest as training/stopping split. dry-run prints all resolved paths/variants/weights/budget without training. User must see `--help`, short smoke invocation, full run invocation and artifact paths in actual implemented README.

## 10. Acceptance

- Zero delta ⇒ FR 0, identical predictions across paired evaluation.
- Hand-constructed predictions ⇒ exact FR/accuracy/cc-ASR counts.
- Channel/range/clamp/normalization order fixtures.
- Split disjointness/label mapping, no duplicate IDs.
- No gradient to detached crop source; gradient to added delta; unchanged model weight tensor hashes.
- Crop ranges and local/mixed/large accepted; one iteration metrics show valid actual crop areas.
- Reject NaN/Inf; never silently repair frozen loss.
- Matched seeds/exogenous RNG fixture, real checkpoint snapshot, CPU resume equality.
- Original loss.py hash and upstream source preservation.
- Document original-vs-research numerical differences; no FR claim based on CPU tests.
