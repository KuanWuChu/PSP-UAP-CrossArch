# Kaggle validation acquisition and engineering smoke

Historical acquisition/smoke record: the original devkit was unavailable during these runs. It was subsequently obtained and all mapping entries crosschecked successfully; see [devkit follow-up](DEVKIT_VERIFICATION_ZH.md). Earlier run provenance and this report's source hashes are preserved.

Kaggle API uses the user's local credential file `~/.kaggle/kaggle.json`, directory0700/file0600. Credentials, signed download URLs, source images and raw private logs are excluded from Git. The user accepted the ImageNet competition rules; a successful data download confirmed access. No full training is authorized in this stage.

## Dataset and integrity route

The official ImageNet website points users to the Kaggle ImageNet competition:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge

The Kaggle competition ZIP contains all splits. The acquisition utilities inspect its central directory with strict HTTP206 range checks, record a stable ETag and select exactly the 50000 original validation JPEG filenames. They download only the contiguous validation-image span (6,665,331,665 compressed/header bytes), in bounded chunks. No training images are selected. A server that ignores a Range request is rejected before its full response body is read.

The extracted images are checked against ZIP central-directory CRC32, original file sizes and JPEG validity; every extracted image and each downloaded chunk gets a SHA256. `tools/extract_validation_parts.py` supports ZIP64 local-header sizes and reads across chunk boundaries. This is a Kaggle ZIP acquisition, so the MD5 of the original `ILSVRC2012_img_val.tar` is **not** claimed verified.

At the time of the smoke, the original `ILSVRC2012_devkit_t12.tar.gz` was unavailable from the currently unreachable ImageNet host, and it is not present in the competition ZIP. It is **not** synthesized or relabeled as downloaded. Official expected MD5 remains `fa75699e90414af021442c21a62c3abf`.

## Explicit alternative label provenance

The official competition `LOC_val_solution.csv` provides WNIDs and original image IDs. Require exactly 50000 unique IDs and a single class per image (multiple bounding boxes of the same class are accepted). Crosscheck every ID against `tensorflow/datasets`' `validation_labels.txt` at a pinned Git commit. Require 1000 sorted unique WNIDs from the same commit's `labels.txt`, matching torchvision ImageFolder class ordering.

`tools/make_manifests.py kaggle-labels` produces a label-source bundle that explicitly records:

- `label_source=official_kaggle_crosschecked_tfds`
- `official_devkit_verified=false`, `devkit_sha256=null`
- competition source, Kaggle CSV SHA256, pinned TFDS commit/URLs and metadata SHA256
- all50000 labels matched, class-index and ground-truth hashes

This supports a traceable engineering smoke on genuine validation images. It does not establish faithful legacy reproduction or original-devkit verification. Any later formal result must retain this data-provenance distinction.

## Reproduce

The ImageNet/Kaggle account must already have authorized access; accept rules in the browser. Keep credential entry local. Install the optional acquisition environment separately:

```bash
python3 -m venv outputs/kaggle-api/venv
outputs/kaggle-api/venv/bin/python -m pip install -r environments/kaggle.txt
outputs/kaggle-api/venv/bin/python tools/kaggle_inspect_zip.py
outputs/kaggle-api/venv/bin/python tools/kaggle_download_validation.py
```

Download `LOC_val_solution.csv` with the authorized Kaggle API. If Kaggle wraps it in ZIP, check CRC and extract only the expected CSV. Fetch `tensorflow_datasets/datasets/imagenet2012/labels.txt` and `validation_labels.txt` from the same explicit40-character `tensorflow/datasets` commit and record their hashes.

```bash
.venv/bin/python tools/make_manifests.py kaggle-labels --kaggle-csv data/imagenet2012/kaggle_metadata/LOC_val_solution.csv --reference-labels data/imagenet2012/independent_label_metadata/labels.txt --reference-truth data/imagenet2012/independent_label_metadata/validation_labels.txt --reference-commit 252301af4a62f264bddf16491fd0bfe57db8140f --out labels/imagenet2012_kaggle_tfds_v1
.venv/bin/python tools/extract_validation_parts.py --index outputs/kaggle-api/competition-zip-validation-index.json --chunks outputs/kaggle-api/validation-chunks.json --parts-dir data/imagenet2012/kaggle_validation_parts --labels-dir labels/imagenet2012_kaggle_tfds_v1 --data-root data/imagenet2012/val --inventory outputs/kaggle-api/image_inventory.csv
.venv/bin/python tools/make_manifests.py split --data-root data/imagenet2012/val --labels-dir labels/imagenet2012_kaggle_tfds_v1 --seed 126 --out manifests/imagenet_val_kaggle_tfds_v1
```

The split builder verifies all image-content hashes, filenames and label mappings before freezing selection1000, disjoint holdout49000 and the fixed smoke_holdout1000 subset. Raw datasets, references and manifests remain local.

After the CPU checks, evaluate actual zero/random `.npy` fixtures on ResNet50 and ViT B/16 with the fixed smoke manifest. A 100-step full-PSP ResNet152 run uses `--smoke-steps 100`, followed by the same target-only smoke if finite. No final holdout metrics, full E0 run or15-run E2/E3 matrix is executed here. Nonfinite original weighted loss remains a recorded failure, not an excuse to modify `loss.py`.

## 2026-11-26 actual results

`reports/kaggle_gpu_smoke_validation.json` contains full hashes, raw counts, checkpoint identity, source commits and per-target timing. Each public path is relative to the project root. Images, weights, credentials and raw private logs remain local.

- Hardware: RTX4090, driver580.178.04, torch2.5.1+cu121 / torchvision0.20.1+cu121. Modern compatibility; faithful legacy environment unvalidated.
- Data: 50000 images /1000 classes, 6,706,296,736 extracted bytes. All images pass ZIP CRC32/size and JPEG verification, with individual SHA256 inventory. All50000 original IDs/labels match pinned TFDS metadata.
- Manifests: seed126; selection1000 and holdout49000 are disjoint; smoke1000 is a fixed subset of holdout.
- CPU: `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q` →62 passed. `python3 scripts/check_upstream_integrity.py` → all8 protected files PASS.
- Zero delta: ResNet50 and ViT B/16 each1000 images, zero flips; clean/adv predictions identical. Random `.npy` fixture also evaluated on both, separately labeled untrained.
- Strict100-step full PSP attempt: failed at step1 before any completed update, because CUDA `grid_sampler_2d_backward_cuda` has no deterministic implementation. Failure retained in its own directory.
- Explicit `--determinism warn`100-step full PSP: seed126, ResNet152, original T/RW enabled,100 finite steps with10 samples per step. Wall time8.378s includes setup/serialization; peak allocated CUDA memory2,892,090,368 bytes. This is allocated tensor memory, not total GPU memory. No GPU bitwise replay claim.

The same resulting UAP was evaluated once per target on the same1000 smoke images. This table reports engineering outputs, not formal architecture comparisons or reproduced paper results; it was not used to select parameters/checkpoints.

| Target | FR | Clean accuracy | Adv accuracy | Clean-correct ASR |
|---|---:|---:|---:|---:|
| ResNet50 |58.7%|78.1%|37.2%|54.55%|
| ConvNeXt Base |29.1%|84.1%|66.0%|24.14%|
| Swin T |19.0%|81.9%|73.9%|13.06%|
| ViT B/16 |14.2%|82.5%|76.4%|9.70%|

All eight target runs' raw counts/ratios were independently recomputed from prediction CSVs; repeated clean predictions agree across fixtures and trained UAP.

### Local artifacts and hashes

| Artifact | Local path | SHA256 |
|---|---|---|
| Image inventory |`outputs/kaggle-api/image_inventory.csv`|`9d76a4f7ee0a3d3344cc1e7c9090063c8c40d44f8d4a463d090ec3fe72c41981`|
| Selection1000 |`manifests/imagenet_val_kaggle_tfds_v1/selection.csv`|`9d85658f9542ae90708b3a35373662d50e7740e9d59aa03304a7e6da35854362`|
| Holdout49000 |`manifests/imagenet_val_kaggle_tfds_v1/holdout.csv`|`a96015afdaeae9040d1be127d715d1a742a39712da308fe8a11bba4fafaa4233`|
| Smoke1000 |`manifests/imagenet_val_kaggle_tfds_v1/smoke_holdout.csv`|`51802ec2f25e8d0a6715818409e74b0f66e76bb158baccec7a36dd5287cc277e`|
| Weight lock (seven full checkpoint hashes inside) |`configs/targets/weights.lock.json`|`d5e2544442bf7135d571be19f7c15594eafa9879191864d0d81a4b2e42c571e0`|
| Actual100-step UAP |`outputs/gpu-smoke-20261126/full-psp-100-seed126-warn/uap.npy`|`eff09a14d4751dfe3840f214327e02dd8d410e6322e41d56b5ff014f8112166c`|

Images: `data/imagenet2012/val/<wnid>/`. Source metadata: `labels/imagenet2012_kaggle_tfds_v1/source.json`. Checkpoint: `outputs/gpu-smoke-20261126/full-psp-100-seed126-warn/checkpoints/000100.pt`. Target outputs: `outputs/gpu-smoke-20261126/trained-evaluation/<target>/{metrics.json,predictions.csv}`. Full artifact hashes are in the report.

### Directly runnable next check

From the project root, rerun only the target smoke into a new output directory:

```bash
OMP_NUM_THREADS=4 .venv/bin/python tools/evaluate.py --uap outputs/gpu-smoke-20261126/full-psp-100-seed126-warn/uap.npy --training-record outputs/gpu-smoke-20261126/full-psp-100-seed126-warn/provenance.json --data-root data/imagenet2012/val --manifest manifests/imagenet_val_kaggle_tfds_v1/smoke_holdout.csv --targets resnet50 convnext_base swin_t vit_b_16 --stage smoke --device cuda:0 --batch-size 16 --workers 2 --output outputs/recheck-kaggle-smoke
```

To repeat the100-step engineering run, use a new output directory:

```bash
OMP_NUM_THREADS=4 .venv/bin/python tools/train.py --config configs/experiments/full_original.json --seed 126 --manifest-dir manifests/imagenet_val_kaggle_tfds_v1 --smoke-steps 100 --determinism warn --device cuda:0 --output outputs/recheck-psp-100
```

During this historical smoke, warn-only mode was limited to smoke. The current full-execution CLI additionally permits it after verified E0 evidence; see FULL_EXECUTION_ZH.md. Numerical flags are preserved in checkpoints; a different policy is rejected at resume. `loss.py` remains unchanged. E0 and formal E1/E2/E3, final49000-image evaluation and full training remain NOT_RUN.

## Original Task1&2 devkit: historical blocker, now resolved

Before the follow-up, the user had no local copy; the official ImageNet host timed out. The authenticated competition ZIP central-directory scan contains no devkit. Missing original archive: `ILSVRC2012_devkit_t12.tar.gz`, including `ILSVRC2012_devkit_t12/data/meta.mat` and `data/ILSVRC2012_validation_ground_truth.txt`.

Legitimate acquisition: use the [official ILSVRC2012 download page](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) with the user's authorized ImageNet account when reachable; let the user handle any login/terms. The expected MD5 is recorded by [torchvision0.20's ImageNet loader](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/imagenet.html). Do not substitute an unverified archive or call the Kaggle/TFDS bundle a devkit.

After placing the original file locally, these commands verify its MD5 before parsing, and crosscheck its derived label mapping against the current source:

```bash
.venv/bin/python tools/make_manifests.py devkit --devkit data/imagenet2012/ILSVRC2012_devkit_t12.tar.gz --out labels/imagenet2012_official_devkit_v1
cmp labels/imagenet2012_official_devkit_v1/validation_wnids.txt labels/imagenet2012_kaggle_tfds_v1/validation_wnids.txt
cmp labels/imagenet2012_official_devkit_v1/class_index.json labels/imagenet2012_kaggle_tfds_v1/class_index.json
```

The CLI requires MD5 `fa75699e90414af021442c21a62c3abf`; a mismatch fails before extraction. These devkit commands were subsequently run against the checksum-matching original archive; all comparisons passed. See the [follow-up report and repeatable verification command](DEVKIT_VERIFICATION_ZH.md).
