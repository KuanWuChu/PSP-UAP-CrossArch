# Kaggle validation acquisition and engineering smoke

Kaggle API uses the user's local credential file `~/.kaggle/kaggle.json`, directory0700/file0600. Credentials, signed download URLs, source images and raw private logs are excluded from Git. The user accepted the ImageNet competition rules; a successful data download confirmed access. No full training is authorized in this stage.

## Dataset and integrity route

The official ImageNet website points users to the Kaggle ImageNet competition:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge

The Kaggle competition ZIP contains all splits. The acquisition utilities inspect its central directory with strict HTTP206 range checks, record a stable ETag and select exactly the 50000 original validation JPEG filenames. They download only the contiguous validation-image span (6,665,331,665 compressed/header bytes), in bounded chunks. No training images are selected. A server that ignores a Range request is rejected before its full response body is read.

The extracted images are checked against ZIP central-directory CRC32, original file sizes and JPEG validity; every extracted image and each downloaded chunk gets a SHA256. `tools/extract_validation_parts.py` supports ZIP64 local-header sizes and reads across chunk boundaries. This is a Kaggle ZIP acquisition, so the MD5 of the original `ILSVRC2012_img_val.tar` is **not** claimed verified.

The original `ILSVRC2012_devkit_t12.tar.gz` is still unavailable from the currently unreachable ImageNet host, and it is not present in the competition ZIP. It is **not** synthesized or relabeled as downloaded. Official expected MD5 remains `fa75699e90414af021442c21a62c3abf`.

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
.venv/bin/python tools/make_manifests.py kaggle-labels --kaggle-csv data/imagenet2012/kaggle_metadata/LOC_val_solution.csv --reference-labels data/imagenet2012/independent_label_metadata/labels.txt --reference-truth data/imagenet2012/independent_label_metadata/validation_labels.txt --reference-commit "$TFDS_LABEL_COMMIT" --out labels/imagenet2012_kaggle_tfds_v1
.venv/bin/python tools/extract_validation_parts.py --index outputs/kaggle-api/competition-zip-validation-index.json --chunks outputs/kaggle-api/validation-chunks.json --parts-dir data/imagenet2012/kaggle_validation_parts --labels-dir labels/imagenet2012_kaggle_tfds_v1 --data-root data/imagenet2012/val --inventory outputs/kaggle-api/image_inventory.csv
.venv/bin/python tools/make_manifests.py split --data-root data/imagenet2012/val --labels-dir labels/imagenet2012_kaggle_tfds_v1 --seed 126 --out manifests/imagenet_val_kaggle_tfds_v1
```

The split builder verifies all image-content hashes, filenames and label mappings before freezing selection1000, disjoint holdout49000 and the fixed smoke_holdout1000 subset. Raw datasets, references and manifests remain local.

After the CPU checks, evaluate actual zero/random `.npy` fixtures on ResNet50 and ViT B/16 with the fixed smoke manifest. A 100-step full-PSP ResNet152 run uses `--smoke-steps 100`, followed by the same target-only smoke if finite. No final holdout metrics, full E0 run or15-run E2/E3 matrix is executed here. Nonfinite original weighted loss remains a recorded failure, not an excuse to modify `loss.py`.
