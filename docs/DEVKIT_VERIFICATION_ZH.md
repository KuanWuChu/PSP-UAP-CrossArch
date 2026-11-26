# 原始 ILSVRC2012 devkit：取得與 mapping 驗證

2026-11-26，原始 `ILSVRC2012_devkit_t12.tar.gz` 已取得，官方 MD5 與 mapping 交叉驗證全部通過。本次沒有啟動訓練或額外 target evaluation。

## 來源與授權

官方 ImageNet 主機仍連線逾時。已搜尋並檢查 [0429charlie/ImageNet_metadata](https://github.com/0429charlie/ImageNet_metadata)，其 README 明確提供此原始 devkit 的公開保存副本；下載固定在 commit `2e9df66a95fe0ce3a7ad1f957a7d1031e9119568`，未使用會變動的 master 作為產物 identity。

[固定版本下載連結](https://raw.githubusercontent.com/0429charlie/ImageNet_metadata/2e9df66a95fe0ce3a7ad1f957a7d1031e9119568/ILSVRC2012_devkit_t12.tar.gz)。這是第三方保存副本，檔案內容以原始官方 MD5 核對；不把該 GitHub 專案描述成 ImageNet 官方主機。

封包內 `ILSVRC2012_devkit_t12/COPYING` 的著作權人為 The ImageNet Project。其授權允許使用、複製及再散布軟體與附帶文件，要求保留著作權／授權聲明，另禁止未經同意以著作權人或貢獻者名稱背書。原始 archive 與完整聲明都已保留；公開文字副本見 [COPYING](licenses/ILSVRC2012_DEVKIT_COPYING.txt)。這段授權說明只適用於 devkit，不延伸宣稱 ImageNet 影像可自由再散布。

此來源沒有要求登入或接受新條款，因此不需要帳號交接。沒有使用 Kaggle/Hugging Face 憑證向第三方網站登入。

## 真實驗證結果

- 封包大小：2,568,145 bytes。
- MD5：`fa75699e90414af021442c21a62c3abf`，與 [torchvision 官方 ImageNet loader](https://docs.pytorch.org/vision/0.20/_modules/torchvision/datasets/imagenet.html) 記錄一致。
- 本機計算 SHA256：`b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953`。
- COPYING SHA256：`5f4d5e6ba018a1c22c12bcb5d7d9c42c5ffe9672ab0165491203a4111a0b4463`。
- devkit 的 `meta.mat` 解析出1,000個 leaf WNIDs；依排序建立的 class index 與目前 mapping 完全一致。
- 官方 `ILSVRC2012_validation_ground_truth.txt` 經 class-ID→WNID 轉換後，全部50,000筆與既有 Kaggle／TFDS mapping 一致。
- 新增 `tools/check_devkit_mapping.py`，使用 torchvision 自己的 `parse_devkit_archive` 再次驗證官方 MD5，獨立於本專案的 label builder 解析封包。它檢查全部原始影像編號、class_idx、WNID、manifest checksum、切分數量與不重疊條件，拒絕自洽 hash 掩蓋的錯誤 mapping。
- 用官方 devkit label bundle 重新掃描50,000張影像，建立獨立 manifests；selection／holdout／smoke 三份 CSV 與原版本逐位元相同。
- CPU：`OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q` → **67 passed in6.30s**。
- `python3 scripts/check_upstream_integrity.py` →八個保護檔案全部 PASS；CLI help／compileall 通過。

完整機器可讀結果、原始來源 commit／URLs、hash 與檢查命令見 [驗證報告](../reports/devkit_mapping_verification.json)。

## 本機產物與 provenance

所有路徑相對於 repo 根目錄：

| 產物 | 位置 |
|---|---|
| 原始 devkit | `data/imagenet2012/ILSVRC2012_devkit_t12.tar.gz` |
| 下載來源與授權 identity | `data/imagenet2012/devkit-acquisition.json` |
| 官方 devkit label bundle | `labels/imagenet2012_official_devkit_v1/` |
| 新 manifests／官方 source index | `manifests/imagenet_val_devkit_v1/` |
| 既有 mapping 的獨立驗證 | `outputs/devkit-acquisition-20261126/crosscheck-existing.json` |
| 新官方 bundle 的獨立驗證 | `outputs/devkit-acquisition-20261126/crosscheck-official.json` |

新 `source.json` 明確包含 `label_source=official_devkit`、`official_devkit_verified=true`、MD5、SHA256 與第三方取得來源；新 manifests index 引用這份來源。

既有 `manifests/imagenet_val_kaggle_tfds_v1/`、[GPU smoke 報告](../reports/kaggle_gpu_smoke_validation.json)、training/evaluation provenance 均保留原樣，正確記錄當時還沒有 devkit。這份後續驗證證明其 mapping 與官方相符，不回寫過去的執行來源。

## 可直接重跑的下一組命令

以下只做 checksum／mapping 檢查，不訓練：

```bash
md5sum data/imagenet2012/ILSVRC2012_devkit_t12.tar.gz
.venv/bin/python tools/check_devkit_mapping.py --devkit data/imagenet2012/ILSVRC2012_devkit_t12.tar.gz --labels-dir labels/imagenet2012_kaggle_tfds_v1 --manifest-dir manifests/imagenet_val_kaggle_tfds_v1 --output outputs/devkit-acquisition-20261126/recheck-existing.json
.venv/bin/python tools/check_devkit_mapping.py --devkit data/imagenet2012/ILSVRC2012_devkit_t12.tar.gz --labels-dir labels/imagenet2012_official_devkit_v1 --manifest-dir manifests/imagenet_val_devkit_v1 --output outputs/devkit-acquisition-20261126/recheck-official.json
```

官方 devkit 缺檔阻礙已解除。原始 validation tar 的 MD5 仍未核對（影像來自已驗證 Kaggle ZIP）；legacy faithful 環境／baseline、正式 E0–E3、完整訓練與 final49,000張評估仍未執行。
