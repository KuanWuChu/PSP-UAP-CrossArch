# 可執行工具：基礎與跨架構評估

新功能在 `psp_research/` 與 `tools/`。八個 upstream 保護檔案保持原樣。
目前 CPU fixtures 是工程驗收，沒有宣稱 E0／E1 成果。

## 環境

本機驗證路徑是 **modern_compatibility**：Python 3.10、torch 2.5.1+cu121、torchvision 0.20.1+cu121。
RTX 4090 compute capability 8.9，driver 580.178.04，CUDA runtime 12.1，24 GiB。
`nvidia-smi` 的 CUDA 13.0 是 driver 可支援上限，不代表 torch 使用 CUDA 13。

官方 faithful 環境為 Python3.8／torch1.11／torchvision0.12／CUDA11.3；這台機器尚未驗證該路徑。
Ada 原生 cubin 需 CUDA11.8，但部分較舊 Ampere cubin／PTX 可相容；不能僅憑新 GPU 宣稱所有舊 binary 必定不能跑。
參考：[NVIDIA Ada compatibility](https://docs.nvidia.com/cuda/ada-compatibility-guide/)。
此專案不硬裝舊 torch，也不將新版執行稱為 faithful reproduction。

新建環境可用以下命令（torch 與其餘套件的 index 分開）：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r environments/modern.txt
.venv/bin/python tools/doctor.py --output outputs/doctor.json
.venv/bin/python -m pytest -q
python3 scripts/check_upstream_integrity.py
```

初次本機測試使用既有 Python 環境建立 `--system-site-packages` venv，再將缺少依賴安裝至專案 venv；沒有修改共用環境。`environments/modern.txt` 固定所用主要版本，doctor 記錄實際版本；並非所有相依套件的 hermetic lock。

## 合法資料與 label mapping

自行取得合法 ImageNet validation 與官方 `ILSVRC2012_devkit_t12.tar.gz`，不下載／公開 ImageNet 圖片。
資料必須已整理為 `n########/ILSVRC2012_val_########.JPEG`。
第一步從 devkit leaf synsets 取 WNID，依 torchvision ImageFolder 的 alphabetic WNID order 對應 class index；使用官方 ground truth 逐張檢查 folder label。
會拒絕不完整 50k、錯標、重複 filename、未知檔案及 path escape。輸出只包含相對 ID、標籤與檔案 SHA256。

```bash
.venv/bin/python tools/make_manifests.py devkit --devkit "$IMAGENET_DEVKIT" --out labels/imagenet2012
.venv/bin/python tools/make_manifests.py split --data-root "$IMAGENET_VAL_ROOT" --labels-dir labels/imagenet2012 --seed 126 --out manifests/imagenet_val_v1
```

固定 selection 1000、holdout 49000；smoke 1000 是 holdout 的固定子集。三個 CSV 與 `index.json` 必須一起保留。
評估會驗證整組 split 的 hash、selection/holdout 不重疊、smoke 子集內容一致，並在讀圖時驗證每張圖的 SHA256。
smoke 僅供工程檢查，不得拿來選方法／調 crop；若曾用於調參，須另訂 final 保留集與 protocol。

## 權重

```bash
.venv/bin/python tools/lock_weights.py --weights-dir models_cache --output configs/targets/weights.lock.json
.venv/bin/python tools/doctor.py --weights-lock configs/targets/weights.lock.json --weights-dir models_cache --output outputs/doctor-locked.json
```

下載 torchvision 官方明確 enum，不使用 DEFAULT。CNN V1 保留舊 `pretrained=True` checkpoint identity。
支援 ResNet152（surrogate）、VGG16、ResNet50、DenseNet121、ConvNeXt Base、Swin T、ViT B/16。
lock 固定 checkpoint URL、完整 SHA256、resize/crop/interpolation/antialias/mean/std 與 transform hash。
不同 checkpoint recipe 並非純架構干預；結果須連同 clean accuracy、pretraining 與 preprocessing 解讀。
DeiT3 尚未配置，不捏造 timm 名稱；原 supplement 的 DeiT3 使用哪些 layers 仍未確認。

## 同一張真正 UAP 的跨架構評估

只接受 `float32 [1,3,224,224]` 的 `.npy`，不接受 PNG、隱式 resize、NaN/Inf 或超出 epsilon。
不自動 clip delta 來掩蓋壞輸入。raw RGB 加 delta、clip image、最後 normalization。
所有 target 使用同一個未變動的 UAP、相同 manifest IDs；每個 checkpoint 使用自己的確定性 geometry。
目標模型 eval、inference_mode、requires_grad false。

```bash
.venv/bin/python tools/evaluate.py --help
.venv/bin/python tools/evaluate.py --uap outputs/RUN/uap.npy --data-root "$IMAGENET_VAL_ROOT" --manifest manifests/imagenet_val_v1/smoke_holdout.csv --targets resnet50 vit_b_16 --weights-dir models_cache --weights-lock configs/targets/weights.lock.json --stage smoke --device cuda:0 --batch-size 16 --output outputs/RUN/smoke
# 僅在 E0 baseline 正確、方法預先固定之後執行 final：
.venv/bin/python tools/evaluate.py --uap outputs/RUN/uap.npy --data-root "$IMAGENET_VAL_ROOT" --manifest manifests/imagenet_val_v1/holdout.csv --targets vgg16 resnet50 densenet121 convnext_base swin_t vit_b_16 --stage final --device cuda:0 --batch-size 16 --output outputs/RUN/evaluation
```

`outputs/` 不可重用既有 run directory，以免覆蓋產物。每次保存 provenance、resolved config、target metrics JSON 與逐圖 predictions CSV。
FR=label flip／total；clean/adv accuracy 各自依 ground truth；CC-ASR=clean correct became wrong／clean correct。
零 clean-correct 分母回傳 null 與原因。失敗保存 failed/error_reason，沒有填成零的 metrics。
`elapsed_seconds` 包含讀圖與 hash，是 wall time，不冒稱純 GPU seconds。

## 狀態界線

權重成功下載不是 E1 實驗。CPU fixture 的 synthetic images／toy model 不是真實 ImageNet accuracy。
本 PR 未提供 trainer；controlled trainer／E2／E3 由後續獨立 PR 接續。
`configs/planned/` 仍為歷史規格，不可直接當 CLI training config。

### 已執行驗收

`python -m pytest -q`：22 passed；`tools/check_models.py`：七個官方 checkpoint 均 strict load 成功且 CPU synthetic forward 為有限 1×1000 logits；ResNet50 的零／非零 delta fixture 與原版 `get_fooling_rate` 一致。
可重跑：`.venv/bin/python tools/check_models.py`。這不是合法 ImageNet 上的 CNN+ViT smoke，沒有據此聲稱 E1 完成。
`configs/research_contract.json` 可由 `tools/validate_config.py --config ...` 檢查；foundation PR 尚無 training runner。
