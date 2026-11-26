# Controlled trainer／E2 操作與差異

這是 **research_fixed_budget**，不是 `upstream_faithful`。`loss.py` 及其餘官方保護檔案 byte-identical。
沿用 `loss.l2_layer_loss`／`loss.l2_layer_loss_weight`、prior generator、curriculum、paired transforms；不訓練分類模型，不實作新 Loss 或 ViT surrogate。

## A/B/C 介入

- A `e2_noise_crops`：`crop(z).detach() + delta`。
- B `e2_psp_only`：`crop(z + delta.detach()).detach() + delta`。
- C `full_original`：B 加原始 T／RW。

A/B 都關閉 T／RW，crop area、ratio、loss branch、樣本數、prior schedule、初始化、LR、預算與 seed 相同。
主要研究 B−A；A 是 crop-matched random-prior control，不冒稱它等於所有原 TRM pipelines。
C 的 weighted activation branch 與 A/B 的 unweighted branch 有原碼差異，不能稱 C−B 為純 sample weighting 因果效果。

`RNGStreams` 為 init／prior／crop／transform 分別保存 Python、NumPy、torch CPU 與 CUDA RNG。
包住未改動的原 helper，使 transform 額外消耗亂數也不改變 prior/crop stream。
每步記錄 prior hash、crop boxes hash、實際 crop area min/max/mean、實際 prior_batch 與 samples；A/B 的學習軌跡預期不同，外生 draws 要相同。

## 與官方路徑的已知差異

- 固定 max_steps，使用 final iterate；不呼叫 target FR、不 early stop、不依 heldout 選 checkpoint。
- 正式 checkpoint 是 delta detached clone；Adam／step／全部 RNG／curriculum／saturation／config／source+implementation hash 一起保存。改 code／model／device type／seed／config 時拒絕 resume。
- 官方 best_uap alias、累積 failure counter 與 zero-FR 未賦值問題保留在官方檔案；此 runner 有自己的明確預算與 snapshot 行為。
- saturation 偵測閾值保留；研究 runner 沒有 FR failure counter，因此觸發時固定除以 2，不進入原 near-patience 的 0.8 分支。
- 現代 torchvision crop resize 明確 `antialias=True`。原環境 torchvision0.12 的預設與此可能不同，不能稱完全忠實。
- paired transform 直接呼叫原 helper，但以 stack 建立輸出，避免新版 autograd 的 in-place version 問題。
- 原 jigsaw always-true boolean branch 保留，A/B/C 一致，不偷偷修正。
- 原 synthetic-prior preprocessing／loss 輸入照原 helper，不擅自增加 normalization 或 image clipping。目標評估才使用 raw RGB add→clip→normalization。
- 模型 `eval + requires_grad_(False)`，optimizer 僅有 delta；CPU fixture 驗證模型 tensor hash 不變。
- loss／sample／gradient／optimizer state／delta NaN/Inf 會失敗並留下 failed 狀態，不修改 frozen loss 加 epsilon 修補。
- 預設 `--determinism strict`。RTX4090／torch2.5.1 實測原 paired rotation 的 `grid_sampler_2d_backward_cuda` 在此模式第一步失敗。工程 smoke 可明確加 `--determinism warn`，保留原 transform 並接受 PyTorch 的 CUDA 非確定性警告；完整research目前亦可明確指定warn模式，但必須通過新增E0 baseline gate（`--baseline-record`）；完整執行命令見FULL_EXECUTION。數值模式寫入 provenance/checkpoint，resume 拒絕模式不同的 checkpoint，不宣稱 GPU bitwise replay。

## 命令

```bash
.venv/bin/python tools/train.py --help
.venv/bin/python tools/train.py --config configs/experiments/e2_noise_crops.json --seed 126 --dry-run
.venv/bin/python tools/train.py --config configs/experiments/e2_psp_only.json --seed 126 --dry-run
```

資料與 checkpoint lock 先依 `TOOLS_RUNBOOK_ZH.md` 完成。沒有 verified manifest bundle 時，CLI 在載入模型／訓練之前明確 BLOCKED。

有合法資料後，先做 100–200 step smoke（不是 E0，也不是正式 E2）：

```bash
.venv/bin/python tools/train.py --config configs/experiments/full_original.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --smoke-steps 100 --device cuda:0 --output outputs/full-original-smoke126
.venv/bin/python tools/evaluate.py --uap outputs/full-original-smoke126/uap.npy --training-record outputs/full-original-smoke126/provenance.json --data-root "$IMAGENET_VAL_ROOT" --manifest manifests/imagenet_val_v1/smoke_holdout.csv --targets resnet50 vit_b_16 --stage smoke --device cuda:0 --output outputs/full-original-smoke126/evaluation-smoke
```

CPU fixtures 通過不代表 ResNet152 full RW 在 GPU 一定 finite；原始 reciprocal KL 可能失敗。需記錄真實 smoke 時間、VRAM 與失敗，再判斷是否可開始 full。

E0 須先依官方 README／`OPERATING_RUNBOOK_ZH.md` 做正式 faithful baseline，現代相容路徑另行標記。這個 runner 不把自己的 fixed-budget run 冒稱 E0。
只有 baseline 正確後，才執行正式 A/B/C（每個 seed 同樣 10000 steps，不看 final targets 調參）：

```bash
.venv/bin/python tools/train.py --config configs/experiments/e2_noise_crops.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --output outputs/e2-a126
.venv/bin/python tools/train.py --config configs/experiments/e2_psp_only.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --output outputs/e2-b126
.venv/bin/python tools/train.py --config configs/experiments/full_original.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --output outputs/full-original126
```

其餘預先固定 seeds 2025、3407，同命令替換 seed／output。CLI 沒有 target gradient 或 target checkpoint selection。

### 中斷與恢復

```bash
.venv/bin/python tools/train.py --config configs/experiments/e2_noise_crops.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --stop-after 200 --output outputs/e2-a126-part
.venv/bin/python tools/train.py --config configs/experiments/e2_noise_crops.json --seed 126 --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --resume outputs/e2-a126-part/checkpoints/000200.pt --output outputs/e2-a126-resumed
```

未達總預算者標 `incomplete`、不輸出 final `uap.npy`；resume 用新 output 目錄。意外中斷可用最近已完整寫入的 checkpoint。
checkpoint 用 `torch.load(weights_only=True)`，不要求執行任意 pickle；只恢復本專案同 config／source／model 的 checkpoint。
CPU fixture 已驗證 loss records、delta、Adam、RNG、curriculum 逐位元相同。GPU exact replay 仍需實測，不以 CPU 結果代替。

評估請加 `--training-record .../provenance.json`，驗證 UAP hash／manifest／epsilon，並將 seed、variant、crop、T/RW 與 training identity 傳給 metric row。
`--stage final` 拒絕 smoke training 或 incomplete record。外來 UAP 可獨立評估，但不會憑空補寫未知訓練 metadata。

## 實際執行界線

67 項 CPU fixtures 通過；2026-11-26 已在 Kaggle validation／TFDS 交叉核對 manifests 上完成 ResNet152100-step GPU smoke，並用其 UAP 做四架構 target-only smoke。嚴格確定性模式第一步失敗；明確 `--determinism warn` 模式完成100步。原始 devkit 已於後續取得並完成 [官方 mapping 交叉驗證](DEVKIT_VERIFICATION_ZH.md)，正式 E0/E1/E2/E3 與完整訓練未執行。詳見 [實測紀錄](KAGGLE_DATA_SMOKE_ZH.md)。
原始 Loss SHA256：`6855a7310c30fbd5bddfefbdb9ca84c2c5732a9aaaecfc316657b4fb8f108982`。
