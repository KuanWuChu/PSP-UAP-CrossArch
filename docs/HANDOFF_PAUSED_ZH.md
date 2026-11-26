# 暫停交接：E0完成，正式矩陣未啟動

使用者要求在可交接點暫停實驗並push一次。自動stage controller已停止，啟動器已加入暫停guard；15組研究訓練／90個final target evaluations沒有啟動。GitHub自動push／PR／release callback保持關閉，不自動merge。

## 已完成的真實執行

- 現代相容環境：未改動的官方ResNet152／seed126完整流程及五CNN50,000張原生評估完成；[原始報告](../reports/e0_modern_original.json)。
- 作者宣告核心環境：Ubuntu20.04／Python3.8.20／torch1.11+cu113／vision0.12+cu113／CUDA11.3；官方原碼訓練依原始early stopping結束，五CNN原生評估完成。
- 後者產出的同一張UAP重新載入，以現代獨立evaluator對五CNN各50,000張計算FR、clean/adv accuracy、clean-correct ASR與原始計數。native/reloaded差異與checkpoint官方clean reference通過事先固定的工程檢查。
- CPU fixtures：72 passed in 5.59s；八個官方保護檔案維持byte-identical。原始首次teacher的BatchNorm狀態更新保留於E0，研究版eval/frozen差異另列，不宣稱完全相同。
- 這是單seed原碼執行和工程驗證，不代表已重現論文五seed平均／std。原版未設定NumPy seed、alias、stopping與jigsaw行為均保留。

| Target | Native FR (%) | Reloaded FR (%) | Clean acc (%) | Adv acc (%) | Clean-correct ASR (%) |
|---|---:|---:|---:|---:|---:|
|alexnet|58.586|58.598|56.556|32.968|46.626|
|vgg16|88.258|88.256|71.586|10.942|85.654|
|vgg19|87.514|87.508|72.394|11.538|85.013|
|resnet152|86.932|86.934|78.314|12.216|85.116|
|googlenet|74.924|74.926|69.772|22.400|69.750|

Native欄來自舊環境原生程式；reloaded各欄來自相同UAP／V1 checkpoint在torch2.5.1+cu121的獨立評估。兩者皆包含原始stopping selection，不能與研究版49,000張holdout混表。完整raw counts／版本／hash／差異見 [verified E0](../reports/e0_legacy_verified.json)。

## 本機產物

- Verified baseline：`outputs/full-experiments-20261126/baseline-validated.json`。
- 舊環境原始UAP：`outputs/full-experiments-20261126/e0-legacy-original/source/results/additional/uap_reweight_3.0_6_10_resnet152_dataset=imagenet_p_rate=0.65_seed=126_prior=jigsaw_True.npy`。
- UAP SHA256：`83a6710107b7d8b06d06350559c7f6c6c8b600369ec4b28eba67509138136049`。
- 每模型原始計數與predictions：`outputs/full-experiments-20261126/e0-legacy-original/reloaded/`。
- 固定資料：`data/imagenet2012/val`；官方devkit manifests：`manifests/imagenet_val_devkit_v1/`。
- E0權重lock：`configs/targets/e0_weights.lock.json`；E1/E2/E3權重lock：`configs/targets/weights.lock.json`。分類權重本體在`models_cache/`，不進Git。
- 未執行矩陣的固定程式worktree：`outputs/full-experiments-20261126/execution-source`，commit `2fdf46e7610c8c4d184655f1246afe5e17587e0d`。

既有baseline／smoke／training provenance不回寫；本次補充報告保留原始baseline JSON hash。影像、分類權重、憑證、私有log不推送；本次不建立release。

## 目前可直接執行的讀取檢查

```bash
.venv/bin/python tools/experiment_status.py
python3 scripts/check_upstream_integrity.py
```

## 恢復時的下一組命令

目前維持暫停；以下只供使用者之後指示恢復時執行。先保留既有`baseline-validated.json`，不重新寫入其provenance。從repo根目錄進入固定source worktree：

```bash
PSP_PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd outputs/full-experiments-20261126/execution-source
OMP_NUM_THREADS=4 .venv/bin/python tools/run_matrix.py --baseline-record "$PSP_PROJECT_ROOT/outputs/full-experiments-20261126/baseline-validated.json" --output "$PSP_PROJECT_ROOT/outputs/full-experiments-20261126/matrix"
```

此命令尚未執行。會跑既定五variants ×三seeds ×10000 steps，完成全部訓練後才做六target49,000張評估。沒有變更loss或參數，也沒有按final FR挑選checkpoint。

## 尚未完成與限制

正式E1、E2、E3、三seed研究統計及完整矩陣圖表尚未執行；這些是因使用者暫停而保留的後續工作，不是缺資料／GPU。原始validation tar MD5仍未驗證，資料採已驗證的Kaggle ZIP路徑。GPU逐位元checkpoint replay未建立；research CUDA warn-only模式可能非確定。
