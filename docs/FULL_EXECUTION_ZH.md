# 完整實驗執行與 GitHub 維護

先完成 E0 原碼執行、實際 UAP／五個 CNN 的50,000張重新評估及來源檢查，才允許15-run研究矩陣啟動。完整訓練已獲授權；不自動 merge。

## 已固定的範圍

- E0 pilot：官方固定 commit，ResNet152、seed126、官方 jigsaw／p_rate0.65／T／RW／temper3，原始 stopping/max10000。原始 alias、failure counter、未設定 NumPy seed 與 jigsaw boolean branch 均保留。
- E1：重用 full_original 的 CNN-trained UAP，六個 targets：VGG16、ResNet50、DenseNet121、ConvNeXt Base、Swin T、ViT B/16。
- E2：noise_crops A vs psp_crops B，T/RW均關閉；相同 crops／prior／樣本數／loss／seed／預算。C保留完整T/RW。
- E3：full_local[.08,.35]、full_original[.08,1]、full_large[.5,1]，其他條件固定。
- 每種variant seeds126/2025/3407，每run10000 steps；共15張UAP、90個49,000張target records。先凍結所有training artifacts，才查看final targets，絕不按FR選參數／seed／checkpoint。
- 不新增loss，不訓練classifier，不做ViT surrogate adaptation；不宣稱首次測Transformer或確定作者DeiT3層選擇。

## 環境與數值模式

Ubuntu20.04容器、Python3.8.20、torch1.11.0+cu113、torchvision0.12.0+cu113、CUDA11.3已在RTX4090通過真正的卷積backward。原始PyTorch容器是Ubuntu18.04，因此另建立作者指定OS的runtime。`environments/Dockerfile.legacy`固定Ubuntu20.04 base digest；`environments/legacy-core.lock.txt`記錄補充套件。GPU/driver與未公開的作者精確補充套件版本仍不能宣稱完全相同。

另保留 modern compatibility：torch2.5.1+cu121 / torchvision0.20.1+cu121。受控矩陣使用此環境；crop antialias、stack assembly與fixed-budget差異依TRAINER_RUNBOOK記錄。`loss.py`與其餘官方保護檔案不變。

正式research可明確使用`--determinism warn`，但必須提供驗證通過的`--baseline-record`。CUDA grid-sampler backward存在非確定性，不宣稱GPU逐位元重現。數值模式、torch/CUDA/cuDNN版本隨checkpoint保存；resume拒絕不同模式。彙整也把training environment／數值模式列入配對identity。

## 可執行入口

先確認矩陣解析，不啟動GPU：

```bash
.venv/bin/python tools/run_matrix.py --baseline-record outputs/full-experiments-20261126/baseline-validated.json --output outputs/full-experiments-20261126/matrix --dry-run
```

五個官方baseline CNN的獨立checkpoint lock：

```bash
.venv/bin/python tools/lock_weights.py --targets alexnet vgg16 vgg19 resnet152 googlenet --output configs/targets/e0_weights.lock.json
.venv/bin/python tools/evaluate_e0.py --uap ACTUAL_E0_UAP.npy --data-root data/imagenet2012/val --manifest-dir manifests/imagenet_val_devkit_v1 --execution-label upstream_legacy_core --output outputs/e0-reloaded
```

E0工具使用原始50,000張（明確包含stopping selection），輸出FR、accuracy、clean-correct ASR、raw counts與predictions；不與研究版49,000張混成同一表。原生程式輸出的UAP必須重新載入並檢查finite／epsilon，不能只讀training期間最佳FR。

完成實際baseline檢查後，`baseline-validated.json`需包含五target的完整計數、UAP實際path/hash、manifest hashes、execution label、upstream commit與`baseline_checks_passed=true`。這是工程來源／資料／指標檢查的證據，不等於單seed已重現論文五seed的平均/std；paper comparison另列。

```bash
OMP_NUM_THREADS=4 .venv/bin/python tools/run_matrix.py --baseline-record outputs/full-experiments-20261126/baseline-validated.json --output outputs/full-experiments-20261126/matrix
```

執行器鎖定source commit／configs／weights／data，拒絕來源變更後沿用同一root。中斷後用同一命令繼續：只重用identity一致的complete產物，其他從最近immutable checkpoint恢復；數值failed不偷偷修loss或挑FR重跑。status.json持續記錄目前phase、run與process；每個run有原始log。最終summary必須完整通過aggregate，否則標incomplete並保留missing／failed。

長run在固定commit的獨立worktree執行，讓結果與GitHub文件維護不改動正在跑的原始碼。

## GitHub交付

- 各功能與結果用獨立branch／draft PR推到KuanWuChu/PSP-UAP-CrossArch，不推作者repo、不自動merge。
- GitHub Actions自動跑CPU fixtures與八檔上游完整性；不下載ImageNet或啟動GPU。
- 完成後整理三seed mean／sample std、E1矩陣、E2 B−A、E3配對差值與標準plotting工具生成的圖；missing/failed不填0。
- 公開只交付可稽核的結果、hash、相對路徑與必要provenance；影像、分類權重、憑證與私有原始log不進Git。UAP只在大小／內容／metadata／hash驗證後作版本化artifact。
