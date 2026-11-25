# 第一次操作與後續跑法

## A. 這次可以真的執行的部分

從 ZIP 解壓後先讀 RESEARCH_README.md；在有 gh 登入的遠端機器恢復完整 bundle：

```bash
git clone PSP_UAP_CrossArch_Planning.bundle PSP-UAP-CrossArch
cd PSP-UAP-CrossArch
python3 scripts/check_upstream_integrity.py
bash scripts/publish_fork.sh
```

腳本不會下載 ImageNet、不會啟動 GPU 訓練、不會邀請同學，也不會自動合併 PR。它只 fork／驗證來源／設定 remotes／推 planning branch 與 tag。

## B. 原作者環境：需先檢查 GPU 相容性

官方確認環境是 Ubuntu20.04/Python3.8/torch1.11/torchvision0.12/CUDA11.3。
先由 doctor（待實作）或 `nvidia-smi` 記錄 GPU/driver。如果 GPU 不支援此舊版 CUDA，請建立 modern-compatible 分支的環境紀錄，不要把一次 pip 安裝成功當 CUDA kernel 可以跑。

可相容的機器上，照作者 README：

```bash
conda create -y -n PSP python=3.8
conda activate PSP
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install matplotlib tqdm opencv-python scikit-image imageio
```

把最後實際可安裝的套件版本保存為 lock；這段是官方舊環境指令，不是我們在此機器驗證通過的 env。

## C. 既有官方完整訓練命令（真實存在）

`IMAGENET_VAL_ROOT` 請填你有權使用的 ImageFolder validation 根目錄（包含 nXXXXXXXX 類別子目錄）。路徑不放進公開 config。

```bash
export IMAGENET_VAL_ROOT='/absolute/path/to/imagenet/val'
python main.py \
  --surrogate_model resnet152 \
  --data_path "$IMAGENET_VAL_ROOT" \
  --p_rate 0.65 --p_active --prior jigsaw \
  --re_weight --temper 3 --input_transform --seed 126
```

這會跑原版 training＋CNN evaluation，不是短 smoke。啟動之前應先完成環境／資料核對。輸出 .npy 真正位置以程式輸出為準；不得把文件範例路徑寫成實際產物。

既有 .npy 測試：

```bash
python main.py --data_path "$IMAGENET_VAL_ROOT" --uap_path '/actual/path/to/uap.npy'
```

上述官方 testing targets 是 CNN，加入 Transformer 的新版 evaluator 尚待遠端實作。官方 whole-val evaluation 包含原 stopping selection 影像；表格須註記。

## D. 研究版未完成的介面

`tools/doctor.py`、`tools/train.py`、`tools/evaluate.py`、`tools/aggregate.py` 都是下一波要開發的程式，不可現在執行。遠端完成後必須補上：

- 精確 conda/venv 環境和 lock。
- 1 個 CPU test command、1 個短 GPU smoke command、1 個正式 full command。
- 每個 command 的資料需求、預估資源（實測後才估）和輸出位置。
- E0 faithful 與 E1/E2/E3 research_fixed_budget 的 protocol ID。
- 如果未拿到 ImageNet、weights 或 GPU，輸出 BLOCKED 原因與可繼續的 code/tests，不產生假 FR。

## E. 每次跑完回報格式

```
Branch / Commit:
Stage and protocol:
Config / seed:
Data manifest hash:
Surrogate / target weights:
UAP hash:
Completed iterations:
Raw counts / FR / clean accuracy / CC-ASR:
Tests actually executed:
Artifacts:
Failures and limitations:
Next exact command:
```
