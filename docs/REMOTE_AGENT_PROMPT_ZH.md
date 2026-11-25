# 可直接貼給遠端開發 Agent 的完整 Prompt

你要開發 115_1作業的 PSP-UAP cross-architecture 研究。本輪不是新 loss、不是 ViT surrogate adaptation，也不是重新做文獻空白的絕對主張。任務以這個 repo 的 AGENTS.md 和 docs 為準。

## 0. 來源與實際狀態

官方 `ChanhuiLee1111/PSP-UAP`，baseline `dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c`，MIT。預定公開 fork `KuanWuChu/PSP-UAP-CrossArch`。本次已完成 local clone 與 docs planning，但 fork 尚未建立、研究程式尚未實作、GPU 實驗尚未執行。不要把 planning configs 當成 runnable CLI。

如果從交付 ZIP 開始，先 `git clone PSP_UAP_CrossArch_Planning.bundle PSP-UAP-CrossArch`，進入 repo，執行 `bash scripts/publish_fork.sh`。此 script 需要已有 `gh auth`。使用者已同意公開 fork，毋須重新問公開與否；但若未登入，要如實回報登入需求，不讀取聊天外或不相關的憑證。

檢查 origin=自己的 fork、upstream=作者 repo，驗證 fork parent/visibility。保留 MIT 和全部歷史。使用者之後自己加同學，這輪不要邀請任何人。

## 1. 開始前必讀

RESEARCH_README.md、AGENTS.md、docs/UPSTREAM_AUDIT.md、docs/EXPERIMENT_PROTOCOL.md、docs/WORK_PACKAGES.md、docs/STATUS.md。跑 `python scripts/check_upstream_integrity.py`。

不可修改 loss.py；原六個 Python 作 faithful route 保留。新功能用 `psp_research/` 和 `tools/`。

## 2. 第一個 PR：provenance + data + evaluator

從目前計畫分支開後續 feature branch，或將 docs PR 合併後從 fork main 開。不要直接在作者 repo 寫入。

完成以下**真的可執行**功能：

- doctor：版本/GPU compute capability/VRAM/weight identity/data status，legacy 和 modern 路徑分開，無 GPU 必須清楚寫；新 GPU 不硬裝 torch1.11。
- config validation：strict enums、epsilon/crop/ratio/steps/source_mode/T/RW、fixed seeds；dry-run 印出 resolved config。
- manifests：合法 ImageNet val class mapping、selection1000/holdout49000、固定 IDs、hash、拒絕 overlap。
- target registry：torchvision CNN/ConvNeXt/Swin/ViT 的 exact pretrained weights、transform config；DeiT3 optional registry check，未確認不得填假 model ID。
- evaluator：只載入 UAP .npy；raw RGB add→clip→normalize；不使用 target gradient；zero-delta identity；FR + clean/adv accuracy + clean-correct ASR raw counts。
- artifacts：run_id/protocol_id、source/research commit、weights/manifest/UAP SHA256、error status，不把 missing 當零。

寫 README（中文操作＋英文命令）、CLI --help、CPU tests，若有合法 data/GPU 才跑短 evaluator smoke。不得宣稱已完成 E0 reproduction。

## 3. 第二個 PR：受控 trainer 與 E2

- 沿用原始 loss 函式，不新增 objective。
- uniform delta init、原 prior curriculum、同 clipping、正確 detach gradient path。
- fixed-budget runner 與 upstream-faithful 兩種結果分開。
- A noise_crops = crop(z).detach()+delta；B PSP-only = crop(z+delta.detach()).detach()+delta；A/B same crop/N/prior schedule/activation loss/T=false/RW=false。這是 crop-matched source contrast，不冒稱它完全等於原 TRM。
- C full = B+原 T/RW。A/B 同 unweighted objective；C weighted loss 是既有 function，不能說加權完全相同。
- Dedicated RNG streams／replay exogenous draws、保存 Python/NumPy/CPU/CUDA/curriculum state。
- 真 checkpoint = delta.detach().clone()；不要複製 upstream 的 alias bug。
- 保持 upstream jigsaw behavior 在全受控組一致；若另外修正要有獨立 config/表，不能默默改。
- NaN/Inf fail-fast；loss 不插 epsilon 假裝 unchanged。
- resume CPU fixture、finite GPU 100–200 iter smoke，記錄而不是猜測時間和 VRAM。

## 4. 第三個 PR：E3 與正式 runs

crop **area** ranges local[.08,.35]、original[.08,1]、large[.50,1]，ratio[.75,4/3]；full T/RW 及其餘設定固定。重用同 config 的 baseline UAP，不為 E1 重訓。

先 E0 官方 ResNet152 seed126 full run、核對 CNN baseline；成功才 E1 target-only、E2配對、E3 crop。其他 seeds2025/3407，最小受控15 runs。大量 batch 前 profile GPU/資料 loading。缺 ImageNet 或 GPU就完成 code/tests＋精確 run commands，將實驗列 BLOCKED 而不是捏造數據。

## 5. 研究解讀限制

不要宣稱第一次發現 Transformer 限制；supplement 已測 DeiT3。公開 loss 的 Conv2d-only 可驗證，但不能說原 supplement 的 DeiT3 一定只用 patch embedding。小 ViT FR 不代表 prior沒用；主要比較逐 target matched ΔFR。Crop改善不證明語義／頻率／局部性根因；本階段不做真正 objective因果替換。

## 6. 驗收及交付

每個 PR 交付：問題與結果行為、變更檔、original loss hash、測試命令與實際結果、smoke和full狀態、run_ids/hash、後續缺件。Draft PR 指向自己 fork，未被要求不自動 merge。

最終產物：可跑 environment config、train/evaluate/aggregate CLI、固定 manifests（不含圖）、run records、UAP hash/檔位置、E1 matrix、E2 ΔFR、E3 crop表、seed mean/std、限制說明；每份結果標 faithful/research_fixed_budget 及 weights/data/version。

開始時先回報 repo/origin/upstream、工作分支、可用GPU與data、目前 blockers；完成第一個可驗收 PR 後給 branch/commit/PR link，並列明下一組命令。不要只寫一份新計畫而沒有實作，也不要把本次文檔的預定介面當完成品。
