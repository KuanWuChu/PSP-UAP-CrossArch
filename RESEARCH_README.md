# PSP-UAP Cross-Architecture Study — 開發入口

2026-11-25，115_1作業。這是以 CVPR 2025 PSP-UAP 官方程式為基礎的研究擴充計畫。

**2026-11-26：公開 native fork 已驗證。Kaggle validation 全部 50,000 張已下載並驗證，標籤與固定 TFDS 版本逐張核對，manifests 已建立。62 項 CPU 測試、100-step ResNet152 相容模式及 CNN／ConvNeXt／Swin／ViT GPU smoke 通過。原始官方 devkit 仍缺；完整訓練與正式 E0–E3 尚未執行。**

本次實測與重跑命令：[Kaggle 資料與 GPU smoke](docs/KAGGLE_DATA_SMOKE_ZH.md)；完整 hash／原始計數摘要：[驗證報告](reports/kaggle_gpu_smoke_validation.json)。

實際可執行入口：[工具操作文件](docs/TOOLS_RUNBOOK_ZH.md)、[受控 trainer／E2](docs/TRAINER_RUNBOOK_ZH.md)、[E3 與結果彙整](docs/EXPERIMENT_EXECUTION_ZH.md)。最新分項狀態：[STATUS](docs/STATUS.md)。

- 官方： https://github.com/ChanhuiLee1111/PSP-UAP
- 固定原始 commit：`dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c`
- 預定公開 fork：[KuanWuChu/PSP-UAP-CrossArch](https://github.com/KuanWuChu/PSP-UAP-CrossArch)
- 本次 feature branch：`feat/kaggle-data-smoke`（以 PR 交付，未自動 merge）
- 原碼基準 tag：`research/upstream-20261125`
- 授權：沿用官方 MIT License，保留作者著作權與論文引用。

## 遠端從這裡開始

1. [完整工程與實驗規劃](docs/DEVELOPMENT_PLAN_ZH.md)
2. [官方程式稽核／不可直接假設的事項](docs/UPSTREAM_AUDIT.md)
3. [E0–E3 精確實驗協定](docs/EXPERIMENT_PROTOCOL.md)
4. [PR 分工、相依性與驗收](docs/WORK_PACKAGES.md)
5. [可直接貼給遠端的開發指令](docs/REMOTE_AGENT_PROMPT_ZH.md)
6. [公開 fork／clone／協作流程](docs/REPOSITORY_SETUP.md)
7. [第一次操作與原作者命令](docs/OPERATING_RUNBOOK_ZH.md)
8. [本次交付狀態](docs/STATUS.md)

## 我們要做什麼

- E0：按照作者原始設定，先復現 CNN 結果。
- E1：同一張 CNN-trained UAP，對 CNN、ConvNeXt、Swin、ViT 做 target-only evaluation。
- E2：配對比較「隨機來源」與「UAP-derived source」的取樣增益；不是只比較完整 PSP 與 TRM。
- E3：只改 crop area range，固定 activation loss 與其他條件。

**`loss.py` 保持 byte-identical。分類模型不重新訓練。不把原文 DeiT3 補充實驗說成已確認只用 patch embedding。**

本 repo 暫時保留官方 `README.md` 與全部 Python 原檔；後續功能請使用新的 research package，與官方復現路徑分離。
`configs/planned/` 只是規格資料，沒有可執行 training runner；不能把它們當成已完成程式。
