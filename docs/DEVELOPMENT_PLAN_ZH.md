# PSP-UAP 公開 fork 與遠端開發完整規劃

版本：2026-11-25。此文件是工程交接與實驗規格，並不是實驗結果。

## 1. 任務範圍與正式研究故事

指定論文是 CVPR 2025《Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior》。我們先理解原作者如何從 evolving UAP 建立 pseudo-semantic samples，再透過固定模型的 convolutional activations 更新 UAP。原作者已展示 CNN 結果，補充材料也包含 ConvNeXt-B 和 DeiT3；因此不宣稱第一次測 Transformer。公開程式可確認 Conv2d-based loss，但不能據此斷言補充 DeiT3 實驗的確切 layer mapping。

我們的研究問題是：**當同一個 CNN surrogate 學出的 UAP 被移到不同 target architecture，pseudo-semantic sampling 帶來的額外效益是否改變？**

H1：PSP sampling 對 CNN targets 的增益可能較大。
H2：它對不同架構都有類似增益。
其他可能：差異來自 objective、權重訓練配方、preprocessing 或個別 target，不一定是 sampling。

本階段不改 loss.py、不設計 Transformer objective、不更新分類模型、不承諾一定改善 FR。

## 2. 已完成／尚未完成

| 事項 | 狀態 |
|---|---|
| 確認官方 repo、MIT 授權及原始 commit | 完成 |
| 完整 git clone、保留原歷史 | 完成 |
| 官方 Python 檔案 hash 基準 | 完成 |
| E0–E3、PR 分工、驗收與遠端 prompt | 完成 |
| GitHub 公開 fork、推送計畫 | 尚未完成：目前終端未登入 gh；現有 GitHub 連線沒有 create-fork API |
| 邀請同學 | 使用者之後自行設定 |
| research package、GPU training、FR 結果 | 尚未實作／尚未執行 |

注意：local clone 和 GitHub fork 是不同動作。前者已做，後者不能假稱完成。

## 3. Repo 結構與維護方式

GitHub fork 預定名稱 `KuanWuChu/PSP-UAP-CrossArch`。`upstream` 指向作者 repo，`origin` 指向自己的 fork。完整保留歷史、LICENSE、README、原始 Python；將研究入口放在 RESEARCH_README.md。

| 路徑 | 狀態／目的 |
|---|---|
| 原始 main.py / PSP_UAP.py / functions.py / loss.py / semantic_aug.py / strategy.py | 已 clone；faithful 路徑原樣保留 |
| docs/ | 已建立；規劃、來源稽核、協定與交接 |
| scripts/check_upstream_integrity.py | 已提供；無需 ML 套件的 hash 檢查 |
| scripts/publish_fork.sh | 已提供；需登入 gh 後執行的 repo bootstrap |
| configs/planned/ | 已提供；待實作 runner 的規格資料，非完成品 |
| psp_research/ | 待建立：config、models、sampling、training、evaluate、metrics、artifacts |
| tools/ | 待建立：doctor.py、make_manifests.py、train.py、evaluate.py、aggregate.py |
| tests/ | 待建立：有意義的數學、數值、流程測試 |
| outputs/ | 執行時生成，不入一般 Git |
| reports/ | 小型 summary CSV、圖表、方法與結果說明，人工確認後入 Git |

## 4. 開發兩種流程，禁止混表

### A. upstream-faithful

完整保留官方原碼與原本環境。用官方 ResNet152 command，先 seed 126、再追加 seeds。原版包含 validation-based stopping、原始 sampling、原始 bugs 和 weights 版本。這條用於回答『是否大致重現公開程式』。

### B. research-controlled

保留相同 activation loss function，另外建立 runner。固定可追蹤的 RNG、資料 manifest、保存真正獨立的 checkpoint、有限數值檢查和固定步數。所有研究 variants 使用同一套 runner 與同一 stopping/checkpoint policy。

這條與 A 的差別要逐項列出；它不是完全相同環境的逐位元複現。legacy torch 1.11 可能不支援新 GPU，先跑 doctor。若用新版 PyTorch/torchvision，列為 modern-compatible，不直接稱為精確復現。

## 5. 實驗內容

### E0：復現及工程正確性

第一輪 1 個 ResNet152 surrogate、seed 126、官方 prior=jigsaw、p_rate=0.65、p_active=true、temper=3、input_transform=true、re_weight=true、epsilon=10/255。用官方設定測五個 CNN targets。後續 seeds 建議 2025、3407。

- 確認 UAP shape=1×3×224×224、float dtype、finite、L∞≤epsilon。
- 記錄初始化 Uniform(-epsilon,epsilon)；Gaussian/jigsaw 是 synthetic input prior，不能混淆。
- 官方最大迭代 10000，實際可能早停；記錄實際 iterations、selection FR、returned tensor 的 reevaluated FR。
- 不以偏差 2% 內當任意硬門檻；報差異與變異，先查版本與評估流程。若偏差明顯、loss NaN、資料 mapping 不對，停止擴大 run。
- CNN 模型權重保持不變，記錄 hash；eval mode 與 requires_grad 是不同事情。

### E1：target-only 跨架構

不改 UAP tensor，不用 target gradient，不因 target 換模型而重訓。初版 targets：VGG16、ResNet50、DenseNet121、ConvNeXt-B、Swin-T、ViT-B/16。DeiT3 待選定公開 checkpoint/timm 版本後再加，不能寫成已支援。

主比較採全部支援 224 的 checkpoint，擾動在 target 的 resize/crop 完成、normalize 之前加進 raw RGB，clip [0,1] 後 normalize。每個模型記錄 pretrained recipe、input size、interpolation、crop、mean/std。若不相同，承認 estimand 是『model + preprocessing pipeline』，不宣稱只改架構；增加共同幾何 preprocessing 的 sensitivity 表。ConvNeXt 是 CNN；Swin 是 hierarchical Transformer，不能把兩者都叫純 ViT。

### E2：取樣來源消融（核心）

只比 full PSP 與 TRM 不能隔離 prior，因為 T/RW 等也不同。更乾淨的是：

- A：`noise_crops`：先 crop/resize 隨機 prior，再 detach，最後加 learnable delta。
- B：`psp_crops`：先 crop/resize 同一種 prior + stop-gradient(delta)，detach 後再加 learnable delta。
- C：Full PSP：B + 原本 T/RW，作完整法參考。

A/B 同樣 crop、N、噪聲 schedule、p_rate、activation loss、steps、RNG replay，T=false、RW=false。B−A 比較 UAP-derived source 的增益，避免把『有沒有裁切』也混進去。可另外加 `noise_direct` 作 paper-style random prior 參考，但不得叫它最乾淨的 source-only contrast。

重要：E2 的 weighted/unweighted 切換改變既有 loss 的加權方式，所以只能說『不設計新 objective，沿用既有 functions』，不能說 full 與 only 的完整 loss 完全一致。A/B 才固定同一個既有 unweighted loss。

### E3：crop area range

僅改 `scale_crop` 面積比例：local [0.08,0.35]、original [0.08,1.00]、large [0.50,1.00]；aspect ratio 固定 [0.75,1.333333]。不是 crop 邊長比例，不把 large crop 稱為已證明的 global semantics。

初版固定 full PSP 的 T/RW；original mixed 是 E0/research 基準，可重用同 config/seed/checkpoint。比較新範圍需重訓 UAP，E1 的 target-only 測試則不重訓。超參數預先列出，不看 ViT final test FR 再挑最佳範圍。

## 6. 資料、指標與結果追蹤

ImageNet validation 由研究者自行合法取得，使用 ImageFolder class layout，核對 wnid→class_index 與 checkpoint 的 label 順序。程序不自動下載未授權 ImageNet，不公開 images。先固定 1000 張 selection manifest，再從其餘 49000 張中選 smoke subset（1000 張、與 selection 不重疊），最後在完整 49000 張 final holdout 評估。

原碼 final analyze=True 會跑全部 50000 張，包含用於 stopping 的部分；faithful 表照原碼但揭露 overlap。正式受控表用同一份 holdout，不把兩者混稱相同 protocol。

主指標 FR = count(clean_pred != adv_pred)/N。次指標 clean accuracy、adversarial accuracy、clean-correct ASR = count(clean_pred==label and adv_pred!=label)/count(clean_pred==label)。FR 不是 ground-truth ASR。

ΔFR 用 percentage points，逐 target、逐 matched seed 比較 B−A，先報 3 seeds 的 mean/std，再圖示 model family summaries。每家族只有一兩種 target 時，不做整個架構族的普遍因果推論。3 seeds 是 pilot，不能把數千張 image 當成數千次獨立 UAP 訓練。

每個 run 留 config、upstream commit、research commit、dirty flag、Python/torch/CUDA/GPU、weights identity/hash、manifest hash、rng、UAP hash、loss trace、iter、checkpoint policy、raw counts 和耗時。errors 需標記 FAILED，不可用 0 FR 代替。

## 7. 程式介面（待實作，不可現在直接使用）

目標 CLI：`python tools/doctor.py`、`python tools/make_manifests.py`、`python tools/train.py --config ...`、`python tools/evaluate.py --uap ... --config ...`、`python tools/aggregate.py --runs ...`。完整指令與 contract 在 EXPERIMENT_PROTOCOL.md。只有官方 README commands 與目前 bootstrap/integrity scripts 已存在。

## 8. 工作量控制

先單 seed smoke，再正式 3 seeds。1 surrogate 的最小完整受控組合：A noise_crops×3 + B PSP-only×3 + full/original×3 + full/local×3 + full/large×3 = 15 runs。E1 可重用 full/original 的 3 個 UAP，不另外新增 training。faithful E0 若另跑 3 seeds，則總計 18；真實 GPU 小時先 profile 100–200 iterations 和 eval throughput 再估，不先捏造。

第一波 PR 僅完成 setup、metadata/manifest/metrics 和 E1 evaluator，做 finite smoke；第二波建立受控 training 與 E2；第三波 E3 + repeat seeds + figures。詳見 WORK_PACKAGES.md。

## 9. 不允許的過度結論

- 低 DeiT3 FR ≠ 所有 Transformer 都抗 UAP。
- B−A 在 ViT 小 ≠ CNN activation 已被證實為根因。
- crop 改善 ≠ 真正語義／architecture invariant representation 被證明。
- 未檢查完整文獻不能宣稱 first / nobody studied。
- CPU fixtures 或單 GPU smoke ≠ E0 reproduced / E1–E3 completed。

## 10. 參考來源

- 原論文：https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Data-free_Universal_Adversarial_Perturbation_with_Pseudo-semantic_Prior_CVPR_2025_paper.pdf
- 補充：https://openaccess.thecvf.com/content/CVPR2025/supplemental/Lee_Data-free_Universal_Adversarial_CVPR_2025_supplemental.pdf
- 固定程式：https://github.com/ChanhuiLee1111/PSP-UAP/tree/dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c

本次重新讀取並稽核官方 README、LICENSE 與本地 clone 的 Python；supplement 表格是先前提案依據，遠端應下載原 PDF 核對後才能把數值寫進正式 results。沒有必要把未核對的 2026 數字寫到工程開發規格。
