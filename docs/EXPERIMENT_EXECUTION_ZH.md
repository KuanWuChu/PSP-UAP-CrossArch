# 已實作 E3 與結果彙整

三個完整 PSP 設定只改 **crop area** 與 variant 名稱；ratio 固定 [0.75,4/3]，T/RW 開啟：

| config | area |
|---|---|
| `full_local.json` | [0.08,0.35] |
| `full_original.json` | [0.08,1.00] |
| `full_large.json` | [0.50,1.00] |

`configs/run_matrix.json` 預先固定五種 variants × seeds126/2025/3407，共十五個 fixed-budget runs，各10000 steps。
E1 重用 full_original 的 UAP，不另行訓練。這些是可執行設定，不代表實驗已跑。
本階段不拿 final target 輸出挑 crop／seed／checkpoint，不更換 objective。

## 下一步依序執行

先依 `TOOLS_RUNBOOK_ZH.md` 提供官方 devkit、驗證合法 ImageNet mapping/splits/weights，並執行 CPU fixtures。
有資料後先做 `TRAINER_RUNBOOK_ZH.md` 的短 GPU smoke；記時間、VRAM 與 checkpoint。未跑過的 smoke 不填數據。

E0 官方 ResNet152 seed126 命令如下（需另外準備並驗證官方 faithful 環境；若在新版環境執行，必須標記 modern compatibility）：

```bash
python main.py --surrogate_model resnet152 --data_path "$IMAGENET_VAL_ROOT" --p_rate 0.65 --p_active --prior jigsaw --re_weight --temper 3 --input_transform --seed 126 --device cuda:0
```

原路徑保留上游 stopping、selection overlap、best_uap alias、NumPy seed 與 jigsaw bugs。它產生的 UAP 必須重新載入評估；不得把研究版 snapshot 修正說成官方未改行為。
先核對 CNN baseline，解決可疑結果後，再跑正式 matrix。以下迴圈是基準驗證後的命令，這次尚未執行：

```bash
for variant in e2_noise_crops e2_psp_only full_original full_local full_large; do
  for seed in 126 2025 3407; do
    .venv/bin/python tools/train.py --config "configs/experiments/$variant.json" --seed "$seed" --manifest-dir manifests/imagenet_val_v1 --device cuda:0 --output "outputs/$variant-$seed" || break 2
  done
done
```

以每個 seed 同一張 UAP 測六個 targets：

```bash
for variant in e2_noise_crops e2_psp_only full_original full_local full_large; do
  for seed in 126 2025 3407; do
    .venv/bin/python tools/evaluate.py --uap "outputs/$variant-$seed/uap.npy" --training-record "outputs/$variant-$seed/provenance.json" --data-root "$IMAGENET_VAL_ROOT" --manifest manifests/imagenet_val_v1/holdout.csv --targets vgg16 resnet50 densenet121 convnext_base swin_t vit_b_16 --stage final --device cuda:0 --output "outputs/$variant-$seed/evaluation" || break 2
  done
done
.venv/bin/python tools/aggregate.py --runs outputs --out reports/first_complete_comparison
```

## 彙整驗證與輸出

只納入 complete／full-budget／final／49000張 holdout，驗證 raw counts 與每個比率相符、必要 hashes 完整。
配對還要求同 seed、eval protocol、target weights、preprocessing、manifest、precision／environment、surrogate weights 與 research implementation。
A/B 除 source_mode／variant 外設定須相同，累積 prior+crop draw hash 須相同。
E3 除 crop area／variant 外設定須相同，累積 prior draw hash 須相同。所有 draw hash 均隨 checkpoint 保存，resume 不會重設。

多份同 variant/seed/target 記錄會標 ambiguous_duplicate；不挑 max FR。若有意重跑，請用獨立的 `--runs` 根目錄只放預先指定的 canonical evaluation records，保留其他記錄供稽核，不得依 target FR 決定哪份入表。
缺 seed 的正式平均與 std 保留空值；三個 seed 全部齊全且 identity 相同才計算 mean／sample std（ddof1）。
零 clean-correct 時 ASR 為 null；沒有合法分母的 seed 不會補零。

| 產物 | 內容 |
|---|---|
| `status.json` | complete/incomplete、預期十五訓練／九十target records、缺失與重複數 |
| `seed_metrics.csv` | 每 variant/seed/target 的 raw counts、比率、UAP／weight／manifest identity |
| `seed_summary.csv` | 三 seed 的 mean 與 sample std，單位 ratio |
| `e1_matrix.csv` | 重用 full_original 的六target比較 |
| `paired_delta_fr.csv` | E2 B−A；E3 local−original／large−original，單位percentage points |
| `paired_delta_summary.csv` | 配對 ΔFR 的三seed mean／sample std |
| `rejected_records.json` | smoke／failed／壞 count 或 hash 等未納入原因 |

彙整不能把不同預訓練 recipe 的模型比較解讀成純架構因果效果，不能據 crop 改善斷言語義／頻率／局部性根因。
原 supplement 已測 DeiT3；未確認其實驗 layers，這個工程不宣稱首次測 Transformer，也不作 ViT surrogate adaptation。

## 本次執行狀態

E0/E1/E2/E3 均未執行；沒有 real ImageNet manifests 或 CNN-trained UAP。
`reports/blocked_comparison/` 是對空的工程輸入目錄執行彙整後的 **缺失狀態表**：九十個 target records 全 missing、所有 FR／ΔFR／mean／std 均空白，不是研究結果。
CPU analytic fixtures 包含為計數測試合成的 records；它們只存在測試暫存目錄，不納入該狀態表。
