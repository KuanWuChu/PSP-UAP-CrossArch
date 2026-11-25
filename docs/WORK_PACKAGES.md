# PR 工作包、分工及驗收

未建立 GitHub issues；以下 ID 是文件內任務編號，不冒充 issue number。

| ID/PR | 工作內容 | 相依 | 驗收／交付 |
|---|---|---|---|
| PR00 docs/bootstrap | 公開 fork、來源 pin、維護入口、計畫與限制 | gh 登入 | fork=true、public、parent 正確；原始 code unchanged；計畫 commit 可讀回 |
| PR01 env/provenance | doctor、legacy/modern env notes、weight registry、run IDs、manifest tool | PR00 | 檢查 GPU 相容性；解析 config；資料 mapping、disjointness fixtures；不啟動 full training |
| PR02 eval-crossarch | 通用 target-only evaluator、FR/accuracy/CC-ASR、CSV、zero-delta test | PR01 | CNN parity；至少 CNN+ViT 短測；raw counts/shape/hash；one UAP unchanged |
| PR03 controlled-trainer | research loop、immutable checkpoint、fixed budget、RNG/curriculum resume | PR01 | original loss unchanged；數值/gradient fixture；100–200 iter GPU smoke；checkpoint/resume |
| PR04 matched-sampling | A noise_crops/B psp_crops/C full、T/RW explicit、crop RNG replay | PR03 | A/B 唯一預定差異是 UAP-derived source；source.detach test；與 E1 接通 |
| PR05 crop-policy | local/mixed/large area range configs | PR04 | ranges logged、ratio 同設定、same objective、no target selection |
| PR06 experiment/report | 15-run matrix、repeat seeds、summary CSV、transfer/gain/crop figures | PR02/05 | 實測結果與 failed runs；與 paper 表分開；重現命令和限制 |

## 第一波遠端任務（先做到可開始實驗）

先 PR00–02，再 PR03 的小型 smoke。不要開始就同時跑三種 surrogate×所有 seeds×所有 variants。先證明 data path、weights、epsilon、FR 正確。

1. Fork/connect origin → build from fixed upstream.
2. 跑 CPU fixtures，驗證 original code hash。
3. 有 GPU 和合法 ImageNet 時，在 1000 holdout smoke images 上測 zero delta/random delta（不是方法成果）。
4. ResNet152 跑 100–200 iterations 做 finite smoke，記時間、峰值顯存、checkpoint 和 raw metrics；不稱 E0 completed。
5. Pipeline 正確後才跑一次 official full E0 seed126。
6. E0 合理後 E1；E2/E3 實作可繼續，但在可疑 baseline 上不要批次消耗 GPU。

## 人員分工（使用者自行指定人員）

- Maintainer：repo integration、branch/PR、來源與實驗狀態。
- Developer A：環境／資料 manifest／target evaluator／模型權重。
- Developer B：受控 trainer／sampling 消融／checkpoint。
- Analyst：結果匯整、模型家族分組、簡報圖表與限制。

這些是角色，不會自動邀請任何人，也不代表需要多 agent。若一個遠端 agent 實作，按 PR dependency 順序即可。

## PR 模板必備欄

Problem and resulting behavior; exact changed interfaces; activation objective hash; source version; data/weights/config identity; tests actually run; commands to reproduce; experiments not run; risks.

full E0/E1/E2/E3 completion must name run IDs and UAP hashes. A code merge is not an experiment completion. No automatic merge into author repo or force push.

## 最小完整實驗預算

| Variant | seeds | runs |
|---|---:|---:|
| E2 A noise_crops, T/RW off | 3 | 3 |
| E2 B PSP-only, T/RW off | 3 | 3 |
| full PSP / original crop | 3 | 3 |
| full PSP / local crop | 3 | 3 |
| full PSP / large crop | 3 | 3 |
| 合計（同一 research protocol） | | 15 |

E1 重用 full/original；E0 official-faithful 另計最多 3 runs。先 seed126 跑 pilot，從小批量 profile 估算成本。不預設 GPU hours 或取得資料時間。
