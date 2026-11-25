# 公開 Fork、Clone、推送與協作

## 現在狀態

本地完整 clone 已完成；來源 commit `dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c`。此套件附完整 history bundle，離線也能恢復；不是把原始碼重新 git init 的假 fork。

GitHub 公開 fork 尚未建立。終端 `gh auth status` 顯示未登入；GitHub App 已連線但工具集沒有 create-fork/repo。此限制不表示要使用者提供 token 到聊天。

## 在已登入 GitHub 的遠端電腦上

先安裝 git、gh，執行 `gh auth status`，若未登入由使用者正常 `gh auth login`（不要把任何密碼/token 寫進 repo 或訊息）。確認 `gh api user --jq .login` 是 `KuanWuChu`。

### 路徑 A：使用交付 bundle（包含完整歷史與計畫 commit）

將 ZIP 解壓後，在合適的開發根目錄：

```bash
git clone PSP_UAP_CrossArch_Planning.bundle PSP-UAP-CrossArch
cd PSP-UAP-CrossArch
bash scripts/publish_fork.sh
```

bootstrap 會：檢查登入帳號 → 查是否已有指定 fork → 不存在才 native fork → 等可讀 → 驗證 parent/public → 設 origin/upstream → 推送目前規劃 branch 與來源 tag。若同名 repo 已存在但不是正確 fork，會停止，不覆蓋。

### 路徑 B：已有官方 clone

```bash
git remote -v
gh repo fork ChanhuiLee1111/PSP-UAP --fork-name PSP-UAP-CrossArch --clone=false --remote=false
```

這裡是 native public fork，因為 parent 本身 public。不需要先建立一個同名空 repo，也不重新初始化 Git。
若你已有上面的 bundle clone，直接用 script 即可，不要重複做 B。

## 初次推送後

1. 到自己的 fork 比對 branch `plan/crossarch-handoff-20261125` 與 `main`。
2. 開 docs-only PR 到**自己 fork 的 main**；確認只新增 docs/scripts/config planned、.gitignore extension，沒有修改原始六個 Python。
3. 合併文件 PR 後，隊友 clone 自己的 fork、看 RESEARCH_README.md。後續 feature branches 都從 fork main 開。
4. 原作者 repo 僅作 upstream，不建立 unsolicited PR、不推作者 branch。

## 使用者加入同學

在自己的 fork：Settings → Collaborators（或 Manage access）→ Add people → 選正確 GitHub username → 一般使用 write/collaborator 權限即可；由同學接受邀請。不要因為開發就給管理員權限。

此文件不會替使用者邀請人；使用者已說之後自行加入。公開 repo 讓任何人可讀，但只有擁有 write 的協作者能直接 push。一般流程仍用 branch + PR。

## Branch 規則

- `main`：可整合的研究主線。
- `plan/crossarch-handoff-20261125`：本次規劃。
- `feat/provenance-and-manifests`、`feat/crossarch-evaluator`、`feat/controlled-training`、`feat/sampling-ablation`：各工作 PR。
- tag `research/upstream-20261125`：固定 original commit。

## 公開內容

可以公開：程式、設定模板、說明、MIT 授權、乾淨小型 CSV、彙整圖、UAP metadata/hash。ImageNet 不隨 repo 發布；權重用官方下載；大型 UAP/checkpoint 以有 provenance 的 Release 或研究儲存另管。報告中的示意圖不可當作實際 .npy perturbation。
