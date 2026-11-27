# AnoDDAE 完整重現、稽核與改良

> **在這個 repo 裡的位置**：本文件原本是獨立 repo 的 README，併入後路徑有以下差異。
> - 結果檔在 `raw_full/`（不是 `results/`）。分析前先設 `ANODDAE_RESULTS=raw_full`，例如
>   `ANODDAE_RESULTS=raw_full python scripts/summarize.py`。
> - 用 GPU 跑要設 `ANODDAE_DEVICE=cuda`（本 repo 的 `models.py` 預設 CPU）。
> - `scripts/queue*.sh` 沒有一起放進來；實驗指令見下方「新增實驗」一節與 `anoddae_repro/runner.py` 的實驗清單。


對 KDD 2025 論文 **Diffusion-Scheduled Denoising Autoencoders for Anomaly Detection in Tabular Data**
（Sattarov, Schreyer, Borth；[arXiv:2508.00758](https://arxiv.org/abs/2508.00758)）與官方程式碼
[`sattarov/AnoDDAE`](https://github.com/sattarov/AnoDDAE) 的完整復現、稽核與改進。

完整結果、圖表與討論見 **`report/AnoDDAE_report.html`**（單一自足 HTML 檔）；程式稽核清單見 **`AUDIT.md`**。

## 目錄

```
anoddae_repro/            復現用的 Python 套件
  data.py                 ADBench 載入；三種評估協定（dte / dte2 / repo），57 個資料集清單
  models.py               DDAE、DDAE-C（論文 Eq. 7-8）、DAE；五種噪聲排程；多種異常分數
  runner.py               多行程、可續跑的實驗執行器（結果寫成 JSONL，逐樣本分數存成 npz）
  analyze.py              彙總工具（家族平均 → 57 資料集平均 → seed 平均 ± 標準差）
scripts/
  queue*.sh               實際執行過的實驗佇列（照順序即可完整重跑）
  validate_knn.py         以 kNN 驗證資料流程與 DTE 公布數字逐值一致
  check_vs_official.py    重新實作 vs 官方程式碼（同資料、同 seed）的分佈比較
  knn_scores.py           kNN 參考分數（供集成）
  centroid_baseline.py    零訓練的「到均值距離」偵測器
  knn_protocol_effect.py  資料協定本身對 kNN 的影響
  score_correlation.py    DDAE 分數與簡單偵測器的逐樣本 Spearman 相關
  summarize.py            產生 results/summary/*（Table 1 對照、排名、檢定、掃描）
  build_report.py         產生 report/AnoDDAE_report.html
official_patched/         修正後的官方程式碼（可直接 python run.py）
official_fixes.patch      對官方 repo 的 diff
reference/                從 DTE 論文附錄解析出的 26 個基線 × 57 資料集逐值表
results/                  原始結果（*.jsonl）、逐樣本分數（scores/）、彙總（summary/）
report/                   報告頁（template.html、body.html、charts.js、render.js、style.css）
AUDIT.md                  官方程式碼 × 論文 的差異與 bug 清單
```

## 安裝

```bash
pip install torch numpy pandas scikit-learn scipy pyyaml matplotlib joblib
```

資料（ADBench，約 580 MB）：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/Minqi824/ADBench data_src/adbench_repo
git -C data_src/adbench_repo sparse-checkout set adbench/datasets/Classical adbench/datasets/CV_by_ResNet18 adbench/datasets/NLP_by_BERT
export ADBENCH_ROOT=$PWD/data_src/adbench_repo/adbench/datasets
```

## 快速開始

單一資料集（半監督，論文設定 T=50 linear）：

```python
from anoddae_repro.data import make_split, batch_size_rule
from anoddae_repro.models import DDAE
from anoddae_repro.runner import _metrics

s = make_split("6_cardio", "semi", seed=0, protocol="dte")
m = DDAE(s.x_train.shape[1], T=50, batch_size=batch_size_rule(s.n_total)).fit(s.x_train)
scores = m.score(s.x_test, x_calib=s.x_train[:4096])
print({k: round(_metrics(s.y_test, v)["ap"], 2) for k, v in scores.items()})
```

`scores` 同時包含 `official`（官方程式碼的 Σ‖e‖）、`paper`（論文 Eq. 6 的 Σ‖e‖²）、
`stepnorm`、`featnorm`（改進版評分，見報告第 8 節）。

## 完整重跑

```bash
python -m anoddae_repro.runner --exp ddae_semi_T50 --setting semi --seeds 0 1 2 3 4 --protocol dte
python -m anoddae_repro.runner --exp ddae_unsup_T1000 --setting unsup --seeds 0 1 2 --protocol dte2
python scripts/summarize.py && python scripts/build_report.py
```

所有實驗名稱與超參數集中在 `anoddae_repro/runner.py` 的 `EXPS`。實際執行順序見 `scripts/queue.sh`、`queue2b.sh`、`queue3b.sh`。
在 4 核 CPU 上，全部實驗約需 10–12 小時；有 GPU 時把 `DDAE(..., )` 的張量移到 GPU 即可大幅加速。

## 評估協定

| 名稱 | 用途 | 說明 |
|---|---|---|
| `dte` | 半監督主實驗 | DTE（Livernoche et al., ICLR 2024）公開程式碼的流程：<1000 筆重抽樣到 1000、>50k 筆抽樣到 50k、前 50% 正常樣本為訓練集、Scaler 只擬合訓練集、CV/NLP 以家族平均。kNN 在 57 個資料集上與 DTE 公布值逐位吻合。 |
| `dte2` | 無監督主實驗 | 同上，但無監督不做小資料集重抽樣（與 DTE 公布的無監督數字較一致；DTE 公開程式碼無法重現其公布的無監督基線，見 AUDIT B4）。訓練集為 bootstrap，評估在全部資料。 |
| `repo` | 協定影響對照 | AnoDDAE 官方程式碼：Scaler 擬合全部資料、隨機 50% 正常樣本訓練、無監督 train = test。 |

指標：AP（= ADBench / DTE 的「AUC-PR」）與 ROC-AUC，皆 ×100。另記錄梯形法 PR-AUC。

## 結果摘要

完整數字與圖表見 `report/AnoDDAE_report.html`。AP 即 ADBench／DTE 的 AUC-PR，皆 ×100。± 為「每個資料集跨 seed 的標準差再取平均」，與論文 Table 1 的算法相同（已用 DTE 附錄驗證）。

| | 無監督 AP | 無監督 ROC | 半監督 AP | 半監督 ROC |
|---|---|---|---|---|
| DDAE（論文 Table 1） | 32.77 ± 0.69 | 74.08 ± 0.51 | 61.36 ± 2.23 | 83.17 ± 1.00 |
| DDAE（本復現；無監督 3 seeds、半監督 5 seeds） | **33.06 ± 0.94** | **73.95 ± 0.60** | **59.39 ± 1.69** | **82.36 ± 0.65** |
| 到均值距離 ‖x−μ‖²（零訓練） | 32.70 | 71.95 | 47.26 | 74.36 |
| kNN k=5（本次重跑） | 27.67 | 70.60 | 56.34 | 80.88 |

主要發現：

1. **重現程度**：無監督幾乎完全一致；半監督低約 2 AP，在論文的 ± 範圍內。在 26 個 DTE 基線中，半監督平均 AP 與平均排名都排第 1；無監督 AP 第 3、ROC 第 2，與論文描述相符。
2. **無監督 DDAE ≈ 到均值距離**：逐樣本 Spearman ρ 平均 0.95。T=1000 時約 1/3 的步驟輸入已是純雜訊，該步最佳重建就是均值。
3. **官方 cosine 排程 bug**：ᾱₜ ≡ 0，結果等同到均值距離。論文 Fig. 7 的 cosine 結論受此影響。
4. **評估協定的人為效應**：DTE 流程把小資料集有放回抽樣後才切分，91–100% 的測試正常樣本在訓練集有完全相同的副本；切分也沒有打亂順序（letter 上前半與後半正常樣本可被 AUC 0.997 區分）。DDAE 相對 DTE-NP 的領先約 72% 來自 13 個被重抽樣的資料集；在其餘 44 個資料集上與 kNN、DTE-NP 差距不顯著。
5. **改進**：污染穩健訓練（每 10 epoch 剔除分數最高的 10% 訓練列）使無監督 T=50 的 AP +8.1（輕量協定），追平 T=1000 與到均值距離但未超越；在乾淨的半監督資料上會降 7.3，只應在訓練資料可能含異常時開啟；修正 cosine、逐特徵／逐步驟正規化評分的效果則較小或只在特定設定有效。詳見報告第 08 節。
6. **DDAE-C**：官方 repo 沒有實作。依論文 Eq. 7–8 重做後，無監督與論文一致（32.66 vs 31.97）；半監督隨對比權重 α 單調變差（α=0.01 / 0.1 / 0.5 相對 DDAE 為 −3.2 / −8.0 / −11.9 AP，輕量協定），論文的數字需要未公開的實作細節才能達到。
7. **DAE**：σ=0.5 的 DAE 在無監督得到 19.90 / 62.97，與論文的 DAE（19.85 / 63.73）幾乎相同，可重現「DDAE 比 DAE 高 65%」；但 σ=0.1 的 DAE 已有 25.30，零訓練到均值距離有 32.70。


## 新增實驗（剔除比例、消融、完整規模剔除）

以下實驗的結果都在 `raw_full/`，統計在 `raw_full/summary/`。

| 實驗 | 指令 | 協定 |
|---|---|---|
| 剔除比例掃描（q = 0 到 30%、oracle、頻率） | `python -m anoddae_repro.runner --exp trimq0.0_T50 trimq0.02_T50 trimq0.05_T50 trimq0.1_T50 trimq0.15_T50 trimq0.2_T50 trimq0.3_T50 trimq_oracle_T50 trimq0.1_every5_T50 trimq0.1_every20_T50 --setting unsup semi --seeds 0 1 2 --protocol dte2 --lite --tag _lite` | 精簡版 |
| 消融 T=50 | `--exp abl_temb0_T50 abl_temb32_T50 abl_cosfix_T50 abl_ddaec0.1_T50 abl_ddaec0.5_T50 --setting unsup semi --seeds 0 1 2 --protocol dte2 --lite --tag _lite` | 精簡版 |
| 消融 T=1000 | `--exp abl1k_base abl1k_temb0 abl1k_temb32 abl1k_cosfix abl1k_trim0.1 abl1k_ddaec0.1 abl1k_ddaec0.5 --setting unsup --seeds 0 1 2 --protocol dte2 --lite --tag _lite` | 精簡版 |
| 完整規模剔除 | `--exp impr_trim0.1_T1000 impr_trim0.1_T50 --setting unsup --seeds 0 1 2 --protocol dte2` | 完整 |

精簡版（`--lite`）：CV 與 NLP 每類只取第一個檔案，每個資料集最多 10000 筆，數字不能直接和論文比。

統計：`python scripts/trim_sweep_analysis.py`、`python scripts/ablation_analysis.py`（都要先設 `ANODDAE_RESULTS=raw_full`）。

### 完整規模無監督結果（57 個資料集，3 次重複）

| 方法 | AP | ROC |
|---|---|---|
| 論文 DDAE | 32.77 ± 0.69 | 74.08 ± 0.51 |
| 重現 DDAE T=1000 | 33.06 ± 0.94 | 73.95 ± 0.60 |
| T=1000 + 剔除 10% | 33.49 ± 0.70 | 74.34 ± 0.56 |
| T=1000 + 剔除 10% + 逐步正規化分數 | **33.85 ± 0.84** | **74.89 ± 0.68** |
| T=50 + 剔除 10% | 33.24 ± 1.96 | 73.70 ± 1.66 |
