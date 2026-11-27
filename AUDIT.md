# AnoDDAE 官方程式碼 × 論文 稽核紀錄

對象：`github.com/sattarov/AnoDDAE`（commit `bc1c5cc`）與論文 *Diffusion-Scheduled Denoising Autoencoders for Anomaly Detection in Tabular Data*（KDD 2025, arXiv:2508.00758v1）。

每一項都附上證據（程式行號或可重跑的腳本）。嚴重度：
**高** = 會改變論文結論或數字；**中** = 影響可重現性或與論文描述不符；**低** = 易用性/程式品質。

## A. 會影響結論的問題

| # | 嚴重度 | 問題 | 證據 |
|---|---|---|---|
| A1 | 高 | **cosine 排程實作錯誤**：`cosine_noise_schedule()` 回傳的是 ᾱ（累積訊號比例），卻被當成 β 使用。β₀ = 1 ⇒ α₀ = 0 ⇒ 所有 t 的 ᾱ_t = 0，模型看到的輸入永遠是純高斯雜訊。最佳重建只能是訓練均值，異常分數退化為「到均值的距離」。論文 Fig. 7「cosine 在無監督最好、在半監督最差」的結論很可能是這個 bug 的產物。 | `src/model.py:164-178`；`scripts/` 中的排程比對（官方 vs 我的實作逐數值相同，cosine 的 ᾱ 全為 0）；`sched_cosine-official_T100` 與 centroid 基準的對照實驗 |
| A2 | 高 | **DDAE-C 沒有實作**。論文的第二個貢獻（對比學習版本，Eq. 7–8、Alg. 1）在 repo 中完全不存在，無法從官方程式碼重現 Table 1 的 DDAE-C 列。 | `src/model.py` 只有 `DDAE`；README 自稱實作 DDAE-C |
| A3 | 中 | **異常分數公式與論文不一致**：論文 Eq. 6 為 Σₜ‖x₀−x̂₀‖²（平方），程式為 Σₜ‖x₀−x̂₀‖₂（未平方），且 t 只取 1…T−1。兩者排序不同，AUC 會有差異。 | `src/model.py:324-333` vs 論文 Eq. 6 |
| A4 | 中 | **評估流程與論文宣稱不符**。論文說沿用 DTE（Livernoche et al.）的流程，但程式：(1) 用全部資料（含測試異常）擬合 StandardScaler；(2) 無監督設定沒有 bootstrap（train = test = X）；(3) 沒有 50,000 筆上限、沒有把 <1000 筆的小資料集重抽樣到 1000 筆；(4) 沒有把 CV/NLP 的多個子檔案（如 CIFAR10_0…9）做家族平均。DTE 的基線數字全部建立在後者流程上。 | `run.py:33-36`、`src/data.py`；DTE `data_generator.py`、`run.py`；`scripts/validate_knn.py` 證明 DTE 流程 |
| A5 | 中 | **超參數以測試集結果挑選**。附錄 A 的網格搜尋（T、層數、寬度、embedding 維度）都以 57 個資料集的測試 PR/ROC 選最佳值，而 DTE 的基線使用原論文預設值，比較不對等（選擇偏差）。 | 論文 Sec. 4、App. A/B、Fig. 5/8/10/11 |

## B. 可重現性與一致性問題

| # | 嚴重度 | 問題 | 證據 |
|---|---|---|---|
| B1 | 中 | `time_emb_type: learned` 會直接拋 `ValueError`（程式只接受 `learnable`）；改成 `learnable` 後，`predict()` 把 int64 的 t 丟進 `nn.Linear`，推論時崩潰。 | `src/config.yaml:25`、`src/model.py:13,49-51,330-331` |
| B2 | 中 | 論文寫 Glorot 初始化，程式使用 PyTorch 預設（Kaiming-uniform）。 | 論文 App. B vs `nn.Linear` 預設 |
| B3 | 中 | 論文的「PR-AUC」與 DTE 基線使用的 Average Precision（ADBench `utils.metric`）不是同一個量；程式同時算 AP 與梯形法 PR-AUC，訓練過程列印的是梯形法 PR-AUC。 | `src/utils.py:25-30` |
| B4 | 中 | DTE 已發布的**無監督**基線數字無法以 DTE 公開程式碼重現（kNN：我方 66.65/24.07 vs 公布 72.81/30.77 ROC/AP）；半監督則逐位吻合。AnoDDAE 的無監督比較因此建立在無法驗證的基線上。 | `results/knn_validation_*.csv` |
| B5 | 低 | `predict()` 每次呼叫都抽新的雜訊，分數具隨機性且未固定 seed；把 numpy 加到 torch tensor 產生大量 DeprecationWarning。 | `src/model.py:330-333` |
| B6 | 低 | `fit()` 每 `eval_epochs` 用測試標籤計算並列印 PR/ROC；若用來挑 epoch 即為測試集偷看。 | `src/model.py:301-305` |

## C. 易用性

| # | 問題 |
|---|---|
| C1 | README 的 `pip install -r requirements.txt`，但 repo 沒有 requirements.txt |
| C2 | README 的 `python run.py --config configs/default.yaml` 路徑錯誤（實際為 `src/config.yaml`）；notebook 路徑 `notebooks/method_demo.ipynb` 錯誤（實際為 `notebook/demo.ipynb`） |
| C3 | 沒有提供 `ad_bench/` 資料或下載方式 |
| C4 | `config.yaml` 的 `train.batch_size: 64` 完全沒被使用（batch size 由資料量決定）；`get_batch_size` 的 docstring 描述了不存在的參數 |
| C5 | notebook 以 `from data import ...` 匯入，需要手動把 `src/` 加入路徑；資料檔路徑寫死在當前目錄 |
| C6 | `split_data` 的 docstring 寫 `'semi_supervised'`，實際只接受 `'semi-supervised'`（`src/data.py:28,40`） |

## 修正

`official_patched/` 為修正後的官方程式碼，`official_fixes.patch` 為對原 repo 的 diff：
修正 A1（正確的 cosine β）、A3（`score_type: squared` 選項）、A2（`contrastive_alpha` 啟用 DDAE-C）、
A4-(1)（Scaler 只擬合訓練集）、B1、B5、C1–C4。預設行為除 A1/A4-(1) 外與原版相同。
