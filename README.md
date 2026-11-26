# AnoDDAE 重現與訓練樣本剔除實驗

資料探勘期末報告的程式與實驗結果。論文：T. Sattarov, M. Schreyer, D. Borth, "Diffusion-Scheduled Denoising Autoencoders for Anomaly Detection in Tabular Data", KDD 2025（[arXiv:2508.00758](https://arxiv.org/abs/2508.00758)，[作者程式](https://github.com/sattarov/AnoDDAE)）。

本 repo 重新實作無監督 DDAE（linear 排程），並測試一個改法：訓練時每 10 個 epoch 依異常分數剔除最高的 10% 訓練列（下文稱「依分數剔除」或「改良版」）。比較對象是未剔除的 DDAE、隨機剔除同樣比例，以及不訓練模型的平方距離基準（每筆資料到訓練集平均向量的平方距離）。所有方法使用相同的 dte2 資料流程與 official 評分。

## 主要結果

57 組 ADBench 資料的等權平均，AP（×100，越高越好）：

| T | 未剔除 | 依分數剔除 | 平方距離 | 來源 |
|---|---|---|---|---|
| 50 | 24.85 | 32.77 | 32.70 | raw/（Linux CPU） |
| 1000 | 32.96 | 33.46 | 32.70 | raw_gpu/（GPU） |

論文 Table 1 的無監督 DDAE 為 32.77（PR-AUC）、74.08（ROC-AUC），從論文 Fig. 5 判讀對應 T=1000；本 repo 的 T=1000 未剔除為 32.96 / 73.91。論文沒有說明 PR-AUC 的算法，所以只能說接近。T=50 的改良版 AP 恰好也是 32.77，但設定不同，不能說成追平論文。

逐資料集的 AP、ROC-AUC 在 tables/datasets.csv，57 組平均在 tables/mean57.csv。47 個表格資料集各對應一個檔案。其餘 10 個影像、文字家族共 74 個檔案，amazon、imdb、yelp 各只有一個檔。原始資料共 121 個檔案，每版各跑 seed 0 至 5。先在各組內平均成員及 seed，再平均 57 組。AP 使用 average_precision_score。

讀結果時注意三點：

- 改良版明顯勝過未剔除版，但跟平方距離基準的差異不顯著（見下方統計檢定）。目前的證據只能說剔除讓 T=50 的平均 AP 接近簡單基準。逐組看仍有輸有贏：6_cardio 上平方距離領先改良版 25.84 點，40_vowels 上改良版領先 40.28 點。
- dte2 無監督流程先在整批資料上做標準化，再從同一批資料有放回抽樣當訓練集，最後對整批評分。以 6_cardio、seed 0 為例，1,831 筆評估資料中有 1,165 筆進過訓練。這種設定衡量的是「在手上這批資料裡找異常」，不能用來估計模型對新資料的表現。三個方法都在同一個設定下比較。
- 10% 是每次剔除的比例，不是資料的異常率。各資料集的異常率從 0.026% 到 39.91% 不等。
- 剔除的效果取決於 T。在 GPU 補跑的 T=1000 上，依分數剔除對未剔除只多 0.51 個 AP 點（Holm 校正後 p=0.09）；T=50 加剔除跟 T=1000 不剔除差 0.16 點，不顯著。在這個設定下，剔除和較大的 T 帶來的改善大致重疊，見「T=1000 對照」。

## 實驗環境與裝置

實驗分別在三種環境執行。同一個工作在不同平台或裝置上的結果不會逐位相同（57 組平均差在 0.5 個 AP 點內），所以每個比較只在同一個資料夾的結果之間做，不跨資料夾混用。

| 資料夾 | 內容 | 環境 |
|---|---|---|
| raw/ | T=50 主實驗（未剔除、依分數剔除），含逐樣本分數 NPZ | Linux，CPU |
| raw_win/ | 隨機剔除對照（三個方法一起重跑） | Windows，CPU |
| raw_trimlog/ | 剔除紀錄：每輪被剔除的訓練列 | Windows，CPU |
| raw_more/ | 三個佐證實驗（依標籤剔除、污染比例、剔除比例） | Windows，CPU |
| raw_gpu/ | T=1000 對照與 T=50 重跑 | Windows，GPU（RTX 3080） |
| verification/ | 6_cardio、seed 0 的試跑紀錄 | Linux，CPU |

各資料夾的 README 記錄確切的套件版本、日期與指令。主實驗用 T=50，是因為 CPU 上跑完 57 組 × 6 seed 的 T=1000 太久；T=1000 改在 GPU 上補跑。

其他檔案：anoddae_repro/ 是模型與執行程式；tables/ 是所有彙總表與檢定結果；data_src/ 附一份 6_cardio 資料供試跑；根目錄的 *.py 是各實驗的彙總與檢定程式，用法見下文各節。

## 建立環境

在 Linux 或 WSL 的repo 根目錄執行。本次使用 Omarchy 主機，環境為 Linux 7.2.15-1-omarchy x86_64、Python 3.12.16、CPU PyTorch。environment.txt 記錄實驗環境的完整套件版本。以下使用已安裝的 uv。

~~~bash
uv venv --python 3.12.16 .venv
source .venv/bin/activate
uv pip install torch==2.15.0+cpu --index https://download.pytorch.org/whl/cpu
uv pip install -r environment.txt
~~~

## 從原始結果重算

~~~bash
python recompute.py
~~~

程式預設讀取 raw/ 的歷史結果，檢查兩份 raw/*_full.jsonl 是否各有 726 筆成功且不重複的資料檔與 seed 配對，再從 raw/scores 的 1,452 個 NPZ 讀取 y 和 official 分數，重算兩種指標並寫入 tables。centroid 的逐樣本分數沒有保存，程式從未剔除版 JSONL 的 metrics.centroid 重新彙總。這一步不必下載 ADBench 或訓練模型。

接著執行以下指令做成對統計檢定：

~~~bash
python stats_tests.py
~~~

程式預設讀取 tables/datasets.csv（重跑後可用 --tables results/tables），以 57 組為成對樣本做雙尾 Wilcoxon signed-rank 檢定，六個比較一起做 Holm 校正，並算 rank-biserial 效果量，結果寫入 tables/stats.csv。改良版對未剔除版的 AP 與 ROC-AUC 校正後 p 值為 0.00072 與 0.00042，效果量 0.58 與 0.61。改良版對平方距離基準為 1 與 0.57，效果量 0.03 與 0.20，差異不顯著。

30_satellite 與 31_satimage-2 幾乎共用相同的資料列（satimage-2 的 5,803 列中有 5,801 列也在 satellite 裡），兩組不算獨立。stats.csv 中 units=56 的列是把這兩組先平均成一組的敏感度分析，結論不變。這個檢定只說明 57 組的勝負與排名是否偏向一邊，不代表兩個方法相同，也不涵蓋 seed 之間的變異。

## 實跑一個資料集

repo 已附該版本的 6_cardio.npz 及 runner 需要的資料大小清單。在repo 根目錄執行：

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m anoddae_repro.runner --exp sweepT_50 impr_trim0.1_T50 --setting unsup --protocol dte2 --datasets 6_cardio --seeds 0 --workers 1 --tag _smoke
~~~

首次會顯示 2 jobs。檢查 results/*_smoke.jsonl 的 metrics.official。預設用 CPU；設定 `ANODDAE_DEVICE=cuda` 改用 GPU，結果會跟 CPU 有浮點差異（見 raw_gpu/README.md）。2026-11-26 加入 GPU 選項後，CPU 結果跟 raw_win/ 的紀錄逐位相同（6_cardio、seed 0）。在原 Linux 主機上，seed 0 的未剔除版 AP / ROC-AUC 為 18.7494 / 58.7184，改良版為 36.3927 / 63.6841，紀錄在 verification。原生 Windows 也能執行（runner 改用 spawn），但數值會有平台差異：2026-11-24 在 Windows 上測得 18.76 / 58.53 與 39.03 / 67.20，改良方向相同。

逐樣本分數存在 results/scores/<實驗名>/<protocol>_cap<資料上限>/tag<tag>/，換 tag、protocol 或 --max_size 都會寫到不同的資料夾。分數檔先寫成暫存檔再換名，中斷時不會留下寫一半的檔案。runner 只略過「同一個 tag、protocol、資料上限下已成功，而且分數檔能開啟、列數正確、分數都是有限值」的工作；紀錄檔最後一行若因中斷只寫了一半，runner 會先備份成 .jsonl.bak 再刪掉那一行；會讓檔名跟其他實驗撞在一起的 tag（例如 sweepT_5 配 0_win，會變成 sweepT_50_win.jsonl）直接拒絕；重跑過的工作以最後一筆成功紀錄為準；沒有記錄資料上限的舊紀錄一律重跑。有任何工作失敗時，runner 結束碼為 1，最後一行會列出成功與失敗數。results 是重跑輸出，raw 是完整實驗的原始結果（raw/scores/ 沿用舊的資料夾結構，recompute.py 只在預設參數下讀它）。

## 完整訓練的資料版本

ADBench commit 見 adbench-commit.txt。重跑全部資料時，將該版本下載到另一個目錄，並設定 ADBENCH_ROOT。以下在repo 根目錄執行：

~~~bash
git clone --filter=blob:none --no-checkout https://github.com/Minqi824/ADBench adbench-full
git -C adbench-full sparse-checkout set adbench/datasets/Classical adbench/datasets/CV_by_ResNet18 adbench/datasets/NLP_by_BERT
git -C adbench-full checkout "$(cat adbench-commit.txt)"
export ADBENCH_ROOT="$PWD/adbench-full/adbench/datasets"
python -m anoddae_repro.runner --exp sweepT_50 impr_trim0.1_T50 --setting unsup --protocol dte2 --seeds 0 1 2 3 4 5 --workers 4 --tag _rerun
python recompute.py --raw results --tag _rerun --out results/tables
python stats_tests.py --tables results/tables
~~~

最後兩行用重跑的結果重建表格並做檢定，寫到 results/tables，不會蓋掉歷史的 tables/。recompute.py 會逐檔用 NPZ 重算 AP、ROC-AUC，跟 JSONL 紀錄不符、缺檔或兩版標籤不同時直接報錯。

raw/ 的歷史結果由 2026-11-20 的 anoddae_repro 產生。2026-11-24 之後的版本加了隨機剔除選項、新的分數資料夾結構、續跑檢查和 Windows 支援；在同一台電腦上，既有兩個實驗的訓練與分數跟舊版逐位元組相同（cardio、seed 0 實測）。run.sh 保留當時首批 seed 0 至 2 的歷史指令；完整結果另含 seed 3 至 5，run.sh 會重寫 environment.txt，請勿當作安裝入口。

## 隨機剔除對照

剔除同時減少了資料量；多數資料組的每輪更新次數也跟著變少（批次大小固定，列數少一成）。為了分開「挑對樣本」和「資料變少」兩種效果，另跑了一個每十輪「隨機」剔除同樣 10% 的版本（impr_randtrim0.1_T50），列數和更新次數都跟依分數剔除相同。三個方法在同一台 Windows 電腦上一起重跑，紀錄與環境見 raw_win/README.md。

~~~bash
python random_trim_control.py
~~~

57 組平均（×100）：

| 方法 | AP | ROC-AUC |
|---|---|---|
| 未剔除版 | 24.89 | 70.17 |
| 隨機剔除 10% | 25.22 | 70.74 |
| 依分數剔除 10% | 32.65 | 73.23 |
| 平方距離 | 32.70 | 71.95 |

依分數剔除比隨機剔除高 7.44 個 AP 點（36 勝 21 敗，Holm 校正後 p=0.005，效果量 0.52），ROC-AUC 高 2.49 點（p=0.042）。隨機剔除對未剔除版只高 0.33 個 AP 點，不顯著（p=0.25）。在這批資料與 dte2 協定下，依分數剔除明顯優於同樣列數的隨機剔除；隨機剔除對未剔除版則未檢出顯著差異，逐組差值從 −8.78 到 +8.98 點都有。這支持「挑哪些列」有影響；被剔除的列是否偏向真異常，見下一節的剔除紀錄。依分數剔除對平方距離仍不顯著（AP p=1、ROC-AUC p=0.56）。十項比較一起做 Holm 校正；把 satellite 和 satimage-2 合併成 56 組時結論相同（tables/random_trim_stats.csv）。

兩種剔除消耗的亂數不同（依分數剔除要對訓練列加噪評分），所以第 11 輪之後的訓練亂數路徑也不同。

## 剔除紀錄：被剔除的列是不是異常

重跑依分數剔除與隨機剔除，這次把每輪（第 10、20、…、90 輪）丟掉的訓練列索引和訓練列的標籤存下來（runner 寫到 results/trimlog/，本 repo收在 raw_trimlog/）。記錄不改變訓練：1,452 次的 AP、ROC-AUC 跟 raw_win/ 逐筆相同。執行環境見 raw_trimlog/README.md。

~~~bash
python trimlog_analysis.py
~~~

輸出 tables/trimlog_datasets.csv（逐組）與 tables/trimlog_summary.txt。57 組平均，被剔除列中的異常比例（%）：

| 列 | 異常比例 |
|---|---|
| 整批訓練列 | 10.97 |
| 隨機剔除的列 | 10.92 |
| 依分數剔除的列 | 25.57 |

依分數剔除的列異常比例高於整批的有 50/57 組，倍數中位數 2.54；對隨機剔除的 Wilcoxon p=3.6×10⁻⁸。最後一輪依分數剔除涵蓋訓練列中 37.4% 的異常（每輪只剔 10%，可涵蓋的上限平均 83.2%），隨機剔除為 9.7%。逐組看，依分數剔除的富集程度越高，它對隨機剔除的 AP 增益越大（Spearman ρ=0.49，p=9.4×10⁻⁵）。7 組沒有富集：9_census、26_optdigits、33_skin、35_SpamBase、39_vertebral、46_WPBC、imdb，它們的 AP 增益在 −2.32 到 +0.17 之間。

讀法：

- 這表示依分數剔除確實偏向丟掉異常，但富集本身幾乎是預期內的：剔除用的就是模型的異常分數，模型只要有一點偵測能力，分數最高的 10% 就會比整體更常是異常。比較有資訊量的是「富集越多、增益越大」這個逐組相關。
- 這是相關，不是因果。兩種剔除的訓練路徑不同，不能說改良版的進步「是因為」丟掉了異常。要回答因果，可以做「依標籤剔除真異常」的對照，或固定測試集、控制訓練集污染比例；兩者都已完成，見下一節。
- dte2 是傳導式設定：訓練列是從評分資料有放回抽樣而來，結論只限這個設定。

## 三個佐證實驗

2026-11-25 在實驗室的 Windows 電腦上加跑三組實驗（同一台電腦重跑剔除紀錄時，1,452 次結果跟 raw_win/ 逐筆相同），原始紀錄在 raw_more/，用以下指令重算：

~~~bash
python extra_analysis.py
~~~

輸出 tables/extra_oracle_datasets.csv、tables/extra_contam<c>_datasets.csv、tables/extra_trimshare_datasets.csv（逐組）、tables/extra_stats.csv（檢定，每個區塊各自做 Holm 校正）與 tables/extra_summary.json（57 組平均）。

### A. 依標籤剔除（上限）

impr_oracletrim0.1_T50：每 10 輪用真實標籤先剔除訓練列中的異常，不足 10% 時隨機補正常列，列數跟依分數剔除相同。現實中拿不到標籤，只當上限。dte2、seed 0 至 5，其餘三個方法取自 raw_win/。

| 方法 | AP | ROC-AUC |
|---|---|---|
| 未剔除版 | 24.89 | 70.17 |
| 隨機剔除 10% | 25.22 | 70.74 |
| 依分數剔除 10% | 32.65 | 73.23 |
| 依標籤剔除 10% | 49.67 | 82.87 |
| 平方距離 | 32.70 | 71.95 |

依標籤剔除對未剔除版 AP 高 24.78（57 勝 0 敗），對依分數剔除高 17.01（54 勝 3 敗），對平方距離高 16.97（48 勝 9 敗），Holm 校正後 p 都小於 10⁻⁷。依分數剔除只補回「未剔除 → 依標籤剔除」這段 AP 差距的 31%。

### B. 固定測試集、控制訓練集污染比例

protocol clean<c>（anoddae_repro/data.py）：每個資料集先分層抽 30% 正常列與 30% 異常列當測試集，訓練集是其餘 70% 正常列加上異常列，使訓練集異常比例為 c；異常列不夠時能加多少就加多少。標準化只用訓練列。為了控制時間使用 --lite（每個影像、文字家族只取第一個檔，最多 10,000 列），seed 0 至 2。

| 目標 c | 實際平均 | 達標組數 | 未剔除 | 依分數剔除 | 依標籤剔除 | 平方距離 |
|---|---|---|---|---|---|---|
| 0 | 0.0% | 57 | 53.64 | 46.98 | 53.07 | 40.89 |
| 0.05 | 4.0% | 36 | 37.01 | 40.29 | 52.11 | 38.07 |
| 0.1 | 6.1% | 18 | 34.72 | 38.42 | 51.76 | 36.97 |
| 0.2 | 8.7% | 13 | 33.20 | 36.89 | 48.63 | 36.02 |

（AP，57 組平均）

- 訓練資料乾淨時，未剔除版比平方距離高 12.75（p_holm=7.9×10⁻⁵）；有污染的三級跟平方距離都沒有顯著差異。
- 依分數剔除對未剔除版：乾淨時低 6.67（p_holm=4.2×10⁻⁴），有污染時高 3.28 至 3.70，但都不顯著（p_holm 0.44 至 0.90）。
- 依標籤剔除：有污染的三級都比未剔除版高 15.10 至 17.04（p_holm<10⁻⁸）；乾淨時沒有異常可剔，等於隨機剔除正常列，跟未剔除版的差異不顯著（p_holm=0.25）。
- 限制：只有 3 個 seed、每家族一個檔，多數資料集達不到目標污染比例。

### C. 剔除比例 5%、10%、20%

impr_trim0.05_T50 與 impr_trim0.2_T50，dte2、seed 0 至 2；0% 與 10% 取自 raw_win/ 的同樣 seed。

| 剔除比例 | AP | ROC-AUC |
|---|---|---|
| 0%（未剔除） | 24.97 | 70.54 |
| 5% | 30.83 | 72.22 |
| 10% | 32.62 | 73.38 |
| 20% | 33.47 | 74.66 |
| 平方距離 | 32.70 | 71.95 |

三種比例的 AP 都顯著高於未剔除（+5.86、+7.65、+8.50，p_holm 都小於 0.01）。20% 對 10% 只高 0.85，不顯著（p_holm=0.24）；ROC-AUC 則高 1.27，顯著（p_holm=0.0077）。20% 對平方距離的 AP 差 +0.77，不顯著（p_holm=0.38）。

## T=1000 對照（GPU）

論文的無監督主結果用 T=1000，本 repo主實驗用 T=50。為了回答「改良在論文的設定下還有沒有用」，2026-11-25 至 26 在實驗室電腦的 GPU 上補跑 T=1000 的四個方法，並在同一張 GPU 上重跑 T=50 的兩個方法，六個方法只彼此比較。環境、溫度保護與中途重啟的紀錄見 raw_gpu/README.md。

~~~bash
python t1000_analysis.py
~~~

輸出 tables/t1000_datasets.csv（逐組）、tables/t1000_mean57.csv、tables/t1000_stats.csv（檢定，兩個區塊各自做 Holm 校正）與 tables/t1000_summary.json。

57 組平均（×100），ρ 是官方分數與平方距離分數的 Spearman 相關（先逐次計算再平均）：

| 方法 | AP | ROC-AUC | ρ |
|---|---|---|---|
| T=50 未剔除 | 24.67 | 69.93 | 0.44 |
| T=50 依分數剔除 10% | 33.12 | 73.39 | 0.46 |
| T=1000 未剔除 | 32.96 | 73.91 | 0.95 |
| T=1000 隨機剔除 10% | 32.80 | 73.91 | 0.95 |
| T=1000 依分數剔除 10% | 33.46 | 74.31 | 0.94 |
| T=1000 依標籤剔除 10%（上限） | 39.44 | 77.28 | 0.95 |
| 平方距離 | 32.70 | 71.95 | |
| 論文 Table 1 的 DDAE（參考） | 32.77 | 74.08 | |

主要檢定（雙尾 Wilcoxon，57 組成對）：

| 比較 | AP 差 | 勝-敗 | p_holm | ROC-AUC 差 | p_holm |
|---|---|---|---|---|---|
| T=1000 依分數剔除 vs 未剔除 | +0.51 | 36-21 | 0.091 | +0.40 | 0.13 |
| T=1000 依分數剔除 vs 隨機剔除 | +0.67 | 33-24 | 0.091 | +0.40 | 0.12 |
| T=1000 隨機剔除 vs 未剔除 | −0.16 | 27-30 | 0.73 | −0.00 | 0.73 |
| T=1000 依標籤剔除 vs 依分數剔除 | +5.98 | 53-3（1 平） | 2.8×10⁻⁹ | +2.98 | 3.9×10⁻⁸ |
| T=1000 未剔除 vs 平方距離 | +0.26 | 41-16 | 0.13 | +1.96 | 7.9×10⁻⁵ |
| T=1000 依分數剔除 vs 平方距離 | +0.77 | 40-17 | 0.13 | +2.36 | 4.8×10⁻⁵ |
| T=1000 未剔除 vs T=50 未剔除 | +8.28 | 32-25 | 0.20 | +3.98 | 1 |
| T=50 依分數剔除 vs T=1000 未剔除 | +0.16 | 28-29 | 1 | −0.53 | 1 |
| T=50 依分數剔除 vs T=50 未剔除 | +8.45 | 37-20 | 0.003 | +3.46 | 0.005 |

完整十二項比較見 tables/t1000_stats.csv。

讀法：

- 在 T=1000，依分數剔除只多 0.51 個 AP 點，校正後不顯著；隨機剔除沒有效果。依分數剔除只補回「未剔除 → 依標籤剔除」這段 AP 差距的 7.8%（T=50 為 31%）。
- T=50 加剔除跟 T=1000 不剔除幾乎一樣（差 0.16 點，28 勝 29 敗）。在這個設定下，剔除和較大的 T 帶來的改善大致重疊，兩者疊加的增益沒有被檢出。
- T=1000 未剔除對 T=50 未剔除平均高 8.28 個 AP 點，但只有 32 勝 25 敗，校正後不顯著：平均值由少數資料集拉高，不能說成「T=1000 全面較好」。
- T=1000 未剔除的 AP 跟平方距離沒有顯著差異，ROC-AUC 顯著較高（+1.96）。它的分數排名跟平方距離的相關是 0.95，T=50 只有 0.44。
- T=50 加剔除的 AP 也接近平方距離，但排名相關只有 0.46，代表它不是靠模仿平方距離得到這個分數。

### 每個時間步的誤差（T=1000 未剔除）

sweepT_1000 另外記錄只用單一時間步評分時的 AP，以及該步誤差跟平方距離的相關（57 組平均，tables/t1000_summary.json 的 per_step）：

| t | 1 | 41 | 92 | 193 | 293 | 495 | 747 | 999 |
|---|---|---|---|---|---|---|---|---|
| ρ（該步誤差 vs 平方距離） | 0.54 | 0.47 | 0.42 | 0.43 | 0.53 | 0.69 | 0.88 | 0.86 |
| 只用該步的 AP | 28.08 | 26.68 | 24.63 | 21.83 | 21.12 | 26.35 | 31.75 | 31.12 |

後段（雜訊很大）的步驟跟平方距離的相關接近 0.9，中段最低，單步 AP 也最低。這支持「雜訊很大時，模型最好的猜法就是猜平均值，誤差因此趨近到平均點的距離」的推論，但這只是相關證據。

限制：dte2 是傳導式設定（訓練列從評分資料有放回抽樣）；AP 用 average_precision_score，論文的 PR-AUC 算法未說明；GPU 與 CPU 的結果不逐位相同，所以本節數字不跟 raw/、raw_win/ 混用。

## 半監督的平方距離基準

半監督（訓練集只有正常資料）時，論文的 DDAE 明顯勝過平方距離，跟無監督的情況不同。在repo 根目錄設定 ADBENCH_ROOT 後執行：

~~~bash
python semi_centroid.py
~~~

程式用 dte 協定（DTE 公開程式的流程）、seed 0 至 4，不訓練模型，寫出 tables/semi_centroid.json（每組 [AP, ROC-AUC]）。57 組平均為 47.26 / 74.36。論文 Table 1 的半監督 DDAE 為 61.36 / 83.17。論文的 PR-AUC 算法與本組 AP 可能不同，這組數字只說明差距的量級，不是嚴格的同條件比較。
