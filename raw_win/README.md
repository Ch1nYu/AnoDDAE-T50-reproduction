# raw_win：隨機剔除對照的原始紀錄

三個方法在同一台 Windows 電腦上重跑，彼此比較，不跟 raw/ 的 Linux 歷史結果混用。

- 方法：未剔除版（sweepT_50）、依分數剔除（impr_trim0.1_T50）、隨機剔除（impr_randtrim0.1_T50）。隨機剔除在第 11、21、…、91 輪開始前，從整批訓練列隨機保留 round(0.9×n) 列，列數和更新次數跟依分數剔除相同，但不看分數。
- 規模：57 組（121 個檔）× seed 0–5，每個方法 726 次，共 2,178 次，全部成功。
- 日期：2026-11-24 01:11 到 04:42（18 個 worker）。
- 環境：Windows 11（10.0.26200）、Intel Core i7-13700F、Python 3.12.10、PyTorch 2.6.0+cu124（只用 CPU）、NumPy 2.4.6、scikit-learn 1.9.1、SciPy 1.18.1。
- 資料：ADBench commit 3dac8221081e190f157d78e93bfa8867f90d0965 的 121 個檔，跟 raw/ 相同。
- 指令（在repo 根目錄，ADBENCH_ROOT 指向上面的資料）：

~~~bash
python -m anoddae_repro.runner --exp impr_randtrim0.1_T50 sweepT_50 impr_trim0.1_T50 --setting unsup --protocol dte2 --seeds 0 1 2 3 4 5 --workers 18 --tag _win
~~~

- 程式版本：models.py 跟本 repo相同。data.py 只差在說明文字。runner.py 是加入新分數路徑和續跑檢查之前的版本，指標計算相同。
- 只附 JSONL（每次訓練的指標與資料規模）。逐樣本分數 NPZ 沒有放進 repo。
- 執行時 Windows 會把背景程序限流到節能核心；前 15 分鐘的 worker 只用到約 6.4 顆核心，關掉限流後約 17.7 顆。這段期間 JSONL 的 wall、train_time 較長。沒有在開、關限流下重跑同一工作比較分數。

彙總：`python random_trim_control.py`，輸出 tables/random_trim_*.csv。

平台對照：同一台 Windows 重跑的未剔除版與依分數剔除，57 組平均 AP / ROC-AUC 為 24.89 / 70.17 與 32.65 / 73.23；raw/ 的 Linux 結果為 24.85 / 70.13 與 32.77 / 73.28。平方距離基準沒有隨機性，兩個平台都是 32.70 / 71.95。
