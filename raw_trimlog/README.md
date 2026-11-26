# raw_trimlog：剔除紀錄實驗的原始結果

- 方法：依分數剔除（impr_trim0.1_T50）與隨機剔除（impr_randtrim0.1_T50），57 組（121 個檔）× seed 0–5，各 726 次，全部成功。
- 程式：本 repo anoddae_repro（models.py 的 trim_log、runner.py 寫 results/trimlog/）。指令：

~~~bash
python -m anoddae_repro.runner --exp impr_trim0.1_T50 impr_randtrim0.1_T50 --setting unsup --protocol dte2 --seeds 0 1 2 3 4 5 --workers 18 --tag _trimlog
~~~

- 日期與環境：2026-11-24。前 6 次在家用 Windows 電腦（raw_win/ 同一台）跑，其餘 1,446 次在另一台 Windows 11 電腦（20 執行緒、32 GB）跑；Python 3.12.10、PyTorch 2.6.0+cpu、NumPy 2.4.6、scikit-learn 1.9.1、SciPy 1.18.1。兩台的 6_cardio seed 0 逐位元相同，全部 1,452 次的 AP、ROC-AUC 跟 raw_win/ 逐筆相同。
- 資料：ADBench commit 3dac8221081e190f157d78e93bfa8867f90d0965（同 raw/）。
- 檔案：<exp>_trimlog.jsonl 是每次的指標（格式同 raw_win/）。npz/<exp>/<file>_s<seed>.npz 有 y_train（訓練列標籤，dte2 無監督下是有放回抽樣的列）、epochs（10…90）、dropped（9×k，每輪被剔除的訓練列索引，對應 y_train）。逐樣本分數 NPZ 沒有收進本 repo。
- 彙總：python trimlog_analysis.py。
