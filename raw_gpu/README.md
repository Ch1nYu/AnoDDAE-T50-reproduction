# raw_gpu：T=1000 對照與 T=50 GPU 重跑的原始紀錄

六個方法在同一台電腦、同一份程式、同一張 GPU 上跑，只彼此比較，不跟 raw/（Linux CPU）或 raw_win/（Windows CPU）的結果混用。

- 方法（全部 unsup、dte2、seed 0–5、tag `_gpu`）：
  - T=1000：未剔除（sweepT_1000）、依分數剔除 10%（impr_trim0.1_T1000）、隨機剔除 10%（impr_randtrim0.1_T1000）、依標籤剔除 10%（impr_oracletrim0.1_T1000，上限）。T=1000 評分時從 1 到 999 等距取 100 個時間步（score_steps=100）。
  - T=50：未剔除（sweepT_50）、依分數剔除 10%（impr_trim0.1_T50），讓 T=50 跟 T=1000 在同一裝置上比。
- 規模：57 組（121 個檔）× seed 0–5，每個方法 726 次，共 4,356 次，全部成功。`*_gpusmoke`、`*_cpusmoke` 是正式執行前在 6_cardio 上的試跑，分析不使用。
- 日期：2026-11-25 19:15 到 2026-11-26 03:14（實驗室電腦的當地時間）。
- 環境：實驗室的 Windows 電腦（Intel Core i7-12700、NVIDIA RTX 3080 12 GB）、Python 3.12、PyTorch 2.6.0+cu124、NumPy 2.4.6、scikit-learn 1.9.1、SciPy 1.18.1。
- 資料：ADBench commit 3dac8221081e190f157d78e93bfa8867f90d0965 的 121 個檔，跟 raw/ 相同。dataset_sizes.json 跟 raw/ 的逐位元組相同。
- 指令（在repo 根目錄，ADBENCH_ROOT 指向上面的資料）：

~~~bash
export ANODDAE_DEVICE=cuda
python -m anoddae_repro.runner --exp sweepT_1000 impr_trim0.1_T1000 impr_randtrim0.1_T1000 impr_oracletrim0.1_T1000 --setting unsup --protocol dte2 --seeds 0 1 2 3 4 5 --workers 5 --tag _gpu
python -m anoddae_repro.runner --exp sweepT_50 impr_trim0.1_T50 --setting unsup --protocol dte2 --seeds 0 1 2 3 4 5 --workers 3 --tag _gpu
~~~

- 每筆紀錄比舊版多三個欄位：`device`（cuda）、`official_cen_rho`（官方分數與平方距離分數的 Spearman 相關）。sweepT_1000 另有 `per_step_cen_rho`（每個評分時間步的誤差與平方距離的 Spearman 相關）。
- 只附 JSONL。逐樣本分數 NPZ（約 591 MB）留在實驗室電腦，沒有放進 repo。

## GPU 與 CPU 的差異

GPU 的亂數與浮點運算路徑跟 CPU 不同，同一個工作的結果不會逐位相同。例如 6_cardio、seed 0、T=50 未剔除，GPU 為 AP 22.26，CPU 為 18.76。GPU 上同一個工作重跑兩次結果相同。57 組平均則接近：T=50 未剔除 GPU 24.67、Linux CPU 24.85、Windows CPU 24.89；T=50 依分數剔除 GPU 33.12、Linux CPU 32.77、Windows CPU 32.65。

`ANODDAE_DEVICE` 沒設定時預設用 CPU。2026-11-26 在家用 Windows 電腦以 CPU 跑 6_cardio、seed 0 的兩個 T=50 方法，AP、ROC-AUC 跟 raw_win/ 的紀錄逐位相同（18.763975… / 58.530966… 與 39.033196… / 67.199945…）。

## 溫度保護與中途重啟

- 為了避免 GPU 長時間高溫，目標是不超過 70°C。另開一個程序每 3–5 秒記錄 GPU 溫度，超過門檻就建立暫停檔，程式在每輪訓練開頭、每個評分區塊前檢查暫停檔（環境變數 `ANODDAE_PAUSE_FILE`）。暫停只是延後，不改變計算內容。
- 第一版門檻為 68°C 暫停，最高到 72°C（19:17–19:20 共 12 筆超過 70°C）。之後改為 63°C 暫停、60°C 恢復，最高 68°C。整段約 58% 的時間處於暫停。
- 22:57–23:18：6 個 worker 的 PyTorch 快取把 GPU 記憶體塞到 11.8/12 GB，單一工作被換到共享記憶體而變得極慢。改為每個工作結束時釋放快取（runner.py 的 `torch.cuda.empty_cache()`）、5 個 worker 後續跑；已完成的 1,590 次沿用。
- 01:20：T=50 的前段大資料集讓溫度到 69°C，改用 3 個 worker 續跑。
- runner 只略過「已成功、分數檔完整」的工作，續跑不會重複或漏掉；每個方法最後都是 726 筆成功紀錄。

彙總：`python t1000_analysis.py`，輸出 tables/t1000_*。
