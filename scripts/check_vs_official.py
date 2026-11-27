"""Re-implementation vs. released code on the README example (InternetAds,
semi-supervised, config.yaml: T=50 linear, [512,512], seed 111, repo protocol).

Runs both implementations over several seeds so that the comparison is about
distributions, not a single draw.  Uses the *released* src/ package unchanged.
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL = os.environ.get("ANODDAE_OFFICIAL", os.path.join(ROOT, "official_patched"))
sys.path.insert(0, ROOT)
sys.path.insert(0, OFFICIAL)
from src.model import DDAE as OfficialDDAE  # noqa: E402
from src.utils import set_seed, evaluate_anomaly_detection  # noqa: E402
from anoddae_repro.data import make_split, batch_size_rule  # noqa: E402
from anoddae_repro.models import DDAE  # noqa: E402
from anoddae_repro.runner import _metrics  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "4")))
f, T = "17_InternetAds", 50
rows = []
for seed in [111, 0, 1, 2, 3]:
    s = make_split(f, "semi", seed, protocol="repo", max_size=0)
    bs = batch_size_rule(s.n_total)
    # released implementation (eval_epochs=epochs -> no intermediate test evaluation)
    set_seed(seed)
    t0 = time.time()
    m = OfficialDDAE(input_dim=s.x_train.shape[1], hidden_dim=[512, 512], activation="lrelu",
                     num_timesteps=T, beta_start=1e-4, beta_end=0.02, scheduler="linear",
                     time_emb_dim=4, time_emb_type="sinusoidal", epochs=100, batch_size=bs,
                     learning_rate=1e-3, eval_epochs=10 ** 9)
    xtr, xte = torch.tensor(s.x_train), torch.tensor(s.x_test)
    m.fit(xtr, xte, None, torch.tensor(s.y_test, dtype=torch.float32))
    so = m.predict(xte).numpy()
    t_off = time.time() - t0
    mo = evaluate_anomaly_detection(so, s.y_test)
    # re-implementation
    set_seed(seed)
    t0 = time.time()
    r = DDAE(s.x_train.shape[1], T=T, batch_size=bs).fit(s.x_train)
    sr = r.score(s.x_test)["official"]
    t_re = time.time() - t0
    mr = _metrics(s.y_test, sr)
    rows.append((seed, 100 * mo["AP"], 100 * mo["ROC-AUC"], 100 * mo["PR-AUC"], mr["ap"], mr["roc"], mr["prauc"], t_off, t_re))
    print(f"seed {seed}: official AP {rows[-1][1]:.2f} ROC {rows[-1][2]:.2f} PRAUC(trapz) {rows[-1][3]:.2f} ({t_off:.0f}s) | "
          f"re-impl AP {mr['ap']:.2f} ROC {mr['roc']:.2f} PRAUC(trapz) {mr['prauc']:.2f} ({t_re:.0f}s)", flush=True)
a = np.array(rows)
print("MEAN official AP %.2f±%.2f ROC %.2f±%.2f | re-impl AP %.2f±%.2f ROC %.2f±%.2f" % (
    a[:, 1].mean(), a[:, 1].std(), a[:, 2].mean(), a[:, 2].std(), a[:, 4].mean(), a[:, 4].std(), a[:, 5].mean(), a[:, 5].std()))
np.savetxt(os.path.join(os.environ.get("ANODDAE_RESULTS", os.path.join(ROOT, "results")), "check_vs_official_internetads.csv"), a, delimiter=",",
           header="seed,off_ap,off_roc,off_prauc,re_ap,re_roc,re_prauc,t_off,t_re", fmt="%.4f")
