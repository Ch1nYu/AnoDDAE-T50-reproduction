"""DDAE / DDAE-C / DAE re-implementation.

Defaults reproduce the released AnoDDAE code (github.com/sattarov/AnoDDAE,
src/model.py) exactly: MLP [d+e -> 512 -> 512 -> d] with LeakyReLU, 4-dim
sinusoidal timestep embedding concatenated to the input, t ~ U{1..T-1},
x0-reconstruction MSE, Adam(1e-3), 100 epochs, batch = pow2 closest to |X|/10,
score = sum_{t=1}^{T-1} ||x0 - x0_hat(x_t, t)||_2 with one fresh noise draw
per step.  Every deviation is an explicit, documented option:

* ``cosine_impl``  'official' reproduces the released schedule, which returns
  alpha_bar instead of beta (beta_0 = 1  =>  alpha_bar_t = 0 for every t, i.e.
  pure-noise inputs); 'fixed' is the Nichol & Dhariwal cosine schedule.
* ``contrastive_alpha`` > 0 turns on DDAE-C (paper Eq. 7-8, Alg. 1), which is
  *not* contained in the released code.
* ``score_steps`` = K < T-1 evaluates the step-sum on K evenly spaced steps and
  rescales by (T-1)/K (Monte-Carlo approximation, validated in the report).
* Several score variants are produced from the same forward passes
  (see ``score``), so improvements to the scoring rule cost no extra training.
"""
from __future__ import annotations

import math
import time

import os

import numpy as np
import torch
import torch.nn as nn

# Device for training and scoring: ANODDAE_DEVICE=cuda puts the model and data on the GPU
# (results then differ from CPU runs at float precision).  Default: cpu, as in all earlier runs.
DEVICE = torch.device(os.environ.get("ANODDAE_DEVICE", "cpu"))
# While this file exists, training waits at the next epoch boundary (GPU temperature guard).
PAUSE_FILE = os.environ.get("ANODDAE_PAUSE_FILE")


def _wait_if_paused():
    if PAUSE_FILE:
        while os.path.exists(PAUSE_FILE):
            time.sleep(5)


# --------------------------------------------------------------------------- #
# Noise schedules
# --------------------------------------------------------------------------- #
class NoiseSchedule:
    def __init__(self, T: int, kind: str = "linear", beta_start: float = 1e-4,
                 beta_end: float = 0.02, cosine_impl: str = "official"):
        self.T, self.kind = T, kind
        if kind == "linear":
            beta = torch.linspace(beta_start, beta_end, T)
        elif kind == "quadratic":
            beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
        elif kind == "sigmoid":
            beta = torch.sigmoid(torch.linspace(-6, 6, T)) * (beta_end - beta_start) + beta_start
        elif kind == "exponential":
            beta = torch.logspace(math.log10(beta_start), math.log10(beta_end), T)
        elif kind == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
            abar = f / f[0]
            if cosine_impl == "official":      # released code: returns alpha_bar as beta
                beta = abar[:-1].float()
            else:                              # Nichol & Dhariwal (2021)
                beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999).float()
        else:
            raise ValueError(kind)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_ab = self.alpha_bar.sqrt().to(DEVICE)
        self.sqrt_1mab = (1 - self.alpha_bar).sqrt().to(DEVICE)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_ab[t].view(-1, 1) * x0 + self.sqrt_1mab[t].view(-1, 1) * noise, noise


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _act(name):
    return {"relu": nn.ReLU, "lrelu": nn.LeakyReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]()


class Denoiser(nn.Module):
    """x_t (+) e_t -> hidden -> ... -> d.

    Encoder f_theta = every layer up to and including the last hidden Linear
    (its pre-activation output is the latent z used by DDAE-C); decoder
    g_phi = remaining activation + output Linear.  With hidden=[512, 512] this
    is *exactly* the released DDAE network, so DDAE-C(alpha=0) == DDAE.
    """

    def __init__(self, d, hidden=(512, 512), t_dim=4, act="lrelu", init="default"):
        super().__init__()
        self.t_dim = t_dim
        layers, prev = [], d + t_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            if i < len(hidden) - 1:
                layers.append(_act(act))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(_act(act), nn.Linear(prev, d))
        if init == "xavier":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def inp(self, x, t):
        if self.t_dim > 0:
            return torch.cat([x, sinusoidal_embedding(t, self.t_dim)], dim=1)
        return x

    def encode(self, x, t):
        return self.encoder(self.inp(x, t))

    def forward(self, x, t):
        z = self.encode(x, t)
        return self.decoder(z), z


# --------------------------------------------------------------------------- #
# DDAE (+ contrastive DDAE-C, + fixed-noise DAE)
# --------------------------------------------------------------------------- #
class DDAE:
    def __init__(self, d, hidden=(512, 512), act="lrelu", T=50, schedule="linear",
                 beta_start=1e-4, beta_end=0.02, cosine_impl="official", t_dim=4,
                 epochs=100, batch_size=64, lr=1e-3, init="default",
                 contrastive_alpha=0.0, dae_sigma=None, score_steps=None,
                 score_chunk_rows=32768, seed=0, trim_frac=0.0, trim_every=10,
                 trim_mode="score"):
        self.d, self.T, self.epochs, self.bs, self.lr = d, T, epochs, batch_size, lr
        self.alpha = contrastive_alpha
        self.dae_sigma = dae_sigma          # not None -> plain DAE: x + sigma*eps, no t
        self.score_steps = score_steps
        self.chunk = score_chunk_rows
        self.sched = NoiseSchedule(T, schedule, beta_start, beta_end, cosine_impl)
        self.net = Denoiser(d, hidden, 0 if dae_sigma is not None else t_dim, act, init).to(DEVICE)
        self.history = []
        # contamination-robust training (improvement, unsupervised setting): every
        # `trim_every` epochs, drop the `trim_frac` training rows with the highest
        # step-normalised reconstruction error from the next epochs.
        self.trim_frac, self.trim_every = trim_frac, trim_every
        # trim_mode="random" is the control: keep the same number of rows, chosen
        # uniformly at random with its own generator, so it changes data volume and
        # update count like the scored trim but selects rows without looking at them.
        # trim_mode="oracle" drops true anomalies (labels) first, then random normals, to the same row count
        assert trim_mode in ("score", "random", "oracle")
        self.trim_mode = trim_mode
        self._trim_gen = torch.Generator().manual_seed(seed)

    # ---------------------------------------------------------------- training
    def _corrupt(self, x0, t):
        if self.dae_sigma is not None:
            return x0 + self.dae_sigma * torch.randn_like(x0)
        return self.sched.q_sample(x0, t)[0]

    def fit(self, x_train: np.ndarray, y_train: np.ndarray | None = None):
        x_all = torch.as_tensor(x_train, dtype=torch.float32).to(DEVICE)
        x = x_all
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(x)
        t0 = time.time()
        self.trim_log = []  # (epoch, indices into x_train dropped for the next trim_every epochs)
        for ep in range(self.epochs):
            _wait_if_paused()
            if self.trim_frac > 0 and ep > 0 and ep % self.trim_every == 0:
                keep = int(round(len(x_all) * (1 - self.trim_frac)))
                if self.trim_mode == "random":
                    order = torch.randperm(len(x_all), generator=self._trim_gen)
                elif self.trim_mode == "oracle":
                    r = torch.randperm(len(x_all), generator=self._trim_gen)
                    anom = torch.as_tensor(np.asarray(y_train) == 1)[r]
                    order = torch.cat([r[~anom], r[anom]])  # the tail (dropped) holds the anomalies
                else:
                    s_tr = self._train_scores(x_all)
                    order = torch.argsort(s_tr)
                order = order.cpu()
                x = x_all[order[:keep].to(DEVICE)]
                self.trim_log.append((ep, order[keep:].numpy().astype(np.int32)))
                n = len(x)
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            nb = 0
            for i in range(0, n, self.bs):
                x0 = x[perm[i:i + self.bs].to(DEVICE)]
                b = len(x0)
                t = torch.randint(1, self.T, (b,)).to(DEVICE)
                xt = self._corrupt(x0, t)
                x0_hat, zt = self.net(xt, t)
                loss_rec = ((x0_hat - x0) ** 2).mean()
                loss = loss_rec
                if self.alpha > 0:
                    # DDAE-C, paper Eq. 7-8 / Alg. 1:
                    #   positive pairs (x0_i, x0_j) both drawn from the training set -> y=0
                    #   negative pairs (x0_i, x_t,i)                                  -> y=1
                    #   delta = ||z_0 - z_t||_2, margin m = 1 + t/T
                    z0 = self.net.encode(x0, t)
                    z0p = self.net.encode(x0[torch.randperm(b).to(DEVICE)], t)
                    d_pos = (z0 - z0p).norm(dim=1)
                    d_neg = (z0 - zt).norm(dim=1)
                    m = 1.0 + t.float() / self.T
                    l_cont = torch.cat([d_pos ** 2, torch.relu(m - d_neg) ** 2]).mean()
                    loss = (1 - self.alpha) * loss_rec + self.alpha * l_cont
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss_rec.item()
                nb += 1
            self.history.append(tot / max(nb, 1))
        self.train_time = time.time() - t0
        return self

    @torch.no_grad()
    def _train_scores(self, x_all, n_steps=8):
        """Step-normalised error on a few fixed timesteps (used for trimming)."""
        self.net.eval()
        if self.dae_sigma is not None:
            ts = np.array([0])
        else:
            ts = np.unique(np.round(np.linspace(1, self.T - 1, n_steps)).astype(int))
        sq = torch.zeros(len(ts), len(x_all), device=DEVICE)
        for i, nb, e in self._errors(x_all, ts):
            sq[:, i:i + nb] = (e ** 2).sum(dim=2)
        sq = sq / (sq.median(dim=1, keepdim=True).values + 1e-8)
        return sq.sum(0).cpu()

    # ----------------------------------------------------------------- scoring
    def score_timesteps(self):
        if self.dae_sigma is not None:
            return np.array([0])
        all_t = np.arange(1, self.T)
        K = self.score_steps
        if K is None or K >= len(all_t):
            return all_t
        return np.unique(np.round(np.linspace(1, self.T - 1, K)).astype(int))

    @torch.no_grad()
    def _errors(self, x, ts):
        """Yields (t_index, row_slice, e) with e = x0 - x0_hat (rows x d)."""
        n = len(x)
        rows_per = max(1, self.chunk // max(1, len(ts)))
        for i in range(0, n, rows_per):
            _wait_if_paused()
            xb = x[i:i + rows_per]
            nb = len(xb)
            k = len(ts)
            tt = torch.as_tensor(np.repeat(ts, nb), dtype=torch.long, device=DEVICE)
            x0 = xb.repeat(k, 1)
            if self.dae_sigma is not None:
                xt = x0                      # DAE: clean input at test time
            else:
                xt = self.sched.q_sample(x0, tt)[0]
            x0_hat, _ = self.net(xt, tt)
            e = (x0 - x0_hat).view(k, nb, -1)
            yield i, nb, e

    @torch.no_grad()
    def score(self, x_test: np.ndarray, x_calib: np.ndarray | None = None,
              per_step: bool = False):
        """Returns dict of score variants (higher = more anomalous).

        official : sum_t ||e_t||_2            (released code)
        paper    : sum_t ||e_t||_2^2          (paper Eq. 6)
        stepnorm : sum_t ||e_t||^2 / c_t      c_t = mean calib ||e_t||^2
        featnorm : sum_t sum_j e_tj^2 / v_tj  v_tj = mean calib e_tj^2
        Calibration rows come from the *training* set (no test labels used).
        """
        self.net.eval()
        ts = self.score_timesteps()
        w = (self.T - 1) / len(ts) if self.dae_sigma is None else 1.0
        x = torch.as_tensor(x_test, dtype=torch.float32).to(DEVICE)
        K = len(ts)
        c_t = v_tj = None
        if x_calib is not None:
            xc = torch.as_tensor(x_calib, dtype=torch.float32).to(DEVICE)
            sq_sum = torch.zeros(K, self.d, device=DEVICE)
            for _, nb, e in self._errors(xc, ts):
                sq_sum += (e ** 2).sum(dim=1)
            v_tj = sq_sum / len(xc) + 1e-8          # K x d
            c_t = v_tj.sum(dim=1)                   # K
        out = {k: torch.zeros(len(x), device=DEVICE) for k in ("official", "paper")}
        if c_t is not None:
            out["stepnorm"] = torch.zeros(len(x), device=DEVICE)
            out["featnorm"] = torch.zeros(len(x), device=DEVICE)
        step_sq = torch.zeros(K, len(x), device=DEVICE) if per_step else None
        for i, nb, e in self._errors(x, ts):
            sq = (e ** 2).sum(dim=2)                 # K x nb
            out["official"][i:i + nb] += sq.sqrt().sum(0) * w
            out["paper"][i:i + nb] += sq.sum(0) * w
            if c_t is not None:
                out["stepnorm"][i:i + nb] += (sq / c_t[:, None]).sum(0)
                out["featnorm"][i:i + nb] += ((e ** 2) / v_tj[:, None, :]).sum(dim=(0, 2))
            if per_step:
                step_sq[:, i:i + nb] = sq
        res = {k: v.cpu().numpy() for k, v in out.items()}
        if per_step:
            res["_per_step"] = (ts, step_sq.cpu().numpy())
        return res
