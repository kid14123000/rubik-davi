"""3x3 dual-head net: DAVI value + self-supervised policy (undo the last move)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cube3 as k3

INV = np.array(k3.INV_ACTION)
NA = k3.NUM_ACTIONS          # 12


class DualNet3(nn.Module):
    def __init__(self, hidden=512, indim=256, n_actions=NA):
        super().__init__()
        h = hidden
        self.trunk = nn.Sequential(nn.Linear(indim, h), nn.SiLU(),
                                   nn.Linear(h, h), nn.SiLU())
        self.vhead = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))
        self.phead = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, n_actions))

    def forward(self, x):
        z = self.trunk(x)
        return self.vhead(z).squeeze(-1), self.phead(z)


@torch.no_grad()
def predict(state, net, dev, kmax=40.0):
    x = torch.from_numpy(k3.piece_feat(state)).to(dev)
    v, logits = net(x)
    return v.clamp_(0.0, kmax).cpu().numpy(), logits.cpu().numpy()


@torch.no_grad()
def value(state, net, dev, kmax=40.0):
    return predict(state, net, dev, kmax)[0]


def cat_states(a, b):
    return tuple(np.concatenate([x, y], 0) for x, y in zip(a, b))


def index_state(state, idx):
    return tuple(x[idx] for x in state)


def move_state(state, a, idx=None):
    sub = state if idx is None else index_state(state, idx)
    out = k3.apply_move(*sub, a)
    if idx is None:
        return out
    new = tuple(x.copy() for x in state)
    for arr, val in zip(new, out):
        arr[idx] = val
    return new


class Trainer3:
    def __init__(self, iters, batch=768, kmax=20, hidden=512, lr=1e-3,
                 gamma=1.0, target_every=20, seed=0, device="cpu",
                 w_value=1.0, w_policy=1.0, kmax_start=4, curriculum_frac=0.35,
                 deep_bias=0.0):
        self.batch, self.kmax, self.gamma = batch, kmax, gamma
        self.target_every, self.w_v, self.w_p = target_every, w_value, w_policy
        self.kmax_start = kmax_start
        self.curriculum_frac = curriculum_frac
        self.deep_bias = deep_bias
        self.dev = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = DualNet3(hidden).to(self.dev)
        self.tgt = DualNet3(hidden).to(self.dev)
        self.tgt.load_state_dict(self.net.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[int(iters * 0.6), int(iters * 0.85)], gamma=0.3)
        self.iters = iters
        self.iter = 0
        self.samples = 0
        self._pin = 16

    def current_kmax(self):
        frac = min(1.0, (self.iter + 1) / max(1.0, self.curriculum_frac * self.iters))
        return int(round(self.kmax_start + (self.kmax - self.kmax_start) * frac))

    def step(self):
        B = self.batch
        dmax = self.current_kmax()
        if self.deep_bias > 0.0:
            # focus capacity on the hardest band once the curriculum is done:
            # a fraction of the batch is scrambled to the full depth
            ks = self.rng.integers(1, dmax + 1, size=B)
            deep = self.rng.random(B) < self.deep_bias
            ks[deep] = dmax
            state, last = k3.scramble_with_last(self.rng, B, dmax, ks=ks)
        else:
            state, last = k3.scramble_with_last(self.rng, B, dmax)

        succ = [k3.apply_move(*state, a) for a in range(NA)]
        big = tuple(np.concatenate([s[i] for s in succ], 0) for i in range(4))
        sv = k3.piece_feat(big)
        sol = np.concatenate([k3.is_solved(*s) for s in succ])
        sol = sol.reshape(NA, B).astype(np.float32)

        with torch.no_grad():
            vt, _ = self.tgt(torch.from_numpy(sv).to(self.dev))
            vt = vt.view(NA, B) * (1.0 - torch.from_numpy(sol).to(self.dev))
            yv = (1.0 + self.gamma * vt).min(0).values.clamp_(0.0, float(self.kmax) * 1.5)
            psm = k3.solved(self._pin)
            xs = torch.from_numpy(k3.piece_feat(psm)).to(self.dev)
            yv = torch.cat([yv, torch.zeros(self._pin, device=self.dev)])

        x = torch.cat([torch.from_numpy(k3.piece_feat(state)).to(self.dev), xs], 0)
        v, logits = self.net(x)
        loss_v = F.smooth_l1_loss(v, yv.to(v.dtype))
        yp = torch.from_numpy(INV[last].astype(np.int64)).to(self.dev)
        loss_p = F.cross_entropy(logits[:B], yp)
        loss = self.w_v * loss_v + self.w_p * loss_p

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self.sched.step()

        acc = float((logits[:B].argmax(1) == yp).float().mean().detach())
        self.iter += 1
        self.samples += B
        if self.iter % self.target_every == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return dict(it=self.iter, kmax=dmax, loss=float(loss.detach()),
                    loss_v=float(loss_v.detach()), loss_p=float(loss_p.detach()),
                    p_acc=acc, grad_norm=float(gn),
                    lr=self.opt.param_groups[0]["lr"],
                    v_mean=float(v[:B].mean().detach()))

    def save(self, path, meta=None):
        torch.save({"state": self.net.state_dict(), "meta": meta or {}}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.net.load_state_dict(ck["state"])
        self.tgt.load_state_dict(ck["state"])
        return ck.get("meta", {})
