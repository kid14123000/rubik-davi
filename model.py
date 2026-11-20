"""Dual-head solver net: DAVI value + self-supervised policy.

Why two heads
-------------
*  value  : cost-to-go learned by DAVI/DeepCubeA-style bootstrapping from the
            goal state only (no labels).  Measured on a 2x2: an MLP fits the
            shallow bands well but forgets deeper ones, so pure value iteration
            alone is too slow -> keep it as the search heuristic.
*  policy : trained to predict the move that UNDOES the last scrambling move
            (a free, dense label produced by the walk itself).  This is
            AlphaCube-style self-supervision and it converges in minutes.

Solver = beam search guided by value, with the policy as prior/tie-break.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cube2

INV = np.array(cube2.INV_ACTION)


class DualNet(nn.Module):
    def __init__(self, hidden=512, indim=88, n_actions=6):
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
def predict(perm, ori, net, dev, kmax=14):
    x = torch.from_numpy(cube2.corner_feat(perm, ori)).to(dev)
    v, logits = net(x)
    return v.clamp_(0.0, float(kmax)).cpu().numpy(), logits.cpu().numpy()


@torch.no_grad()
def value(perm, ori, net, dev, kmax=14):
    return predict(perm, ori, net, dev, kmax)[0]


@torch.no_grad()
def policy(perm, ori, net, dev):
    """-> (B,6) logits"""
    return predict(perm, ori, net, dev)[1]


class Trainer:
    def __init__(self, iters, batch=2048, kmax=14, hidden=512, lr=1e-3,
                 gamma=1.0, target_every=20, seed=0, device="cpu",
                 w_value=1.0, w_policy=1.0, kmax_start=3):
        self.batch, self.kmax, self.gamma = batch, kmax, gamma
        self.target_every, self.w_v, self.w_p = target_every, w_value, w_policy
        self.kmax_start = kmax_start
        self.dev = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = DualNet(hidden).to(self.dev)
        self.tgt = DualNet(hidden).to(self.dev)
        self.tgt.load_state_dict(self.net.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[int(iters * 0.6), int(iters * 0.85)], gamma=0.3)
        self.iters = iters
        self.iter = 0
        self.samples = 0
        self._pin = 32

    def current_kmax(self):
        """Depth curriculum 3 -> kmax over the first 40% of the run."""
        frac = min(1.0, (self.iter + 1) / max(1.0, 0.4 * self.iters))
        return int(round(self.kmax_start + (self.kmax - self.kmax_start) * frac))

    def step(self):
        B = self.batch
        dmax = self.current_kmax()
        perm, ori, last = cube2.scramble_with_last(self.rng, B, dmax)

        succ = [cube2.apply_move(perm, ori, a) for a in range(cube2.NUM_ACTIONS)]
        sv = np.concatenate([cube2.corner_feat(p, o) for p, o in succ], 0)
        sol = np.concatenate([cube2.is_solved(p, o) for p, o in succ])
        sol = sol.reshape(cube2.NUM_ACTIONS, B).astype(np.float32)

        with torch.no_grad():
            vt, _ = self.tgt(torch.from_numpy(sv).to(self.dev))
            vt = vt.view(cube2.NUM_ACTIONS, B) * (1.0 - torch.from_numpy(sol).to(self.dev))
            yv = (1.0 + self.gamma * vt).min(0).values.clamp_(0.0, float(self.kmax) * 1.5)
            psm, osm = cube2.solved(self._pin)
            xs = torch.from_numpy(cube2.corner_feat(psm, osm)).to(self.dev)
            yv = torch.cat([yv, torch.zeros(self._pin, device=self.dev)])
        x = torch.cat([torch.from_numpy(cube2.corner_feat(perm, ori)).to(self.dev), xs], 0)

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
