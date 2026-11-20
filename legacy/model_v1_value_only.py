"""DAVI-style value iteration (McAleer 2018 / Agostinelli's DeepCubeA 2019).

Key differences from a bare DQN, and why they matter:
  1. data comes from *reverse* random walks out of the solved state (curriculum,
     and the goal state is actually visited);
  2. the head regresses **cost-to-go in moves** (J(s) = min_a 1 + J(s')) instead
     of a sparse +-1 terminal reward, so the signal is dense and shift-free;
  3. the net is used as a *heuristic* by a search (greedy / beam / weighted A*),
     not as a policy that is executed blindly.
"""
import numpy as np
import torch
import torch.nn as nn

import cube2


class ValueNet(nn.Module):
    def __init__(self, hidden=256, indim=144):
        super().__init__()
        h = hidden
        self.f = nn.Sequential(
            nn.Linear(indim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.f(x).squeeze(-1)


@torch.no_grad()
def value(x_perm, x_ori, net, dev, kmax=14):
    """Cost-to-go estimate for (B,8) perm/ori numpy arrays -> (B,) numpy."""
    x = torch.from_numpy(cube2.one_hot(x_perm, x_ori)).to(dev)
    return net(x).clamp_(0.0, float(kmax)).cpu().numpy()


class DAVI:
    def __init__(self, iters, batch=2048, kmax=14, hidden=256, lr=1e-3,
                 gamma=1.0, target_every=20, seed=0, device="cpu", log=None,
                 mixed_depth=True):
        self.batch, self.kmax, self.gamma = batch, kmax, gamma
        self.target_every = target_every
        self.dev = torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = ValueNet(hidden).to(self.dev)
        self.tgt = ValueNet(hidden).to(self.dev)
        self.tgt.load_state_dict(self.net.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[int(iters * 0.6), int(iters * 0.85)], gamma=0.3)
        # the solved state is never produced by a k>=1 scramble, so pin it down
        self._pin = 32
        self.mixed_depth = mixed_depth
        self.iter = 0
        self.samples = 0
        self.log = log or (lambda *a, **k: None)

    def step(self):
        B = self.batch
        if self.mixed_depth:
            ks = self.rng.integers(1, self.kmax + 1, size=B)
            K = float(ks.mean())
        else:
            K = int(self.rng.integers(1, self.kmax + 1))
            ks = K
        perm, ori, _ = cube2.scramble_mixed(self.rng, B, self.kmax, k=ks)

        succ = [cube2.apply_move(perm, ori, a) for a in range(cube2.NUM_ACTIONS)]
        sv = np.concatenate([cube2.one_hot(p, o) for p, o in succ], 0)
        sol = np.concatenate([cube2.is_solved(p, o) for p, o in succ])
        sol = sol.reshape(cube2.NUM_ACTIONS, B).astype(np.float32)      # (6,B)

        with torch.no_grad():
            vt = self.tgt(torch.from_numpy(sv).to(self.dev)).view(cube2.NUM_ACTIONS, B)
            # cost-to-go Bellman step:  J(s) = min_a [ 1 + J(s') ],  J(goal) = 0
            vt = vt * (1.0 - torch.from_numpy(sol).to(self.dev))
            y = (1.0 + self.gamma * vt).min(0).values                   # (B,)
            y = y.clamp_(0.0, float(self.kmax) * 1.5)

            # pin V(solved) = 0
            ps, os_ = cube2.solved(self._pin)
            xs = torch.from_numpy(cube2.one_hot(ps, os_)).to(self.dev)
            y = torch.cat([y, torch.zeros(self._pin, device=self.dev)])

        x = torch.from_numpy(cube2.one_hot(perm, ori)).to(self.dev)
        x = torch.cat([x, xs], 0)

        v = self.net(x)
        loss = torch.nn.functional.smooth_l1_loss(v, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        self.sched.step()

        self.iter += 1
        self.samples += B
        if self.iter % self.target_every == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return dict(it=self.iter, K=K, loss=float(loss.detach()), grad_norm=float(gn),
                    lr=self.opt.param_groups[0]["lr"],
                    v_mean=float(v[:B].mean().detach()), v_max=float(v[:B].max().detach()))

    def save(self, path, meta=None):
        torch.save({"state": self.net.state_dict(), "meta": meta or {}}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.dev)
        self.net.load_state_dict(ck["state"])
        self.tgt.load_state_dict(ck["state"])
        return ck.get("meta", {})
