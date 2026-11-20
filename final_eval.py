#!/usr/bin/env python3
"""Final report card for a trained run: solve rate / optimality per solver."""
import argparse
import json
import os
import time

import numpy as np
import torch

import model as M
import scorecard as SC

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/main/model.pt")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default=None)
a = ap.parse_args()

ck = torch.load(a.model, map_location="cpu", weights_only=False)
meta = ck.get("meta", {})
net = M.DualNet(meta.get("hidden", 512))
net.load_state_dict(ck["state"])
net.eval()
dev = "cpu"

perm, ori = SC.test_set(a.n, seed=999)
opt = SC.optimal_len(perm, ori)
v = M.value(perm, ori, net, dev)
res = dict(model_it=meta.get("it"), n=a.n,
           opt_len_mean=float(opt.mean()),
           v_mae=float(np.abs(v - opt).mean()),
           v_corr=float(np.corrcoef(v, opt)[0, 1]),
           p_acc_selfsup=SC.policy_acc(net, dev, n=800),
           solvers=[])

gl, gs = SC.solve_greedy_policy(net, dev, perm, ori)
res["solvers"].append(dict(name="greedy-policy", solved=float(gs.mean()),
                           len=float(gl[gs].mean()) if gs.any() else 0.0,
                           optimal=float((gl[gs] == opt[gs]).mean()) if gs.any() else 0.0))

for mode, w in (("policy", 96), ("policy", 192), ("value", 96)):
    t0 = time.time()
    bl, bs = SC.beam(net, dev, perm, ori, width=w, max_steps=20, mode=mode)
    res["solvers"].append(dict(name=f"beam-{mode}-w{w}", solved=float(bs.mean()),
                               len=float(bl[bs].mean()) if bs.any() else 0.0,
                               optimal=float((bl[bs] == opt[bs]).mean()) if bs.any() else 0.0,
                               excess=float((bl[bs] - opt[bs]).mean()) if bs.any() else 0.0,
                               sec=round(time.time() - t0, 1)))

print(json.dumps(res, indent=2, ensure_ascii=False))
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
