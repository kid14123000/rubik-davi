#!/usr/bin/env python3
"""Final report card for the 3x3 run (solve rate is the exactly verifiable
metric; every returned solution is replayed and checked)."""
import argparse
import json
import time

import numpy as np
import torch

import cube3 as k3
import model3 as M
import scorecard3 as SC

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs3/main/model.pt")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", default=None)
ap.add_argument("--kmax", type=int, default=20)
a = ap.parse_args()

ck = torch.load(a.model, map_location="cpu", weights_only=False)
meta = ck.get("meta", {})
net = M.DualNet3(meta.get("hidden", 512))
net.load_state_dict(ck["state"])
net.eval()
dev = "cpu"

rng = np.random.default_rng(4242)
states, scr = [], []
for _ in range(a.n):
    k = int(rng.integers(8, a.kmax + 1))         # 8..20 move scrambles
    st, seq = k3.scramble(rng, k, 1)
    states.append(st)
    scr.append(len(seq))
states = tuple(np.concatenate([s[i] for s in states], 0) for i in range(4))
scr = np.array(scr)

res = dict(model_it=meta.get("it"), n=a.n, scramble_len=float(scr.mean()),
           p_acc=SC.policy_acc(net, dev, n=1024), solvers=[])

gl, gs = SC.greedy_policy(net, dev, states, max_steps=40)
res["solvers"].append(dict(name="greedy-policy", solved=float(gs.mean()),
                           len=float(gl[gs].mean()) if gs.any() else 0.0,
                           max_len=int(gl[gs].max()) if gs.any() else 0))

for mode, w in (("policy", 32), ("policy", 128), ("value", 128)):
    t0 = time.time()
    bl, bs = SC.beam(net, dev, states, width=w, max_steps=45, mode=mode)
    res["solvers"].append(dict(name=f"beam-{mode}-w{w}", solved=float(bs.mean()),
                               len=float(bl[bs].mean()) if bs.any() else 0.0,
                               max_len=int(bl[bs].max()) if bs.any() else 0,
                               sec_per_cube=round((time.time() - t0) / a.n, 2)))

print(json.dumps(res, indent=2, ensure_ascii=False))
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
