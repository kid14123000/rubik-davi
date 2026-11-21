"""Probes: is the bottleneck the net (information missing) or the search (budget)?"""
import sys
import numpy as np
import torch

import cube3 as k3
import model3 as M
import scorecard3 as SC

CK = sys.argv[1] if len(sys.argv) > 1 else "runs3/main/model.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
net = M.DualNet3(ck["meta"].get("hidden", 512))
net.load_state_dict(ck["state"])
net.eval()
DEV = "cpu"
print(f"checkpoint it={ck['meta'].get('it')}")

rng = np.random.default_rng(31337)

# ---------------- probe A: is the correct move inside the policy's top-k?
print("\n[A] policy top-k coverage of the 'undo the last scramble move' label")
print("    (states where beam-128 fails vs succeeds, depth 16/20)")
for depth in (16, 20):
    n = 12
    states, labels, beam_ok = [], [], []
    for _ in range(n):
        st, last = k3.scramble_with_last(rng, 1, depth, ks=depth)
        states.append(st)
        labels.append(int(k3.INV_ACTION[int(last[0])]))
    S = tuple(np.concatenate([s[i] for s in states], 0) for i in range(4))
    logits = M.predict(S, net, DEV)[1]
    order = np.argsort(-logits, axis=1)
    for i in range(n):
        one = tuple(x[i:i + 1] for x in S)
        ln, ok, _ = SC.beam_one(net, DEV, one, width=128, mode="policy", max_steps=45)
        beam_ok.append(ok)
    beam_ok = np.array(beam_ok)
    for grp, mask in (("beam FILL", ~beam_ok), ("beam OK  ", beam_ok)):
        if not mask.any():
            print(f"  depth {depth} {grp}: n=0")
            continue
        ranks = np.array([int(np.where(order[i] == labels[i])[0][0]) for i in np.where(mask)[0]])
        cov = {kk: float((ranks < kk).mean()) for kk in (1, 3, 5, 12)}
        print(f"  depth {depth} {grp}: n={mask.sum():2d}  mean rank {ranks.mean():4.1f}  "
              f"top1 {cov[1]*100:3.0f}%  top3 {cov[3]*100:3.0f}%  "
              f"top5 {cov[5]*100:3.0f}%  top12 {cov[12]*100:3.0f}%")

# ---------------- probe B: does a wider beam buy anything?
print("\n[B] beam width scaling (n=8 per cell)")
print(f"  {'depth':>5} {'w128':>6} {'w512':>6} {'w1024':>6}")
for depth in (16, 20):
    states, _ = [], None
    for _ in range(8):
        st, _ = k3.scramble(rng, depth, 1)
        states.append(st)
    S = tuple(np.concatenate([s[i] for s in states], 0) for i in range(4))
    row = []
    for w in (128, 384, 1024):
        _, bs = SC.beam(net, DEV, S, width=w, max_steps=45, mode="policy")
        row.append(float(bs.mean()))
    print(f"  {depth:5d} " + " ".join(f"{v*100:5.0f}%" for v in row))
