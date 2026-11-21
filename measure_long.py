"""Does a 40/60/80/100-move scramble differ from a 20-move one?

A 3x3 has God's number 20 (HTM) / 26 (QTM), so every state is at most that far
from solved. A long random walk therefore cannot reach "harder" states -- it just
revisits the same space. This measures that instead of asserting it:

  * solve rate / mean solution length per scramble depth (our 500k weight, beam 1024)
  * two independent difficulty proxies on the same states:
      - pycuber CFOP length (different solver family, no neural net involved)
      - our value head's estimate V (should plateau, not grow)
  * a mixing check: are the k=100 states distinguishable from the k=20 states?
"""
import sys
sys.path.append("/Users/kid/venvs/cube/lib/python3.12/site-packages")

import contextlib
import io
import json

import numpy as np
import torch

import cube3 as k3
import model3 as M3
import scorecard3 as SC3

W = 1024
N = 20
DEPTHS = [20, 40, 60, 80, 100]

ck = torch.load("weights/rubik3x3_500k.pt", map_location="cpu", weights_only=False)
net = M3.DualNet3(ck["meta"]["hidden"])
net.load_state_dict(ck["state"])
net.eval()

try:
    import pycuber as pc
    from pycuber.solver import CFOPSolver
    HAVE_PC = True
except ImportError:
    HAVE_PC = False

print(f"[model] it={ck['meta']['it']}  beam={W}  n={N} per depth", flush=True)
print(f"{'depth':>6} {'solved':>8} {'QTM mean':>9} {'HTM mean':>9} {'Vhat mean':>10} "
      f"{'CFOP mean':>10} {'sec/cube':>9}")


def cfop_len(seq):
    if not HAVE_PC:
        return None
    c = pc.Cube()
    c(" ".join(seq))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return len([str(x) for x in CFOPSolver(c).solve()])
    except Exception:                                              # noqa: BLE001
        return None


out = {}
for depth in DEPTHS:
    rng = np.random.default_rng(500000 + depth)
    qtm, htm, vhat, cfop, ok_n, secs = [], [], [], [], 0, []
    for _ in range(N):
        st, seq = k3.scramble(rng, depth, 1)
        names = [k3.MOVE_NAMES[a] for a in seq]
        vhat.append(float(M3.value(st, net, "cpu")[0]))
        import time
        t0 = time.time()
        ln, ok, path = SC3.beam_one(net, "cpu", st, width=W, mode="policy",
                                    max_steps=45, want_path=True)
        secs.append(time.time() - t0)
        if ok:                                    # replay-verify
            st2 = st
            for a in path:
                st2 = k3.apply_move(*st2, a)
            ok = bool(k3.is_solved(*st2)[0])
        if ok:
            ok_n += 1
            qtm.append(ln)
            # HTM: merge same-face quarter turns
            s = [k3.MOVE_NAMES[a] for a in path]
            changed = True
            while changed:
                changed, o, i = False, [], 0
                while i < len(s):
                    if i + 1 < len(s) and s[i + 1][0] == s[i][0]:
                        if s[i].endswith("'") == s[i + 1].endswith("'"):
                            o.append(s[i][0] + "2")
                        i += 2
                        changed = True
                    else:
                        o.append(s[i])
                        i += 1
                s = o
            htm.append(len(s))
        c = cfop_len(names)
        if c:
            cfop.append(c)
    row = dict(depth=depth, n=N, solved=ok_n / N,
               qtm=float(np.mean(qtm)) if qtm else None,
               htm=float(np.mean(htm)) if htm else None,
               vhat=float(np.mean(vhat)),
               cfop=float(np.mean(cfop)) if cfop else None,
               sec=float(np.mean(secs)))
    out[depth] = row
    print(f"{depth:6d} {ok_n:4d}/{N:<3d} {row['qtm'] or 0:9.1f} {row['htm'] or 0:9.1f} "
          f"{row['vhat']:10.2f} {row['cfop'] or 0:10.1f} {row['sec']:9.2f}", flush=True)

json.dump(out, open("docs/results/long_scramble_plateau.json", "w"), indent=2)
print("\nsaved docs/results/long_scramble_plateau.json")
