"""How many moves does our trained solver actually use?

Measures with the shipped 500k weight at beam 1024:
  A) 20-move scrambles            -> solve rate + mean length (solved only)
  B) uniform depth k ~ U[1,20]     -> same + per-depth breakdown

Every returned solution is replayed and checked with is_solved.

Notes on metrics:
  * our move set is QTM (12 quarter turns), so lengths are QTM;
  * also reported as HTM after merging same-face quarter turns (QQ -> 2);
  * pycuber's solver is CFOP (a human method, counts whole-cube rotations),
    so it is a sanity reference only -- NOT an optimality bound.
"""
import sys
sys.path.append("/Users/kid/venvs/cube/lib/python3.12/site-packages")

import contextlib
import io
import json
import time

import numpy as np
import torch

import cube3 as k3
import model3 as M3
import scorecard3 as SC3

W = 1024
ck = torch.load("weights/rubik3x3_500k.pt", map_location="cpu", weights_only=False)
net = M3.DualNet3(ck["meta"]["hidden"])
net.load_state_dict(ck["state"])
net.eval()
print(f"[model] weights/rubik3x3_500k.pt it={ck['meta']['it']} beam={W}", flush=True)

try:
    import pycuber as pc
    from pycuber.solver import CFOPSolver
    HAVE_PC = True
except ImportError:
    HAVE_PC = False
    print("[pycuber] unavailable; skipping the CFOP reference", flush=True)


def qtm_to_htm(path):
    """count moves in the half-turn metric: QQ on one face = one double turn."""
    seq = [k3.MOVE_NAMES[a] for a in path] if path and isinstance(path[0], (int, np.integer)) else list(path)
    seq = [s for s in seq]
    changed = True
    while changed:
        changed = False
        out = []
        i = 0
        while i < len(seq):
            cur = seq[i]
            if i + 1 < len(seq) and seq[i + 1][0] == cur[0]:
                a, b = cur, seq[i + 1]
                if a.endswith("'") == b.endswith("'"):      # same direction: QQ -> double
                    out.append(cur[0] + "2")
                # opposite directions (U U') cancel -> append nothing
                i += 2
                changed = True
            else:
                out.append(cur)
                i += 1
        seq = out
    return len(seq)


def solve_ours(st):
    ln, ok, path = SC3.beam_one(net, "cpu", st, width=W, mode="policy",
                                max_steps=45, want_path=True)
    if ok:                                    # replay-verify; never trust the flag
        st2 = st
        for a in path:
            st2 = k3.apply_move(*st2, a)
        ok = bool(k3.is_solved(*st2)[0])
    return int(ln), bool(ok), (qtm_to_htm(path) if ok else None)


def solve_cfop(seq_names):
    if not HAVE_PC:
        return None
    c = pc.Cube()
    if seq_names:
        c(" ".join(seq_names))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sol = CFOPSolver(c).solve()
        return len([str(x) for x in sol])
    except Exception:                                              # noqa: BLE001
        return None


def batch(depths, seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in depths:
        st, seq = k3.scramble(rng, int(k), 1)
        out.append((int(k), st, [k3.MOVE_NAMES[a] for a in seq]))
    return out


def run(label, items):
    print(f"\n=== {label} (n={len(items)}) ===", flush=True)
    rows = []
    t0 = time.time()
    for k, st, seq in items:
        ln, ok, htm = solve_ours(st)
        rows.append(dict(depth=k, qtm=ln if ok else None, htm=htm, ok=ok,
                         scramble=k))
        print(f"  depth {k:2d}  {'FAIL' if not ok else 'qtm %-3d htm %-3d' % (ln, htm)}", flush=True)
    ok = [r for r in rows if r["ok"]]
    q = np.array([r["qtm"] for r in ok])
    h = np.array([r["htm"] for r in ok])
    print(f"  -> solved {len(ok)}/{len(rows)} ({100*len(ok)/len(rows):.0f}%)   "
          f"({time.time()-t0:.1f}s total)")
    if len(ok):
        print(f"  -> mean  QTM {q.mean():.2f}  HTM {h.mean():.2f}")
        print(f"  -> median QTM {np.median(q):.1f}  HTM {np.median(h):.1f}   "
              f"range QTM {q.min()}..{q.max()}")
        if len(ok) < len(rows):
            print("  -> NOTE: the mean is over solved cubes only; "
                  f"{len(rows)-len(ok)} unsolved cubes are excluded (not counted as infinity)")
    return rows


out = {}
items20 = batch([20] * 40, seed=777001)
out["A_20move"] = run("A) 20-move QTM scrambles", items20)
cf = [solve_cfop(seq) for _, _, seq in items20[:20]]
cf = [x for x in cf if x]
print(f"\n  [ref] pycuber CFOP on 20 of the same cubes: mean {np.mean(cf):.1f} moves "
      f"(human method, includes y/y2 rotations)") if cf else None

mixed = [int(x) for x in np.random.default_rng(777002).integers(1, 21, size=60)]
out["B_mixed"] = run("B) arbitrary depth, k ~ U[1,20]", batch(mixed, seed=777003))

# per-depth summary for B
print("\n=== B per depth ===")
print(f"  {'depth':>5} {'n':>3} {'solved':>7} {'QTM mean':>9} {'HTM mean':>9}")
for k in sorted({r["depth"] for r in out["B_mixed"]}):
    rs = [r for r in out["B_mixed"] if r["depth"] == k]
    ok = [r for r in rs if r["ok"]]
    q = f"{np.mean([r['qtm'] for r in ok]):.1f}" if ok else "-"
    h = f"{np.mean([r['htm'] for r in ok]):.1f}" if ok else "-"
    print(f"  {k:5d} {len(rs):3d} {len(ok):3d}/{len(rs):<3d} {q:>9} {h:>9}")

json.dump(out, open("docs/results/avg_solution_length.json", "w"), indent=2)
print("\nsaved docs/results/avg_solution_length.json")
