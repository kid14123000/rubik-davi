"""EXAM PAPER ONLY.

Full BFS over the 2x2 cube graph -> exact optimal distance for every state.

Used *only* to grade the learned value function (optimality rate, V-vs-true
MAE, demo scoring).  It must never be imported by the trainer; using it as
labels would be label injection, not learning.
"""
import os
import sys
import time
import numpy as np

import cube2 as c

N_STATES = 40320 * 2187
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist2x2.npz")


def build(chunk=200_000, verbose=True):
    dist = np.full(N_STATES, -1, dtype=np.int8)
    fp, fo = c.solved()
    dist[c.state_index(fp, fo)] = 0
    depth = 0
    t0 = time.time()
    while len(fp):
        depth += 1
        idxs = []
        for s in range(0, len(fp), chunk):
            cp, co = fp[s:s + chunk], fo[s:s + chunk]
            for a in range(c.NUM_ACTIONS):
                p2, o2 = c.apply_move(cp, co, a)
                i2 = c.state_index(p2, o2)
                keep = dist[i2] < 0
                if keep.any():
                    idxs.append(i2[keep])
        if not idxs:
            break
        allidx = np.concatenate(idxs)
        uniq = np.unique(allidx)
        dist[uniq] = depth
        # rebuild the frontier states from their indices (needed for expansion)
        fp, fo = _decode(uniq)
        if verbose:
            print(f"depth {depth:2d}  +{len(uniq):8d}  frontier={len(uniq):8d}  "
                  f"{time.time()-t0:6.1f}s", flush=True)
    return dist


_RANK = np.arange(40320, dtype=np.int64)


def _perms_table():
    """All 8! permutations in Lehmer order (cached in module)."""
    global _PERMS
    if "_PERMS" in globals():
        return _PERMS
    perms = np.zeros((40320, 8), dtype=np.int8)
    factorials = [5040, 720, 120, 24, 6, 2, 1, 1]
    for r in range(40320):
        digits = []
        rem = r
        avail = list(range(8))
        for f in factorials:
            k, rem = divmod(rem, f)
            digits.append(avail.pop(k))
        perms[r] = digits
    _PERMS = perms
    return perms


def _decode(idx):
    """index -> (perm (B,8), ori (B,8))"""
    perms = _perms_table()
    r = idx // 2187
    o = idx % 2187
    perm = perms[r].copy()
    ori = np.zeros((len(idx), 8), dtype=np.int8)
    for i in range(6, -1, -1):
        ori[:, i] = o % 3
        o = o // 3
    ori[:, 7] = (3 - ori[:, :7].sum(1) % 3) % 3      # twist sum == 0 mod 3
    return perm, ori


def load(verbose=False):
    if os.path.exists(CACHE):
        d = np.load(CACHE)["dist"]
        if verbose:
            print(f"[oracle] loaded {CACHE}")
        return d
    if verbose:
        print("[oracle] building BFS distance table (one-off, ~1-2 min)...")
    d = build(verbose=verbose)
    np.savez_compressed(CACHE, dist=d)
    return d


if __name__ == "__main__":
    d = load(verbose=True)
    counts = [(d == k).sum() for k in range(d.max() + 1)]
    print("reachable states:", int((d >= 0).sum()))
    print("max depth:", int(d.max()))
    print("histogram:", counts)
    # published 2x2 QTM distance distribution
    PUB = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508,
           930588, 1350852, 782536, 90280, 276]
    print("matches published distribution:", list(map(int, counts)) == PUB)
