"""Derive the exact facelet view axes from pycuber, by colour matching.

For each face we try all 8 (right, up) in-plane axis pairs and keep the one
that reproduces pycuber's 3x3 colour grid on scrambled cubes.
"""
import sys
try:
    import pycuber
except ImportError:
    sys.exit("pip install pycuber numpy   # external reference for verification")
import itertools
import numpy as np
import pycuber as pc
import cube3 as k

FACES = "URFDLB"
IDX = {n: i for i, n in enumerate(k.MOVE_NAMES)}
_c0 = pc.Cube()
CMAP = {str(_c0.get_face(f)[1][1])[1]: i for i, f in enumerate(FACES)}
NORMALS = {0: (0, 1, 0), 1: (1, 0, 0), 2: (0, 0, 1),
           3: (0, -1, 0), 4: (-1, 0, 0), 5: (0, 0, -1)}
UNITS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def color_at(state, pos, n):
    """colour sitting on `pos` facing normal `n`."""
    cub = k.cubies(state, 0)
    for c in cub:
        if tuple(c['p']) == pos:
            f = k.FACE_OF_NORMAL[n]
            return c['c'][f]
    raise KeyError(pos)


def ref_state(seq):
    c = pc.Cube()
    if seq:
        c(" ".join(seq))
    return c


def ref_grid(c, f):
    m = c.get_face(FACES[f])
    return np.array([[CMAP[str(m[r][cc])[1]] for cc in range(3)] for r in range(3)])


def my_state(seq):
    st = k.solved()
    for nm in seq:
        st = k.apply_move(*st, IDX[nm])
    return tuple(np.array(x) for x in st)


rng = np.random.default_rng(1)
scrambles = [[k.MOVE_NAMES[int(a)] for a in rng.integers(12, size=int(rng.integers(12, 22)))]
             for _ in range(3)]

cands = {}
for f in range(6):
    n = NORMALS[f]
    inplane = [u for u in UNITS if abs(sum(a * b for a, b in zip(u, n))) == 0]
    pairs = [(rv, uv) for rv in inplane for uv in inplane
             if rv != uv and rv != tuple(-x for x in uv)]
    good = []
    for rv, uv in pairs:
        ok = True
        for seq in scrambles:
            c = ref_state(seq)
            st = my_state(seq)
            g = ref_grid(c, f)
            for r in range(3):
                for cc in range(3):
                    tv, su = 1 - r, cc - 1
                    pos = (n[0] + su * rv[0] + tv * uv[0],
                           n[1] + su * rv[1] + tv * uv[1],
                           n[2] + su * rv[2] + tv * uv[2])
                    if color_at(st, pos, n) != g[r, cc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            good.append((rv, uv))
    cands[f] = good
    print(FACES[f], "candidates:", good)

print()
print("axes = {", ", ".join(f"{f}: (({rv[0]}, {rv[1]}, {rv[2]}), ({uv[0]}, {uv[1]}, {uv[2]}))"
                            for f, (rv, uv) in ((f, cands[f][0]) for f in range(6) if cands[f])), "}")

# now decide move directions with the derived axes
AX = {f: cands[f][0] for f in range(6)}


def mine_grid(st, f):
    n = NORMALS[f]
    rv, uv = AX[f]
    out = np.zeros((3, 3), dtype=np.int8)
    for r in range(3):
        for cc in range(3):
            tv, su = 1 - r, cc - 1
            pos = (n[0] + su * rv[0] + tv * uv[0],
                   n[1] + su * rv[1] + tv * uv[1],
                   n[2] + su * rv[2] + tv * uv[2])
            out[r, cc] = color_at(st, pos, n)
    return out


print("\nmove direction check (mine(f) vs ref(f), mine(f') vs ref(f)):")
for f in FACES:
    a = ref_state([f])
    st1 = my_state([f])
    st2 = my_state([f + "'"])
    same = all(np.array_equal(mine_grid(st1, x), ref_grid(a, x)) for x in range(6))
    inv = all(np.array_equal(mine_grid(st2, x), ref_grid(a, x)) for x in range(6))
    print(f"  {f}: direct={same}  inverted={inv}")
