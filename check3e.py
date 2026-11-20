"""Piece-level cross-check against pycuber (no facelet views involved)."""
import sys
try:
    import pycuber
except ImportError:
    sys.exit("pip install pycuber numpy   # external reference for verification")
import numpy as np
import pycuber as pc
import cube3 as k

FACES = "URFDLB"
IDX = {n: i for i, n in enumerate(k.MOVE_NAMES)}
_c0 = pc.Cube()
CMAP = {str(_c0.get_face(f)[1][1])[1]: i for i, f in enumerate(FACES)}
NORMAL_OF_FACE = {'U': (0, 1, 0), 'R': (1, 0, 0), 'F': (0, 0, 1),
                  'D': (0, -1, 0), 'L': (-1, 0, 0), 'B': (0, 0, -1)}

print("sample Square attrs:", [a for a in dir(_c0.get_face('U')[0][0]) if not a.startswith('_')])


WORD = {"yellow": 0, "orange": 1, "green": 2, "white": 3, "red": 4, "blue": 5}


def colour_of(sq):
    c = getattr(sq, "colour", None)
    if isinstance(c, (list, tuple)):
        c = c[0]
    s = str(c)
    if s.startswith("["):
        return CMAP[s[1]]
    return WORD[s]


def state_from_pycuber(c):
    cperm = np.zeros((1, 8), np.int8)
    cori = np.zeros((1, 8), np.int8)
    eperm = np.zeros((1, 12), np.int8)
    eori = np.zeros((1, 12), np.int8)
    for cb in list(c.children):
        loc = cb.location
        facings = cb.facings
        if len(loc) == 1:                                    # centre: never moves
            continue
        if len(loc) == 3:                                    # corner
            pos = tuple(sum(v) for v in zip(*[NORMAL_OF_FACE[ch] for ch in loc]))
            j = k.CORN_IDX[pos]
            cols = []
            for n in k.CORN_SLOT[j]:
                fname = [f for f, nn in NORMAL_OF_FACE.items() if nn == n][0]
                cols.append(colour_of(facings[fname]))
            piece = k._CORN_BY_SET[frozenset(cols)]
            cperm[0, j] = piece
            cori[0, j] = cols.index(int(k.CORN_PRIM_COLOR[piece]))
        else:                                                # edge
            pos = tuple(sum(v) for v in zip(*[NORMAL_OF_FACE[ch] for ch in loc]))
            j = k.EDGE_IDX[pos]
            cols = []
            for n in k.EDGE_SLOT[j]:
                fname = [f for f, nn in NORMAL_OF_FACE.items() if nn == n][0]
                cols.append(colour_of(facings[fname]))
            piece = k._EDGE_BY_SET[frozenset(cols)]
            eperm[0, j] = piece
            eori[0, j] = cols.index(int(k.EDGE_PRIM_COLOR[piece]))
    return (cperm, cori, eperm, eori)


rng = np.random.default_rng(0)
bad = 0
for t in range(40):
    n = int(rng.integers(1, 25))
    seq = [k.MOVE_NAMES[int(a)] for a in rng.integers(12, size=n)]
    c = pc.Cube()
    if seq:
        c(" ".join(seq))
    ref = state_from_pycuber(c)
    st = k.solved()
    for nm in seq:
        st = k.apply_move(*st, IDX[nm])
    ok = all((a == b).all() for a, b in zip(ref, st))
    if not ok:
        bad += 1
        if bad == 1:
            print("MISMATCH seq", seq)
            for name, a, b in (("cperm", ref[0], st[0]), ("cori", ref[1], st[1]),
                               ("eperm", ref[2], st[2]), ("eori", ref[3], st[3])):
                print("  ", name, "ref ", a.tolist(), "\n   ", name, "mine", b.tolist())
print(f"piece-level agreement: {40 - bad}/40 sequences")
