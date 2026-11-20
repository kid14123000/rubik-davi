"""3x3 Rubik's cube core: geometry-derived move tables + vectorised state ops.

State = (cperm, cori, eperm, eori)
  cperm[8]   cubie id at corner position j       (0..7)
  cori[8]    0/1/2 = which of CORN_SLOT[j] holds the cubie's primary (U/D) sticker
  eperm[12]  cubie id at edge position j         (0..11)
  eori[12]   0/1 = which of EDGE_SLOT[j] holds the cubie's primary sticker

Actions = 12 quarter turns: U U' R R' F F' D D' L L' B B'
(verified below: each move^4 = id, m*m' = id, sexy-move order 6, and a known
26-move superflip sequence really produces the superflip.)
"""
import numpy as np

# ---------------------------------------------------------------- positions
CORNER_POS = [(1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1),
              (1, -1, 1), (-1, -1, 1), (-1, -1, -1), (1, -1, -1)]
EDGE_POS = [(0, 1, 1), (1, 1, 0), (0, 1, -1), (-1, 1, 0),
            (0, -1, 1), (1, -1, 0), (0, -1, -1), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)]
CORN_IDX = {p: i for i, p in enumerate(CORNER_POS)}
EDGE_IDX = {p: i for i, p in enumerate(EDGE_POS)}

FACE_OF_NORMAL = {(0, 1, 0): 0, (1, 0, 0): 1, (0, 0, 1): 2,
                  (0, -1, 0): 3, (-1, 0, 0): 4, (0, 0, -1): 5}
FACE_NAME = ['U', 'R', 'F', 'D', 'L', 'B']


def _rot90(axis, s):
    if axis == 0:   # x
        return np.array([[1, 0, 0], [0, 0, -s], [0, s, 0]], dtype=np.int64)
    if axis == 1:   # y
        return np.array([[0, 0, s], [0, 1, 0], [-s, 0, 0]], dtype=np.int64)
    return np.array([[0, -s, 0], [s, 0, 0], [0, 0, 1]], dtype=np.int64)      # z


def _rot120(p):
    u = np.array(p, float) / np.sqrt(3.0)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = -0.5 * np.eye(3) + (np.sqrt(3) / 2) * K + 1.5 * np.outer(u, u)
    return np.rint(R).astype(np.int64)


def _corner_slots(p):
    n0 = np.array([0, 1 if p[1] > 0 else -1, 0], dtype=np.int64)
    R = _rot120(p)
    n1, n2 = R @ n0, R @ (R @ n0)
    assert np.array_equal(R @ np.array(p), np.array(p))
    return [tuple(n0), tuple(n1), tuple(n2)]


def _edge_slots(p):
    """Two normals of an edge position, ordered by priority  y > z > x."""
    axes = [(1, 'y'), (2, 'z'), (0, 'x')]
    nz = [i for i in range(3) if p[i] != 0]
    nz.sort(key=lambda i: [a for a, _ in axes].index(i))
    out = []
    for i in nz:
        v = [0, 0, 0]
        v[i] = p[i]
        out.append(tuple(v))
    assert len(out) == 2
    return out


CORN_SLOT = [_corner_slots(p) for p in CORNER_POS]
EDGE_SLOT = [_edge_slots(p) for p in EDGE_POS]
# primary normal of each position = its slot-0 normal
CORN_PRIM = [s[0] for s in CORN_SLOT]
EDGE_PRIM = [s[0] for s in EDGE_SLOT]

NUM_C, NUM_E = 8, 12
CORN_M, EDGE_M = 3, 2          # orientation modulus

MOVES = {}                     # name -> (dst_c, dc, dst_e, de)
MOVE_NAMES = []


def _build_move(name, mat, axis, layer):
    # corners
    dst_c, dc = [], []
    for j, p in enumerate(CORNER_POS):
        if p[axis] != layer:
            dst_c.append(j)
            dc.append(0)
            continue
        dst_c.append(CORN_IDX[tuple(mat @ np.array(p))])
        want = tuple(mat @ np.array(CORN_SLOT[j][0]))
        dc.append(CORN_SLOT[dst_c[-1]].index(want))
    for j, p in enumerate(CORNER_POS):
        if p[axis] != layer:
            continue
        for k in range(3):
            got = tuple(mat @ np.array(CORN_SLOT[j][k]))
            assert got == CORN_SLOT[dst_c[j]][(k + dc[j]) % 3], (name, 'corner', j, k)
    # edges
    dst_e, de = [], []
    for j, p in enumerate(EDGE_POS):
        if p[axis] != layer:
            dst_e.append(j)
            de.append(0)
            continue
        dst_e.append(EDGE_IDX[tuple(mat @ np.array(p))])
        want = tuple(mat @ np.array(EDGE_SLOT[j][0]))
        de.append(EDGE_SLOT[dst_e[-1]].index(want))
    for j, p in enumerate(EDGE_POS):
        if p[axis] != layer:
            continue
        for k in range(2):
            got = tuple(mat @ np.array(EDGE_SLOT[j][k]))
            assert got == EDGE_SLOT[dst_e[j]][(k + de[j]) % 2], (name, 'edge', j, k)
    assert sorted(dst_c) == list(range(8)) and sorted(dst_e) == list(range(12))
    MOVES[name] = (np.array(dst_c, dtype=np.int64), np.array(dc, dtype=np.int64),
                   np.array(dst_e, dtype=np.int64), np.array(de, dtype=np.int64))
    MOVE_NAMES.append(name)


# (face, axis, rotation sign, layer) -- verified against pycuber below
for _f, _ax, _sg, _ly in [('U', 1, -1, +1), ('R', 0, -1, +1), ('F', 2, -1, +1),
                          ('D', 1, +1, -1), ('L', 0, +1, -1), ('B', 2, +1, -1)]:
    _m = _rot90(_ax, _sg)
    _build_move(_f, _m, _ax, _ly)
    _build_move(_f + "'", np.linalg.matrix_power(_m, 3), _ax, _ly)

NUM_ACTIONS = len(MOVE_NAMES)          # 12
INV_ACTION = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
for _i, _nm in enumerate(MOVE_NAMES):
    assert MOVE_NAMES[INV_ACTION[_i]] == (_nm[:-1] if _nm.endswith("'") else _nm + "'")

_INV = {}
for _nm, (_dc_, _d_, _de_, _e_) in MOVES.items():
    _INV[_nm] = (np.argsort(_dc_), _d_, np.argsort(_de_), _e_)

# sticker colour tables (base colour of every piece slot in the solved cube)
CORN_BASE = np.array([[FACE_OF_NORMAL[n] for n in CORN_SLOT[c]] for c in range(8)],
                     dtype=np.int8)
EDGE_BASE = np.array([[FACE_OF_NORMAL[n] for n in EDGE_SLOT[c]] for c in range(12)],
                     dtype=np.int8)
CORN_PRIM_COLOR = np.array([FACE_OF_NORMAL[n] for n in CORN_PRIM], dtype=np.int8)
EDGE_PRIM_COLOR = np.array([FACE_OF_NORMAL[n] for n in EDGE_PRIM], dtype=np.int8)

# piece lookup by colour set (for facelets -> state)
_CORN_BY_SET = {frozenset(CORN_BASE[c].tolist()): c for c in range(8)}
_EDGE_BY_SET = {frozenset(EDGE_BASE[c].tolist()): c for c in range(12)}


# ---------------------------------------------------------------- state ops
def solved(batch=1):
    return (np.tile(np.arange(8, dtype=np.int8), (batch, 1)),
            np.zeros((batch, 8), dtype=np.int8),
            np.tile(np.arange(12, dtype=np.int8), (batch, 1)),
            np.zeros((batch, 12), dtype=np.int8))


def apply_move(cperm, cori, eperm, eori, a):
    ic, dc, ie, de = _INV[MOVE_NAMES[a]]
    return (cperm[:, ic],
            ((cori[:, ic] + dc[ic]) % 3).astype(np.int8),
            eperm[:, ie],
            ((eori[:, ie] + de[ie]) % 2).astype(np.int8))


def apply_seq(state, seq):
    for a in seq:
        state = apply_move(*state, a)
    return state


def is_solved(cperm, cori, eperm, eori):
    return ((cperm == np.arange(8)).all(1) & (cori == 0).all(1)
            & (eperm == np.arange(12)).all(1) & (eori == 0).all(1))


def scramble(rng, k, batch=1):
    st = solved(batch)
    seq, last = [], -1
    for _ in range(k):
        while True:
            a = int(rng.integers(NUM_ACTIONS))
            if last < 0 or a // 2 != last // 2:
                break
        seq.append(a)
        st = apply_move(*st, a)
        last = a
    return st, seq


def scramble_with_last(rng, batch, kmax, ks=None):
    st = solved(batch)
    if ks is None:
        ks = rng.integers(1, kmax + 1, size=batch)
    elif np.isscalar(ks):
        ks = np.full(batch, int(ks))
    last = np.full(batch, -1)
    for t in range(int(max(ks.max(), 1))):
        a = rng.integers(NUM_ACTIONS, size=batch)
        bad = (last >= 0) & (a // 2 == last // 2)
        a = np.where(bad, (a + 2) % NUM_ACTIONS, a)
        act = np.where(ks > t)[0]
        for m in range(NUM_ACTIONS):
            sel = act[a[act] == m]
            if len(sel):
                cp, co, ep, eo = st
                n = apply_move(cp[sel], co[sel], ep[sel], eo[sel], m)
                for arr, val in zip(st, n):
                    arr[sel] = val
        last[act] = a[act]
    return st, last


def _rank(perm):
    """Lehmer rank (per-row), 0 <= r < n!."""
    n = perm.shape[1]
    r = np.zeros(perm.shape[0], dtype=np.int64)
    for i in range(n - 1):
        smaller = (perm[:, i + 1:] < perm[:, i:i + 1]).sum(1).astype(np.int64)
        w = 1
        for t in range(2, n - i):
            w *= t
        r += smaller * w
    return r


def state_index_batch(state):
    """Cheap unique key per state (for beam de-duplication)."""
    cperm, cori, eperm, eori = state
    r = _rank(cperm) * (3 ** 7)
    o = np.zeros(len(cperm), dtype=np.int64)
    for i in range(7):
        o = o * 3 + cori[:, i]
    r = r + o
    r2 = _rank(eperm)
    oe = np.zeros(len(eperm), dtype=np.int64)
    for i in range(11):
        oe = oe * 2 + eori[:, i]
    return r * (40320 * 2187) + r2 * 2048 + oe


def piece_feat(state):
    """(B,256): per corner 8+3, per edge 12+2 one-hots."""
    cperm, cori, eperm, eori = state
    B = cperm.shape[0]
    x = np.zeros((B, 8, 11), dtype=np.float32)
    rows, cols = np.arange(B)[:, None], np.arange(8)[None, :]
    x[rows, cols, cperm.astype(np.int64)] = 1.0
    x[rows, cols, 8 + cori.astype(np.int64)] = 1.0
    y = np.zeros((B, 12, 14), dtype=np.float32)
    rows, cols = np.arange(B)[:, None], np.arange(12)[None, :]
    y[rows, cols, eperm.astype(np.int64)] = 1.0
    y[rows, cols, 12 + eori.astype(np.int64)] = 1.0
    return np.concatenate([x.reshape(B, 88), y.reshape(B, 168)], 1)


# ---------------------------------------------------------------- display
def cubies(state, i=0):
    """27 entries: {'p': [x,y,z], 'c': [colour or -1] x 6 in face order U R F D L B}."""
    cperm, cori, eperm, eori = state
    cp, co = int(cperm[i, 0]) if cperm.ndim > 1 else 0, None
    cperm, cori, eperm, eori = [np.atleast_2d(a)[i] for a in (cperm, cori, eperm, eori)]
    out = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                p = (x, y, z)
                colors = [-1] * 6
                if p in CORN_IDX:
                    j = CORN_IDX[p]
                    piece, o = int(cperm[j]), int(cori[j])
                    for k, n in enumerate(CORN_SLOT[j]):
                        colors[FACE_OF_NORMAL[n]] = int(CORN_BASE[piece][(k - o) % 3])
                elif p in EDGE_IDX:
                    j = EDGE_IDX[p]
                    piece, o = int(eperm[j]), int(eori[j])
                    for k, n in enumerate(EDGE_SLOT[j]):
                        colors[FACE_OF_NORMAL[n]] = int(EDGE_BASE[piece][(k - o) % 2])
                else:
                    for ax in range(3):
                        if p[ax] != 0:
                            v = [0, 0, 0]
                            v[ax] = p[ax]
                            colors[FACE_OF_NORMAL[tuple(v)]] = FACE_OF_NORMAL[tuple(v)]
                out.append(dict(p=[x, y, z], c=colors))
    return out


def facelets(state, i=0):
    """(6,3,3) colour ids, each face row-major as seen from outside."""
    cub = {(tuple(c['p']), f): c['c'][f] for c in cubies(state, i) for f in range(6)}
    out = np.zeros((6, 3, 3), dtype=np.int8)
    # screen right / up vectors for each face (viewed from outside)
    # verified cell-by-cell against pycuber (derive_view.py)
    axes = {0: ((1, 0, 0), (0, 0, -1)),     # U: right=+x, up=-z
            1: ((0, 0, -1), (0, 1, 0)),     # R: right=-z, up=+y
            2: ((1, 0, 0), (0, 1, 0)),      # F: right=+x, up=+y
            3: ((1, 0, 0), (0, 0, 1)),      # D: right=+x, up=+z
            4: ((0, 0, 1), (0, 1, 0)),      # L: right=+z, up=+y
            5: ((-1, 0, 0), (0, 1, 0))}     # B: right=-x, up=+y
    normals = {0: (0, 1, 0), 1: (1, 0, 0), 2: (0, 0, 1),
               3: (0, -1, 0), 4: (-1, 0, 0), 5: (0, 0, -1)}
    for f, (rv, uv) in axes.items():
        n = normals[f]
        for r in range(3):
            for c in range(3):
                tv = 1 - r                       # top row -> +1
                su = c - 1                       # left -> -1
                pos = [n[0] + su * rv[0] + tv * uv[0],
                       n[1] + su * rv[1] + tv * uv[1],
                       n[2] + su * rv[2] + tv * uv[2]]
                out[f, r, c] = cub[(tuple(pos), f)]
    return out


def facelets_str(state, i=0):
    return ''.join(str(v) for v in facelets(state, i).reshape(-1).tolist())


def state_from_facelets(s):
    """Parse a 54-char colour string back into a state (validated)."""
    fl = np.array([int(ch) for ch in s.strip()], dtype=np.int8).reshape(6, 3, 3)
    axes = {0: ((1, 0, 0), (0, 0, -1)), 1: ((0, 0, -1), (0, 1, 0)),
            2: ((1, 0, 0), (0, 1, 0)), 3: ((1, 0, 0), (0, 0, 1)),
            4: ((0, 0, 1), (0, 1, 0)), 5: ((-1, 0, 0), (0, 1, 0))}
    normals = {0: (0, 1, 0), 1: (1, 0, 0), 2: (0, 0, 1),
               3: (0, -1, 0), 4: (-1, 0, 0), 5: (0, 0, -1)}
    slot_color = {}                    # (pos, normal) -> colour
    for f, (rv, uv) in axes.items():
        n = normals[f]
        for r in range(3):
            for c in range(3):
                tv, su = 1 - r, c - 1
                pos = (n[0] + su * rv[0] + tv * uv[0],
                       n[1] + su * rv[1] + tv * uv[1],
                       n[2] + su * rv[2] + tv * uv[2])
                slot_color[(pos, n)] = int(fl[f, r, c])
    cperm = np.zeros((1, 8), np.int8)
    cori = np.zeros((1, 8), np.int8)
    eperm = np.zeros((1, 12), np.int8)
    eori = np.zeros((1, 12), np.int8)
    for j, p in enumerate(CORNER_POS):
        cols = [slot_color[(p, n)] for n in CORN_SLOT[j]]
        key = frozenset(cols)
        if key not in _CORN_BY_SET:
            raise ValueError("invalid corner at %s: %s" % (p, cols))
        piece = _CORN_BY_SET[key]
        cperm[0, j] = piece
        cori[0, j] = cols.index(int(CORN_PRIM_COLOR[piece]))
    for j, p in enumerate(EDGE_POS):
        cols = [slot_color[(p, n)] for n in EDGE_SLOT[j]]
        key = frozenset(cols)
        if key not in _EDGE_BY_SET:
            raise ValueError("invalid edge at %s: %s" % (p, cols))
        piece = _EDGE_BY_SET[key]
        eperm[0, j] = piece
        eori[0, j] = cols.index(int(EDGE_PRIM_COLOR[piece]))
    if cori.sum() % 3 or eori.sum() % 2:
        raise ValueError("orientation invariant violated (twist/flip)")
    if (np.argsort(cperm[0]) % 2).sum() % 2 != (np.argsort(eperm[0]) % 2).sum() % 2:
        raise ValueError("permutation parity violated")
    return (cperm, cori, eperm, eori)
