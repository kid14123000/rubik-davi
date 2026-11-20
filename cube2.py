"""2x2 Rubik's cube core: geometry-derived move tables, vectorised state ops.

State = (perm, ori)
  perm[j] = cubie id sitting at position j   (0..7)
  ori[j]  = 0/1/2  = which of ord[j] holds the cubie's U/D sticker

Corners (position -> sign vector):
  0 URF (+1,+1,+1)   1 UFL (-1,+1,+1)   2 ULB (-1,+1,-1)   3 UBR (+1,+1,-1)
  4 DFR (+1,-1,+1)   5 DLF (-1,-1,+1)   6 DBL (-1,-1,-1)   7 DRB (+1,-1,-1)

Move set = QTM on {U, R, F} x {quarter, prime} = 6 actions (D,L,B are whole-cube
rotations of these on a 2x2, so they add nothing).
"""
import numpy as np

POS = [(1, 1, 1), (-1, 1, 1), (-1, 1, -1), (1, 1, -1),
       (1, -1, 1), (-1, -1, 1), (-1, -1, -1), (1, -1, -1)]
POSI = {p: i for i, p in enumerate(POS)}

# colour id per face normal
NORMAL_COLOR = {(0, 1, 0): 0, (1, 0, 0): 1, (0, 0, 1): 2,
                (0, -1, 0): 3, (-1, 0, 0): 4, (0, 0, -1): 5}
COLOR_NAME = ['U', 'R', 'F', 'D', 'L', 'B']


def _rot90(axis, s):
    """+90 deg about axis (s=+1) as an integer 3x3 matrix."""
    c, sn = 0, s
    if axis == 0:   # x: (x,y,z) -> (x, -s*z, s*y)
        return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]], dtype=np.int64)
    if axis == 1:   # y: (x,y,z) -> (s*z, y, -s*x)
        return np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]], dtype=np.int64)
    return np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=np.int64)  # z


def _rot120(p):
    """+120 deg about the corner axis p (Rodrigues, rounded to ints)."""
    u = np.array(p, float) / np.sqrt(3.0)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = -0.5 * np.eye(3) + (np.sqrt(3) / 2) * K + 1.5 * np.outer(u, u)
    return np.rint(R).astype(np.int64)


# ord[j] = [U/D normal, +120deg, +240deg]  -> canonical 3-cycle per corner
ORD = []
for p in POS:
    n0 = np.array([0, 1 if p[1] > 0 else -1, 0], dtype=np.int64)
    R = _rot120(p)
    n1 = R @ n0
    n2 = R @ n1
    assert np.array_equal(R @ np.array(p), np.array(p))
    ORD.append([tuple(n0), tuple(n1), tuple(n2)])

MOVES = {}      # name -> (dst[j] position, d[j] ori delta)
MOVE_NAMES = []


def _build_move(name, mat, axis, layer):
    """A face turn only moves the 4 corners in the `axis/layer` slice."""
    dst, d = [], []
    for j in range(8):
        if POS[j][axis] != layer:          # fixed half of the cube
            dst.append(j)
            d.append(0)
            continue
        dst.append(POSI[tuple(mat @ np.array(POS[j]))])
        want = tuple(mat @ np.array(ORD[j][0]))
        d.append(ORD[dst[-1]].index(want))
    # verify the offset is a pure cyclic shift for all three slots
    for j in range(8):
        if POS[j][axis] != layer:
            continue
        for k in range(3):
            got = tuple(mat @ np.array(ORD[j][k]))
            assert got == ORD[dst[j]][(k + d[j]) % 3], (name, j, k)
    # verify dst is a permutation
    assert sorted(dst) == list(range(8))
    MOVES[name] = (np.array(dst, dtype=np.int64), np.array(d, dtype=np.int64))
    MOVE_NAMES.append(name)


for _nm, _ax, _sg in [('U', 1, -1), ('R', 0, +1), ('F', 2, -1)]:
    _m = _rot90(_ax, _sg)
    _build_move(_nm, _m, _ax, +1)
    _build_move(_nm + "'", np.linalg.matrix_power(_m, 3), _ax, +1)

NUM_ACTIONS = len(MOVE_NAMES)          # 6

# pre-compute inverse-position orderings for vectorised scatter
_INV = {nm: (np.argsort(dst), d) for nm, (dst, d) in MOVES.items()}


def solved(batch=1):
    perm = np.tile(np.arange(8, dtype=np.int8), (batch, 1))
    ori = np.zeros((batch, 8), dtype=np.int8)
    return perm, ori


def apply_move(perm, ori, a):
    """Vectorised move on (B,8) perm/ori arrays."""
    inv, d = _INV[MOVE_NAMES[a]]
    return perm[:, inv], ((ori[:, inv] + d[inv]) % 3).astype(np.int8)


def apply_seq(perm, ori, seq):
    for a in seq:
        perm, ori = apply_move(perm, ori, a)
    return perm, ori


def is_solved(perm, ori):
    return (perm == np.arange(8, dtype=np.int8)).all(1) & (ori == 0).all(1)


def scramble(rng, k, batch=1):
    """k random QTM moves, no immediate undo / no same face twice."""
    perm, ori = solved(batch)
    seq = []
    last = -1
    for _ in range(k):
        while True:
            a = int(rng.integers(NUM_ACTIONS))
            if last < 0 or (a // 2 != last // 2):
                break
        seq.append(a)
        perm, ori = apply_move(perm, ori, a)
        last = a
    return perm, ori, seq


def scramble_mixed(rng, batch, kmax, k=None):
    """Independent random walk per sample (this is what a batch must be).

    k: int -> same depth for the whole batch but *different* move sequences;
       array -> per-sample depth; None -> per-sample depth ~ U[1, kmax].
    """
    perm, ori = solved(batch)
    if k is None:
        ks = rng.integers(1, kmax + 1, size=batch)
    elif np.isscalar(k):
        ks = np.full(batch, int(k))
    else:
        ks = np.asarray(k)
    last = np.full(batch, -1)
    for t in range(int(ks.max())):
        act = rng.integers(NUM_ACTIONS, size=batch)
        bad = (last >= 0) & (act // 2 == last // 2)
        act = np.where(bad, (act + 2) % NUM_ACTIONS, act)   # hop to another face
        active = np.where(ks > t)[0]
        for a in range(NUM_ACTIONS):
            sel = active[act[active] == a]
            if len(sel):
                p2, o2 = apply_move(perm[sel], ori[sel], a)
                perm[sel], ori[sel] = p2, o2
        last[active] = act[active]
    return perm, ori, ks


def scramble_with_last(rng, batch, kmax, ks=None):
    """One walk per sample, returning the LAST action taken (self-supervised
    policy label: the move that undoes it is INV[last])."""
    perm, ori = solved(batch)
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
                p2, o2 = apply_move(perm[sel], ori[sel], m)
                perm[sel], ori[sel] = p2, o2
        last[act] = a[act]
    return perm, ori, last


INV_ACTION = [1, 0, 3, 2, 5, 4]      # index of the inverse of each action


_FACT = [1, 1, 2, 6, 24, 120, 720, 5040]


def perm_rank(perm):
    """Lehmer rank of (B,8) permutation, 0..40319."""
    B = perm.shape[0]
    rank = np.zeros(B, dtype=np.int64)
    for i in range(8):
        c = (perm[:, i + 1:] < perm[:, i:i + 1]).sum(1)
        rank += c.astype(np.int64) * _FACT[7 - i]
    return rank


def state_index(perm, ori):
    """Dense index in [0, 8!*3^7); only reachable states are used."""
    r = perm_rank(perm)
    o = np.zeros(perm.shape[0], dtype=np.int64)
    for i in range(7):
        o = o * 3 + ori[:, i]
    return r * 2187 + o


def slot_colors(perm, ori):
    """(B,8,3) colour id of the sticker sitting on ord[j][k]."""
    B = perm.shape[0]
    base = np.array([[NORMAL_COLOR[n] for n in ORD[c]] for c in range(8)], dtype=np.int8)
    out = np.empty((B, 8, 3), dtype=np.int8)
    for j in range(8):
        for k in range(3):
            idx = (k - ori[:, j]) % 3                      # (B,)
            out[:, j, k] = base[perm[:, j], idx]
    return out


def face_grid(col):
    """6 faces x 2x2 grid of colour ids for display.  col: (B,8,3).

    Layout of the unfolded net:   U
                                L F R B
                                  D
    """
    from collections import defaultdict
    cell = defaultdict(list)          # (face, row, col) -> append
    # screen right / up axes of each face in the unfolded net
    axes = {(0, 1, 0): ((1, 0, 0), (0, 0, -1)),   # U
            (1, 0, 0): ((0, 0, 1), (0, 1, 0)),    # R
            (0, 0, 1): ((1, 0, 0), (0, 1, 0)),    # F
            (0, -1, 0): ((1, 0, 0), (0, 0, 1)),   # D
            (-1, 0, 0): ((0, 0, -1), (0, 1, 0)),  # L
            (0, 0, -1): ((-1, 0, 0), (0, 1, 0)),  # B
            }
    for j in range(8):
        p = np.array(POS[j])
        for k, n in enumerate(ORD[j]):
            u, v = axes[n]
            cc = 1 if p @ np.array(u) > 0 else 0
            rr = 0 if p @ np.array(v) > 0 else 1
            cell[(n, rr, cc)].append((j, k))
    order = [(0, 1, 0), (-1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, -1), (0, -1, 0)]
    out = np.zeros((col.shape[0], 6, 2, 2), dtype=np.int8)
    for f, n in enumerate(order):
        for rr in range(2):
            for cc in range(2):
                j, k = cell[(n, rr, cc)][0]
                out[:, f, rr, cc] = col[:, j, k]
    return out, [COLOR_NAME[NORMAL_COLOR[n]] for n in order]


def corner_feat(perm, ori):
    """(B,88): per position, one-hot of which cubie sits there + its twist.

    This is the natural 'which piece is where' encoding (the one DeepCubeA and
    AlphaCube use); raw sticker one-hot has no locality, so an MLP cannot
    generalise from it.
    """
    B = perm.shape[0]
    x = np.zeros((B, 8, 11), dtype=np.float32)
    rows = np.arange(B)[:, None]
    cols = np.arange(8)[None, :]
    x[rows, cols, perm.astype(np.int64)] = 1.0
    x[rows, cols, 8 + ori.astype(np.int64)] = 1.0
    return x.reshape(B, 88)


def one_hot(perm, ori):
    """(B,144) sticker one-hot: 24 slots x 6 colours. No hand features."""
    col = slot_colors(perm, ori).reshape(perm.shape[0], 24).astype(np.int64)
    out = np.zeros((perm.shape[0], 144), dtype=np.float32)
    out[np.arange(perm.shape[0])[:, None], np.arange(24)[None, :] * 6 + col] = 1.0
    return out
