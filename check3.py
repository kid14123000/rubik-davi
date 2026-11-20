"""Cross-validate cube3 against an independent implementation (pycuber).

If the 54 facelets agree for random sequences, the move tables, orientation
conventions and facelet layout are all correct.
"""
import sys
try:
    import pycuber
except ImportError:                                  # optional dev dependency
    sys.exit("pip install pycuber numpy   # external reference for verification")

import numpy as np
import pycuber as pc

import cube3 as k

FACES = "URFDLB"
_c0 = pc.Cube()                       # derive the palette from pycuber itself
CMAP = {str(_c0.get_face(f)[1][1])[1]: i for i, f in enumerate(FACES)}
IDX = {n: i for i, n in enumerate(k.MOVE_NAMES)}


def ref(seq):
    c = pc.Cube()
    if seq:
        c(" ".join(seq))
    out = np.zeros((6, 3, 3), dtype=np.int8)
    for f, name in enumerate(FACES):
        m = c.get_face(name)
        for r in range(3):
            for cc in range(3):
                out[f, r, cc] = CMAP[str(m[r][cc])[1]]
    return out


def mine(seq):
    st = k.solved()
    for nm in seq:
        st = k.apply_move(*st, IDX[nm])
    return k.facelets(st)


def ref_solved(c):
    for f in FACES:
        m = c.get_face(f)
        first = str(m[0][0])
        for row in m:
            for x in row:
                if str(x) != first:
                    return False
    return True


def main():
    rng = np.random.default_rng(0)
    bad = tot = 0
    for _ in range(60):
        n = int(rng.integers(1, 30))
        seq = [k.MOVE_NAMES[int(a)] for a in rng.integers(12, size=n)]
        a, b = ref(seq), mine(seq)
        tot += 1
        if not (a == b).all():
            bad += 1
            if bad == 1:
                print("FIRST MISMATCH seq:", seq)
                for f in range(6):
                    print("  face", FACES[f], "ref", a[f].tolist(), "mine", b[f].tolist())
    print(f"facelet agreement with pycuber: {tot - bad}/{tot} sequences")

    # group order of R U / R U' (independent reference)
    for seq in (["R", "U"], ["R", "U'"]):
        c = pc.Cube()
        got = None
        for t in range(1, 400):
            c(" ".join(seq))
            if ref_solved(c):
                got = t
                break
        print(f"pycuber order({' '.join(seq)}) =", got)
        st = k.solved()
        for t in range(1, 400):
            for nm in seq:
                st = k.apply_move(*st, IDX[nm])
            if k.is_solved(*st)[0]:
                print(f"   cube3  order({' '.join(seq)}) =", t)
                break


if __name__ == "__main__":
    main()
