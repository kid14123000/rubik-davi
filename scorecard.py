"""Scorecard: grades the learned nets against exact BFS distances.

The oracle (dist2x2.npz) is the *exam paper* only -- nothing here feeds a loss.
"""
import numpy as np
import torch

import cube2
import model as M

NA = cube2.NUM_ACTIONS
_ORACLE = None
_OK = None


def oracle():
    global _ORACLE
    if _ORACLE is None:
        import oracle as _o
        _ORACLE = _o.load()
    return _ORACLE


def _ok_indices():
    global _OK
    if _OK is None:
        _OK = np.where(oracle() >= 0)[0]
    return _OK


def test_set(n=300, seed=12345):
    """Uniformly random states -- the right definition of 'random scramble'."""
    import oracle as _o
    rng = np.random.default_rng(seed)
    idx = rng.choice(_ok_indices(), size=n, replace=False)
    return _o._decode(np.sort(idx))


def optimal_len(perm, ori):
    return oracle()[cube2.state_index(perm, ori)].astype(np.int32)


def policy_acc(net, dev, n=400, seed=7, kmax=14):
    """Self-supervised accuracy: does it undo the last scrambling move?"""
    rng = np.random.default_rng(seed)
    p, o, last = cube2.scramble_with_last(rng, n, kmax)
    lg = M.policy(p, o, net, dev)
    return float((lg.argmax(1) == np.array(cube2.INV_ACTION)[last]).mean())


def _log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


# ---------------------------------------------------------------- solvers
def solve_greedy_policy(net, dev, perm, ori, max_steps=30):
    """Execute argmax policy.  Weak on its own -- useful as a baseline."""
    B = len(perm)
    cur_p, cur_o = perm.copy(), ori.copy()
    left = np.ones(B, bool)
    lens = np.zeros(B, np.int32)
    for _ in range(max_steps):
        idx = np.where(left)[0]
        if len(idx) == 0:
            break
        a = M.policy(cur_p[idx], cur_o[idx], net, dev).argmax(1)
        for m in range(NA):
            sel = idx[a == m]
            if len(sel):
                p2, o2 = cube2.apply_move(cur_p[sel], cur_o[sel], m)
                cur_p[sel], cur_o[sel] = p2, o2
        lens[idx] += 1
        ok = cube2.is_solved(cur_p[idx], cur_o[idx])
        left[idx[ok]] = False
    return lens, ~left


def beam_one(net, dev, p, o, width=64, max_steps=20, mode="value", want_path=False):
    """Beam search from a single state.

    mode='value' : f = g + V(s)      (DeepCubeA / BWAS style)
    mode='policy': f = -sum log p(a) (AlphaCube style)
    """
    P, O = p[None].copy(), o[None].copy()
    G = np.zeros(1, np.float32)
    C = np.zeros(1, np.float32)
    paths = [[]] if want_path else None
    L = M.policy(P, O, net, dev)
    for step in range(1, max_steps + 1):
        cp = np.concatenate([cube2.apply_move(P, O, a)[0] for a in range(NA)])
        co = np.concatenate([cube2.apply_move(P, O, a)[1] for a in range(NA)])
        W = len(P)
        cg = np.tile(G + 1.0, NA)
        cc = (C[:, None] - _log_softmax(L)).T.reshape(-1)
        ok = cube2.is_solved(cp, co)
        if ok.any():
            i = int(np.where(ok)[0][0])
            parent, action = i % W, i // W      # cp is laid out [action][parent]
            return step, True, (paths[parent] + [action] if want_path else None)
        v, lg = M.predict(cp, co, net, dev)
        f = (cg + v) if mode == "value" else cc
        order = np.argsort(f)[: width * 3]
        _, first = np.unique(cube2.state_index(cp[order], co[order]), return_index=True)
        keep = order[np.sort(first)][:width]
        if want_path:
            paths = [paths[i % W] + [i // W] for i in keep]
        P, O = cp[keep], co[keep]
        G, C, L = (cg[keep], cc[keep], lg[keep]) if mode == "policy" else (cg[keep], cc[keep], lg[keep])
    return max_steps, False, (paths[0] if want_path else None)


def beam(net, dev, perm, ori, width=64, max_steps=20, mode="value"):
    B = len(perm)
    lens = np.zeros(B, np.int32)
    solved = np.zeros(B, bool)
    for b in range(B):
        ln, ok, _ = beam_one(net, dev, perm[b], ori[b], width, max_steps, mode)
        lens[b], solved[b] = ln, ok
    return lens, solved


# ---------------------------------------------------------------- scoring
def score(net, dev, n=200, seed=12345, do_beam=True, beam_n=60, width=64,
          mode="value"):
    perm, ori = test_set(n, seed=seed)
    opt = optimal_len(perm, ori)
    v = M.value(perm, ori, net, dev)
    mae = float(np.abs(v - opt).mean())
    corr = float(np.corrcoef(v, opt)[0, 1]) if v.std() > 1e-6 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    out = dict(n=int(n), v_mae=mae, v_corr=corr, p_acc=policy_acc(net, dev),
               opt_len=float(opt.mean()))
    gl, gs = solve_greedy_policy(net, dev, perm, ori)
    out.update(greedy_solved=float(gs.mean()),
               greedy_len=float(gl[gs].mean()) if gs.any() else 0.0)
    if do_beam:
        k = min(beam_n, n)
        bl, bs = beam(net, dev, perm[:k], ori[:k], width=width, mode=mode)
        out.update(beam_n=int(k), beam_width=int(width), beam_mode=mode,
                   beam_solved=float(bs.mean()),
                   beam_len=float(bl[bs].mean()) if bs.any() else 0.0,
                   beam_optimal=float((bl[bs] == opt[:k][bs]).mean()) if bs.any() else 0.0,
                   beam_excess=float((bl[bs] - opt[:k][bs]).mean()) if bs.any() else 0.0)
    return out


def demo(net, dev, seed=0, k=14, mode="value", width=64):
    """One scramble + the solver's actual path, for the web UI."""
    rng = np.random.default_rng(seed)
    p, o, seq = cube2.scramble(rng, k, 1)
    opt0 = int(optimal_len(p, o)[0])
    faces = lambda pp, oo: cube2.face_grid(cube2.slot_colors(pp, oo))[0][0].tolist()
    ln, sol, path = beam_one(net, dev, p[0], o[0], width=width, mode=mode, want_path=True)
    steps = [dict(i=0, move=None, v=float(M.value(p, o, net, dev)[0]), faces=faces(p, o))]
    cp, co = p, o
    for t, a in enumerate(path or []):
        cp, co = cube2.apply_move(cp, co, a)
        steps.append(dict(i=t + 1, move=cube2.MOVE_NAMES[a],
                          v=float(M.value(cp, co, net, dev)[0]), faces=faces(cp, co)))
    return dict(scramble=[cube2.MOVE_NAMES[a] for a in seq], steps=steps,
                solved=bool(sol), length=ln, optimal=opt0)
