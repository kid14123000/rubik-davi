"""Scorecard for 3x3: no oracle possible (4.3e19 states) -- solve rate is the
honest, exactly verifiable metric, since we replay every solution."""
import numpy as np
import torch

import cube3 as k3
import model3 as M

NA = k3.NUM_ACTIONS


def test_set(n=40, kmax=20, seed=12345):
    rng = np.random.default_rng(seed)
    st, _ = k3.scramble(rng, 0, n)             # placeholder
    states, lengths = [], []
    for _ in range(n):
        k = int(rng.integers(1, kmax + 1))
        s, seq = k3.scramble(rng, k, 1)
        states.append(s)
        lengths.append(len(seq))
    return tuple(np.concatenate([s[i] for s in states], 0) for i in range(4)), np.array(lengths)


def policy_acc(net, dev, n=512, seed=7, kmax=20):
    rng = np.random.default_rng(seed)
    st, last = k3.scramble_with_last(rng, n, kmax)
    lg = M.predict(st, net, dev)[1]
    return float((lg.argmax(1) == np.array(k3.INV_ACTION)[last]).mean())


def _log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def beam_one(net, dev, state, width=64, max_steps=40, mode="policy", want_path=False,
             return_frontier=False):
    """Beam over states.  mode='policy': f = -sum log p;  mode='value': f = g + V."""
    P = state
    G = np.zeros(1, np.float32)
    C = np.zeros(1, np.float32)
    paths = [[]] if want_path else None
    L = M.predict(P, net, dev)[1]
    for step in range(1, max_steps + 1):
        W = P[0].shape[0]
        succ = [k3.apply_move(*P, a) for a in range(NA)]
        cp = tuple(np.concatenate([s[i] for s in succ], 0) for i in range(4))
        cg = np.tile(G + 1.0, NA)
        cc = (C[:, None] - _log_softmax(L)).T.reshape(-1)
        ok = k3.is_solved(*cp)
        if ok.any():
            i = int(np.where(ok)[0][0])
            parent, action = i % W, i // W
            path = paths[parent] + [action] if want_path else None
            if return_frontier:
                return step, True, path, None
            return step, True, path
        v, lg = M.predict(cp, net, dev)
        f = (cg + v) if mode == "value" else cc
        order = np.argsort(f)[: width * 3]
        sub_idx = k3.state_index_batch(tuple(x[order] for x in cp))
        _, first = np.unique(sub_idx, return_index=True)
        keep = order[np.sort(first)][:width]
        if want_path:
            paths = [paths[i % W] + [i // W] for i in keep]
        P = tuple(x[keep] for x in cp)
        G, C, L = cg[keep], cc[keep], lg[keep]
    if return_frontier:
        return max_steps, False, (paths[0] if want_path else None), (P, paths)
    return max_steps, False, (paths[0] if want_path else None)


def solve_iterative(net, dev, state, width=128, steps_per_round=20, rounds=8,
                    mode="policy"):
    """Beam search with restarts: when a round gets stuck, jump to the frontier
    node with the lowest value estimate and search again."""
    cur = state
    moves = []
    for r in range(rounds):
        ln, ok, path, front = beam_one(net, dev, cur, width=width,
                                       max_steps=steps_per_round, mode=mode,
                                       want_path=True, return_frontier=True)
        if ok:
            moves += path
            return True, moves, r + 1
        P, paths = front
        v = M.value(P, net, dev)
        j = int(np.argmin(v))
        moves += paths[j]
        cur = tuple(x[j:j + 1].copy() for x in P)
    return False, moves, rounds


def beam(net, dev, states, width=64, max_steps=40, mode="policy"):
    n = states[0].shape[0]
    lens = np.zeros(n, np.int32)
    solved = np.zeros(n, bool)
    for b in range(n):
        one = tuple(x[b:b + 1] for x in states)
        ln, ok, _ = beam_one(net, dev, one, width, max_steps, mode)
        lens[b], solved[b] = ln, ok
    return lens, solved


def greedy_policy(net, dev, states, max_steps=40):
    n = states[0].shape[0]
    cur = tuple(x.copy() for x in states)
    left = np.ones(n, bool)
    lens = np.zeros(n, np.int32)
    for _ in range(max_steps):
        idx = np.where(left)[0]
        if len(idx) == 0:
            break
        a = M.predict(M.index_state(cur, idx), net, dev)[1].argmax(1)
        for m in range(NA):
            sel = idx[a == m]
            if len(sel):
                sub = k3.apply_move(*M.index_state(cur, sel), m)
                for arr, val in zip(cur, sub):
                    arr[sel] = val
        lens[idx] += 1
        ok = k3.is_solved(*M.index_state(cur, idx))
        left[idx[ok]] = False
    return lens, ~left


def score(net, dev, n=40, do_beam=True, beam_n=20, width=64, mode="policy", seed=12345):
    states, scr_len = test_set(n, seed=seed)
    gl, gs = greedy_policy(net, dev, states)
    out = dict(n=int(n), p_acc=policy_acc(net, dev), v_mean=float(M.value(states, net, dev).mean()),
               scramble_len=float(scr_len.mean()),
               greedy_solved=float(gs.mean()),
               greedy_len=float(gl[gs].mean()) if gs.any() else 0.0)
    if do_beam:
        m = min(beam_n, n)
        bl, bs = beam(net, dev, tuple(x[:m] for x in states), width=width, mode=mode)
        out.update(beam_n=int(m), beam_width=int(width), beam_mode=mode,
                   beam_solved=float(bs.mean()),
                   beam_len=float(bl[bs].mean()) if bs.any() else 0.0,
                   beam_max=int(bl.max()) if bs.any() else 0)
    return out


def demo(net, dev, seed=0, k=20, mode="policy", width=64, max_steps=40):
    rng = np.random.default_rng(seed)
    st, seq = k3.scramble(rng, k, 1)
    ln, sol, path = beam_one(net, dev, st, width=width, mode=mode,
                             max_steps=max_steps, want_path=True)
    return dict(scramble=[k3.MOVE_NAMES[a] for a in seq],
                moves=[k3.MOVE_NAMES[a] for a in (path or [])],
                solved=bool(sol), length=ln, scramble_len=len(seq),
                facelets=k3.facelets_str(st))
