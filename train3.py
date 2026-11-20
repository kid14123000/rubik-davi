#!/usr/bin/env python3
"""Train the 3x3 dual-head solver (DAVI value + self-supervised policy)."""
import argparse
import json
import os
import time

import numpy as np
import torch

import cube3 as k3
import model3 as M
import scorecard3 as SC


def pick_device(pref="auto"):
    if pref != "auto":
        return pref
    if not torch.backends.mps.is_available():
        return "cpu"
    st, _ = k3.scramble(np.random.default_rng(0), 20, 512)
    x = torch.from_numpy(k3.piece_feat(st))
    res = {}
    for dev in ("cpu", "mps"):
        try:
            net = M.DualNet3(512).to(dev)
            xx = x.to(dev)
            for _ in range(3):
                v, lg = net(xx)
                (v.sum() + lg.sum()).backward()
            t0 = time.time()
            for _ in range(10):
                v, lg = net(xx)
                (v.sum() + lg.sum()).backward()
            if dev == "mps":
                torch.mps.synchronize()
            res[dev] = (time.time() - t0) / 10
        except Exception:                                        # noqa: BLE001
            res[dev] = 1e9
    best = min(res, key=res.get)
    print(f"[device] {dict((k, round(v*1000,1)) for k,v in res.items())} ms/step -> {best}",
          flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--kmax-start", type=int, default=4)
    ap.add_argument("--curriculum-frac", type=float, default=0.35)
    ap.add_argument("--deep-bias", type=float, default=0.0,
                    help="fraction of each batch scrambled to full depth")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--w-value", type=float, default=1.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--target-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=40)
    ap.add_argument("--beam-every", type=int, default=5000)
    ap.add_argument("--beam-n", type=int, default=16)
    ap.add_argument("--beam-width", type=int, default=32)
    ap.add_argument("--beam-mode", default="policy", choices=["policy", "value"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    run = args.out or time.strftime("runs3/%Y%m%d-%H%M%S")
    os.makedirs(run, exist_ok=True)
    dev = pick_device(args.device)
    cfg = {k: v for k, v in vars(args).items() if k not in ("out", "resume")}
    cfg["device"] = dev
    with open(os.path.join(run, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    tr = M.Trainer3(args.iters, batch=args.batch, kmax=args.kmax, hidden=args.hidden,
                    lr=args.lr, gamma=args.gamma, target_every=args.target_every,
                    seed=args.seed, device=dev, w_value=args.w_value,
                    w_policy=args.w_policy, kmax_start=args.kmax_start,
                    curriculum_frac=args.curriculum_frac, deep_bias=args.deep_bias)
    if args.resume:
        tr.load(args.resume)

    t0 = time.time()
    hist, evals = [], []
    status_path = os.path.join(run, "status.json")
    logf = open(os.path.join(run, "metrics.jsonl"), "a")
    n_params = sum(p.numel() for p in tr.net.parameters())

    def publish(done=False):
        n = len(hist)
        step = max(1, n // 400)
        curve = [[hist[i][0]] + [float(np.mean([h[j] for h in hist[i:i + step]]))
                                 for j in (1, 2, 3)] for i in range(0, n, step)]
        st = dict(run=os.path.basename(run), started=t0, iters=tr.iter,
                  total_iters=args.iters, samples=tr.samples,
                  elapsed=time.time() - t0, device=dev, params=n_params,
                  kmax_now=tr.current_kmax(),
                  cfg={k: cfg[k] for k in ("batch", "kmax", "kmax_start", "hidden",
                                           "lr", "gamma", "beam_mode", "beam_width")},
                  loss_v=hist[-1][1] if hist else None,
                  loss_p=hist[-1][2] if hist else None,
                  p_acc=hist[-1][3] if hist else None,
                  v_mean=hist[-1][4] if hist else None,
                  loss_v_avg=float(np.mean([h[1] for h in hist[-100:]])) if hist else None,
                  loss_p_avg=float(np.mean([h[2] for h in hist[-100:]])) if hist else None,
                  p_acc_avg=float(np.mean([h[3] for h in hist[-100:]])) if hist else None,
                  curve=curve, evals=evals[-80:], done=done, cube=3)
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, status_path)

    publish()
    for it in range(args.iters):
        r = tr.step()
        hist.append((r["it"], r["loss_v"], r["loss_p"], r["p_acc"], r["v_mean"]))
        logf.write(json.dumps(r) + "\n")
        if it % 20 == 0:
            logf.flush()
            print(f"it {r['it']:7d} kmax={r['kmax']:2d} loss_v={r['loss_v']:7.3f} "
                  f"loss_p={r['loss_p']:6.3f} p_acc={r['p_acc']:.3f} "
                  f"Vmean={r['v_mean']:5.2f} gn={r['grad_norm']:6.2f} "
                  f"{(it+1)/(time.time()-t0):6.1f}it/s", flush=True)
        if (it + 1) % 200 == 0:
            publish()

        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            do_beam = ((it + 1) % args.beam_every == 0) or (it + 1 == args.iters)
            te = time.time()
            ev = SC.score(tr.net, dev, n=args.eval_n, do_beam=do_beam,
                          beam_n=args.beam_n, width=args.beam_width,
                          mode=args.beam_mode, seed=12345 + it // args.eval_every)
            ev.update(it=r["it"], samples=tr.samples, t=time.time() - t0,
                      eval_s=round(time.time() - te, 1), kmax=r["kmax"])
            evals.append(ev)
            tr.save(os.path.join(run, "model.pt"),
                    meta=dict(it=tr.iter, hidden=args.hidden, device=dev))
            publish()
            print("EVAL " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in ev.items()}), flush=True)

    publish(done=True)
    logf.close()
    print(f"[done] {run}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
