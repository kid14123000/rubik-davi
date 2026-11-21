"""Load the shipped weights and solve something — the shortest possible usage demo.

    python3 solve.py --cube 3 --state <54 facelet digits> --width 1024
    python3 solve.py --cube 3 --scramble 20            # random scramble, then solve
    python3 solve.py --cube 2 --scramble 14 --width 192
"""
import argparse
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, default=3, choices=[2, 3])
    ap.add_argument("--state", default=None, help="facelet string (54 digits for 3x3, 24 for 2x2)")
    ap.add_argument("--scramble", type=int, default=0, help="random scramble length")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    seed = a.seed if a.seed is not None else int(time.time() * 1000) % 100000

    if a.cube == 3:
        import cube3 as k
        import model3 as M
        import scorecard3 as SC
        ck = torch.load(os.path.join(HERE, "weights/rubik3x3_417k.pt"),
                        map_location=dev, weights_only=False)
        net = M.DualNet3(ck["meta"]["hidden"]).to(dev)
        net.load_state_dict(ck["state"])
        net.eval()
        rng = np.random.default_rng(seed)
        if a.state:
            st = k.state_from_facelets(a.state)
            scr = None
        else:
            n = a.scramble or 20
            st, seq = k.scramble(rng, n, 1)
            scr = [k.MOVE_NAMES[x] for x in seq]
        t0 = time.time()
        ln, ok, path = SC.beam_one(net, dev, st, width=a.width, mode="policy",
                                   max_steps=a.max_steps, want_path=True)
        moves = [k.MOVE_NAMES[x] for x in (path or [])]
    else:
        import cube2 as k
        import model as M
        import scorecard as SC
        ck = torch.load(os.path.join(HERE, "weights/rubik2x2_20k.pt"),
                        map_location=dev, weights_only=False)
        net = M.DualNet(ck["meta"]["hidden"]).to(dev)
        net.load_state_dict(ck["state"])
        net.eval()
        rng = np.random.default_rng(seed)
        if a.state:
            raise SystemExit("--state for 2x2 is not supported yet; use --scramble")
        n = a.scramble or 14
        (perm, ori, seq) = k.scramble(rng, n, 1)
        scr = [k.MOVE_NAMES[x] for x in seq]
        t0 = time.time()
        # scorecard.beam_one takes a single state as 1-D arrays
        ln, ok, path = SC.beam_one(net, dev, perm[0], ori[0], width=a.width,
                                   max_steps=a.max_steps, mode="policy",
                                   want_path=True)
        st = (perm, ori)
        moves = [k.MOVE_NAMES[x] for x in (path or [])]

    print(json.dumps(dict(cube=a.cube, width=a.width, seed=seed,
                          scramble=scr, solved=bool(ok), length=int(ln),
                          sec=round(time.time() - t0, 2),
                          moves=moves), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
