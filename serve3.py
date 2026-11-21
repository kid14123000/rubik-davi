#!/usr/bin/env python3
"""3x3 dashboard server: 3D viewer, manual moves, net hints, auto-solve.

Cube logic lives here only (single source of truth); the browser just renders
what it is told and animates the turn it is told to animate.
"""
import argparse
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

import cube3 as k3
import model3 as M
import scorecard3 as SC

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs3", "main")
# search device: keep it off MPS while a training run is using the GPU;
# switch with DEVICE=mps once training is done.
DEV = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
_LOCK = threading.Lock()
_CACHE = {"mtime": None, "net": None, "it": None}

# per-move geometry for the browser animation (axis index, layer sign, rotation sign)
MOVE_GEOM = {
    'U': (1, +1, -1), "U'": (1, +1, +1),
    'D': (1, -1, +1), "D'": (1, -1, -1),
    'R': (0, +1, -1), "R'": (0, +1, +1),
    'L': (0, -1, +1), "L'": (0, -1, -1),
    'F': (2, +1, -1), "F'": (2, +1, +1),
    'B': (2, -1, +1), "B'": (2, -1, -1),
}


def _offset():
    p = os.path.join(RUN, "offset.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return int(json.load(f).get("offset", 0))
        except Exception:                                        # noqa: BLE001
            return 0
    return 0


def load_net():
    path = os.path.join(RUN, "model.pt")
    with _LOCK:
        if not os.path.exists(path):
            return None
        mt = os.path.getmtime(path)
        if _CACHE["mtime"] != mt:
            ck = torch.load(path, map_location=DEV, weights_only=False)
            meta = ck.get("meta", {})
            net = M.DualNet3(meta.get("hidden", 512)).to(DEV)
            net.load_state_dict(ck["state"])
            net.eval()
            _CACHE.update(mtime=mt, net=net, it=_offset() + int(meta.get("it") or 0))
        return _CACHE["net"]


def payload(state):
    return dict(facelets=k3.facelets_str(state),
                cubies=k3.cubies(state),
                solved=bool(k3.is_solved(*state)[0]))


def hint(state):
    net = load_net()
    if net is None:
        return None
    v, lg = M.predict(state, net, DEV)
    p = np.exp(lg[0] - lg[0].max())
    p = p / p.sum()
    child = [k3.apply_move(*state, a) for a in range(k3.NUM_ACTIONS)]
    cv = M.value(tuple(np.concatenate([c[i] for c in child], 0) for i in range(4)),
                 net, DEV)
    rows = [dict(move=k3.MOVE_NAMES[a], p=float(p[a]), v_after=float(cv[a]))
            for a in range(k3.NUM_ACTIONS)]
    best_p = int(np.argmax(p))
    best_v = int(np.argmin(cv))
    return dict(v=float(v[0]), best_policy=k3.MOVE_NAMES[best_p],
                p_policy=float(p[best_p]), best_value=k3.MOVE_NAMES[best_v],
                v_value=float(cv[best_v]), moves=rows, model_it=_CACHE.get("it"))


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        g = lambda n, d=None: q.get(n, [d])[0]
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "web3", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api3/meta":
                return self._send(200, dict(moves=[dict(name=n, axis=g_, layer=l_, s=s_)
                                                   for n, (g_, l_, s_) in MOVE_GEOM.items()],
                                            state=payload(k3.solved()),
                                            status=self._status()))
            if u.path == "/api3/status":
                return self._send(200, self._status())
            if u.path == "/api3/move":
                st = k3.state_from_facelets(g("state"))
                m = g("m")
                a = k3.MOVE_NAMES.index(m)
                st = k3.apply_move(*st, a)
                out = payload(st)
                out["anim"] = dict(axis=MOVE_GEOM[m][0], layer=MOVE_GEOM[m][1],
                                   s=MOVE_GEOM[m][2])
                out["move"] = m
                out["hint"] = hint(st)
                return self._send(200, out)
            if u.path == "/api3/scramble":
                k = int(g("k", "20"))
                seed = int(g("seed", str(int(time.time() * 1000) % 100000)))
                rng = np.random.default_rng(seed)
                st = k3.solved()
                seq = []
                last = -1
                for _ in range(k):
                    while True:
                        a = int(rng.integers(k3.NUM_ACTIONS))
                        if last < 0 or a // 2 != last // 2:
                            break
                    st = k3.apply_move(*st, a)
                    seq.append(k3.MOVE_NAMES[a])
                    last = a
                out = payload(st)
                out["seq"] = seq
                out["hint"] = hint(st)
                return self._send(200, out)
            if u.path == "/api3/solve":
                st = k3.state_from_facelets(g("state"))
                width = int(g("width", "48"))
                mode = g("mode", "policy")
                t0 = time.time()
                ln, ok, path = SC.beam_one(load_net(), DEV, st, width=width,
                                           mode=mode, max_steps=45, want_path=True)
                return self._send(200, dict(moves=[k3.MOVE_NAMES[a] for a in (path or [])],
                                            solved=bool(ok), length=int(ln),
                                            sec=round(time.time() - t0, 2)))
            if u.path == "/api3/hint":
                st = k3.state_from_facelets(g("state"))
                return self._send(200, hint(st) or dict(no_model=True))
            return self._send(404, {"error": "not found"})
        except Exception as e:                                       # noqa: BLE001
            return self._send(500, {"error": repr(e)})

    def _status(self):
        p = os.path.join(RUN, "status.json")
        if not os.path.exists(p):
            return {"waiting": True, "run": os.path.basename(RUN)}
        with open(p) as f:
            st = json.load(f)
        st["net_it"] = _CACHE.get("it")
        return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--port", type=int, default=8851)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    RUN = os.path.abspath(a.run)
    print(f"[serve3] run={RUN}")
    print(f"[serve3] http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
