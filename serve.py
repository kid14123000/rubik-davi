#!/usr/bin/env python3
"""Live dashboard server for a rubik_davi training run (stdlib only + torch).

  python3 serve.py --run runs/main --port 8848
  -> http://127.0.0.1:8848

Reads runs/<id>/status.json (written atomically by train.py) and runs the
current checkpoint to animate a solve on demand.
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

import model as M
import scorecard as SC

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = None
DEV = "cpu"
_LOCK = threading.Lock()
_CACHE = {"mtime": None, "net": None, "demo": {}}


def load_net():
    path = os.path.join(RUN, "model.pt")
    with _LOCK:
        if not os.path.exists(path):
            return None
        mt = os.path.getmtime(path)
        if _CACHE["mtime"] != mt:
            ck = torch.load(path, map_location=DEV, weights_only=False)
            meta = ck.get("meta", {})
            net = M.DualNet(meta.get("hidden", 512)).to(DEV)
            net.load_state_dict(ck["state"])
            net.eval()
            _CACHE.update(mtime=mt, net=net, demo={}, it=meta.get("it"))
        return _CACHE["net"]


class H(BaseHTTPRequestHandler):
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
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                p = os.path.join(RUN, "status.json")
                if not os.path.exists(p):
                    return self._send(200, {"waiting": True, "run": os.path.basename(RUN)})
                with open(p) as f:
                    st = json.load(f)
                rd = os.path.join(HERE, "runs")
                st["runs"] = sorted(os.listdir(rd)) if os.path.isdir(rd) else []
                return self._send(200, st)
            if u.path == "/api/demo":
                net = load_net()
                if net is None:
                    return self._send(200, {"waiting": True})
                seed = int(q.get("seed", ["0"])[0])
                mode = q.get("mode", ["policy"])[0]
                width = int(q.get("width", ["64"])[0])
                key = (round(_CACHE["mtime"] or 0, 3), seed, mode, width)
                with _LOCK:
                    hit = _CACHE["demo"].get(key)
                if hit is not None:
                    return self._send(200, hit)
                t0 = time.time()
                d = SC.demo(net, DEV, seed=seed, mode=mode, width=width)
                d["gen_s"] = round(time.time() - t0, 2)
                d["model_it"] = _CACHE.get("it")
                with _LOCK:
                    if len(_CACHE["demo"]) > 40:
                        _CACHE["demo"].clear()
                    _CACHE["demo"][key] = d
                return self._send(200, d)
            if u.path == "/api/runs":
                rd = os.path.join(HERE, "runs")
                return self._send(200, sorted(os.listdir(rd)) if os.path.isdir(rd) else [])
            return self._send(404, {"error": "not found"})
        except Exception as e:                                       # noqa: BLE001
            return self._send(500, {"error": repr(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "runs", "main"))
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    RUN = os.path.abspath(a.run)
    print(f"[serve] run={RUN}")
    print(f"[serve] http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
