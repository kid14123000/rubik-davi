#!/usr/bin/env bash
# One command: start (or resume) training + the live dashboard.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}

$PY - <<'EOF' || { echo "missing deps: pip install -r requirements.txt"; exit 1; }
import torch, numpy
print(f"[deps] torch {torch.__version__}  numpy {numpy.__version__}  "
      f"mps={torch.backends.mps.is_available()}")
EOF

if [ ! -f dist2x2.npz ]; then
  echo "[1/2] building the exact BFS distance table (exam paper, ~5s)..."
  $PY oracle.py
fi

if [ "${TRAIN:-1}" = "1" ]; then
  echo "[2/2] training -> ${RUN:-runs/main}   (dashboard: http://127.0.0.1:${PORT:-8849})"
  nohup $PY train.py --iters "${ITERS:-20000}" --eval-every 500 --eval-n 200 \
        --beam-n 80 --beam-width 96 --beam-mode policy --out "${RUN:-runs/main}" \
        > "${RUN:-runs/main}".log 2>&1 &
  echo "      pid $!  log ${RUN:-runs/main}.log"
else
  echo "[2/2] dashboard only -> http://127.0.0.1:${PORT:-8849}"
fi
sleep 1
exec $PY serve.py --run "${RUN:-runs/main}" --port "${PORT:-8849}"
