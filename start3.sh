#!/usr/bin/env bash
# 3x3: train (background) + 3D dashboard.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
RUN=${RUN:-runs3/main}
PORT=${PORT:-8851}

$PY - <<'EOF' || { echo "missing deps: pip install -r requirements.txt"; exit 1; }
import torch, numpy
print(f"[deps] torch {torch.__version__}  numpy {numpy.__version__}  "
      f"mps={torch.backends.mps.is_available()}")
EOF

if [ "${TRAIN:-1}" = "1" ]; then
  mkdir -p runs3
  echo "[1/2] training -> $RUN   (dashboard: http://127.0.0.1:$PORT)"
  nohup $PY train3.py --iters "${ITERS:-300000}" --batch 768 --kmax 20 --kmax-start 4 \
        --curriculum-frac 0.3 --eval-every 1000 --eval-n 40 --beam-every 5000 \
        --beam-n 16 --beam-width 32 --beam-mode policy --out "$RUN" \
        > "$RUN".log 2>&1 &
  echo "      pid $!  log $RUN.log"
else
  echo "[1/2] dashboard only -> http://127.0.0.1:$PORT"
fi
sleep 1
exec $PY serve3.py --run "$RUN" --port "$PORT"
