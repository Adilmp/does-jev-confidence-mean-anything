#!/usr/bin/env bash
# Run the whole pipeline.
#   ./run_all.sh hard              real API calls, needs JEV_API_KEY
#   ./run_all.sh hard --simulate   no key needed, invented data
#   ./run_all.sh natural --strict  tightened criteria (see step 7)
set -euo pipefail

NAME="${1:-hard}"
FLAG="${2:-}"

if [ ! -f "data/${NAME}.csv" ]; then
  python3 1_get_data.py
fi

# --strict changes the question wording, so step 2 writes to a separate
# results name. Everything downstream reads that name and takes no flag -
# only step 2 knows about wording at all.
if [ "$FLAG" = "--strict" ]; then
  python3 2_ask_jev.py "$NAME" --strict
  NAME="${NAME}_strict"
  FLAG=""
else
  python3 2_ask_jev.py "$NAME" $FLAG
fi

python3 3_measure.py    "$NAME" $FLAG
python3 4_chart.py      "$NAME" $FLAG
python3 5_thresholds.py "$NAME" $FLAG

echo
echo "Charts are in results/. The one for the post is:"
echo "  results/${NAME}_calibration${FLAG:+.SIMULATED}.png"
