#!/bin/bash
# Score one finished training run on the dev set with the official staff tools.
#
#   bash scc/score_run.sh RUN [SETTING=VALUE ...]
#
# Examples (run from the project root, on a GPU node, with the env active):
#   bash scc/score_run.sh base_r2
#   bash scc/score_run.sh d8_r2       lookahead=8
#   bash scc/score_run.sh blstm_r2    bidirectional=True rnn_units=142
#   bash scc/score_run.sh conf_causal conformer=True lookahead=0
#   bash scc/score_run.sh conf_full   conformer=True lookahead=-1
#
# It copies cli/submission/model.py to runs/RUN/model.py, sets CONFIG to the
# run's settings there (the real CONFIG is not touched), then runs enhance.py,
# evaluate.py (noisy and clean inputs) and score.py. Outputs:
#   preds/RUN, preds/RUN_clean, reports/RUN.json, reports/RUN_clean.json, reports/RUN_score.txt
set -euo pipefail

RUN=$1; shift
[ -f "runs/$RUN/best.safetensors" ] || { echo "no runs/$RUN/best.safetensors"; exit 1; }
SRC=runs/$RUN/model.py
cp cli/submission/model.py "$SRC"
for kv in "$@"; do
    key=${kv%%=*}; value=${kv#*=}
    grep -q -E "^    $key=" "$SRC" || { echo "CONFIG has no setting '$key'"; exit 1; }
    sed -i -E "s/^    $key=[^,]*,/    $key=$value,/" "$SRC"
done
echo "CONFIG used for $RUN:"
sed -n '/^CONFIG = dict(/,/^)/p' "$SRC"
mkdir -p preds reports

cd cli
for kind in noisy clean; do
    tag=$RUN; [ "$kind" = clean ] && tag=${RUN}_clean
    python enhance.py --source "../$SRC" --checkpoint "../runs/$RUN/best.safetensors" \
        --input "../dev/$kind" --output "../preds/$tag" --device cuda
    python evaluate.py --clean ../dev/clean --enhanced "../preds/$tag" --report "../reports/$tag.json"
done
python score.py --noisy-report "../reports/$RUN.json" --clean-report "../reports/${RUN}_clean.json" \
    | tee "../reports/${RUN}_score.txt"