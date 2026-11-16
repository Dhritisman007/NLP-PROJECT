#!/usr/bin/env bash
# End-to-end runner. Phases are appended here as they are implemented and tested.
# Usage: scripts/run_all.sh [pair=en-hi] [profile=mac_dev]
set -euo pipefail
cd "$(dirname "$0")/.."
PAIR="${1:-en-hi}"
PROFILE="${2:-mac_dev}"

echo "== Phase 1: setup + smoke test =="
python -m pytest -q -m "not slow"
python -m mml.cli smoke-test --pair "$PAIR" --profile "$PROFILE"

echo "== Phase 2: synthetic errors =="
python -m mml.cli download-data --pair "$PAIR" --what flores
python -m mml.cli synth --pair "$PAIR" --profile "$PROFILE"

echo "== Phase 3: detection signals (cometkiwi/xcomet are skipped unless run on Colab with HF_TOKEN) =="
python -m mml.cli signals --pair "$PAIR" --profile "$PROFILE" --split val --limit 60

echo "== Phase 4: detection classifier (full feature build takes ~1h per pair on a laptop) =="
python -m mml.cli convert-ct2 --profile "$PROFILE"
python -m mml.cli features --pair "$PAIR" --profile "$PROFILE" --splits train,val,test
python -m mml.cli train-detector --pair "$PAIR" --profile "$PROFILE"
# Final held-out evaluation is deliberately NOT run here (once, after freezing choices):
#   python -m mml.cli train-detector --pair "$PAIR" --evaluate-test

echo "== Phase 5: correction (LLM strategies need ANTHROPIC_API_KEY in .env; they are skipped with a warning otherwise) =="
python -m mml.cli correct-compare --pair "$PAIR" --profile "$PROFILE" --n 80
python -m mml.cli rerank-eval --pair "$PAIR" --profile "$PROFILE" --n 60

echo "== Phase 6: evaluation on the held-out TEST split (run once, after Phases 4-5 are frozen) =="
python -m mml.cli evaluate --pair "$PAIR" --profile "$PROFILE" --real-n 300
python -m mml.cli report --pairs "$PAIR"

echo "== Phase 7: human-evaluation kit (annotation itself is manual) =="
python -m mml.cli human-eval-kit --pair "$PAIR" --profile "$PROFILE"
echo "   send results/human_eval/$PAIR/for_annotators/ to the annotators, then: mml human-eval-iaa --pair $PAIR --a A.csv --b B.csv"
echo "== Phase 8: demo (interactive; run manually) ==  python -m mml.cli demo --pair $PAIR"
