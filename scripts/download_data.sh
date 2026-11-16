#!/usr/bin/env bash
# Download FLORES+ (needs HF_TOKEN), HalOmi and WMT MQM for a language pair.
# Usage: scripts/download_data.sh [pair=en-hi] [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
PAIR="${1:-en-hi}"
shift || true
if [[ -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "NOTE: no HF token found; FLORES+ (gated) will be skipped. Set HF_TOKEN or run 'huggingface-cli login'." >&2
fi
python -m mml.cli download-data --pair "$PAIR" "$@"
