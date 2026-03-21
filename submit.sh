#!/usr/bin/env bash
set -e

COMP="mva-dlmi-2026-histopathology-ood-classification"
WAIT=false

if [ "$1" = "--wait" ]; then
  WAIT=true
  shift
fi

SUB_FILE="$1"
MESSAGE="$2"

uvx --from git+https://github.com/Kaggle/kaggle-api/ \
kaggle competitions submit \
-c "$COMP" \
-f "$SUB_FILE" \
-m "$MESSAGE"

if [ "$WAIT" = true ]; then
  echo "Waiting 10s for scoring..."
  sleep 20

  uvx --from git+https://github.com/Kaggle/kaggle-api/ \
  kaggle competitions leaderboard \
  -c "$COMP" --show 10
fi