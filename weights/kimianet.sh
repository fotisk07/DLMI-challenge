#!/usr/bin/env bash

set -euo pipefail

URL="https://github.com/KimiaLabMayo/KimiaNet/raw/refs/heads/main/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth"
OUT="weights/KimiaNetPyTorchWeights.pth"

wget -O "$OUT" "$URL"