#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# download_data.sh — fetch the Kaggle "Quality Prediction in a Mining Process" dataset
#
# Pre-req: Kaggle API token configured at:
#   ~/.kaggle/kaggle.json (Linux/Mac)
#   %USERPROFILE%\.kaggle\kaggle.json (Windows)
#
# Get your token at: https://www.kaggle.com/settings → API → Create New Token
# ------------------------------------------------------------------------------
set -euo pipefail

DATASET="edumagalhaes/quality-prediction-in-a-mining-process"
DEST="data/raw/flotation"
mkdir -p "$DEST"

if [[ -f "$DEST/MiningProcess_Flotation_Plant_Database.csv" ]]; then
  echo "✅ Dataset already present at $DEST — skipping download."
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "❌ kaggle CLI not installed. Run: pip install kaggle"
  echo "   Then configure your API token (see header of this script)."
  exit 1
fi

echo "→ Downloading $DATASET (~125 MB) ..."
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

echo "✅ Done. Files in $DEST:"
ls -lh "$DEST" | head -10
