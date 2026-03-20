#!/bin/bash
set -e  # stop on error


# Download
echo "Downloading dataset..."
uvx --from git+https://github.com/Kaggle/kaggle-api/ \
    kaggle competitions download -c mva-dlmi-2026-histopathology-ood-classification

# Unzip
echo "Unzipping..."
unzip mva-dlmi-2026-histopathology-ood-classification.zip

# Organize
mkdir -p data
mv test.h5 train.h5 val.h5 data/

# Cleanup
echo "Cleaning up..."
rm mva-dlmi-2026-histopathology-ood-classification.zip
rm getting_started.ipynb
echo "Done."