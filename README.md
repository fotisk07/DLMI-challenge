# DLMI Challenge — Patch-level Cancer Grading

Histopathology patch classification challenge (MVA DLMI).

## Setup

### Using UV
```bash
uv sync
```

Then prefix any script with `uv run`:

```bash
uv run scripts/train.py
```

### Without uv (using venv)
Create a virtual environment and install the requirements in the `requirements.txt`

## Data

You can download the data using the `data/downloand.sh` script. You will need a kaggle token. 
Otherwise just place `train.h5`, `val.h5` and `test.h5` in the data folder.

## KimiaNet Weights

Download weights from [KimiaLabMayo/KimiaNet](https://github.com/KimiaLabMayo/KimiaNet) and place them in a `weights/` folder.

## Stain Normalization

Fit a normalizer on the training data:

```bash
uv run scripts/fit_normalizer.py
```

## Training

```bash
uv run scripts/train.py \
  --arch kimianet \
  --pretrained \
  --HEAug 0.7 0.7 0.7 \
  --freeze-backbone 1 \
  --num-workers 4 \
  --epochs 20 \
  --batch-size 128 \
  --scheduler step \
  --gaussian-blur \
  --lora 8 16
```

Checkpoints are saved to `checkpoints/`.

## Prediction

```bash
uv run scripts/predict.py checkpoints/bestmodel.pt
```

Results are saved to `output/`.

## Other Scripts

| Script | Description |
|---|---|
| `scripts/linear_probing.py` | Linear probing evaluation |
| `scripts/dinoV2_embds.py` | Extract DINOv2 embeddings |
| `scripts/phikon_embds.py` | Extract Phikon embeddings |
| `scripts/sweep.py` | Hyperparameter sweep |
| `scripts/visu_normalizer.py` | Visualize stain normalization |
