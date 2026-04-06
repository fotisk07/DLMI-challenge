# Data

The scripts in the `scripts` folder assume that the data is located in a folder named `data`.

# Download KimiaNet

Download the weights from: https://github.com/KimiaLabMayo/KimiaNet  
Then place them in a `/weights` folder.

# Reproduce Results

The script used to train the model is `train.py`. It performs training and saves checkpoints in the `checkpoints` folder.

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
````

# Predict

Once training is complete, you can run predictions on the test set. Results will be saved in the `output` folder.

```bash
uv run scripts/predicts.py checkpoints/bestmodel.pt
```
