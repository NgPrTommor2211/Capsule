# Capsule Endoscopy

## Folder layout

```
checkpoint_model/   trained model checkpoints (.pth)
dataset/             pose CSVs (train/test) + train_stats.pt (mean/std for normalization)
                      + train_data.pt (full EMF tensor, optional, regenerable, not in git)
generate_train_data.py
generate_test_data.py
train.py
infer.py
```

## 1. Check results with an existing checkpoint

Ready-to-run commands for the 3 test scenarios already in `dataset/`:

**Fixed orientation (helix)**
```bash
python3 infer.py --checkpoint checkpoint_model/7residual_checkpoint.pth --test_csv dataset/test_fixed.csv --out inference_results.csv
```

**Varying orientation (helix)**
```bash
python3 infer.py --checkpoint checkpoint_model/7residual_checkpoint.pth --test_csv dataset/test_varying.csv --out inference_results.csv
```

**Random position + orientation**
```bash
python3 infer.py --checkpoint checkpoint_model/7residual_checkpoint.pth --test_csv dataset/test_random.csv --out inference_results.csv
```

## 2. Generate data

Train set
```bash
python3 generate_train_data.py --out dataset/train_data.csv
```

Test set — 3 scenarios:
```bash
# fixed orientation, helix trajectory
python3 generate_test_data.py --num_points 1000 --out dataset/test_fixed.csv

# continuously varying orientation, helix trajectory
python3 generate_test_data.py --num_points 1000 --varying_orientation --out dataset/test_varying.csv

# fully random position + orientation
python3 generate_test_data.py --num_points 1000 --random --out dataset/test_random.csv
```

## 3. Train

```bash
python3 train.py \
  --train_csv dataset/train_data.csv \
  --test_csv dataset/test_fixed.csv \
  --epochs xxxx \
  --out checkpoint_model/model_7residual.pth
```

- Computes EMF from `--train_csv` and saves it to `--emf_out` (default `dataset/train_data.pt`) —
  this becomes the normalization-stats source `infer.py` uses for checkpoints without embedded stats.
- The saved checkpoint already contains `mean`/`std`, so `infer.py` works on it directly without `--train_emf`.
