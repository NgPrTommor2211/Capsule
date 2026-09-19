import argparse
import os
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from train import load_poses, cos_to_angle, compute_emf

MODEL_BUILDERS = {
    "DT": lambda: DecisionTreeRegressor(),
    "RF": lambda: RandomForestRegressor(n_jobs=-1),
    "ET": lambda: ExtraTreesRegressor(n_jobs=-1),
    "GB": lambda: GradientBoostingRegressor(),
}


def load_emf_pose(csv_path, device, max_samples=None, seed=0):
    poses = load_poses(csv_path, device)
    if max_samples is not None and poses.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(poses.shape[0], max_samples, replace=False)
        poses = poses[idx]
    poses_rad = poses.clone()
    poses_rad[:, 3:6] = cos_to_angle(poses[:, 3:6])
    emf = compute_emf(poses_rad, device)
    return emf.cpu().numpy(), poses.cpu().numpy()


def evaluate(models, X, Y_true):
    Y_pred = np.stack([m.predict(X) for m in models], axis=1)

    pos_err = (Y_pred[:, :3] - Y_true[:, :3]) * 1000.0
    pos_rmse = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1) / 3))

    cos_true = np.clip(Y_true[:, 3:6], -1.0, 1.0)
    cos_pred = np.clip(Y_pred[:, 3:6], -1.0, 1.0)
    ang_err = np.degrees(np.arccos(cos_pred)) - np.degrees(np.arccos(cos_true))
    ang_rmse = np.sqrt(np.mean(np.sum(ang_err ** 2, axis=1) / 3))

    return pos_rmse, ang_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="dataset/train_data.csv")
    parser.add_argument("--max_train_samples", type=int, default=50000)
    parser.add_argument("--test_csvs", type=str, nargs="+", default=[
        "dataset/test_fixed.csv", "dataset/test_varying.csv", "dataset/test_random.csv",
    ])
    parser.add_argument("--out_dir", type=str, default="ml_baselines")
    args = parser.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Computing EMF from {args.train_csv} ...")
    X_train, Y_train = load_emf_pose(args.train_csv, device, args.max_train_samples)
    print(f"Train samples: {X_train.shape[0]}")

    test_sets = {}
    for csv_path in args.test_csvs:
        if os.path.exists(csv_path):
            X_t, Y_t = load_emf_pose(csv_path, device)
            test_sets[os.path.basename(csv_path)] = (X_t, Y_t)

    results = []
    for name, builder in MODEL_BUILDERS.items():
        print(f"\nTraining {name} ...")
        models = []
        for i in range(Y_train.shape[1]):
            m = builder()
            m.fit(X_train, Y_train[:, i])
            models.append(m)

        out_path = os.path.join(args.out_dir, f"{name.lower()}_models.joblib")
        joblib.dump(models, out_path)
        print(f"Saved -> {out_path}")

        for test_name, (X_t, Y_t) in test_sets.items():
            pos_rmse, ang_rmse = evaluate(models, X_t, Y_t)
            results.append((name, test_name, pos_rmse, ang_rmse))
            print(f"  {test_name:20s} Position RMSE={pos_rmse:7.3f} mm   Orientation RMSE={ang_rmse:7.3f} deg")

    print("\n================ SUMMARY ================")
    print(f"{'Model':6} {'Test set':20} {'Pos RMSE (mm)':15} {'Orient RMSE (deg)'}")
    for name, test_name, pos_rmse, ang_rmse in results:
        print(f"{name:6} {test_name:20} {pos_rmse:15.3f} {ang_rmse:.3f}")


if __name__ == "__main__":
    main()
