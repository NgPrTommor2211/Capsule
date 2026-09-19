import argparse
import os
import numpy as np
import pandas as pd
import torch

from train import WPRefiner, load_poses, cos_to_angle, compute_emf, infer_num_blocks


def load_checkpoint(path, device, norm_stats_path, train_emf_path):
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
        mean = obj["mean"].to(device)
        std = obj["std"].to(device)
        num_blocks = obj.get("num_blocks", infer_num_blocks(state_dict))
        return state_dict, mean, std, num_blocks

    state_dict = obj
    num_blocks = infer_num_blocks(state_dict)

    if os.path.exists(norm_stats_path):
        stats = torch.load(norm_stats_path, map_location=device)
        return state_dict, stats["mean"].to(device), stats["std"].to(device), num_blocks

    if os.path.exists(train_emf_path):
        emf_train = torch.load(train_emf_path, map_location=device)
        mean = emf_train.mean(dim=0, keepdim=True)
        std = emf_train.std(dim=0, keepdim=True)
        std[std == 0] = 1e-8
        return state_dict, mean, std, num_blocks

    return state_dict, None, None, num_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_model/model_7residual.pth")
    parser.add_argument("--test_csv", type=str, default="dataset/test_data.csv")
    parser.add_argument("--norm_stats", type=str, default="dataset/train_stats.pt")
    parser.add_argument("--train_emf", type=str, default="dataset/train_data.pt")
    parser.add_argument("--out", type=str, default="inference_results.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    poses_true = load_poses(args.test_csv, device)
    poses_rad = poses_true.clone()
    poses_rad[:, 3:6] = cos_to_angle(poses_true[:, 3:6])
    emf = compute_emf(poses_rad, device)

    state_dict, mean, std, num_blocks = load_checkpoint(args.checkpoint, device, args.norm_stats, args.train_emf)
    if mean is None:
        mean = emf.mean(dim=0, keepdim=True)
        std = emf.std(dim=0, keepdim=True)
        std[std == 0] = 1e-8

    model = WPRefiner(num_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    emf_norm = (emf - mean) / std * 10
    with torch.no_grad():
        pred = model(emf_norm)

    pos_true_mm = poses_true[:, :3].cpu().numpy() * 1000.0
    pos_pred_mm = pred[:, :3].cpu().numpy() * 1000.0
    pos_err = pos_pred_mm - pos_true_mm
    pos_rmse_per_sample = np.sqrt(np.mean(pos_err ** 2, axis=1))
    pos_rmse = pos_rmse_per_sample.mean()

    cos_true = np.clip(poses_true[:, 3:6].cpu().numpy(), -1.0, 1.0)
    cos_pred = np.clip((pred[:, 3:6] * 10.0).cpu().numpy(), -1.0, 1.0)
    ang_true_deg = np.degrees(np.arccos(cos_true))
    ang_pred_deg = np.degrees(np.arccos(cos_pred))
    ang_err = ang_pred_deg - ang_true_deg
    ang_rmse_per_sample = np.sqrt(np.mean(ang_err ** 2, axis=1))
    ang_rmse = ang_rmse_per_sample.mean()

    print("\n================ INFERENCE RESULTS ================")
    print(f"Checkpoint   : {args.checkpoint} ({num_blocks} residual blocks)")
    print(f"Test set     : {args.test_csv}")
    print(f"Samples      : {poses_true.shape[0]}")
    print(f"Position RMSE: {pos_rmse:.3f} mm")
    print(f"Orientation RMSE: {ang_rmse:.3f} deg")
    print("=====================================================")

    out_df = pd.DataFrame({
        "X_true_mm": pos_true_mm[:, 0], "Y_true_mm": pos_true_mm[:, 1], "Z_true_mm": pos_true_mm[:, 2],
        "X_pred_mm": pos_pred_mm[:, 0], "Y_pred_mm": pos_pred_mm[:, 1], "Z_pred_mm": pos_pred_mm[:, 2],
        "pos_err_mm": pos_rmse_per_sample,
        "ang_err_deg": ang_rmse_per_sample,
    })
    out_df.to_csv(args.out, index=False)
    print(f"Saved details -> {args.out}")


if __name__ == "__main__":
    main()
