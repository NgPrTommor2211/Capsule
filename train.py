import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def rotz(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    B = theta.shape[0]
    rot = torch.zeros((B, 3, 3), device=theta.device)
    rot[:, 0, 0] = c; rot[:, 0, 1] = -s
    rot[:, 1, 0] = s; rot[:, 1, 1] = c
    rot[:, 2, 2] = 1
    return rot


def roty(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    B = theta.shape[0]
    rot = torch.zeros((B, 3, 3), device=theta.device)
    rot[:, 0, 0] = c; rot[:, 0, 2] = s
    rot[:, 1, 1] = 1
    rot[:, 2, 0] = -s; rot[:, 2, 2] = c
    return rot


def rotx(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    B = theta.shape[0]
    rot = torch.zeros((B, 3, 3), device=theta.device)
    rot[:, 0, 0] = 1
    rot[:, 1, 1] = c; rot[:, 1, 2] = -s
    rot[:, 2, 1] = s; rot[:, 2, 2] = c
    return rot


def batch_euler_rotation_matrix(angles):
    return torch.bmm(rotz(angles[:, 2]), torch.bmm(roty(angles[:, 1]), rotx(angles[:, 0])))


def B_Bio_batch(WP, RL, WPL, I, Ntx, device):
    B = WP.shape[0]
    Rc = 0.045
    Ml = torch.pi * Ntx * I * Rc ** 2

    WRL = batch_euler_rotation_matrix(RL)
    z_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).expand(B, 3).unsqueeze(-1)
    D = torch.bmm(WRL, z_axis).squeeze(-1)

    Phi = WP - WPL
    norm_Phi = torch.clamp(torch.norm(Phi, dim=1), min=1e-8)
    dot_DP = torch.sum(D * Phi, dim=1)

    return (1e-7 * Ml).unsqueeze(-1) * (
        3 * dot_DP.unsqueeze(-1) * Phi / norm_Phi.unsqueeze(-1) ** 5 -
        D / norm_Phi.unsqueeze(-1) ** 3
    )


def emf_Bio(AK, f, WPL, RL, WP, WR, Rxk, I, nRx, Ntx, device):
    AK = torch.tensor(AK, dtype=torch.float32, device=device).unsqueeze(0).repeat(WP.shape[0], 1)
    f = torch.tensor(f, dtype=torch.float32, device=device).unsqueeze(0).repeat(WP.shape[0], 1)
    WPL = torch.tensor(WPL, dtype=torch.float32, device=device)
    RL = torch.tensor(RL, dtype=torch.float32, device=device)
    Rxk = torch.tensor(Rxk, dtype=torch.float32, device=device).reshape(WP.shape[0], 3, 3)
    I = torch.tensor(I, dtype=torch.float32, device=device)
    nRx = torch.tensor(nRx, dtype=torch.float32, device=device)
    Ntx = torch.tensor(Ntx, dtype=torch.float32, device=device)

    WRCE = batch_euler_rotation_matrix(WR)
    A = B_Bio_batch(WP, RL[:, 0, :], WPL[:, 0, :], I[:, 0], Ntx[:, 0], device)
    Bc = B_Bio_batch(WP, RL[:, 1, :], WPL[:, 1, :], I[:, 1], Ntx[:, 1], device)
    C = B_Bio_batch(WP, RL[:, 2, :], WPL[:, 2, :], I[:, 2], Ntx[:, 2], device)

    ind_v = torch.bmm(WRCE, Rxk.transpose(1, 2))
    X, Y, Z = ind_v[:, :, 0], ind_v[:, :, 1], ind_v[:, :, 2]

    flux = torch.stack([
        torch.sum(A * X, dim=1), torch.sum(Bc * X, dim=1), torch.sum(C * X, dim=1),
        torch.sum(A * Y, dim=1), torch.sum(Bc * Y, dim=1), torch.sum(C * Y, dim=1),
        torch.sum(A * Z, dim=1), torch.sum(Bc * Z, dim=1), torch.sum(C * Z, dim=1),
    ], dim=1)

    emf = (2 * torch.pi) * \
        torch.repeat_interleave(AK, 3, dim=1) * \
        torch.repeat_interleave(nRx, 3, dim=1) * \
        f.repeat(1, 3) * flux
    return torch.abs(emf)


def initialize_parameters(B):
    Ntx_single = np.array([300, 300, 300])
    nRx_single = np.array([250, 200, 200])
    Rxk_single = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    I_single = np.array([1.588, 1.354, 0.565])
    RL_single = np.array([
        [0, -15, 0],
        [15, 0, 30],
        [15, 0, 120]
    ]) * np.pi / 180
    WPL_single = np.array([
        [0.085, 0, 0.0725],
        [-0.0425, 0.0736, 0.0725],
        [-0.0425, -0.0736, 0.0725]
    ])
    AK = np.array([np.pi * 3 ** 2 / 1000 ** 2, 10 * 7 / 1000 ** 2, 10 * 10 / 1000 ** 2])
    f = np.array([4000, 4500, 5000])

    Ntx = np.tile(Ntx_single.reshape(1, 3), (B, 1))
    nRx = np.tile(nRx_single.reshape(1, 3), (B, 1))
    Rxk = np.tile(Rxk_single.reshape(1, 3, 3), (B, 1, 1))
    I = np.tile(I_single.reshape(1, 3), (B, 1))
    RL = np.tile(RL_single.reshape(1, 3, 3), (B, 1, 1))
    WPL = np.tile(WPL_single.reshape(1, 3, 3), (B, 1, 1))
    return Ntx, nRx, Rxk, I, RL, WPL, AK, f


def cos_to_angle(cos_vec):
    return torch.arccos(torch.clamp(cos_vec, -1.0, 1.0))


def compute_emf(poses_rad, device, batch_size=4000):
    num_samples = poses_rad.shape[0]
    results = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = poses_rad[start:end]
        WP, WR = batch[:, :3], batch[:, 3:]
        B = WP.shape[0]
        Ntx, nRx, Rxk, I, RL, WPL, AK, f = initialize_parameters(B)
        emf_batch = emf_Bio(AK, f, WPL, RL, WP, WR, Rxk, I, nRx, Ntx, device)
        results.append(emf_batch)
    return torch.cat(results, dim=0)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(0.01),
            nn.Linear(dim, dim), nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return x + self.block(x)


class WPRefiner(nn.Module):
    def __init__(self, num_blocks=7):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(9, 512), nn.LeakyReLU(0.01))
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(512, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


def infer_num_blocks(state_dict):
    indices = set()
    for key in state_dict:
        if key.startswith("res_blocks.") and key.endswith(".block.0.weight"):
            indices.add(int(key.split(".")[1]))
    return len(indices) if indices else 7


def load_poses(csv_path, device):
    df = pd.read_csv(csv_path, header=None)
    return torch.tensor(df.values, dtype=torch.float32, device=device)


def evaluate(model, poses_cos, emf_norm):
    model.eval()
    with torch.no_grad():
        pred = model(emf_norm)
    pos_err = (pred[:, :3] - poses_cos[:, :3]).cpu().numpy() * 1000.0
    pos_rmse = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1) / 3))

    cos_true = np.clip(poses_cos[:, 3:6].cpu().numpy(), -1.0, 1.0)
    cos_pred = np.clip((pred[:, 3:6] * 10.0).cpu().numpy(), -1.0, 1.0)
    ang_err = np.degrees(np.arccos(cos_pred)) - np.degrees(np.arccos(cos_true))
    ang_rmse = np.sqrt(np.mean(np.sum(ang_err ** 2, axis=1) / 3))

    model.train()
    return pos_rmse, ang_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="dataset/train_data.csv")
    parser.add_argument("--test_csv", type=str, default="dataset/test_data.csv")
    parser.add_argument("--emf_out", type=str, default="dataset/train_data.pt")
    parser.add_argument("--stats_out", type=str, default="dataset/train_stats.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_blocks", type=int, default=7)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--out", type=str, default="checkpoint_model/model_7residual.pth")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    poses_train = load_poses(args.train_csv, device)
    poses_train_rad = poses_train.clone()
    poses_train_rad[:, 3:6] = cos_to_angle(poses_train[:, 3:6])
    emf_train = compute_emf(poses_train_rad, device)

    emf_out_dir = os.path.dirname(args.emf_out)
    if emf_out_dir:
        os.makedirs(emf_out_dir, exist_ok=True)
    torch.save(emf_train.cpu(), args.emf_out)
    print(f"Computed EMF from {args.train_csv} ({emf_train.shape[0]} samples) -> saved to {args.emf_out}")

    mean = emf_train.mean(dim=0, keepdim=True)
    std = emf_train.std(dim=0, keepdim=True)
    std[std == 0] = 1e-8

    stats_out_dir = os.path.dirname(args.stats_out)
    if stats_out_dir:
        os.makedirs(stats_out_dir, exist_ok=True)
    torch.save({"mean": mean.cpu(), "std": std.cpu()}, args.stats_out)
    print(f"Saved normalization stats -> {args.stats_out}")
    emf_train_norm = (emf_train - mean) / std * 10

    target_train = poses_train.clone()
    target_train[:, 3:6] = target_train[:, 3:6] / 10.0

    dataset = TensorDataset(emf_train_norm, target_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Train samples: {poses_train.shape[0]}")

    has_test = False
    try:
        poses_test = load_poses(args.test_csv, device)
        poses_test_rad = poses_test.clone()
        poses_test_rad[:, 3:6] = cos_to_angle(poses_test[:, 3:6])
        emf_test = compute_emf(poses_test_rad, device)
        emf_test_norm = (emf_test - mean) / std * 10
        has_test = True
        print(f"Test samples : {poses_test.shape[0]}")
    except FileNotFoundError:
        print(f"{args.test_csv} not found, skipping periodic evaluation.")

    model = WPRefiner(num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    loss_history = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            loss_pos = loss_fn(pred[:, :3], y_batch[:, :3])
            loss_ang = loss_fn(pred[:, 3:6], y_batch[:, 3:6])
            loss = loss_pos + loss_ang

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        loss_history.append(avg_loss)
        if len(loss_history) > 3:
            loss_history.pop(0)
        if len(loss_history) == 3:
            has_increase = any(loss_history[i] > loss_history[i - 1] for i in range(1, 3))
            max_loss, min_loss = max(loss_history), min(loss_history)
            is_fluctuating = (max_loss - min_loss) / (min_loss + 1e-8) > 0.0
            if has_increase and is_fluctuating:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.98
                loss_history = []

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            lr = optimizer.param_groups[0]["lr"]
            msg = f"[Epoch {epoch}/{args.epochs}] loss={avg_loss:.6f} lr={lr:.6f}"
            if has_test:
                pos_rmse, ang_rmse = evaluate(model, poses_test, emf_test_norm)
                msg += f" | test pos_rmse={pos_rmse:.3f}mm ang_rmse={ang_rmse:.3f}deg"
            print(msg)

    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean.cpu(),
        "std": std.cpu(),
        "num_blocks": args.num_blocks,
    }, args.out)
    print(f"Saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
