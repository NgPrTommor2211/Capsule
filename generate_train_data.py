import argparse
import os
import numpy as np
import pandas as pd


def generate(nx, ny, nz, na, nb, ng,
             x_min=-0.25, x_max=0.25,
             y_min=-0.25, y_max=0.25,
             z_min=0.15, z_max=0.65,
             cos_min=-0.95, cos_max=1.0):
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    a = np.linspace(cos_min, cos_max, na)
    b = np.linspace(cos_min, cos_max, nb)
    g = np.linspace(cos_min, cos_max, ng)
    grid = np.stack(np.meshgrid(x, y, z, a, b, g, indexing="ij"), axis=-1)
    return grid.reshape(-1, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--na", type=int, default=10)
    parser.add_argument("--nb", type=int, default=10)
    parser.add_argument("--ng", type=int, default=10)
    parser.add_argument("--x_min", type=float, default=-0.25)
    parser.add_argument("--x_max", type=float, default=0.25)
    parser.add_argument("--y_min", type=float, default=-0.25)
    parser.add_argument("--y_max", type=float, default=0.25)
    parser.add_argument("--z_min", type=float, default=0.15)
    parser.add_argument("--z_max", type=float, default=0.65)
    parser.add_argument("--cos_min", type=float, default=-0.95)
    parser.add_argument("--cos_max", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="dataset/train_data.csv")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = generate(args.nx, args.ny, args.nz, args.na, args.nb, args.ng,
                     args.x_min, args.x_max, args.y_min, args.y_max,
                     args.z_min, args.z_max, args.cos_min, args.cos_max)
    pd.DataFrame(data).to_csv(args.out, index=False, header=False)
    total = args.nx * args.ny * args.nz * args.na * args.nb * args.ng
    print(f"Generated {total} samples ({args.nx}x{args.ny}x{args.nz}x{args.na}x{args.nb}x{args.ng} grid) -> {args.out}")
    print(f"X: [{data[:,0].min():.3f}, {data[:,0].max():.3f}] m")
    print(f"Y: [{data[:,1].min():.3f}, {data[:,1].max():.3f}] m")
    print(f"Z: [{data[:,2].min():.3f}, {data[:,2].max():.3f}] m")
    print(f"cos: [{data[:,3].min():.3f}, {data[:,3].max():.3f}]")


if __name__ == "__main__":
    main()
