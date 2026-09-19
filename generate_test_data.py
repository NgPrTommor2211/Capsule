import argparse
import os
import numpy as np
import pandas as pd


def generate_random(num_points, seed,
                     x_min=-0.25, x_max=0.25,
                     y_min=-0.25, y_max=0.25,
                     z_min=0.15, z_max=0.65,
                     cos_min=-1.0, cos_max=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, num_points)
    y = rng.uniform(y_min, y_max, num_points)
    z = rng.uniform(z_min, z_max, num_points)
    cos_a = rng.uniform(cos_min, cos_max, num_points)
    cos_b = rng.uniform(cos_min, cos_max, num_points)
    cos_g = rng.uniform(cos_min, cos_max, num_points)
    wp = np.stack([x, y, z], axis=1)
    wr = np.stack([cos_a, cos_b, cos_g], axis=1)
    return np.concatenate([wp, wr], axis=1)


def generate(num_points, n_turns, fixed_cos, seed,
             varying_orientation=False,
             cos_min=-0.8, cos_max=0.8,
             z_min=0.2, z_max=0.5,
             radius_max=0.2):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, n_turns * 2 * np.pi, num_points)
    angle_offset = rng.uniform(-np.pi / 2, np.pi / 2)
    radius = radius_max * np.linspace(1.0, 0.5, num_points)

    x = radius * np.cos(theta + angle_offset)
    y = radius * np.sin(theta + angle_offset)
    z = np.linspace(z_min, z_max, num_points)

    if varying_orientation:
        t = np.linspace(0.0, 1.0, num_points)
        mid, half_range = (cos_min + cos_max) / 2, (cos_max - cos_min) / 2
        cos_a = mid + half_range * np.sin(2 * np.pi * 1.0 * t)
        cos_b = mid + half_range * np.sin(2 * np.pi * 1.3 * t + 1.0)
        cos_g = mid + half_range * np.sin(2 * np.pi * 0.7 * t + 2.0)
        wr = np.stack([cos_a, cos_b, cos_g], axis=1)
    else:
        wr = np.full((num_points, 3), fixed_cos)

    wp = np.stack([x, y, z], axis=1)
    return np.concatenate([wp, wr], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--fixed_cos", type=float, default=0.5)
    parser.add_argument("--varying_orientation", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--cos_min", type=float, default=None)
    parser.add_argument("--cos_max", type=float, default=None)
    parser.add_argument("--z_min", type=float, default=0.2)
    parser.add_argument("--z_max", type=float, default=0.5)
    parser.add_argument("--radius_max", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="dataset/test_data.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.random:
        cos_min = args.cos_min if args.cos_min is not None else -0.8
        cos_max = args.cos_max if args.cos_max is not None else 0.8
        data = generate_random(args.num_points, args.seed,
                                z_min=args.z_min, z_max=args.z_max,
                                cos_min=cos_min, cos_max=cos_max)
        pd.DataFrame(data).to_csv(args.out, index=False, header=False)
        print(f"Generated {args.num_points} random points (position + orientation) -> {args.out}")
        return

    cos_min = args.cos_min if args.cos_min is not None else -0.8
    cos_max = args.cos_max if args.cos_max is not None else 0.8
    data = generate(args.num_points, args.n_turns, args.fixed_cos, args.seed,
                     args.varying_orientation, cos_min, cos_max,
                     args.z_min, args.z_max, args.radius_max)
    pd.DataFrame(data).to_csv(args.out, index=False, header=False)
    kind = "varying orientation" if args.varying_orientation else "fixed orientation"
    print(f"Generated {args.num_points} helix points ({args.n_turns} turns, {kind}) -> {args.out}")


if __name__ == "__main__":
    main()
