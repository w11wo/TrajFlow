import datetime
import json
import math
import os
import pickle
from argparse import ArgumentParser
from ast import literal_eval
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm.contrib.concurrent import process_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("data/"))
    parser.add_argument("--city", type=str, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument("--trajectory_length", type=int, default=120)
    parser.add_argument(
        "--grid_cell_km",
        type=float,
        default=1.0,
        help="Side length of each grid cell in km (default: 1.0, which corresponds to JISMesh level-3)",
    )
    parser.add_argument(
        "--lat_offset", type=int, default=10_000, help="Offset for latitude cell indices to ensure positivity"
    )
    parser.add_argument(
        "--lon_offset", type=int, default=20_000, help="Offset for longitude cell indices to ensure positivity"
    )
    parser.add_argument(
        "--lon_stride",
        type=int,
        default=50_000,
        help="Stride for longitude cell codes (must be > max possible lon cell index + lon_offset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def trip_distance_m(lats: np.ndarray, lons: np.ndarray) -> float:
    return sum(haversine_m(lats[i], lons[i], lats[i + 1], lons[i + 1]) for i in range(len(lats) - 1))


def latlon_to_cell_code(
    lat: float,
    lon: float,
    lat_cell_deg: float,
    lon_cell_deg: float,
    lat_offset: int = 10_000,
    lon_offset: int = 20_000,
    lon_stride: int = 50_000,
) -> int:
    cell_lat = int(math.floor(lat / lat_cell_deg)) + lat_offset
    cell_lon = int(math.floor(lon / lon_cell_deg)) + lon_offset
    return cell_lat * lon_stride + cell_lon


def resample_trajectory(lats: np.ndarray, lons: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a variable-length trajectory to exactly n points via linear
    interpolation along the index axis.  Returns shape (n, 2) — [lat, lon].
    """
    old_idx = np.arange(len(lats), dtype=float)
    new_idx = np.linspace(0, len(lats) - 1, n)
    new_lats = interp1d(old_idx, lats, kind="linear")(new_idx)
    new_lons = interp1d(old_idx, lons, kind="linear")(new_idx)
    return np.stack([new_lats, new_lons], axis=1)  # (n, 2)


def timestamp_to_slot(ts: pd.Timestamp) -> int:
    """Convert a timestamp to a 5-minute slot index (0-287)."""
    return int((ts.hour * 60 + ts.minute) // 5)


def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_row(
    args,
    geo2coords: dict[int, list],
    lat_cell_deg: float,
    lon_cell_deg: float,
    lat_offset: int = 10_000,
    lon_offset: int = 20_000,
    lon_stride: int = 50_000,
    trajectory_length: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    _, row = args
    rid_list = row["rid_list"]
    time_list = row["time_list"]
    rid_list = literal_eval(row["rid_list"])
    time_list_str = row["time_list"].split(",")
    time_list = [datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in time_list_str]

    geometry_list = [geo2coords[rid] for rid in rid_list]
    geometry_list = [g for geo in geometry_list for g in geo]
    trajectory = np.array(geometry_list)  # [L, 2] (lon, lat)

    lats, lons = trajectory[:, 1], trajectory[:, 0]
    n = len(trajectory)

    t_start, t_end = time_list[0], time_list[-1]
    duration_s = (t_end - t_start).total_seconds()
    dist_m = trip_distance_m(lats, lons)
    avg_speed = dist_m / duration_s if duration_s > 0 else 0.0
    avg_dis = dist_m / n if n > 0 else 0.0
    departure_slot = timestamp_to_slot(t_start)
    origin_code = latlon_to_cell_code(lats[0], lons[0], lat_cell_deg, lon_cell_deg, lat_offset, lon_offset, lon_stride)
    dest_code = latlon_to_cell_code(lats[-1], lons[-1], lat_cell_deg, lon_cell_deg, lat_offset, lon_offset, lon_stride)

    cond = np.array(
        [departure_slot, dist_m, duration_s, float(n), avg_dis, avg_speed, float(origin_code), float(dest_code)],
        dtype=np.float64,
    )

    trajectory = resample_trajectory(trajectory[:, 1], trajectory[:, 0], n=trajectory_length)
    return cond, trajectory


def main(args):
    train_file = pd.read_csv(args.input_dir / args.city / "train.csv")
    test_file = pd.read_csv(args.input_dir / args.city / "test.csv")
    val_file = pd.read_csv(args.input_dir / args.city / "val.csv")
    geo_file = args.input_dir / args.city / "roadmap.geo"
    output_dir = args.input_dir / args.city

    geo = pd.read_csv(geo_file)
    geo2coords = {rid: literal_eval(coords) for rid, coords in zip(geo["geo_id"], geo["coordinates"])}

    min_lat = min([c[0] for coord in geo2coords.values() for c in coord])
    max_lat = max([c[0] for coord in geo2coords.values() for c in coord])
    min_lon = min([c[1] for coord in geo2coords.values() for c in coord])
    max_lon = max([c[1] for coord in geo2coords.values() for c in coord])

    lat_cell_deg = args.grid_cell_km / 111.32  # degrees per km (lat)
    mid_lat_rad = math.radians((min_lat + max_lat) / 2)
    lon_cell_deg = args.grid_cell_km / (111.32 * math.cos(mid_lat_rad))  # degrees per km (lon)

    fn = partial(
        process_row,
        geo2coords=geo2coords,
        lat_cell_deg=lat_cell_deg,
        lon_cell_deg=lon_cell_deg,
        lat_offset=args.lat_offset,
        lon_offset=args.lon_offset,
        lon_stride=args.lon_stride,
        trajectory_length=args.trajectory_length,
    )

    train_results = process_map(fn, train_file.iterrows(), max_workers=32, total=len(train_file), chunksize=1000)
    test_results = process_map(fn, test_file.iterrows(), max_workers=32, total=len(test_file), chunksize=1000)
    val_results = process_map(fn, val_file.iterrows(), max_workers=32, total=len(val_file), chunksize=1000)

    train_conds, train_trajectories = zip(*train_results)
    test_conds, test_trajectories = zip(*test_results)
    val_conds, val_trajectories = zip(*val_results)

    train_conds = np.stack(train_conds, dtype=np.float64)
    test_conds = np.stack(test_conds, dtype=np.float64)
    val_conds = np.stack(val_conds, dtype=np.float64)

    train_trajectories = np.stack(train_trajectories, dtype=np.float64)
    test_trajectories = np.stack(test_trajectories, dtype=np.float64)
    val_trajectories = np.stack(val_trajectories, dtype=np.float64)

    lat_mean = train_trajectories[:, :, 0].mean()
    lat_std = train_trajectories[:, :, 0].std()
    lon_mean = train_trajectories[:, :, 1].mean()
    lon_std = train_trajectories[:, :, 1].std()

    def normalize_trajs(trajs: np.ndarray) -> np.ndarray:
        trajs_norm = trajs.copy()
        trajs_norm[:, :, 0] = (trajs[:, :, 0] - lat_mean) / lat_std
        trajs_norm[:, :, 1] = (trajs[:, :, 1] - lon_mean) / lon_std
        return trajs_norm

    train_segments_norm = normalize_trajs(train_trajectories)
    test_segments_norm = normalize_trajs(test_trajectories)
    val_segments_norm = normalize_trajs(val_trajectories)

    NORM_COLS = [1, 2, 3, 4, 5]  # columns to normalise
    cond_means = train_conds[:, NORM_COLS].mean(axis=0)  # (5,)
    cond_stds = train_conds[:, NORM_COLS].std(axis=0)  # (5,)

    def normalize_conds(conds: np.ndarray) -> np.ndarray:
        conds_norm = conds.copy()
        conds_norm[:, NORM_COLS] = (conds[:, NORM_COLS] - cond_means) / cond_stds
        return conds_norm

    train_conds_norm = normalize_conds(train_conds)
    test_conds_norm = normalize_conds(test_conds)
    val_conds_norm = normalize_conds(val_conds)

    all_cell_codes = np.unique(
        np.concatenate([train_conds[:, 6:8], test_conds[:, 6:8], val_conds[:, 6:8]], axis=0).astype(int)
    )
    mesh_mapping_dict = {int(code): idx for idx, code in enumerate(all_cell_codes)}

    save_pkl(train_conds_norm, os.path.join(output_dir, "train_conditions.pkl"))
    save_pkl(test_conds_norm, os.path.join(output_dir, "test_conditions.pkl"))
    save_pkl(val_conds_norm, os.path.join(output_dir, "val_conditions.pkl"))

    save_pkl(train_segments_norm, os.path.join(output_dir, "train_trajectories.pkl"))
    save_pkl(test_segments_norm, os.path.join(output_dir, "test_trajectories.pkl"))
    save_pkl(val_segments_norm, os.path.join(output_dir, "val_trajectories.pkl"))

    save_pkl(mesh_mapping_dict, os.path.join(output_dir, "mesh_mapping_dict.pkl"))

    traj_stats_path = os.path.join(output_dir, "traj_mean_std.txt")
    with open(traj_stats_path, "w") as f:
        f.write(f"lat_mean: {lat_mean}\n")
        f.write(f"lat_std: {lat_std}\n")
        f.write(f"lon_mean: {lon_mean}\n")
        f.write(f"lon_std: {lon_std}\n")

    norm_col_names = ["total_dis", "total_time", "total_len", "avg_dis", "avg_speed"]
    cond_stats_path = os.path.join(output_dir, "conditions_mean_std.txt")
    with open(cond_stats_path, "w") as f:
        for name, m, s in zip(norm_col_names, cond_means, cond_stds):
            f.write(f"{name}_mean: {m}\n")
            f.write(f"{name}_std: {s}\n")

    grid_meta = {
        "encoding": "custom_grid",
        "city": args.city,
        "grid_cell_km": args.grid_cell_km,
        "lat_cell_deg": lat_cell_deg,
        "lon_cell_deg": lon_cell_deg,
        "lat_offset": args.lat_offset,
        "lon_offset": args.lon_offset,
        "lon_stride": args.lon_stride,
        "num_grid_cells": len(mesh_mapping_dict),
        "bbox": [min_lat, max_lat, min_lon, max_lon],
    }
    grid_meta_path = os.path.join(output_dir, "grid_meta.json")
    with open(grid_meta_path, "w") as f:
        json.dump(grid_meta, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
