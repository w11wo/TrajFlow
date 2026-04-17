import ast
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from fmm import STMATCH, Network, NetworkGraph, STMATCHConfig

model, stmatch_config = None, None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--roadmap_geo_path", type=Path, required=True)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument(
        "--label_trajs_csv", type=Path, required=True, help="Path to the .npy file containing label trajectories"
    )
    parser.add_argument(
        "--gen_trajs_csv", type=Path, required=True, help="Path to the .npy file containing generated trajectories"
    )
    parser.add_argument(
        "--k", type=int, default=8, help="Number of nearest roads to consider for map matching (default: 8)"
    )
    parser.add_argument(
        "--radius", type=float, default=300 / 1.132e5, help="Search radius for roads in degrees (default: 300m)"
    )
    parser.add_argument("--gps_error", type=float, default=50 / 1.132e5, help="GPS error in degrees (default: 50m)")
    parser.add_argument(
        "--vmax", type=float, default=80 / 3.6 / 1.132e5, help="Maximum speed in degrees/s (default: 80 km/h)"
    )
    parser.add_argument("--num_proc", type=int, default=16, help="Number of parallel processes to use (default: 16)")
    return parser.parse_args()


def map_match_trip(task):
    trip_id, wkt = task
    try:
        result = model.match_wkt(wkt, stmatch_config)
        cpath = list(result.cpath)
        if not cpath:
            return None
        return {"id": trip_id, "opath": list(result.opath), "cpath": cpath}
    except Exception:
        return None


def load_trajs_from_csv(csv_path: Path) -> list[np.ndarray]:
    df = pd.read_csv(csv_path, usecols=["uid", "latitude", "longitude"])
    coords = df[["longitude", "latitude"]].values  # (total_rows, 2)
    uids = df["uid"].values

    # Find split points where uid changes
    splits = np.where(np.diff(uids) != 0)[0] + 1
    trajs = np.split(coords, splits)
    return trajs


def main(args):
    global model, stmatch_config

    geo = pd.read_csv(args.roadmap_geo_path)
    geo = geo.rename(columns={"geo_id": "id"})
    geo["geometry"] = geo["coordinates"].apply(lambda x: LineString(ast.literal_eval(x)))
    geo["source"] = geo["geometry"].apply(lambda x: hash(x.coords[0]))
    geo["target"] = geo["geometry"].apply(lambda x: hash(x.coords[-1]))

    gdf = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[["id", "source", "target", "geometry"]]
    output_shp_path = args.dataset_path / f"{args.city}_network.shp"
    gdf.to_file(output_shp_path)

    network = Network(str(output_shp_path), "id", "source", "target")
    graph = NetworkGraph(network)
    stmatch_config = STMATCHConfig(k_arg=args.k, r_arg=args.radius, gps_error_arg=args.gps_error, vmax_arg=args.vmax)
    model = STMATCH(network, graph)

    label_traj = load_trajs_from_csv(args.label_trajs_csv)
    gen_traj = load_trajs_from_csv(args.gen_trajs_csv)
    assert len(label_traj) == len(gen_traj)

    array_to_wkt = lambda arr: "LINESTRING (" + ", ".join(f"{x} {y}" for x, y in arr) + ")"

    def create_traj_df(trajs):
        wkt = [array_to_wkt(traj) for traj in tqdm(trajs, desc="Converting to WKT")]
        traj_df = pd.DataFrame({"id": range(len(trajs)), "wkt": wkt})
        traj_df["point_count"] = [len(t) for t in trajs]
        traj_df = traj_df.sort_values("point_count", ascending=False)
        return traj_df

    label_traj_df = create_traj_df(label_traj)
    gen_traj_df = create_traj_df(gen_traj)

    def map_match_traj_df(traj_df):
        tasks = ((row.id, row.wkt) for row in traj_df.itertuples())
        ctx = mp.get_context("fork")
        results = []
        with ctx.Pool(processes=args.num_proc) as pool:
            for res in tqdm(pool.imap_unordered(map_match_trip, tasks, chunksize=10), total=len(traj_df)):
                if res is not None:
                    results.append(res)
        return pd.DataFrame(results)

    label_result_df = map_match_traj_df(label_traj_df)
    gen_result_df = map_match_traj_df(gen_traj_df)

    label_result_df.to_csv(args.label_trajs_csv.parent / "label_trajs_map_matched.csv", index=False)
    gen_result_df.to_csv(args.gen_trajs_csv.parent / "gen_trajs_map_matched.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
