import json
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from map_manager import MapManager
from metrics import (
    compute_local_trajectory_metrics,
    compute_trajectory_distance,
    compute_trajectory_radius,
    get_metric_distribution,
    js_divergence,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--roadmap_geo_path", type=Path, required=True)
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument(
        "--label_trajs_csv",
        type=Path,
        required=True,
        help="Path to the .csv file containing map-matched label trajectories",
    )
    parser.add_argument(
        "--gen_trajs_csv",
        type=Path,
        required=True,
        help="Path to the .csv file containing map-matched generated trajectories",
    )
    args = parser.parse_args()
    return args


def main(args):
    map_manager = MapManager(args.city)

    # read roadmap file
    geo = pd.read_csv(args.roadmap_geo_path)
    road_gps = []
    for _, row in geo.iterrows():
        coordinates = eval(row["coordinates"])
        road_line = LineString(coordinates=coordinates)
        center_coord = road_line.centroid
        center_lon, center_lat = center_coord.x, center_coord.y
        road_gps.append((center_lon, center_lat))

    label_trajs_df = pd.read_csv(args.label_trajs_csv)
    gen_trajs_df = pd.read_csv(args.gen_trajs_csv)

    # NOTE: some generated trajectories may have failed map-matching, so we do a left join to keep all label trajectories
    merged_df = label_trajs_df.merge(gen_trajs_df, on="id", suffixes=("_label", "_gen"), how="left")
    label_rids = [
        literal_eval(rids) for rids in tqdm(merged_df["cpath_label"].tolist(), desc="Loading RIDs") if pd.notna(rids)
    ]
    prediction_rids = [
        literal_eval(rids) for rids in tqdm(merged_df["cpath_gen"].tolist(), desc="Loading RIDs") if pd.notna(rids)
    ]

    real_distance_list = [
        compute_trajectory_distance(rid_list, road_gps)
        for rid_list in tqdm(label_rids, desc="Computing Real Distances")
    ]
    real_radius_list = [
        compute_trajectory_radius(rid_list, road_gps) for rid_list in tqdm(label_rids, desc="Computing Real Radii")
    ]
    real_distance_distribution, real_distance_bins = get_metric_distribution(real_distance_list)
    real_radius_distribution, real_radius_bins = get_metric_distribution(real_radius_list)

    predicted_distance_list = [
        compute_trajectory_distance(rid_list, road_gps)
        for rid_list in tqdm(prediction_rids, desc="Computing Predicted Distances")
    ]
    predicted_radius_list = [
        compute_trajectory_radius(rid_list, road_gps)
        for rid_list in tqdm(prediction_rids, desc="Computing Predicted Radii")
    ]
    predicted_distance_distribution, _ = get_metric_distribution(
        predicted_distance_list, reference_metric_bins=real_distance_bins
    )
    predicted_radius_distribution, _ = get_metric_distribution(
        predicted_radius_list, reference_metric_bins=real_radius_bins
    )

    distance_js_divergence = js_divergence(real_distance_distribution, predicted_distance_distribution)
    radius_js_divergence = js_divergence(real_radius_distribution, predicted_radius_distribution)

    def group_trajectories_by_grid_od(rid_lists: list[list[int]]):
        od_groups = dict()
        for idx, rid_list in enumerate(rid_lists):
            o_rid, d_rid = rid_list[0], rid_list[-1]
            o_rid_x, o_rid_y = map_manager.gps2grid(*road_gps[o_rid])
            d_rid_x, d_rid_y = map_manager.gps2grid(*road_gps[d_rid])
            key = (o_rid_x * map_manager.img_height + o_rid_y, d_rid_x * map_manager.img_height + d_rid_y)
            od_groups[key] = od_groups.get(key, []) + [idx]
        return od_groups

    real_od2traj_id = group_trajectories_by_grid_od(label_rids)
    predicted_od2traj_id = group_trajectories_by_grid_od(prediction_rids)

    haudorff_list, dtw_list, edr_list = [], [], []
    common_keys = set(real_od2traj_id.keys()) & set(predicted_od2traj_id.keys())
    for key in tqdm(common_keys, desc="Computing Local Trajectory Metrics"):
        num_points = min(len(real_od2traj_id[key]), len(predicted_od2traj_id[key]))
        for i in range(num_points):
            real_idx = real_od2traj_id[key][i]
            pred_idx = predicted_od2traj_id[key][i]
            hausdorff, dtw, edr = compute_local_trajectory_metrics(
                label_rids[real_idx], prediction_rids[pred_idx], road_gps
            )
            haudorff_list.append(hausdorff)
            dtw_list.append(dtw)
            edr_list.append(edr)

    eval_metrics = {
        "distance": distance_js_divergence.item(),
        "radius": radius_js_divergence.item(),
        "hausdorff": np.mean(haudorff_list).item(),
        "dtw": np.mean(dtw_list).item(),
        "edr": np.mean(edr_list).item(),
    }

    print("Eval Metrics:", eval_metrics)

    with open(args.label_trajs_csv.parent / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
