import math

import hausdorff
import numpy as np
from fastdtw import fastdtw
from geopy import distance
from scipy.stats import entropy

rad = math.pi / 180.0
R = 6378137.0


def get_metric_distribution(
    metric_list: list[float], reference_metric_bins: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    max_value = np.max(metric_list)
    # use metric bins from reference/target distribution
    if reference_metric_bins is not None:
        metric_bins = reference_metric_bins
    else:
        metric_bins = np.linspace(0, max_value, 100).tolist()
        metric_bins.append(float("inf"))
        metric_bins = np.array(metric_bins)
    metric_distribution, _ = np.histogram(metric_list, metric_bins)
    return metric_distribution, metric_bins


def js_divergence(p, q):
    p = p / (np.sum(p) + 1e-14)
    q = q / (np.sum(q) + 1e-14)
    m = (p + q) / 2
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def compute_trajectory_distance(rid_list: list[int], road_gps) -> float:
    travel_distance = 0
    for i in range(1, len(rid_list)):
        travel_distance += distance.great_circle(
            (road_gps[rid_list[i - 1]][1], road_gps[rid_list[i - 1]][0]),
            (road_gps[rid_list[i]][1], road_gps[rid_list[i]][0]),
        ).kilometers
    return travel_distance


def compute_trajectory_radius(rid_list: list[int], road_gps) -> float:
    lon_mean = np.mean([road_gps[rid][0] for rid in rid_list])
    lat_mean = np.mean([road_gps[rid][1] for rid in rid_list])
    rad = []
    for rid in rid_list:
        lon = road_gps[rid][0]
        lat = road_gps[rid][1]
        dis = distance.great_circle((lat_mean, lon_mean), (lat, lon)).kilometers
        rad.append(dis)
    return np.mean(rad)


def get_gps_trajectory(rid_list: list[int], road_gps) -> np.ndarray:
    return np.array([road_gps[rid][::-1] for rid in rid_list])


def compute_local_trajectory_metrics(
    real_rids: list[int], pred_rids: list[int], road_gps, edr_threshold: float = 100
) -> tuple[float, float, float]:
    real_gps = get_gps_trajectory(real_rids, road_gps)
    pred_gps = get_gps_trajectory(pred_rids, road_gps)

    # Calculate individual metrics
    h_dist = hausdorff.hausdorff_distance(real_gps, pred_gps, distance="haversine")
    dtw_dist, _ = fastdtw(real_gps, pred_gps, dist=haversine)
    edr_dist = edr(real_gps, pred_gps, edr_threshold)

    return h_dist, dtw_dist, edr_dist


def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = pow(math.sin(dlat / 2.0), 2.0) + math.cos(lat_x) * math.cos(lat_y) * pow(math.sin(dlon / 2.0), 2.0)
    return R * 2 * math.asin(math.sqrt(a))


def great_circle_distance(lon1, lat1, lon2, lat2):
    dlat = rad * (lat2 - lat1)
    dlon = rad * (lon2 - lon1)
    a = math.sin(dlat / 2.0) * math.sin(dlat / 2.0) + math.cos(rad * lat1) * math.cos(rad * lat2) * math.sin(
        dlon / 2.0
    ) * math.sin(dlon / 2.0)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def edr(t0, t1, eps):
    n0 = len(t0)
    n1 = len(t1)
    C = np.full((n0 + 1, n1 + 1), np.inf)
    C[:, 0] = np.arange(n0 + 1)
    C[0, :] = np.arange(n1 + 1)

    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if great_circle_distance(t0[i - 1][0], t0[i - 1][1], t1[j - 1][0], t1[j - 1][1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr
