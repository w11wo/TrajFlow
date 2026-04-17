import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import para2point_batch
from src.models.networks import CNN, MLP, BiLSTMVelocity, ConditionalVelocityModel, TransformerVelocity
from src.models.trajectory_gan import ConditionalTrajectoryGAN, TrajectoryGAN
from src.models.trajectory_vae import ConditionalTrajectoryVAE, TrajectoryVAE
from src.utils.visualization import visualize_density_comparison, visualize_trajectories

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import FlowMatchingDataset
from src.eval.inference import FlowMatchingInference

# Field name mapping
switcher = {
    0: "departure_time",
    1: "total_dis",
    2: "total_time",
    3: "total_len",
    4: "avg_dis",
    5: "avg_speed",
    6: "starting_location",
    7: "ending_location",
}


def resolve_device(device_str):
    """Resolve requested device safely with fallback to available CUDA/CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    requested = device_str or "cuda:0"
    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
        except (IndexError, ValueError):
            idx = 0
        if idx >= torch.cuda.device_count():
            print(f"Requested device {requested} not available; fallback to cuda:0")
            return torch.device("cuda:0")
        return torch.device(requested)

    if requested == "cuda":
        return torch.device("cuda:0")

    print(f"Unknown device '{requested}'; fallback to cpu")
    return torch.device("cpu")


def find_config_by_timestamp(exp_savename_str):
    """Find config file by timestamp in folder name"""
    base_paths = ["./outputs"]
    matching_folders = []

    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            for folder in dirs:
                if exp_savename_str in folder:
                    matching_folders.append(os.path.join(root, folder))
    if not matching_folders:
        return None, None, None
    config_path = os.path.join(matching_folders[0], "config.yaml")

    model_path = os.path.join(os.path.dirname(matching_folders[0]), "models", os.path.basename(matching_folders[0]))

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return matching_folders[0], model_path, config

    return None, None, None


def generate_trajectories(
    model,
    all_gt_data,
    all_head,
    lengths,
    traj_mean,
    traj_std,
    cond_mean,
    cond_std,
    config,
    batch_size,
    result_dir,
    dataset,
    device,
    condition_mode="real",
    save_dir="./test_output",
):
    """Generate trajectories using flow matching model"""
    os.makedirs(save_dir, exist_ok=True)

    SAVE_RAW_TRAJS = False
    if SAVE_RAW_TRAJS:
        all_raw_sol_np = []
        all_raw_ground_truth_np = []

    if config["data"]["parametrized"]:
        M = config["data"]["parametrized_M"]
    else:
        M = config["data"]["trajectory_length"]
    traj_length = config["data"]["trajectory_length"]

    # Initialize collectors
    all_sol_np = []
    all_ground_truth_np = []
    all_indices = []
    all_conditions = []
    raw_gt_trajs = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize inference model
    inference = FlowMatchingInference(config=config, model=model, dataset=dataset, save_dir=save_dir, device=device)

    # Determine sampling parameters
    if config.get("ddpm", {}).get("enabled", False):
        n_steps = config["ddpm"]["ddim_steps"]
    else:
        n_steps = config["inference"]["num_steps"]
    solve_method = config["inference"]["sampling_method"]

    od_finer_enabled = config["data"].get("od_finer", False)
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    # Log file for generation details
    log_file = os.path.join(save_dir, "generation_log.txt")

    # Progress bar over batches
    pbar = tqdm(total=total_batches, desc="Generating", unit="batch")

    for batch_idx, batch_data in enumerate(dataloader):
        # Handle conditional vs unconditional data
        if config["condition"]["enabled"]:
            x_1_sample, condition_sample = batch_data
            x_1_sample = x_1_sample.to(device, non_blocking=True)

            transmode_switcher = {"WALK": 0, "CAR": 1, "BUS": 2, "TRAIN": 3, "BIKE": 4}
            if condition_mode != "real":
                condition_sample[:, 8] = transmode_switcher[condition_mode]
            condition_sample = condition_sample.to(device, non_blocking=True)
            current_batch_size = x_1_sample.size(0)

            # Calculate correct indices for this batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + current_batch_size
            batch_indices = list(range(start_idx, end_idx))
            all_indices.extend(batch_indices)

            # Extract raw ground truth for these indices
            for idx in batch_indices:
                if idx < len(all_gt_data):
                    raw_traj = all_gt_data[idx]
                    if len(raw_traj) != M:
                        raw_traj = resample_trajectory(raw_traj, config["data"]["trajectory_length"])
                    for k in range(2):
                        raw_traj[:, k] = raw_traj[:, k] * traj_std[k] + traj_mean[k]
                    raw_gt_trajs.append(raw_traj)
            sol = inference.sample(
                n_samples=current_batch_size, n_steps=n_steps, method=solve_method, condition=condition_sample
            )
        else:
            x_1_sample = batch_data.to(device, non_blocking=True)
            condition_sample = None
            current_batch_size = x_1_sample.size(0)

            sol = inference.sample(n_samples=current_batch_size, n_steps=n_steps, method=solve_method)

        # Convert to numpy
        sol_np = [s.detach().cpu().numpy() for s in sol][-1]
        ground_truth_np = x_1_sample.detach().cpu().numpy()

        # Handle the parameterized trajectories
        if od_finer_enabled:
            traj_dim = M * 2

            sol_traj = sol_np[:, :traj_dim]
            sol_od_finer = sol_np[:, traj_dim:]

            gt_traj = ground_truth_np[:, :traj_dim]
            gt_od_finer = ground_truth_np[:, traj_dim:]

            if config["data"]["parametrized"]:
                para_method = config["data"].get("para_method", "rdp_k")
                sol_traj = para2point_batch(sol_traj, traj_length, para_method)
                gt_traj = para2point_batch(gt_traj, traj_length, para_method)

            sol_np = sol_traj
            ground_truth_np = gt_traj
            od_finer_params_sol = sol_od_finer
            od_finer_params_gt = gt_od_finer
        else:
            if config["data"]["parametrized"]:
                para_method = config["data"].get("para_method", "rdp_k")
                sol_np = para2point_batch(sol_np, traj_length, para_method)
                ground_truth_np = para2point_batch(ground_truth_np, traj_length, para_method)
            od_finer_params_sol = None
            od_finer_params_gt = None

        # Denormalize if needed
        if (
            config["condition"]["enabled"]
            and config["data"]["norm1by1"]
            and config["visualization"].get("norm1by12origialvis", False)
        ):
            condition_sample_np = condition_sample.detach().cpu().numpy()

            if SAVE_RAW_TRAJS:
                raw_sol_np = np.reshape(sol_np, (sol_np.shape[0], -1))
                raw_ground_truth_np = np.reshape(ground_truth_np, (ground_truth_np.shape[0], -1))

            if od_finer_enabled:
                sol_np = inference.denormalize_trajectories(
                    [sol_np], condition_sample_np, dataset, od_finer_params=od_finer_params_sol
                )[0]
                ground_truth_np = inference.denormalize_trajectories(
                    [ground_truth_np], condition_sample_np, dataset, od_finer_params=od_finer_params_gt
                )[0]
            else:
                sol_np = inference.denormalize_trajectories([sol_np], condition_sample_np, dataset)[0]
                ground_truth_np = inference.denormalize_trajectories([ground_truth_np], condition_sample_np, dataset)[0]
        else:
            sol_np = np.reshape(sol_np, (sol_np.shape[0], traj_length, 2))
            ground_truth_np = np.reshape(ground_truth_np, (ground_truth_np.shape[0], traj_length, 2))
            for k in range(2):
                sol_np[:, :, k] = sol_np[:, :, k] * traj_std[k] + traj_mean[k]
                ground_truth_np[:, :, k] = ground_truth_np[:, :, k] * traj_std[k] + traj_mean[k]

        # Collect results
        all_sol_np.extend(sol_np)
        all_ground_truth_np.extend(ground_truth_np)
        if SAVE_RAW_TRAJS and config["data"]["norm1by1"]:
            all_raw_sol_np.extend(raw_sol_np)
            all_raw_ground_truth_np.extend(raw_ground_truth_np)

        # Clean up batch tensors
        del sol, x_1_sample
        if config["condition"]["enabled"]:
            del condition_sample

        pbar.update(1)

        # Free GPU memory once after all batches
        torch.cuda.empty_cache()

    pbar.close()

    # Extract corresponding condition info if available
    total_cond_info = []
    if config["condition"]["enabled"] and all_indices:
        for idx in all_indices:
            if idx < len(all_head):
                total_cond_info.append(all_head[idx])

    # Convert final trajectories to expected format
    total_gen_trajs = [x.reshape(traj_length, 2) for x in all_sol_np]
    total_gt_trajs = [x.reshape(traj_length, 2) for x in all_ground_truth_np]
    if SAVE_RAW_TRAJS:
        total_raw_gt_trajs = [x.reshape(traj_length, 2) for x in all_raw_ground_truth_np]
        total_raw_gen_trajs = [x.reshape(traj_length, 2) for x in all_raw_sol_np]

    # Visualize results
    visualize_trajectories(
        total_gen_trajs, total_gt_trajs, config["data"]["trajectory_length"], parametrized=False, save_folder=save_dir
    )
    visualize_density_comparison(total_gen_trajs, total_gt_trajs, config["data"]["trajectory_length"], save_dir)

    # Save trajectories to CSV
    gen_df = save_trajectories_to_csv(
        trajs=total_gen_trajs,
        cond_info=total_cond_info,
        cond_std=dataset.cond_std,
        cond_mean=dataset.cond_mean,
        save_dir=save_dir,
        traj_type="generated",
    )

    gt_df = save_trajectories_to_csv(
        trajs=total_gt_trajs,
        cond_info=total_cond_info,
        cond_std=dataset.cond_std,
        cond_mean=dataset.cond_mean,
        save_dir=save_dir,
        traj_type="ground_truth",
    )

    if raw_gt_trajs:
        raw_gt_df = save_trajectories_to_csv(
            trajs=raw_gt_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="raw_ground_truth",
        )

    if SAVE_RAW_TRAJS:
        save_trajectories_to_csv(
            trajs=total_raw_gen_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="generated_before_denormalization",
        )
        save_trajectories_to_csv(
            trajs=total_raw_gt_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="ground_truth_before_denormalization",
        )

    return total_gen_trajs, total_gt_trajs, total_cond_info


def resample_trajectory(traj, target_length):
    """Resample a trajectory to the target length"""
    current_length = traj.shape[0]
    if current_length == target_length:
        return traj

    orig_indices = np.linspace(0, current_length - 1, current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)

    resampled = np.zeros((target_length, traj.shape[1]))
    for dim in range(traj.shape[1]):
        resampled[:, dim] = np.interp(new_indices, orig_indices, traj[:, dim])

    return resampled


def save_trajectories_to_csv(trajs, cond_info, cond_std, cond_mean, save_dir, traj_type="generated"):
    """Save trajectories to CSV file (vectorized)"""
    cond_info = np.array(cond_info)
    n_trajs = cond_info.shape[0]
    traj_len = len(trajs[0])
    trajs_arr = np.array(trajs)  # (n_trajs, traj_len, 2)

    # Vectorized condition denormalization
    head_vals = np.zeros((n_trajs, 6))
    head_vals[:, 0] = cond_info[:, 0]
    for j in range(1, 6):
        head_vals[:, j] = cond_info[:, j] * cond_std[j - 1] + cond_mean[j - 1]

    # Vectorized departure time strings
    temp_time = head_vals[:, 0] * 0.0833333
    hours = temp_time.astype(int)
    minutes = ((temp_time - hours) * 60).astype(int)
    seconds = (((temp_time - hours) * 60 - minutes) * 60).astype(int)
    departure_times = [f"2024-04-01 {h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours, minutes, seconds)]

    # Build arrays by repeating per-traj values across traj_len
    uid = np.repeat(np.arange(n_trajs), traj_len)
    dep_time = np.repeat(departure_times, traj_len)
    total_dis = np.repeat(head_vals[:, 1], traj_len)
    total_time = np.repeat(head_vals[:, 2], traj_len)
    total_len = np.repeat(head_vals[:, 3], traj_len)
    avg_dis = np.repeat(head_vals[:, 4], traj_len)
    avg_speed = np.repeat(head_vals[:, 5], traj_len)
    latitude = trajs_arr[:, :, 0].ravel()
    longitude = trajs_arr[:, :, 1].ravel()

    # Vectorized time offsets
    point_indices = np.tile(np.arange(traj_len), n_trajs)
    base_times = np.repeat(pd.to_datetime(departure_times), traj_len)
    time_col = base_times + pd.to_timedelta(point_indices, unit="min")

    df = pd.DataFrame(
        {
            "uid": uid,
            "departure_time": dep_time,
            "total_dis": total_dis,
            "total_time": total_time,
            "total_len": total_len,
            "avg_dis": avg_dis,
            "avg_speed": avg_speed,
            "time": time_col,
            "latitude": latitude,
            "longitude": longitude,
        }
    )

    df.to_csv(os.path.join(save_dir, f"{traj_type}_trajectories.csv"), index=False)
    return df


def visualize_flow_field(model, config, save_dir, device):
    """Visualize the vector field of the flow model"""
    grid_size = 20
    x_range = (-3, 3)
    y_range = (-3, 3)
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    if hasattr(config, "data"):
        traj_length = config.data.trajectory_length
    else:
        traj_length = 30

    dummy_condition = torch.zeros(1, 8).to(device)

    timesteps = [0.0, 0.25, 0.5, 0.75, 0.99]

    for t_val in timesteps:
        plt.figure(figsize=(10, 8))
        U = np.zeros((grid_size, grid_size))
        V = np.zeros((grid_size, grid_size))

        with torch.inference_mode():
            for i in range(grid_size):
                for j in range(grid_size):
                    point = torch.zeros(1, 2, traj_length).to(device)
                    point[0, 0, :] = Y[i, j]
                    point[0, 1, :] = X[i, j]

                    t = torch.tensor([t_val]).to(device)
                    velocity = model(point, t, c=dummy_condition)

                    U[i, j] = velocity[0, 1, 0].cpu().item()
                    V[i, j] = velocity[0, 0, 0].cpu().item()

        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1
        U = U / max_mag
        V = V / max_mag

        plt.quiver(X, Y, U, V, magnitude, cmap="viridis", scale=25)
        plt.colorbar(label="Velocity Magnitude")
        plt.title(f"Flow Vector Field at t={t_val:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(save_dir, f"vector_field_t{t_val:.2f}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Trajectory Generation")
    parser.add_argument("--exp_savename_str", type=str, required=True, help="Timestamp for model selection")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (reduce to 16 or 8 if OOM)",
    )
    parser.add_argument("--placeholder", type=str, default="real", help="Use placeholder conditions")
    parser.add_argument(
        "--USE_GIVEN_STEPS", type=bool, default=False, help="If using the number of given sampling steps parameters"
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of sampling steps")
    parser.add_argument(
        "--method",
        type=str,
        default="euler",
        choices=["euler", "em", "rk4"],
        help="Integration method: euler, euler-maruyama, or runge-kutta4",
    )
    parser.add_argument(
        "--generate_results_dir",
        type=str,
        default="generate_results_baseline_full_clean",
        help="generate results directory",
    )
    args = parser.parse_args()

    args.output_dir = "./%s/%s" % (args.generate_results_dir, args.exp_savename_str)
    condition_mode = args.placeholder

    # Get config and model path
    if args.config is None:
        config_dir, model_dir, config_dict = find_config_by_timestamp(args.exp_savename_str)
        if config_dict is None:
            print(f"Could not find config with timestamp {args.exp_savename_str}")
            return
        config = config_dict
    else:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = config_dict
        model_dir = os.path.dirname(args.config)

    # Resolve device — no global variable
    configured_device = config.get("training", {}).get("device", "cuda:0")
    device = resolve_device(args.device or configured_device)
    print(f"Using device: {device}")

    # Resolve sampling method
    method_map = {"euler": "em", "em": "em", "rk4": "em"}
    effective_method = method_map.get(args.method, config.get("inference", {}).get("sampling_method", "em"))
    config.setdefault("inference", {})
    config["inference"]["sampling_method"] = effective_method

    # Find best model if checkpoint not specified
    if args.checkpoint is None:
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
    else:
        if os.path.isfile(args.checkpoint):
            checkpoint_path = args.checkpoint
        else:
            candidate_file = os.path.join(model_dir, args.checkpoint)
            candidate_epoch = os.path.join(model_dir, f"checkpoint_epoch_{args.checkpoint}.pt")
            if os.path.isfile(candidate_file):
                checkpoint_path = candidate_file
            elif os.path.isfile(candidate_epoch):
                checkpoint_path = candidate_epoch
            else:
                print(f"Checkpoint not found: {args.checkpoint}")
                return -1

    # Create results directory
    result_dir = os.path.join(args.output_dir, f"generation_{condition_mode}_test")
    if args.USE_GIVEN_STEPS:
        result_dir += f"_steps_{args.steps}"
    os.makedirs(result_dir, exist_ok=True)

    # Apply given steps if specified
    if args.USE_GIVEN_STEPS:
        config["inference"]["num_steps"] = args.steps
        config["ddpm"]["ddim_steps"] = args.steps

    # Save config for reference
    with open(os.path.join(result_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)

    # Load data
    # PROJ_PATH = "."
    # input_folder = os.path.join(PROJ_PATH, config["data"]["dataset_folder"])

    # (all_head, traj_mean, traj_std, lengths, cond_mean, cond_std, all_gt_data, grid_mapping_dict) = (
    #     PrepareDataset.loadExistingData(input_folder, resample_length=config["data"]["trajectory_length"])
    # )

    # Create dataset
    dataset = FlowMatchingDataset(config_dict, mode="test")

    # Create model
    input_dim = config["data"]["trajectory_length"] * 2
    hidden_dim = config["model"]["hidden_dim"]

    # Check if this is a baseline model
    if config.get("baseline", {}).get("enabled", True):
        model_type = config.get("baseline", {}).get("type", "flow_matching")
    else:
        model_type = config.get("model", {}).get("type", "error")

    if model_type == "gan":
        if config["condition"]["enabled"]:
            model = ConditionalTrajectoryGAN(config, dataset)
        else:
            model = TrajectoryGAN(config)
    elif model_type == "vae":
        if config["condition"]["enabled"]:
            model = ConditionalTrajectoryVAE(config, dataset)
        else:
            model = TrajectoryVAE(config)
    else:
        if config["model"]["type"] == "mlp" or config["model"]["type"] == "unet":
            if config["condition"]["enabled"]:
                # Infer training location_dim from the saved embedding weight
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                for key in state_dict:
                    if "sid_embedding.weight" in key:
                        dataset.location_dim = state_dict[key].shape[0]

                model = ConditionalVelocityModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    condition_dim=dataset.location_dim,
                    embedding_dim=hidden_dim,
                    dropout_prob=config["flow_matching"]["dropout_prob"],
                    config=config,
                    dataset=dataset,
                )
            else:
                model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
        elif config["model"]["type"] == "cnn":
            model = CNN(input_dim=2, hidden_dim=hidden_dim)
        elif config["model"]["type"] == "transformer":
            model = TransformerVelocity(input_dim=2, hidden_dim=hidden_dim)
        elif config["model"]["type"] == "bilstm":
            model = BiLSTMVelocity(input_dim=2, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {config['model']['type']}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Generate trajectories
    gen_trajs, gt_trajs, cond_info = generate_trajectories(
        model=model,
        all_gt_data=dataset.traj_segments,
        all_head=dataset.all_head,
        lengths=dataset.lengths,
        traj_mean=dataset.traj_mean,
        traj_std=dataset.traj_std,
        cond_mean=dataset.cond_mean,
        cond_std=dataset.cond_std,
        config=config,
        batch_size=args.batch_size,
        result_dir=result_dir,
        dataset=dataset,
        device=device,
        condition_mode=condition_mode,
        save_dir=result_dir,
    )

    print(f"Done. {len(gen_trajs)} trajectories saved to {result_dir}")


if __name__ == "__main__":
    main()
