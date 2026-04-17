# !/bin/bash
cities=(San_Francisco Porto Beijing)
exp_ids=(20260415_021449 20260415_021951 20260415_021624)

for i in "${!cities[@]}"; do
    city="${cities[$i]}"
    exp_id="${exp_ids[$i]}"

    python generate.py --exp_savename_str $exp_id

    python scripts/map_match.py \
        --dataset_path data/$city \
        --roadmap_geo_path data/$city/roadmap.geo \
        --city $city \
        --label_trajs_csv generate_results_baseline_full_clean/$exp_id/generation_real_test/ground_truth_trajectories.csv \
        --gen_trajs_csv generate_results_baseline_full_clean/$exp_id/generation_real_test/generated_trajectories.csv

    python eval.py \
        --roadmap_geo_path data/${city}/roadmap.geo \
        --city $city \
        --label_trajs_csv generate_results_baseline_full_clean/$exp_id/generation_real_test/label_trajs_map_matched.csv \
        --gen_trajs_csv generate_results_baseline_full_clean/$exp_id/generation_real_test/gen_trajs_map_matched.csv
done