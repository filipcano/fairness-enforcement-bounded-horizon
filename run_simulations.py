import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
import json
from tqdm import tqdm

import make_cpp_input_from_ML_model
import make_one_simulation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="experimental_setups/simulation1.json", help="Path to JSON file with parameters on which simulations to run")
    return parser.parse_args()


def compute_total_iterations(params):
    total_iterations = len(params["dataset_env"])
    total_iterations *= len(params["ml_algos"])
    total_iterations *= len(params["costs"])
    total_iterations *= len(params["lambda_decision"])
    total_iterations *= len(params["time_horizon"])
    total_iterations *= len(params["dp_thresholds"])
    total_iterations *= params["n_simulations"]
    return total_iterations

def main(params):
    total_iterations = compute_total_iterations(params)

    n_cost_bins = params["n_cost_bins"]
    print(f"{total_iterations=}")

    with tqdm(total=total_iterations) as pbar:


        for dataset_env in params["dataset_env"]:
            dataset = dataset_env["dataset"]
            target_attr = dataset_env["target_attr"]
            sensitive_attr = dataset_env["sensitive_attr"]
            for ml_algo in params["ml_algos"]:
                ml_model = f"experimental_results/ML_models/model_{dataset}_{sensitive_attr}_{ml_algo}.pkl"
                for cost_type in params["costs"]:
                    for lambda_decision in params["lambda_decision"]:
                        for time_horizon in params["time_horizon"]:
                            for dp_threshold in params["dp_thresholds"]:
                                for idx_simul in range(params["n_simulations"]):
                                    dp_threshold_str = f"{dp_threshold:.4f}".split('.')[1]
                                    res_df = make_one_simulation.main(ml_model, dataset, sensitive_attr, time_horizon, params["n_cost_bins"], dp_threshold, cost_type, lambda_decision, debug=0)

                                    path_to_save = f"experimental_results/simulation_results/{dataset}_{sensitive_attr}_{ml_algo}_{time_horizon}_{cost_type}_{n_cost_bins}_{dp_threshold_str}_{lambda_decision}_{idx_simul}.csv"
                                    res_df.to_csv(path_to_save)
                                    pbar.update(1)


    


if __name__ == "__main__":
    args = parse_args()
    with open(args.params_file, 'r') as fp:
        params = json.load(fp)
    main(params)