import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
import json, os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import psutil, time

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

def check_memory(threshold=0.2):
    """
    Check if available memory is below the threshold.
    :param threshold: float, minimum fraction of free memory required
    :return: bool, True if memory is above the threshold, else False
    """
    mem = psutil.virtual_memory()
    return mem.available / mem.total > threshold


def run_simulation(task):
    ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, idx_simul = task
    dp_threshold_str = f"{dp_threshold:.4f}".split('.')[1]
    res_df = make_one_simulation.main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, debug=0)
    res_df['sim_num'] = idx_simul
    res_df['step'] = res_df.index
    res_df['acc_util'] = res_df['utility'].cumsum()
    res_df['acc_cost'] = res_df['cost'].cumsum()
    res_df['intervention'] = (res_df['cost'] != 0).astype(int)
    res_df['n_interventions'] = res_df['intervention'].cumsum()
    path_to_save = f"experimental_results/simulation_results/{dataset}_{sensitive_attr}_{ml_algo}_{time_horizon}_{cost_type}_{n_cost_bins}_{dp_threshold_str}_{lambda_decision}.csv"
    if os.path.isfile(path_to_save):
        res_df.to_csv(path_to_save, mode='a', header=False, index=True)
    else:
        res_df.to_csv(path_to_save)

    return 1

def main_parallel(params): # parallel version
    total_iterations = compute_total_iterations(params)
    n_cost_bins = params["n_cost_bins"]
    print(f"{total_iterations=}")

    tasks = []
    for idx_simul in range(params["n_simulations"]):
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
                            
                                task = (ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, idx_simul)
                                tasks.append(task)

    # with tqdm(total=total_iterations) as pbar:
    #     with Pool(processes=int(multiprocessing.cpu_count()/2)) as pool:
    #         for _ in pool.imap_unordered(run_simulation, tasks):
    #             pbar.update(1)


    # with tqdm(total=total_iterations) as pbar:
    #     with Pool(multiprocessing.cpu_count()) as pool:
    #         for task in tasks:
    #             while not check_memory():
    #                 print("Low memory, waiting...")
    #                 time.sleep(5)  # Wait for 5 seconds before checking memory again
    #             pool.apply_async(run_simulation, args=(task,), callback=lambda _: pbar.update(1))
    #         pool.close()
    #         pool.join()

    with tqdm(total=total_iterations) as pbar:
        with Pool(processes=multiprocessing.cpu_count()-1) as pool:
            for task in tasks:
                pool.apply_async(run_simulation, args=(task,), callback=lambda _: pbar.update(1))
            pool.close()
            pool.join()


def main_sequential(params): # sequential version
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

    # params_sequential = params.copy()
    # params_sequential["n_simulations"] = 1
    # main_sequential(params_sequential)
    
    main_parallel(params)