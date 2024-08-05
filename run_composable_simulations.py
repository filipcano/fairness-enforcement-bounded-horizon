import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
import json, os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import psutil, time, sys, logging
import scipy.stats as stats


sys.path.append('ffb')
from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data

import make_cpp_input_from_ML_model
import make_one_composable_simulation


# Configure logging
simulation_errors_file = 'simulation_errors.log'
with open(simulation_errors_file, 'w') as fp:
    fp.write("")
logging.basicConfig(filename='simulation_errors.log', level=logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="experimental_setups/simulation_composable1.json", help="Path to JSON file with parameters on which simulations to run")
    return parser.parse_args()


def compute_total_iterations(params):
    total_iterations = len(params["dataset_env"])
    total_iterations *= len(params["ml_algos"])
    total_iterations *= len(params["costs"])
    total_iterations *= len(params["lambda_decision"])
    total_iterations *= len(params["time_horizon"])
    total_iterations *= len(params["dp_thresholds"])
    total_iterations *= params["n_simulations"]
    total_iterations *= len(params["composability_types"])
    return total_iterations


def check_memory(threshold=0.3):
    """
    Check if available memory is below the threshold.
    :param threshold: float, minimum fraction of free memory required
    :return: bool, True if memory is above the threshold, else False
    """
    mem = psutil.virtual_memory()
    # print(f"{mem.available=}, {mem.total=}", mem.available / mem.total)
    return mem.available / mem.total > threshold



def get_bounds_on_acceptance_rates(dataset, dp_threshold, time_horizon):
    data_path = f"datasets/{dataset}/raw"
    if dataset == "adult":
        X, y, s = load_adult_data(path=data_path)
    if dataset == "german":
        X, y, s = load_german_data(path=data_path)
    if dataset == "compas":
        X, y, s = load_compas_data(path=data_path)
    if dataset == "bank_marketing":
        X, y, s = load_bank_marketing_data(path=data_path)

    mean = np.mean(y.values)
    p = np.mean(s.values)
    min_acc_rate = max(0, mean-dp_threshold/2)
    max_acc_rate = min_acc_rate + dp_threshold
    N = np.ceil(1/(max_acc_rate - min_acc_rate))

    if N >= time_horizon/2:
        probability = 0 
    else:

        P_N_minus_1 = stats.binom.cdf(N-1, time_horizon, p)
        P_T_minus_N = stats.binom.cdf(time_horizon-N, time_horizon, p)
        # Compute the probability of getting and enforceable trace
        probability = P_T_minus_N - P_N_minus_1

    return min_acc_rate, max_acc_rate, probability


def run_simulation(task):
    try:
        ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, composability_type, n_windows, idx_simul = task
        dp_threshold_str = f"{dp_threshold:.4f}".split('.')[1]
        min_acc_rate, max_acc_rate, prob = get_bounds_on_acceptance_rates(dataset, dp_threshold, time_horizon)
        # if prob < 0.4 and composability_type == "bounded_acc_rates":
        #     logging.error(f"In simulation {task}, the probability of gettig and enforceable trace was too small {prob=}")
        #     return 0
        
        res_df = make_one_composable_simulation.main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, n_windows, composability_type, min_acc_rate=min_acc_rate, max_acc_rate=max_acc_rate, debug=0)
        # if len(res_df) == time_horizon*n_windows:
        res_df['sim_num'] = idx_simul
        res_df['step'] = res_df.index
        res_df['acc_util'] = res_df['utility'].cumsum()
        res_df['acc_cost'] = res_df['cost'].cumsum()
        res_df['intervention'] = (res_df['cost'] != 0).astype(int)
        res_df['n_interventions'] = res_df['intervention'].cumsum()
        path_to_save = f"experimental_results/simulation_results/{dataset}_{sensitive_attr}_{ml_algo}_{time_horizon}_{cost_type}_{n_cost_bins}_{dp_threshold_str}_{lambda_decision}_composable_{composability_type}_{n_windows}.csv"
        if os.path.isfile(path_to_save):
            res_df.to_csv(path_to_save, mode='a', header=False, index=True)
        else:
            res_df.to_csv(path_to_save)
        return 1

        #     return 1
        # else:
        #     logging.error(f"In task {task}, the resulting dataframe was incomplete, with {len(res_df)} rows")

    except Exception as e:
        logging.error(f"Error in simulation {task}: {e}")
        return 0  # Indicate failure to the callback

def main_parallel(params): # parallel version
    total_iterations = compute_total_iterations(params)
    n_cost_bins = params["n_cost_bins"]
    n_windows = params["n_time_windows"]
    print(f"{total_iterations=}")

    tasks = []
    
    for idx_simul in range(params["n_simulations"]):
        for dataset_env in params["dataset_env"]:
            dataset = dataset_env["dataset"]
            target_attr = dataset_env["target_attr"]
            sensitive_attr = dataset_env["sensitive_attr"]
            for ml_algo in params["ml_algos"]:
                ml_model = f"experimental_results/ML_models/model_{dataset}_{sensitive_attr}_{ml_algo}.pkl"
                for time_horizon in params["time_horizon"]:
                    for composability_type in params["composability_types"]:
                        if composability_type == "none":
                            # these three parameters are just placeholders, they result does not depend on them for composability_type == "none"
                            cost_type = "constant"
                            dp_threshold = 0.1
                            lambda_decision = 0 
                            task = (ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, composability_type, n_windows, idx_simul)
                            tasks.append(task)
                        else:
                            for cost_type in params["costs"]:
                                for lambda_decision in params["lambda_decision"]:
                                    
                                    for dp_threshold in params["dp_thresholds"]:
                                    
                                        task = (ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, composability_type, n_windows, idx_simul)
                                        tasks.append(task)

    total_iterations = len(tasks)
    n_processes = 16
    if not "long_window" in params["composability_types"]:
        n_processes = multiprocessing.cpu_count()-1
    with tqdm(total=total_iterations) as pbar:
        with Pool(processes=n_processes) as pool:
            for task in tasks:
                pool.apply_async(run_simulation, args=(task,), callback=lambda _: pbar.update(1))
            pool.close()
            pool.join()



def main_sequential(params): # parallel version
    total_iterations = compute_total_iterations(params)
    n_cost_bins = params["n_cost_bins"]
    n_windows = params["n_time_windows"]
    print(f"{total_iterations=}")
    with tqdm(total=total_iterations) as pbar:
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
                                    for composability_type in params["composability_types"]:
                                        task = (ml_model, ml_algo, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, composability_type, n_windows, idx_simul)
                                        # tasks.append(task)
                                        run_simulation(task)
                                        pbar.update(1)



    # with tqdm(total=total_iterations) as pbar:
    #     with Pool(processes=multiprocessing.cpu_count()-1) as pool:
    #         for task in tasks:
    #             time.sleep(np.random.random())
    #             while not check_memory():
    #                 print("Low memory, waiting...")
    #                 time.sleep(5)  # Wait for 5 seconds before checking memory again
    #             pool.apply_async(run_simulation, args=(task,), callback=lambda _: pbar.update(1))
    #         pool.close()
    #         pool.join()


    


if __name__ == "__main__":
    args = parse_args()
    with open(args.params_file, 'r') as fp:
        params = json.load(fp)


    with open("experimental_results/simulation_results/AAnon-enforcement-log.txt", "w") as fp:
        fp.write("")

    # params_sequential = params.copy()
    # params_sequential["n_simulations"] = 1
    # main_sequential(params_sequential)

    print(params["composability_types"])

    
    main_parallel(params)
    # main_sequential(params)
    # check_memory()