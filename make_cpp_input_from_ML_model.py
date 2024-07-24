import torch
import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import subprocess

import sys
import os
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_model", type=str, default="experimental_results/ML_models/model_adult_race_erm.pkl", help="Path to the trained ML model")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "german", "compas", "bank_marketing"], help="Choose a dataset")
    parser.add_argument("--sensitive_attr", type=str, default="race", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--time_horizon", type=int, default=100, help="Time horizon for fairness properties")
    parser.add_argument("--n_cost_bins", type=int, default=10, help="Number of bins for the cost")
    parser.add_argument("--dp_epsilon", type=float, default=0.15, help="Bound on demographic parity")
    parser.add_argument("--cost", type=str, default="paired", choices=["constant", "paired"], 
                        help="Cost for each decision. Can be constant or paired with using the provided ML model")
    return parser.parse_args()

def assert_input_integrity(dataset, sensitive_attr, ml_model):
    if dataset not in ml_model:
        raise ValueError(f"ML model {ml_model} was not trained on {dataset} data")
    if sensitive_attr not in ml_model:
        raise ValueError(f"Attribute {sensitive_attr} was not considered as sensitive when training {ml_model}")

    valid_pairings = {
        "adult" : ["sex", "race"],
        "german" : ["sex", "age"],
        "bank_marketing" : ["age"],
        "compas" : ["sex", "race"]
    }

    for i in valid_pairings.keys():
        if dataset == i:
            if sensitive_attr in valid_pairings[i]:
                return sensitive_attr
            return valid_pairings[i][0]
    raise ValueError(f"Unknown dataset: {dataset}")


def main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon, cost_type, debug=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sensitive_attr = assert_input_integrity(dataset, sensitive_attr, ml_model)

    net = torch.load(ml_model)

    
    
    if dataset == "adult":
        if debug > 0:
            print(f"Dataset: adult")
        X, y, s = load_adult_data(path="datasets/adult/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "german":
        if debug > 0:
            print(f"Dataset: german")
        X, y, s = load_german_data(path="datasets/german/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "compas":
        if debug > 0:
            print(f"Dataset: compas")
        X, y, s = load_compas_data(path="datasets/compas/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "bank_marketing":
        if debug > 0:
            print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path="datasets/bank_marketing/raw", sensitive_attribute=sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if debug > 1:
        print(f"{ml_model=}, {sensitive_attr=}, {time_horizon=}, {n_cost_bins=}, {dp_epsilon=}, {cost_type=}")

    ml_algo = ml_model.split('/')[-1].split('_')[2 + len(dataset.split('_'))].split('.')[0]
    
    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    numurical_cols = X.select_dtypes("float32").columns
    if len(numurical_cols) > 0:

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        scaler = StandardScaler().fit(X[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        X[numurical_cols] = X[numurical_cols].pipe(scale_df, scaler)
    
    data = PandasDataSet(X, y, s)
    loader = DataLoader(data, batch_size=512, shuffle=True)

    probGroup0 = s.value_counts()[0]/(len(s))
    probGroup1 = s.value_counts()[1]/(len(s))

    X0 = X[s[sensitive_attr] == 0]
    s0 = s[s[sensitive_attr] == 0]
    y0 = y[s[sensitive_attr] == 0]

    X1 = X[s[sensitive_attr] == 1]
    s1 = s[s[sensitive_attr] == 1]
    y1 = y[s[sensitive_attr] == 1]

    data0 = PandasDataSet(X0, y0, s0)
    loader0 = DataLoader(data0, batch_size=512, shuffle=True)

    data1 = PandasDataSet(X1, y1, s1)
    loader1 = DataLoader(data1, batch_size=512, shuffle=True)

    outputsG0 = []
    outputsG1 = []

    for data, target, sensitive in loader0:

        if ml_algo == "laftr":
            h, decoded, output, adv_pred = net(data.to(device), sensitive.to(device))
        else:
            h, output = net(data.to(device))
        outputsG0.append(output.detach().cpu().numpy())
    
    for data, target, sensitive in loader1:
        if ml_algo == "laftr":
            h, decoded, output, adv_pred = net(data.to(device), sensitive.to(device))
        else:
            h, output = net(data.to(device))
        outputsG1.append(output.detach().cpu().numpy())

    outputsG0 = np.concatenate(outputsG0)
    outputsG1 = np.concatenate(outputsG1)

    n_bins = 2*n_cost_bins
    counts, _ = np.histogram(outputsG0, bins=n_bins, range=(0, 1))
    relative_frequenciesG0 = counts / sum(counts)
    counts, _ = np.histogram(outputsG1, bins=n_bins, range=(0, 1))
    relative_frequenciesG1 = counts / sum(counts)

    prob_rej_0 = np.sum(relative_frequenciesG0[0:n_cost_bins])
    prob_acc_0 = 1-prob_rej_0
    prob_rej_1 = np.sum(relative_frequenciesG1[0:n_cost_bins])
    prob_acc_1 = 1-prob_rej_1
    
    prob_rej_0_norm_factor = 0 if prob_rej_0 == 0 else 1/prob_rej_0
    prob_acc_0_norm_factor = 0 if prob_acc_0 == 0 else 1/prob_acc_0
    prob_rej_1_norm_factor = 0 if prob_rej_1 == 0 else 1/prob_rej_1
    prob_acc_1_norm_factor = 0 if prob_acc_1 == 0 else 1/prob_acc_1

    probs_rej_0 = prob_rej_0_norm_factor*relative_frequenciesG0[:n_cost_bins]
    probs_acc_0 = prob_acc_0_norm_factor*relative_frequenciesG0[n_cost_bins:]
    probs_rej_1 = prob_rej_1_norm_factor*relative_frequenciesG1[:n_cost_bins]
    probs_acc_1 = prob_acc_1_norm_factor*relative_frequenciesG1[n_cost_bins:]


    # print(relative_frequenciesG0)
    # print(relative_frequenciesG1)

    min_acc_rate = 0
    max_acc_rate = 1
    buff_gAacc = 0
    buff_gAseen = 0
    buff_gBacc = 0
    buff_gBseen = 0

    dp_epsilon_str = f"{dp_epsilon:.4f}".split('.')[1]
    output_str = f"{time_horizon} {n_cost_bins} {dp_epsilon} {min_acc_rate} {max_acc_rate}\n"
    output_str += f"{buff_gAacc} {buff_gAseen} {buff_gBacc} {buff_gBseen}\n"
    for i in range(1,n_cost_bins+1):
        output_str += f"{(1/n_bins)*(i-0.5):.4f} "
    output_str += "\n"

    if cost_type == "constant":
        for i in range(n_cost_bins):
            output_str += f"{0.5*probGroup0*(1/n_cost_bins):.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{0.5*probGroup0*(1/n_cost_bins):.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{0.5*probGroup1*(1/n_cost_bins):.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{0.5*probGroup1*(1/n_cost_bins):.5f} "

    elif cost_type == "hybrid":
        for i in range(n_cost_bins):
            output_str += f"{prob_rej_0*(1/n_cost_bins)*probGroup0:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_acc_0*(1/n_cost_bins)*probGroup0:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_rej_1*(1/n_cost_bins)*probGroup1:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_acc_1*(1/n_cost_bins)*probGroup1:.5f} "
    
    elif cost_type == "paired":
        for i in range(n_cost_bins):
            output_str += f"{prob_rej_0*probs_rej_0[i]*probGroup0:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_acc_0*probs_acc_0[i]*probGroup0:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_rej_1*probs_rej_1[i]*probGroup1:.5f} "
        output_str += "\n"
        for i in range(n_cost_bins):
            output_str += f"{prob_acc_1*probs_acc_1[i]*probGroup1:.5f} "
    clean_ml_model = ml_model.split('/')[-1].split('.')[0]
    tmp_input_path = f"cpp_inputs/{clean_ml_model}_{time_horizon}_{n_cost_bins}_{dp_epsilon_str}_{cost_type}.txt"
    tmp_saved_policy_path = f"experimental_results/dp_enforcer_policies/{clean_ml_model}_{time_horizon}_{n_cost_bins}_{dp_epsilon_str}_{cost_type}.txt"
    with open(tmp_input_path, "w") as fp:
        fp.write(output_str)

    # print("CPP input start:")
    # print(output_str)
    # print("CPP input end")


    # Command and arguments
    command = ['./dp_enforcer.o', '--save_policy', f'--saved_policy_file={tmp_saved_policy_path}']

    # Open the input file
    with open(tmp_input_path, 'r') as input_file:
        # Call the C++ executable with subprocess.run
        result = subprocess.run(
            command,
            stdin=input_file,          # Set stdin to read from the file
            capture_output=True,       # To capture stdout and stderr
            text=True                  # To get outputs as strings (not bytes)
        )

    # Print the outputs and errors, if any
    if debug > 0:
        print("Output:", result.stdout)
    



    # print(output_str)






    




if __name__ == "__main__":
    args = parse_args()    
    # seed_everything(0)
    main(args.ml_model, args.dataset, args.sensitive_attr, args.time_horizon, args.n_cost_bins, args.dp_epsilon, args.cost)