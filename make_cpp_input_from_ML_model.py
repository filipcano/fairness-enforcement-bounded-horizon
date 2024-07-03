import torch
import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import subprocess

import sys
import os
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data

def assert_sensitive_attribute(dataset, sensitive_attr):
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_model", type=str, default="experimental_results/ML_models/model_adult_erm_2024-07-02T17-21-57.pkl", help="Path to the trained ML model")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult","german", "compas" ,"bank_marketing"], help="Choose a dataset from the available options: adult, german, compas, or bank_marketing")
    parser.add_argument("--sensitive_attr", type=str, default="sex", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--T", type=int, default = 10, 
                        help="Time horizon for fairness properties")
    parser.add_argument("--n_cost_bins", type=int, default=10, 
                        help="number of bins for the cost")
    parser.add_argument("--dp_epsilon", type=float, default=0.15, 
                        help="bound on demographic parity")


    args = parser.parse_args()

    args.sensitive_attr = assert_sensitive_attribute(args.dataset, args.sensitive_attr)

    net = torch.load(args.ml_model)

    
    
    if args.dataset == "adult":
        print(f"Dataset: adult")
        X, y, s = load_adult_data(path="datasets/adult/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "german":
        print(f"Dataset: german")
        X, y, s = load_german_data(path="datasets/german/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "compas":
        print(f"Dataset: compas")
        X, y, s = load_compas_data(path="datasets/compas/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "bank_marketing":
        print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path="datasets/bank_marketing/raw", sensitive_attribute=args.sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
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

    X0 = X[s[args.sensitive_attr] == 0]
    s0 = s[s[args.sensitive_attr] == 0]
    y0 = y[s[args.sensitive_attr] == 0]

    X1 = X[s[args.sensitive_attr] == 1]
    s1 = s[s[args.sensitive_attr] == 1]
    y1 = y[s[args.sensitive_attr] == 1]

    data0 = PandasDataSet(X0, y0, s0)
    loader0 = DataLoader(data0, batch_size=512, shuffle=True)

    data1 = PandasDataSet(X1, y1, s1)
    loader1 = DataLoader(data1, batch_size=512, shuffle=True)

    outputsG0 = []
    outputsG1 = []

    for data, target, sensitive in loader0:
        h, output = net(data)
        outputsG0.append(output.detach().numpy())
    
    for data, target, sensitive in loader1:
        h, output = net(data)
        outputsG1.append(output.detach().numpy())

    outputsG0 = np.concatenate(outputsG0)
    outputsG1 = np.concatenate(outputsG1)

    n_bins = 2*args.n_cost_bins
    counts, _ = np.histogram(outputsG0, bins=n_bins, range=(0, 1))
    relative_frequenciesG0 = counts / sum(counts)
    counts, _ = np.histogram(outputsG1, bins=n_bins, range=(0, 1))
    relative_frequenciesG1 = counts / sum(counts)

    prob_rej_0 = np.sum(relative_frequenciesG0[0:args.n_cost_bins])
    prob_acc_0 = 1-prob_rej_0
    prob_rej_1 = np.sum(relative_frequenciesG1[0:args.n_cost_bins])
    prob_acc_1 = 1-prob_rej_1

    probs_rej_0 = relative_frequenciesG0[:args.n_cost_bins]/prob_rej_0
    probs_acc_0 = relative_frequenciesG0[args.n_cost_bins:]/prob_acc_0
    probs_rej_1 = relative_frequenciesG1[:args.n_cost_bins]/prob_rej_1
    probs_acc_1 = relative_frequenciesG1[args.n_cost_bins:]/prob_acc_1


    # print(relative_frequenciesG0)
    # print(relative_frequenciesG1)

    output_str = f"{args.T} {args.n_cost_bins} {args.dp_epsilon}\n"
    for i in range(1,args.n_cost_bins+1):
        output_str += f"{(1/n_bins)*(i-0.5):.4f} "
    output_str += "\n"

    for i in range(args.n_cost_bins):
        output_str += f"{probs_rej_0[i]*probGroup0:.5f} "
    output_str += "\n"
    for i in range(args.n_cost_bins):
        output_str += f"{probs_acc_0[i]*probGroup0:.5f} "
    output_str += "\n"
    for i in range(args.n_cost_bins):
        output_str += f"{probs_rej_1[i]*probGroup1:.5f} "
    output_str += "\n"
    for i in range(args.n_cost_bins):
        output_str += f"{probs_acc_1[i]*probGroup1:.5f} "
    
    tmp_input_path = "cpp_inputs/prova_input.txt"
    tmp_saved_policy_path = "experimental_results/prova_policy.txt"
    with open(tmp_input_path, "w") as fp:
        fp.write(output_str)


    # Command and arguments
    command = ['./a.out', '--save_policy', f'--saved_policy_file={tmp_saved_policy_path}']

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
    print("Output:", result.stdout)
    



    # print(output_str)






    




if __name__ == "__main__":
    main()