import torch
import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data

def parse_val_table(val_table_filepath, time_horizon):
    df = pd.read_csv(val_table_filepath, delim_whitespace=True, header=None, 
                 names=['rem_dec', 'gAseen', 'gAacc', 'gBacc', 'val'])
    df['gBseen'] = 100 - df['rem_dec'] - df['gAseen']
    # df['dp'] = np.abs(df['gAacc']/(1+df['gAseen']) - df['gBacc']/(1+df['gBseen']))
    return df




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_model", type=str, default="experimental_results/ML_models/model_adult_race_erm.pkl", help="Path to the trained ML model")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "german", "compas", "bank_marketing"], help="Choose a dataset")
    parser.add_argument("--target_attr", type=str, default="income", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="sex", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--time_horizon", type=int, default=10, help="Time horizon for fairness properties")
    parser.add_argument("--n_cost_bins", type=int, default=10, help="Number of bins for the cost")
    parser.add_argument("--dp_epsilon", type=float, default=0.15, help="Bound on demographic parity")
    return parser.parse_args()

def main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon):
    net = torch.load(ml_model)
    if dataset == "adult":
        print(f"Dataset: adult")
        X, y, s = load_adult_data(path="datasets/adult/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "german":
        print(f"Dataset: german")
        X, y, s = load_german_data(path="datasets/german/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "compas":
        print(f"Dataset: compas")
        X, y, s = load_compas_data(path="datasets/compas/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "bank_marketing":
        print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path="datasets/bank_marketing/raw", sensitive_attribute=sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
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

    


if __name__ == "__main__":
    args = parse_args()
    main(args.ml_model, args.dataset, args.sensitive_attr, args.time_horizon, args.n_cost_bins, args.dp_epsilon)