import torch
import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data
from ffb.utils import InfiniteDataLoader

import make_cpp_input_from_ML_model


def parse_val_table(val_table_filepath, time_horizon):
    df = pd.read_csv(val_table_filepath, delim_whitespace=True, header=None, 
                 names=['rem_dec', 'gAseen', 'gAacc', 'gBacc', 'val'])
    df['gBseen'] = 100 - df['rem_dec'] - df['gAseen']
    # df['dp'] = np.abs(df['gAacc']/(1+df['gAseen']) - df['gBacc']/(1+df['gBseen']))
    return df



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--ml_model", type=str, default="experimental_results/ML_models/model_adult_race_erm.pkl", help="Path to the trained ML model")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "german", "compas", "bank_marketing"], help="Choose a dataset")
    parser.add_argument("--target_attr", type=str, default="income", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="race", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--time_horizon", type=int, default=100, help="Time horizon for fairness properties")
    parser.add_argument("--n_cost_bins", type=int, default=10, help="Number of bins for the cost")
    parser.add_argument("--dp_epsilon", type=float, default=0.15, help="Bound on demographic parity")
    parser.add_argument("--lambda_decision", type=float, default=1, help="Probability of accepting the shield recommendation")
    parser.add_argument("--cost-type", type=str, default="paired", choices=["constant", "paired"], 
                        help="Cost for each decision. Can be constant or paired with using the provided ML model")
    return parser.parse_args()



def load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon, cost_type, debug = 1):
    dp_epsilon_str = f"{dp_epsilon:.4f}".split('.')[1]
    clean_ml_model = ml_model.split('/')[-1].split('.')[0]
    if debug > 1:
        print(clean_ml_model)

    shield_filepath = f"experimental_results/dp_enforcer_policies/{clean_ml_model}_{time_horizon}_{n_cost_bins}_{dp_epsilon_str}.txt"

    if not os.path.isfile(shield_filepath):
        if debug > 0:
           print("Shield did not exist, starting computing...")
        make_cpp_input_from_ML_model.main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon, cost_type)



    shield_df = pd.read_csv(shield_filepath, delim_whitespace=True, header=None, 
                 names=['rem_dec', 'gAseen', 'gAacc', 'gBacc', 'val'])
    
    shield_df['gBseen'] = time_horizon - shield_df['rem_dec'] - shield_df['gAseen']
    shield_df.set_index(['gAseen', 'gAacc', 'gBseen', 'gBacc'], inplace=True)
    return shield_df



def get_shield_value(df, gAseen, gAacc, gBseen, gBacc, dp_threshold):
    # Retrieve the 'val' series for faster access
    val_series = df['val']
    # Use 'get' with a default value of -1 (assumes cost is always positive)
    val = val_series.get((gAseen, gAacc, gBseen, gBacc), -1)
    if val != -1:
        return val
    dp = np.abs(gAacc/(1.0+gAseen) - gBacc/(1.0+gBseen))
    if dp <= dp_threshold:
        return 0
    return np.inf


def make_one_simulation(net, data_loader, shield_df, dp_threshold, time_horizon, n_windows = 1, lambda_decision = 1):
    """
    net: ML classifier
    data_loader: infinite data loader with the test split of the dataset, shuffle=True, batch_size = 1
    shield_df: df containing shield values. df contains columns [rem_dec, gAseen, gAacc, gBacc, gBseen, val], 
    where val is the expected cost to be minimized.
    n_windows: how many windows of size time_horizon to simulate
    lambda_decision: when expected cost of changing the ML decision is smaller than the expected cost of keeping, the decision will be changed with probability lambda_decision.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{dp_threshold=}")

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    cost_decision = [] # cost of each decision. Add to get cost of the run
    utility = [] # utility of each decision. Add to get utility of run

    gAseen = 0
    gAacc = 0
    gBseen = 0
    gBacc = 0

    for step, (x, y, s) in enumerate(data_loader):
        
        if step > time_horizon:
            break
        h, output = net(x.to(device))
        score = output.detach().cpu().numpy()[0]
        net_proposes_accept = score > 0.5
        cost_of_intervention = np.abs(score - 0.5)

        cost_keep = None
        cost_change = None

        if (s == 0): # Group A, I think
            gAseen += 1
            expected_cost_after_accept = get_shield_value(shield_df, gAseen, gAacc+1, gBseen, gBacc, dp_threshold)
            expected_cost_after_reject = get_shield_value(shield_df, gAseen, gAacc, gBseen, gBacc, dp_threshold)
        else: # Group B, I think
            gBseen += 1
            expected_cost_after_accept = get_shield_value(shield_df, gAseen, gAacc, gBseen, gBacc+1, dp_threshold)
            expected_cost_after_reject = get_shield_value(shield_df, gAseen, gAacc, gBseen, gBacc, dp_threshold)

        
        if net_proposes_accept:
            cost_keep = expected_cost_after_accept
            cost_change = cost_of_intervention + expected_cost_after_reject

            if cost_keep <= cost_change:
                is_decision_accept = True
            else:
                is_decision_accept = not(np.random.random() <= lambda_decision)
                
        else: # net proposes reject
            cost_keep = expected_cost_after_reject
            cost_change = cost_of_intervention + expected_cost_after_accept
            if cost_keep <= cost_change:
                is_decision_accept = False
            else:
                is_decision_accept = np.random.random() <= lambda_decision
        if is_decision_accept:
            if s == 0:
                gAacc += 1
            else:
                gBacc += 1
                    
        final_label = 1 if is_decision_accept else 0
        ml_proposed_label = 1 if net_proposes_accept else 0

        utility.append(1 - np.abs(final_label - y.detach().numpy()[0]))
        if final_label == ml_proposed_label:
            cost_decision.append(0)
        else:
            cost_decision.append(cost_of_intervention)

        log_gAseen.append(gAseen)
        log_gAacc.append(gAacc)
        log_gBseen.append(gBseen)
        log_gBacc.append(gBacc)        


        # print("\nStep: ", step)
        # print(f"{s=}, {y=}")
        # print(f"{score=:2f}, {ml_proposed_label=}, {final_label=}")
        # print(f"{cost_of_intervention=:.2f}")
        # print(f"{expected_cost_after_accept=:.2f}")
        # print(f"{expected_cost_after_reject=:.2f}")
        # print(f"{cost_keep=:.2f}")
        # print(f"{cost_change=:.2f}")
        # print(f"{ml_proposed_label=}, {final_label=}")
        # print(f"{gAseen=}, {gAacc=}, {gBseen=}, {gBacc=}")

        # val_series = shield_df['val']
        # # Use 'get' with a default value of -1 (assumes cost is always positive)
        # val = val_series.get((gAseen, gAacc, gBseen, gBacc), -1)
        # print(f"{val=}")
        # if s == 0:

        #     print(f"Group A: {s == 0}")


        

    res_dict = {
        "gAseen" : log_gAseen,
        "gAacc" : log_gAacc,
        "gBseen" : log_gBseen,
        "gBacc" : log_gBacc,
        "cost" : cost_decision,
        "utility" : utility
    }
    res_df = pd.DataFrame(res_dict)

    return res_df



def main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, debug=1):
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
    
    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X, y_train, y, s_train, s = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=None) # Todo: seed random state
    
    numurical_cols = X.select_dtypes("float32").columns
    if len(numurical_cols) > 0:

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        scaler = StandardScaler().fit(X[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        X[numurical_cols] = X[numurical_cols].pipe(scale_df, scaler)
    
    data = PandasDataSet(X, y, s)
    loader = InfiniteDataLoader(data, batch_size=1, shuffle=True, drop_last=True)

    shield_df = load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, debug=debug)


    res_df = make_one_simulation(net, data, shield_df, dp_threshold, time_horizon)

    res_df['dp'] = np.abs(res_df['gAacc']/(1+res_df['gAseen']) - res_df['gBacc']/(1+res_df['gBseen']))
    pd.set_option('display.max_rows', None)
    if debug > 1:
        print(res_df.tail(1))
        print("cost: ", res_df['cost'].sum())
        print("utility: ", res_df['utility'].sum())
    return res_df




    


if __name__ == "__main__":
    args = parse_args()
    # seed_everything(0)
    main(args.ml_model, args.dataset, args.sensitive_attr, args.time_horizon, args.n_cost_bins, args.dp_epsilon, args.cost_type, args.lambda_decision)
