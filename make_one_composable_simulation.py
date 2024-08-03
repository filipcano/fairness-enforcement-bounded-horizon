import torch
import argparse
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys, os, pathlib, time
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data
from ffb.utils import InfiniteDataLoader

import make_cpp_input_from_ML_model


def parse_val_table(val_table_filepath, time_horizon):
    df = pd.read_csv(val_table_filepath, delim_whitespace=True, header=None, 
                 names=['rem_dec', 'gAseen', 'gAacc', 'gBacc', 'val'])
    df['gBseen'] = 100 - df['rem_dec'] - df['gAseen']
    return df

def compute_dp(gAseen, gAacc, gBseen, gBacc):
    if gAseen == 0:
        gAseen = 1
        gAacc = 0
    if gBseen == 0:
        gBseen = 1
        gBacc = 0
    return np.abs(gAacc/gAseen - gBacc/gBseen)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--ml_model", type=str, default="experimental_results/ML_models/model_adult_race_erm.pkl", help="Path to the trained ML model")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "german", "compas", "bank_marketing"], help="Choose a dataset")
    parser.add_argument("--target_attr", type=str, default="income", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="race", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--time_horizon", type=int, default=30, help="Time horizon for fairness properties")
    parser.add_argument("--n_cost_bins", type=int, default=10, help="Number of bins for the cost")
    parser.add_argument("--dp_epsilon", type=float, default=0.15, help="Bound on demographic parity")
    parser.add_argument("--lambda_decision", type=float, default=1, help="Probability of accepting the shield recommendation")
    parser.add_argument("--cost-type", type=str, default="paired", choices=["constant", "paired"], 
                        help="Cost for each decision. Can be constant or paired with using the provided ML model")
    parser.add_argument("--composability-type", type=str, default="naive", 
                        choices=["naive", "buffered", "long_window", "bounded_acc_rates"], 
                        help="Type of composability")
    parser.add_argument("--n_time_windows", type=int, default=4, help="Number of time windows")
    parser.add_argument("--min_acc_rate", type=float, default=0, 
                        help="Minimum acceptance rate (for bounded_acc_rates shields)")
    parser.add_argument("--max_acc_rate", type=float, default=1, 
                        help="Minimum acceptance rate (for bounded_acc_rates shields)")
    
    return parser.parse_args()


def load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon, cost_type, 
                   min_acc_rate=0, max_acc_rate=1, 
                   buff_gAacc=0, buff_gAseen=0, buff_gBacc=0, buff_gBseen=0, debug = 1):
    


    is_shield_composable = (min_acc_rate != 0) or (max_acc_rate != 1)
    is_shield_composable_buffered = (buff_gAacc != 0) or (buff_gAseen != 0) or (buff_gBacc != 0) or (buff_gBseen != 0)
    is_shield_composable = is_shield_composable or is_shield_composable_buffered

    # print("Is shield composable: ", is_shield_composable)
    # print(f"{min_acc_rate=}, {max_acc_rate=}")


    
    dp_epsilon_str = f"{dp_epsilon:.4f}".split('.')[1]
    clean_ml_model = ml_model.split('/')[-1].split('.')[0]
    if debug > 1:
        print(clean_ml_model)

    base_filename = f"{clean_ml_model}_{time_horizon}_{n_cost_bins}_{dp_epsilon_str}_{cost_type}"

    if is_shield_composable:
        min_acc_rate_str = f"{min_acc_rate:.4f}".split('.')[1]
        max_acc_rate_str = f"{max_acc_rate:.4f}".split('.')[1]
        base_filename = f"{base_filename}_composable_{min_acc_rate_str}_{max_acc_rate_str}_{buff_gAacc}_{buff_gAseen}_{buff_gBacc}_{buff_gBseen}"
    


    shield_filepath = f"experimental_results/dp_enforcer_policies/{base_filename}.txt"

    # if the shield does not exist, it computes it. For parallel executions, if the shield is not there, touches it first before starting the computation, so others can wait for it

    # print(f"{shield_filepath=}")

    if not os.path.isfile(shield_filepath):
        pathlib.Path(shield_filepath).touch()
        if debug > 0:
           print("Shield did not exist, starting computing...")
        make_cpp_input_from_ML_model.main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_epsilon, cost_type, min_acc_rate=min_acc_rate, max_acc_rate=max_acc_rate, buff_gAacc=buff_gAacc, buff_gAseen=buff_gAseen, buff_gBacc=buff_gBacc, buff_gBseen=buff_gBseen,
         debug = debug)
        # print("Done with shield")

    # if the file exists, but it's empty, wait 3s to see if someone else computes it. Eventually stop and raise an error if you waited too much
    max_time_asleep = 300
    time_waited = 0
    while (time_waited < max_time_asleep) and (os.stat(shield_filepath).st_size == 0):
        time.sleep(3)
        time_waited += 3
    # print("Time waited: ", time_waited)


    if time_waited >= max_time_asleep:
        raise Exception("Waited for more than 1 hour for shield to be computed!")
    
    shield_df = pd.read_csv(shield_filepath, delim_whitespace=True, header=None, 
                 names=['rem_dec', 'gAseen', 'gAacc', 'gBacc', 'val'])
    
    # if is_shield_composable_buffered:
        # os.remove(shield_filepath)
    
    shield_df['gBseen'] = time_horizon - shield_df['rem_dec'] - shield_df['gAseen']
    shield_df.set_index(['gAseen', 'gAacc', 'gBseen', 'gBacc'], inplace=True)
    return shield_df

def is_dp_smaller_or_equal_than_kappa(gAseen, gAacc, gBseen, gBacc, kappa):
    LHS = np.abs(gAacc*(gBseen+1.0) - gBacc*(gAseen+1.0))
    RHS = (gAseen+1.0)*(gBseen+1.0)*kappa
    return LHS <= RHS

def get_shield_value(df, gAseen, gAacc, gBseen, gBacc, dp_threshold, 
                     buff_gAseen=0, buff_gAacc=0, buff_gBseen=0, buff_gBacc=0):
    # Retrieve the 'val' series for faster access
    val_series = df['val']
    # Use 'get' with a default value of -1 (assumes cost is always positive)
    val = val_series.get((gAseen, gAacc, gBseen, gBacc), -1)
    if val != -1:
        return val
    # dp = np.abs((gAacc+buff_gAacc)/(1.0+gAseen+buff_gAseen) - (gBacc+buff_gBacc)/(1.0+gBseen+buff_gBseen))
    if is_dp_smaller_or_equal_than_kappa(gAseen+buff_gAseen, gAacc+buff_gAacc, gBseen+buff_gBseen, gBacc+buff_gBacc, dp_threshold):
    # if dp <= dp_threshold:
        return 0
    return np.inf


def make_long_window_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=0):

    """
    See make_one_simulation() for documentation on the input-output of this function
    """
    
    shield_df = load_shield_df(ml_model, dataset, sensitive_attr, time_horizon*n_windows, n_cost_bins, dp_threshold, cost_type, debug=debug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{dp_threshold=}")

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    cost_decision = [] # cost of each decision. Add to get cost of the run
    utility = [] # utility of each decision. Add to get utility of run
    expected_cost_with_intervention = []
    expected_cost_without_intervention = []


    gAseen = 0
    gAacc = 0
    gBseen = 0
    gBacc = 0

    for step, (x, y, s) in enumerate(data_loader):
        
        if step >= time_horizon*n_windows:
            break
        if ml_algo == "laftr":
            h, decoded, output, adv_pred = net(x.to(device), s.to(device))
        else:
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
        expected_cost_with_intervention.append(cost_change)
        expected_cost_without_intervention.append(cost_keep)  


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
        "utility" : utility,
        "exp_cost_int" : expected_cost_with_intervention,
        "exp_cost_no_int" : expected_cost_without_intervention
    }
    res_df = pd.DataFrame(res_dict)

    return res_df


def make_naive_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=0):

    """
    See make_one_simulation() for documentation on the input-output of this function
    """

    shield_df = load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, debug=debug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{dp_threshold=}")

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    cost_decision = [] # cost of each decision. Add to get cost of the run
    utility = [] # utility of each decision. Add to get utility of run
    expected_cost_with_intervention = []
    expected_cost_without_intervention = []


    # cummulative versions
    cum_gAseen = 0
    cum_gAacc = 0
    cum_gBseen = 0
    cum_gBacc = 0

    

    for window_it in range(n_windows):
        gAseen = 0
        gAacc = 0
        gBseen = 0
        gBacc = 0

        for step, (x, y, s) in enumerate(data_loader):
            
            if step >= time_horizon:
                break

            if ml_algo == "laftr":
                h, decoded, output, adv_pred = net(x.to(device), s.to(device))
            else:
                h, output = net(x.to(device))

            score = output.detach().cpu().numpy()[0]
            net_proposes_accept = score > 0.5
            cost_of_intervention = np.abs(score - 0.5)

            cost_keep = None
            cost_change = None

            if (s == 0): # Group A, I think
                gAseen += 1
                cum_gAseen += 1
                expected_cost_after_accept = get_shield_value(shield_df, gAseen, gAacc+1, gBseen, gBacc, dp_threshold)
                expected_cost_after_reject = get_shield_value(shield_df, gAseen, gAacc, gBseen, gBacc, dp_threshold)
            else: # Group B, I think
                gBseen += 1
                cum_gBseen += 1
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
                    cum_gAacc += 1
                else:
                    gBacc += 1
                    cum_gBacc += 1
                        
            final_label = 1 if is_decision_accept else 0
            ml_proposed_label = 1 if net_proposes_accept else 0

            utility.append(1 - np.abs(final_label - y.detach().numpy()[0]))
            if final_label == ml_proposed_label:
                cost_decision.append(0)
            else:
                cost_decision.append(cost_of_intervention)

            log_gAseen.append(cum_gAseen)
            log_gAacc.append(cum_gAacc)
            log_gBseen.append(cum_gBseen)
            log_gBacc.append(cum_gBacc)
            expected_cost_with_intervention.append(cost_change)
            expected_cost_without_intervention.append(cost_keep)
    
    res_dict = {
        "gAseen" : log_gAseen,
        "gAacc" : log_gAacc,
        "gBseen" : log_gBseen,
        "gBacc" : log_gBacc,
        "cost" : cost_decision,
        "utility" : utility,
        "exp_cost_int" : expected_cost_with_intervention,
        "exp_cost_no_int" : expected_cost_without_intervention
    }

    res_df = pd.DataFrame(res_dict)

    return res_df


def make_bounded_acc_rates_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, 
                          min_acc_rate, max_acc_rate, debug=0):
    
    """
    See make_one_simulation() for documentation on the input-output of this function
    """

    shield_df = load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, 
                               min_acc_rate=min_acc_rate, max_acc_rate=max_acc_rate, debug=debug)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{dp_threshold=}")

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    cost_decision = [] # cost of each decision. Add to get cost of the run
    utility = [] # utility of each decision. Add to get utility of run
    expected_cost_with_intervention = []
    expected_cost_without_intervention = []

    # cummulative versions
    cum_gAseen = 0
    cum_gAacc = 0
    cum_gBseen = 0
    cum_gBacc = 0

    failure = False # this flag gets true if in some windows, the problem is unenforceable

    for window_it in range(n_windows):
        gAseen = 0
        gAacc = 0
        gBseen = 0
        gBacc = 0

        for step, (x, y, s) in enumerate(data_loader):
            
            if step >= time_horizon:
                break

            if ml_algo == "laftr":
                h, decoded, output, adv_pred = net(x.to(device), s.to(device))
            else:
                h, output = net(x.to(device))

            score = output.detach().cpu().numpy()[0]
            net_proposes_accept = score > 0.5
            cost_of_intervention = np.abs(score - 0.5)

            cost_keep = None
            cost_change = None

            if (s == 0): # Group A, I think
                gAseen += 1
                cum_gAseen += 1
                expected_cost_after_accept = get_shield_value(shield_df, gAseen, gAacc+1, gBseen, gBacc, dp_threshold)
                expected_cost_after_reject = get_shield_value(shield_df, gAseen, gAacc, gBseen, gBacc, dp_threshold)
            else: # Group B, I think
                gBseen += 1
                cum_gBseen += 1
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
                    cum_gAacc += 1
                else:
                    gBacc += 1
                    cum_gBacc += 1
                        
            final_label = 1 if is_decision_accept else 0
            ml_proposed_label = 1 if net_proposes_accept else 0

            utility.append(1 - np.abs(final_label - y.detach().numpy()[0]))
            if final_label == ml_proposed_label:
                cost_decision.append(0)
            else:
                cost_decision.append(cost_of_intervention)

            log_gAseen.append(cum_gAseen)
            log_gAacc.append(cum_gAacc)
            log_gBseen.append(cum_gBseen)
            log_gBacc.append(cum_gBacc)
            expected_cost_with_intervention.append(cost_change)
            expected_cost_without_intervention.append(cost_keep)

        if gAseen <= np.ceil(1/dp_threshold) or gBseen <= np.ceil(1/dp_threshold):
            failure = True
            break
    
    res_dict = {
        "gAseen" : log_gAseen,
        "gAacc" : log_gAacc,
        "gBseen" : log_gBseen,
        "gBacc" : log_gBacc,
        "cost" : cost_decision,
        "utility" : utility,
        "exp_cost_int" : expected_cost_with_intervention,
        "exp_cost_no_int" : expected_cost_without_intervention
    }
    res_df = pd.DataFrame(res_dict)

    return res_df, failure


def make_buffered_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=0):

    """
    See make_one_simulation() for documentation on the input-output of this function
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"{dp_threshold=}")

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    cost_decision = [] # cost of each decision. Add to get cost of the run
    utility = [] # utility of each decision. Add to get utility of run
    expected_cost_with_intervention = []
    expected_cost_without_intervention = []

    cum_gAseen = 0
    cum_gAacc = 0
    cum_gBseen = 0
    cum_gBacc = 0

    failure = False # this flag gets true if in some windows, the problem is unenforceable

    for window_it in range(n_windows):
        gAseen = 0
        gAacc = 0
        gBseen = 0
        gBacc = 0

        buff_gAseen = cum_gAseen
        buff_gAacc = cum_gAacc
        buff_gBseen = cum_gBseen
        buff_gBacc = cum_gBacc

        shield_df = load_shield_df(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, 
        buff_gAacc=buff_gAacc, buff_gAseen=buff_gAseen, buff_gBacc=buff_gBacc, buff_gBseen=buff_gBseen,
        debug=debug)

        if get_shield_value(shield_df,gAseen,gAacc,gBseen,gBacc,dp_threshold,buff_gAseen,buff_gAacc,buff_gBseen,buff_gBacc) == np.inf:
            failure = True
            break

        for step, (x, y, s) in enumerate(data_loader):
            
            if step >= time_horizon:
                break

            if ml_algo == "laftr":
                h, decoded, output, adv_pred = net(x.to(device), s.to(device))
            else:
                h, output = net(x.to(device))

            score = output.detach().cpu().numpy()[0]
            net_proposes_accept = score > 0.5
            cost_of_intervention = np.abs(score - 0.5)

            cost_keep = None
            cost_change = None

            if (s == 0): # Group A, I think
                gAseen += 1
                cum_gAseen += 1
                expected_cost_after_accept = get_shield_value(shield_df,gAseen,gAacc+1,gBseen,gBacc,dp_threshold,buff_gAseen,buff_gAacc,buff_gBseen,buff_gBacc) 
                expected_cost_after_reject = get_shield_value(shield_df,gAseen,gAacc,gBseen,gBacc,dp_threshold,buff_gAseen,buff_gAacc,buff_gBseen,buff_gBacc)
            else: # Group B, I think
                gBseen += 1
                cum_gBseen += 1
                expected_cost_after_accept = get_shield_value(shield_df,gAseen,gAacc,gBseen,gBacc+1,dp_threshold,buff_gAseen,buff_gAacc,buff_gBseen,buff_gBacc)
                expected_cost_after_reject = get_shield_value(shield_df,gAseen,gAacc,gBseen,gBacc,dp_threshold,buff_gAseen,buff_gAacc,buff_gBseen,buff_gBacc)

            
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
                    cum_gAacc += 1
                else:
                    gBacc += 1
                    cum_gBacc += 1
                        
            final_label = 1 if is_decision_accept else 0
            ml_proposed_label = 1 if net_proposes_accept else 0

            utility.append(1 - np.abs(final_label - y.detach().numpy()[0]))
            if final_label == ml_proposed_label:
                cost_decision.append(0)
            else:
                cost_decision.append(cost_of_intervention)

            log_gAseen.append(cum_gAseen)
            log_gAacc.append(cum_gAacc)
            log_gBseen.append(cum_gBseen)
            log_gBacc.append(cum_gBacc)
            expected_cost_with_intervention.append(cost_change)
            expected_cost_without_intervention.append(cost_keep)
    
    res_dict = {
        "gAseen" : log_gAseen,
        "gAacc" : log_gAacc,
        "gBseen" : log_gBseen,
        "gBacc" : log_gBacc,
        "cost" : cost_decision,
        "utility" : utility,
        "exp_cost_int" : expected_cost_with_intervention,
        "exp_cost_no_int" : expected_cost_without_intervention
    }
    res_df = pd.DataFrame(res_dict)

    return res_df, failure


def make_one_simulation(composability_type, net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, 
                          min_acc_rate=None, max_acc_rate=None, debug=0):
    
    """
    composability_type: Type of composability, can be ["naive", "buffered", "long_window", "bounded_acc_rates"]
    net: ML classifier
    ml_model: path to the ml_model being used, corresponds with net
    ml_algo: name of the ML algo (erm, hsic, laftr, etc)
    dataset: dataset name (adult, compas, german...)
    data_loader: infinite data loader with the test split of the dataset, shuffle=True, batch_size = 1
    shield_df: df containing shield values. df contains columns [rem_dec, gAseen, gAacc, gBacc, gBseen, val], 
    where val is the expected cost to be minimized.
    sensitive_attr: sensitive attribute taken into account when training net and to check in the simulation
    n_cost_bins: discretization of cost bins in the shield
    dp_threshold: threshold in DP to be enforced
    time_horizon: time horizon for the shield, total simulation has time_horizon*n_windows steps
    n_windows: how many windows of size time_horizon to simulate
    lambda_decision: when expected cost of changing the ML decision is smaller than the expected cost of keeping, the decision will be changed with probability lambda_decision.
    cost_type: Type of cost being modeled by the shield (constant, paired...)
    min_acc_rate: minimum accepting rate for bounded_acc_rates shields
    max_acc_rate: maximum accepting rate for bounded_acc_rates shields
    """

    log_str = f"{composability_type=}, {ml_algo=}, {dataset=}, {sensitive_attr=}, {n_cost_bins=}, {dp_threshold=}, {n_windows=}, {lambda_decision=}, {cost_type=}"

    max_tries = 10

    if min_acc_rate != None and max_acc_rate != None:
        log_str += f", {min_acc_rate=:.4f}, {max_acc_rate=:.4f}"
    
    
    if composability_type == "naive":
        return make_naive_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=debug)
    
    elif composability_type == "buffered":
        failure = True
        n_tries = 0
        while failure and n_tries < max_tries:
            res_df, failure = make_buffered_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=debug)
            if failure:
                n_tries += 1

        if n_tries > 0:
            log_str += f", {n_tries=}\n"
            with open("experimental_results/simulation_results/AAnon-enforcement-log.txt", "a") as fp:
                fp.write(log_str)

        if failure:
            res_df = res_df.head(1)

        return res_df
        

    elif composability_type == "long_window":
        return make_long_window_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, debug=debug)
    elif composability_type == "bounded_acc_rates":
        assert (min_acc_rate != None) and (max_acc_rate != None), "Bounds on acc rates not provided"
        failure = True
        n_tries = 0
        while failure and n_tries < max_tries:
            res_df, failure = make_bounded_acc_rates_simulation(net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, 
                          min_acc_rate, max_acc_rate, debug=0)
            if failure:
                n_tries += 1

        if n_tries > 0:
            log_str += f", {n_tries=}\n"
            with open("experimental_results/simulation_results/AAnon-enforcement-log.txt", "a") as fp:
                fp.write(log_str)
        if failure:
            res_df = res_df.head(1) # makes sure that this does not pass the next filter
        return res_df
    else:
        raise Exception(f"Composability type {composability_type} not implemented")


def preprocess(ml_model, dataset, sensitive_attr, debug=0):

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
    

    ml_algo = ml_model.split('/')[-1].split('_')[2 + len(dataset.split('_'))].split('.')[0]
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
    
    data_loader = PandasDataSet(X, y, s)

    return net, ml_algo, data_loader





def main(ml_model, dataset, sensitive_attr, time_horizon, n_cost_bins, dp_threshold, cost_type, lambda_decision, 
         n_windows, composability_type, min_acc_rate=None, max_acc_rate=None, debug=1):

    net, ml_algo, data_loader = preprocess(ml_model, dataset, sensitive_attr, debug=debug)

    res_df = make_one_simulation(composability_type, net, ml_model, ml_algo, dataset, data_loader, sensitive_attr, 
                          n_cost_bins, dp_threshold, time_horizon, n_windows, lambda_decision, cost_type, 
                          min_acc_rate=min_acc_rate, max_acc_rate=max_acc_rate, debug=debug)

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
    res_df = main(args.ml_model, args.dataset, args.sensitive_attr, args.time_horizon, args.n_cost_bins, args.dp_epsilon, args.cost_type, args.lambda_decision, args.n_time_windows, args.composability_type, args.min_acc_rate, args.max_acc_rate, debug=1)

    res_df.to_csv("merda.csv")
    



    """
        parser.add_argument("--composability-type", type=str, default="naive", 
                        choices=["naive", "buffered", "long_window", "bounded_acc_rates"], 
                        help="Type of composability")
    parser.add_argument("--n_time_windows", type=int, default=4, help="Number of time windows")
    parser.add_argument("--min_acc_rate", type=float, default=0, 
                        help="Minimum acceptance rate (for bounded_acc_rates shields)")
    parser.add_argument("--max_acc_rate", type=float, default=1, 
                        help="Minimum acceptance rate (for bounded_acc_rates shields)")
    
    """
