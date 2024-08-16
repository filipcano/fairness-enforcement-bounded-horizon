import numpy as np
import pandas as pd
import subprocess, tqdm, os
import matplotlib.pyplot as plt
from matplotlib import rc

fontsize = 26
plt.rcParams.update({'font.size': fontsize})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

colors = {"eo": '#FFA500', "dp": '#008000'}
markers = {"eo": 'x', "dp": 'o'}
linewidth = 3
markersize = 12
prop_names = {"eo" : "$\mathtt{EqOpp}$", "dp" : "$\mathtt{DP}$"}
xscale = 'linear' # can also be 'linear' or 'log'
yscale = 'linear'

graph_ylabels = {
    "time": "Execution time (seconds)", 
     "mem": "Memory usage (MB)" 
}


def get_output_filename(time_horizons_range, fairness_metric, n_runs):
    return f"experimental_results/resource_usage_{fairness_metric}_{time_horizons_range[0]}_{time_horizons_range[-1]}_{n_runs}runs.csv"

def run_benchmark(time_horizons_range, fairness_metric, n_runs):
    output_filename = get_output_filename(time_horizons_range, fairness_metric, n_runs)

    with open("experimental_setups/input_template_runtime.txt", "r") as fp:
        input_string_template = fp.read()
    
    tmp_input_path = "cpp_inputs/runtime_aux.txt"

    comp_times = []
    comp_memory = []
    time_horizons = []

    total_iterations = len(time_horizons_range)*n_runs

    with tqdm.tqdm(total=total_iterations) as pbar:

        for i in range(n_runs):
            for th in time_horizons_range:
                input_str = f"{th} {input_string_template}"
                with open(tmp_input_path, 'w') as fp:
                    fp.write(input_str)
                command = [f'./{fairness_metric}_enforcer.o']
            
                with open(tmp_input_path, 'r') as input_file:
                    # Call the C++ executable with subprocess.run
                    result = subprocess.run(
                        command,
                        stdin=input_file,          # Set stdin to read from the file
                        capture_output=True,       # To capture stdout and stderr
                        text=True                  # To get outputs as strings (not bytes)
                    )
                # print(result.stdout)

                tmp_time = float(result.stdout.split("Elapsed time: ")[1].split(" seconds")[0])
                comp_times.append(tmp_time)

                tmp_mem = float(result.stdout.split("Memory Usage: ")[1].split(" MB")[0])
                comp_memory.append(tmp_mem)
                time_horizons.append(th)
                pbar.update(1)
        
    res_dict = {
        "time_horizon" : time_horizons,
        "comp_time" : comp_times,
        "comp_memory" : comp_memory
    }
    
    res_df = pd.DataFrame(res_dict)

    res_df.to_csv(output_filename, index = False)


def make_plots(results_filenames):
    fairness_metrics = results_filenames.keys()
    to_plot_time = {}
    to_plot_mem = {}




    for fairness_metric in fairness_metrics:
        df = pd.read_csv(results_filenames[fairness_metric])
        to_plot_time[fairness_metric] = {}
        to_plot_time[fairness_metric]["time_horizon"] = np.sort(df.time_horizon.unique())
        to_plot_time[fairness_metric]["mean"] = df.groupby("time_horizon").mean()["comp_time"].values
        to_plot_time[fairness_metric]["std"] = df.groupby("time_horizon").std()["comp_time"].values
        
        to_plot_mem[fairness_metric] = {}
        to_plot_mem[fairness_metric]["time_horizon"] = np.sort(df.time_horizon.unique())
        to_plot_mem[fairness_metric]["mean"] = df.groupby("time_horizon").mean()["comp_memory"].values
        to_plot_mem[fairness_metric]["std"] = df.groupby("time_horizon").std()["comp_memory"].values

    to_plot = {
        "time" : to_plot_time,
        "mem" : to_plot_mem
    }

    for metric in ["time", "mem"]:

        fig, ax = plt.subplots(figsize=(8,5))


        for fairness_metric in fairness_metrics:
            plt.plot(to_plot[metric][fairness_metric]["time_horizon"], to_plot[metric][fairness_metric]["mean"], 
                    label = prop_names[fairness_metric], color = colors[fairness_metric], 
                    marker=markers[fairness_metric], linewidth=linewidth, markersize=markersize
                    )
            plt.fill_between(to_plot[metric][fairness_metric]["time_horizon"], 
                            to_plot[metric][fairness_metric]["mean"] + to_plot_time[fairness_metric]["std"],
                            to_plot[metric][fairness_metric]["mean"] - to_plot_time[fairness_metric]["std"],
                            color = colors[fairness_metric], alpha = 0.2
                            )


        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)

        # Set the scale of y-axis to linear or logarithmic
        plt.yscale(yscale)
        plt.xscale(xscale)


        # Add labels and title
        plt.xlabel('Time Horizon')
        plt.ylabel(graph_ylabels[metric])
        plt.legend()

        # Show the plot
        plot_filepath = f'experimental_results/graphs/resources_{metric}.pdf'
        plt.savefig(plot_filepath, bbox_inches='tight')
        print(f"Plot saved to {plot_filepath}")
        # plt.show()


    






def main():
    time_horizons_dp = np.arange(10,151,10)
    time_horizons_eo = np.arange(10, 81, 10)
    n_runs = 5

    results_filenames = {
        "dp" : get_output_filename(time_horizons_dp, "dp", n_runs),
        "eo" : get_output_filename(time_horizons_eo, "eo", n_runs)
    }
    if not os.path.isfile(results_filenames["dp"]):
        run_benchmark(time_horizons_dp, "dp", n_runs)

    if not os.path.isfile(results_filenames["eo"]):
        run_benchmark(time_horizons_eo, "eo", n_runs)

    make_plots(results_filenames)

    



        


if __name__ == "__main__":
    main()