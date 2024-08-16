# Fairness Shields: A Neurosymbolic Approach for Guaranteed Fair Decision-Making

This is the accompaning repository to reproduce the experiments in the submitted paper. 

## Setup
The code for shield synthesis is developed in C++, a regular compiler, like g++ can be used to compile the C++ code.

The code for training ML classifiers and doing simulations is in python. The repository includes an `environment.yml` file to reproduce the same environment as used in the development of these experiments.

## Repository structure
- *dp_enforcer.cc*: Shield synthesis algorithm for enforcing demographic parity
- *eo_enforcer.cc*: Shield synthesis algorithm for enforcing equal opportunity
- *make_cpp_input_from_ML_model.py*: Handles the logic of running the shield synthesis algorithm to be used with an ML model.
- *make_one_composable_simulation.py*: Handles the simulation of runs with ML models and different types of shields (Static-Fair, Static-BW and Dynamic) for demographic parity.
- *make_one_composable_simulation.py*: Handles the simulation of runs with ML models and different types of shields (Static-Fair, Static-BW and Dynamic) for equal opportunity.
- *run_composable_simulations.py*: Main script to run simulations for demographic parity. 
- *run_composable_simulations_eqopp.py*: Main script to run simulations for equal opportunity.
- *ffb/*: This folder is a fork of the FFB repository (cited in the paper). We used a modified version of their implementation of the different ML algorithms and dataset handling.
- *synthesis_runtimes.py*: Script used to produce the plots on resource usage of the shield synthesis algorithm
- *datasets/*: Folder containing datasets. To alleviate space, it contains just the instructions on how to download the datasets, as they are publicly available.
- *experimental_setups/*: The main scripts use a parameter file to describe which simulations are to be run. This folder contains the parameter files used to produce the experiments in the paper.


## How to run

To train a classifier for each pair of dataset and machine learning algorithm, run 
```
bash ffb/run_all.sh
```

To run simulations for demographic parity with a single window:
```
python run_composable_simulation.py --params_file experimental_setups/simulation_single_window.json
```

To run simulations for equal opportunity with a single window:
```
python run_composable_simulation_eqopp.py --params_file experimental_setups/simulation_eo_singlewindow75.json
```

To run simulations for demographic parity with many windows, static and dynamic shields:
```
python run_composable_simulation.py --params_file experimental_setups/simulation_composable1.json
```

To run simulations for equal opportunity with many windows, static and dynamic shields:
```
python run_composable_simulation_eqopp.py --params_file experimental_setups/simulation_composable_eo.json
```

