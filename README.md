# Invariant Representation Learning for Treatment Effect Estimation
# Introduction

This repository contains software and data for Invariant Representation Learning for Treatment Effect Estimation.

This paper develops nearly invariant causal estimation (NICE), an estimation procedure for causal inference from observational data where the data comes from multiple datasets. The datasets are drawn from distinct environments, corresponding to distinct distributions. Each environment has the same causal mechanism for the outcome, but the distributions are different in other ways. Using
[invariant risk minimization](https://arxiv.org/abs/1907.02893), NICE takes advantage of the differences and similarities across the datasets to estimate the causal effect.

# Requirements and setup
see src/package-list.txt

# Reproducing results
1. Experiment 1 (src/experiment_synthetic)

  * To reproduce all the plots in experiment 1, run main.ipynb

2. Experiment 2 (src/SpeedDating)

  * To reproduce the tables for experiment 2. 

  * First, simulate data according to generate_simSpeedDate.R.

  * Then, run `./run_script/run_local`.

  * Last, Make tables by running speed_tables.ipynb. 


3. Experiment 3 (src/finite_sample)

  * To reproduce the plots for experiment 3. 

  * First, simulate data by running `gen_dat.py`

  * Then, run `./run_script/run_local` if using cpu, `./run_script/run_gpu` if using gpu.(Make sure you specify where to store the data/result)

  * Last, make plots by running plot.ipynb 

Sample data and sample results are stored in dat/ and res/. 
If you have any questions, feel free to contact claudia.j.shi@gmail.com
