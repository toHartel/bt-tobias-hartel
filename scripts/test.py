import os
import sys
import multiprocessing
import subprocess
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# Import modules
from Data_Generation.data_synthesizer import ds_generate_data
from Data_Generation.synthetic_data_vault import sdv_generate_data
from Data_Evaluation.sd_metrics import sd_metrics
from Data_Evaluation.resemblance import pairwise_correlation_diff, jsd, wd
from Data_Evaluation.utility import run_utility_eval
from Data_Evaluation.privacy import dcr, nndr, mia

# Set the start method of the multiprocessing module to 'fork' to avoid an error
multiprocessing.set_start_method('fork', force=True)

# Number of samples to generate
n = 5000

# Path to the input data file
data_path = "Original_Data/Dataset_2.csv"
# Split the data into training (70%) and testing (30%) sets 
original_data = pd.read_csv(data_path)
train_data, test_data = train_test_split(original_data, test_size=0.3, random_state=42)
train_data.to_csv("Original_Data/train_data.csv", index=False)
test_data.to_csv("Original_Data/test_data.csv", index=False)

# Use train_data.csv to fit SDG models and generate synthetic data
data_path = "Original_Data/train_data.csv"
test_path = "Original_Data/test_data.csv"
arguments = [data_path, str(n)]


print("Sampling synthpop...")
start_time = time.time()
result = subprocess.run(['Rscript', 'Data_Generation/synthpop.R',   *arguments], capture_output=True, text=True)
print("Time taken to fit and sample from synthpop: {:.2f} seconds".format(time.time() - start_time), "\n")

print("Sampling DataSynthesizer...")
start_time = time.time()
ds_generate_data(data_path=data_path, num_samples=n)
print("Time taken to fit and sample from DataSynthesizer: {:.2f} seconds".format(time.time() - start_time), "\n")

print("Sampling SDV...")
sdv_generate_data(data_path=data_path, num_samples=n)

models = ['copula_gan', 'ctgan', 'tvae', 'gaussian', "ds", "synthpop"]
for model in models:
    print(f"Evaluate \033[1m{model}\033[0m...\n")
    synth_data_path = f"Synthetic_Data/{model}_samples.csv"
    print(sd_metrics(data_path, synth_data_path))
    print("Avg. DCR: ", dcr(data_path, synth_data_path, model, save_hist=True), "\n")
    print("Avg. NNDR: ", nndr(data_path, synth_data_path), "\n")
    # print("MIA: ", mia(data_path, synth_data_path, save_plts=True), "\n")
    print("Difference in pairwise correlation: ", pairwise_correlation_diff(data_path, synth_data_path), "\n")
    print("Average JSD: ", jsd(data_path, synth_data_path), "\n")
    print("Average WD: ", wd(data_path, synth_data_path), "\n")
    utility_model = "random_forest"
    acc_diff, f1_diff = run_utility_eval(data_path, test_path, synth_data_path, "math score", utility_model)
    print(f"Model: {utility_model} | Accuracy diff: {acc_diff} | F1 diff: {f1_diff} \n")
    print("------------------------------------------------------------\n")
