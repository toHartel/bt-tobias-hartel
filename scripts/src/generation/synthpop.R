# Read the input arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if the required arguments are provided
if (length(args) < 3) {
  stop("Please enter input data, number of samples and the dataset name as arguments. \nExample: Rscript script.R input.csv 80 1")
}

input_file <- args[1]  # First parameter: path to dataset
k <- as.numeric(args[2])  # Second parameter: number of synthetic samples to generate
dataset_name <- args[3] # Third parameter: dataset name

# Load required libraries
if (!requireNamespace("synthpop", quietly = TRUE)) install.packages("synthpop")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("glue", quietly = TRUE)) install.packages("glue")
library(synthpop)
library(tidyverse)
library(glue)

# Read input data
data <- read_delim(input_file, delim = ",", show_col_types = FALSE)

# Generate synthetic data
syn_data <- syn(data, k = k)

# Show the first few rows of the synthetic data
print(head(syn_data))

# Set output file path
output_file <- glue("../data/synthetic_data/{dataset_name}/synthpop")
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

# Save the synthetic data
try(write.syn(syn_data, file = output_file, filetype = "csv"), silent = TRUE)
cat("Synthetic data saved in: ", output_file, "\n")
