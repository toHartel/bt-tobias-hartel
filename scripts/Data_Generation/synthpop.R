# Argumente aus der Kommandozeile einlesen
args <- commandArgs(trailingOnly = TRUE)

# Argumente prüfen
if (length(args) < 3) {
  stop("Bitte übergeben Sie die Eingabedatei, den Wert für k und die Dataset Nr. als Argumente.\nBeispiel: Rscript script.R input.csv 80 1")
}

input_file <- args[1]  # Erster Parameter: Pfad zur Eingabedatei
k <- as.numeric(args[2])  # Zweiter Parameter: Anzahl der zu generierenden Samples
n <- args[3] # Dritter Parameter: Nummer des gewählten Datensatzes

# Benötigte Pakete laden
if (!requireNamespace("synthpop", quietly = TRUE)) install.packages("synthpop")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("glue", quietly = TRUE)) install.packages("glue")
library(synthpop)
library(tidyverse)
library(glue)

# Eingabedaten lesen
data <- read_delim(input_file, delim = ",", show_col_types = FALSE)

# Synthesedaten generieren
syn_data <- syn(data, k = k)

# Ergebnisse anzeigen
print(head(syn_data))

# Output-Dateipfad festlegen
output_file <- glue("Synthetic_Data/Dataset_{n}/synthpop_samples")
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

# Synthesedaten speichern
try(write.syn(syn_data, file = output_file, filetype = "csv"), silent = TRUE)
cat("Synthesedaten erfolgreich gespeichert unter:", output_file, "\n")
