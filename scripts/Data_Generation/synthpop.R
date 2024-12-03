# Argumente aus der Kommandozeile einlesen
args <- commandArgs(trailingOnly = TRUE)

# Argumente prüfen
if (length(args) < 2) {
  stop("Bitte übergeben Sie die Eingabedatei und den Wert für k als Argumente.\nBeispiel: Rscript script.R input.csv 80")
}

input_file <- args[1]  # Erster Parameter: Pfad zur Eingabedatei
k <- as.numeric(args[2])  # Zweiter Parameter: Anzahl der zu generierenden Samples

# Benötigte Pakete laden
if (!requireNamespace("synthpop", quietly = TRUE)) install.packages("synthpop")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")

library(synthpop)
library(tidyverse)

# Eingabedaten lesen
data <- read_delim(input_file, delim = ",", show_col_types = FALSE)

# Synthesedaten generieren
syn_data <- syn(data, k = k)

# Ergebnisse anzeigen
print(head(syn_data))

# Output-Dateipfad festlegen
output_file <- "Synthetic_Data/synthpop_samples"
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

# Synthesedaten speichern
try(write.syn(syn_data, file = output_file, filetype = "csv"), silent = TRUE)
cat("Synthesedaten erfolgreich gespeichert unter:", output_file, "\n")
