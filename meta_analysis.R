# Meta-analysis script for effect size calculation and analysis
# Author: Yannuo Feng, Bo Hu
# Description: This script performs a meta-analysis on effect sizes (Cohen's d) 
#              across different genera from an Excel dataset

# Load required libraries
library(meta)
library(metafor)
library(readxl)
library(esc)
library(tidyverse)
library(patchwork)

# Import data from Excel file
data_meta <- read_excel("data.xlsx")

# Function to calculate Cohen's d
calculate_d <- function(grp1m, grp2m, grp1sd, grp2sd, grp1n, grp2n) {
  # Calculate pooled standard deviation and Cohen's d
  pooled_sd <- sqrt(((grp1n - 1) * grp1sd^2 + (grp2n - 1) * grp2sd^2) / (grp1n + grp2n - 2))
  d <- (grp1m - grp2m) / pooled_sd
  return(d)
}

# Calculate Cohen's d for the dataset
data_meta$Cohen_d <- calculate_d(
  grp1m = data_meta$ASD_Mean, 
  grp2m = data_meta$TD_Mean, 
  grp1sd = data_meta$ASD_SD, 
  grp2sd = data_meta$TD_SD, 
  grp1n = data_meta$N_ASD, 
  grp2n = data_meta$N_TD
)

# Calculate standard error (SE) for Cohen's d
# Based on Hedges (1981) "Distribution Theory for Glass's Estimator of Effect Size"
data_meta$SE <- sqrt((data_meta$N_ASD + data_meta$N_TD) / 
                    (data_meta$N_ASD * data_meta$N_TD) + 
                    data_meta$Cohen_d^2 / (2 * (data_meta$N_ASD + data_meta$N_TD)))

# Clean data by removing NaN values in Cohen_d
data_meta_new <- data_meta %>% 
  filter(!is.nan(Cohen_d))  # Removed 106 rows with NaN values

# Count occurrences of each Genus
genus_counts <- data_meta_new %>% 
  count(Genus, name = "n")

# Filter data to include only genera with 3 or more occurrences
data_meta_final <- data_meta_new %>%
  inner_join(genus_counts, by = "Genus") %>%
  filter(n >= 3) %>%
  select(-n)  # Remove the count column; 1075 rows removed

# Extract unique genera for analysis
unique_genus <- unique(data_meta_final$Genus)

# Initialize results dataframe
results <- data.frame()

# Perform meta-analysis for each genus
for (genus in unique_genus) {
  # Filter data for the current genus
  data_example <- data_meta_final %>% 
    filter(Genus == genus)
  
  # Conduct random-effects meta-analysis
  meta_data <- metagen(
    TE = Cohen_d,           # Treatment effect (Cohen's d)
    seTE = SE,             # Standard error
    data = data_example,   # Dataset
    studlab = paste(Author), # Study labels
    sm = "SMD",            # Summary measure: standardized mean difference
    fixed = FALSE,         # No fixed-effect model
    random = TRUE          # Random-effects model
  )
  
  # Extract key results
  result_row <- data.frame(
    Genus = genus,
    k = meta_data[["k"]],                # Number of studies
    TE_random = meta_data[["TE.random"]], # Random-effects estimate
    lower_random = meta_data[["lower.random"]], # Lower CI
    upper_random = meta_data[["upper.random"]], # Upper CI
    pval_random = meta_data[["pval.random"]],   # P-value
    Q = meta_data[["Q"]],                # Heterogeneity statistic
    I2 = meta_data[["I2"]],              # I-squared (heterogeneity)
    tau2 = meta_data[["tau2"]]           # Between-study variance
  )
  
  # Append results
  results <- rbind(results, result_row)
}

# Filter significant results (p <= 0.05)
filtered_results <- results %>% 
  filter(pval_random <= 0.05)

# Export results to CSV
write.table(
  filtered_results, 
  file = "result.csv", 
  row.names = FALSE, 
  col.names = TRUE, 
  sep = ","
)

# Optional: Print confirmation
cat("Meta-analysis completed. Results saved to 'result.csv'.\n")
