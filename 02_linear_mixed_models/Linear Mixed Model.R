#!/usr/bin/env Rscript
# ==============================================================================
# Linear Mixed Model Analysis for Preterm Connectome Study
# ==============================================================================
#
# STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
#        connectome of early school-aged children"
#
# PURPOSE: Fit edge-level linear mixed models to test the effect of gestational
#          age on structural connectivity, controlling for age at scan, gender,
#          handedness, and head motion.
#
# MODEL SPECIFICATION (Equation 3 from manuscript):
#   Edge_weight = β₀ + β₁·GA + β₂·Age + β₃·Gender + β₄·Handedness + β₅·Motion + (1|Subject) + ε
#
# KEY METHODOLOGICAL DETAILS:
#   1. PREDICTOR STANDARDIZATION: All continuous predictors (GA, Age, Motion)
#      are standardized (mean-centered, divided by SD) before model fitting.
#
#   2. OUTCOME STANDARDIZATION (PER-EDGE): Edge weights are standardized
#      PER-EDGE across all subjects. This means each edge's values are
#      mean-centered and SD-scaled using that specific edge's population
#      statistics. This allows β coefficients to be interpreted as effect
#      sizes in SD units.
#
#   3. RANDOM EFFECTS: Random intercepts for Subject handle the repeated
#      measures structure (longitudinal data with multiple sessions per subject).
#
# OUTPUT: CSV file with β coefficients and p-values for each edge
#   - Columns: Node_i, Node_j, Intercept, WeeksPreterm, AgeAtScan, GenderMale,
#              HandednessRight, Total_Bad_Volumes, pval_*
#
# INTERPRETATION:
#   - WeeksPreterm β: SD change in edge strength per 1 SD increase in GA
#   - Positive β: Higher GA associated with stronger connectivity
#   - Negative β: Higher GA associated with weaker connectivity
#
# ==============================================================================

library(lme4)
library(lmerTest)
library(dplyr)
library(readr)
library(parallel)

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# ==============================================================================

# Path to demographic data with subject information
demographic_data_path <- '[PATH_TO_DATA]/SCM/Individual_SCM/independent_longitudinal_dataframe.csv'

# Directory containing weighted connectome matrices (CSV format)
matrix_directory <- '[PATH_TO_DATA]/SCM/Individual_SCM/filtered/JULY_2025/Longitudinal/Longitudinal_filtered_weights'

# Output directory for results
output_dir <- '[PATH_TO_DATA]/SCM/Individual_SCM/filtered/JULY_2025/JULY_2025_RESULTS/LMM_Edges_Rough_Pass_Pvalue'

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

complete_data <- read_csv(demographic_data_path)

# Recode Handedness and standardize continuous predictors
# NOTE: This matches the published methods - standardizing predictor variables
complete_data <- complete_data %>%
  mutate(
    # Recode handedness: 0 = missing, negative = Left, positive = Right
    Handedness = ifelse(Handedness == 0, NA,
                        ifelse(Handedness %in% c(-2, -1), "Left", "Right")),
    Handedness = factor(Handedness, levels = c("Left", "Right")),
    
    # STANDARDIZE CONTINUOUS PREDICTORS (z-score transformation)
    WeeksPreterm = scale(WeeksPreterm),         # Gestational age
    AgeAtScan = scale(AgeAtScan),               # Age at scan
    Total_Bad_Volumes = scale(Total_Bad_Volumes) # Motion parameter
  )

# ==============================================================================
# FUNCTION: Load all adjacency matrices from directory
# ==============================================================================

load_all_adjacency_matrices <- function(directory) {
  print(paste("Loading all matrices from directory:", directory))
  files <- list.files(directory, pattern = "*.csv", recursive = TRUE, full.names = TRUE)
  matrices <- list()
  
  if (length(files) == 0) {
    stop("No CSV files found in the specified directory")
  }
  
  for (file in files) {
    filename <- basename(file)
    parts <- strsplit(filename, "_")[[1]]
    subject_session <- paste(parts[2], parts[3], sep = "_")
    matrices[[subject_session]] <- as.matrix(read.csv(file, header = FALSE))
  }
  
  print(paste("Loaded", length(matrices), "adjacency matrices"))
  return(matrices)
}

# ==============================================================================
# FUNCTION: Standardize edge values PER-EDGE across subjects
# ==============================================================================
#
# This implements the outcome standardization described in the methods:
# "outcomes were standardized by mean centering and dividing by the standard deviation"
#
# RATIONALE: Standardizing per-edge allows comparison of effect sizes across
# edges with different baseline weights. Beta coefficients represent how many
# SDs the edge weight changes per 1 SD change in the predictor.
#
# ==============================================================================

standardize_matrices_per_edge <- function(all_adjacency_matrices) {
  print("Calculating per-edge means and SDs across all subjects...")
  print("This implements the outcome standardization described in the methods section.")
  
  subject_names <- names(all_adjacency_matrices)
  num_nodes <- nrow(all_adjacency_matrices[[1]])
  num_subjects <- length(subject_names)
  
  # Initialize arrays to store edge statistics
  edge_means <- matrix(0, nrow = num_nodes, ncol = num_nodes)
  edge_sds <- matrix(1, nrow = num_nodes, ncol = num_nodes)  # Default SD = 1
  edge_counts <- matrix(0, nrow = num_nodes, ncol = num_nodes)
  
  # Collect all values for each edge across subjects
  print("Collecting edge values across subjects...")
  
  for (i in 1:num_nodes) {
    for (j in 1:num_nodes) {
      if (i < j) {  # Only upper triangle (symmetric matrices)
        values <- numeric(num_subjects)
        count <- 0
        
        for (s in 1:num_subjects) {
          value <- all_adjacency_matrices[[subject_names[s]]][i, j]
          if (value > 0) {  # Only include non-zero values
            count <- count + 1
            values[count] <- value
          }
        }
        
        if (count > 1) {  # Need at least 2 values to calculate SD
          values_clean <- values[1:count]
          edge_means[i, j] <- mean(values_clean)
          edge_sds[i, j] <- sd(values_clean)
          edge_counts[i, j] <- count
        }
      }
    }
  }
  
  print(paste("Edge statistics calculated. Min count:", min(edge_counts[edge_counts > 0]),
              "Max count:", max(edge_counts)))
  
  # Standardize each subject's matrix using edge-specific means/SDs
  print("Standardizing matrices using per-edge statistics...")
  print("Formula: (edge_value - edge_population_mean) / edge_population_SD")
  standardized_matrices <- list()
  
  for (subject in subject_names) {
    matrix <- all_adjacency_matrices[[subject]]
    standardized_matrix <- matrix(0, nrow = num_nodes, ncol = num_nodes)
    
    for (i in 1:num_nodes) {
      for (j in 1:num_nodes) {
        if (i < j && matrix[i, j] > 0) {  # Only standardize non-zero upper triangle
          if (edge_sds[i, j] > 0) {
            # STANDARDIZATION: (value - population_mean) / population_SD
            standardized_matrix[i, j] <- (matrix[i, j] - edge_means[i, j]) / edge_sds[i, j]
            standardized_matrix[j, i] <- standardized_matrix[i, j]  # Symmetric
          } else {
            standardized_matrix[i, j] <- 0
            standardized_matrix[j, i] <- 0
          }
        }
      }
    }
    
    standardized_matrices[[subject]] <- standardized_matrix
  }
  
  print("Per-edge standardization complete")
  print("Interpretation: Beta coefficients now represent effect sizes in SD units")
  
  # Print summary statistics
  non_zero_means <- edge_means[edge_means > 0]
  non_zero_sds <- edge_sds[edge_sds > 0 & edge_sds != 1]
  
  print(paste("Edge mean statistics - Min:", round(min(non_zero_means), 2),
              "Max:", round(max(non_zero_means), 2),
              "Median:", round(median(non_zero_means), 2)))
  print(paste("Edge SD statistics - Min:", round(min(non_zero_sds), 2),
              "Max:", round(max(non_zero_sds), 2),
              "Median:", round(median(non_zero_sds), 2)))
  
  return(standardized_matrices)
}

# ==============================================================================
# FUNCTION: Process a single edge with LMM
# ==============================================================================
#
# Fits the model: edge_value ~ GA + Age + Gender + Handedness + Motion + (1|Subject)
#
# Returns: Vector with node indices, fixed effects, and p-values
#
# ==============================================================================

process_edge_per_edge_standardized <- function(k, edge_indices, all_adjacency_matrices_standardized, complete_data) {
  i <- edge_indices[k, 1]
  j <- edge_indices[k, 2]
  edge_data <- data.frame()
  
  for (subject_session in names(all_adjacency_matrices_standardized)) {
    subject_data <- complete_data %>% filter(new_ID == !!subject_session)
    
    if (nrow(subject_data) > 0) {
      WeeksPreterm <- subject_data$WeeksPreterm         # Gestational age (standardized)
      Gender <- subject_data$Gender
      AgeAtScan <- subject_data$AgeAtScan              # Age at scan (standardized)
      Handedness <- subject_data$Handedness
      Total_Bad_Volumes <- subject_data$Total_Bad_Volumes  # Motion (standardized)
      Subject <- subject_data$ID
      
      # Use PER-EDGE standardized value
      edge_value_standardized <- all_adjacency_matrices_standardized[[subject_session]][i, j]
      
      if (length(WeeksPreterm) == length(Gender) && length(Gender) == length(AgeAtScan) &&
          length(AgeAtScan) == length(Handedness) && length(Handedness) == length(Total_Bad_Volumes) &&
          length(Total_Bad_Volumes) == length(Subject) && length(Subject) == length(edge_value_standardized)) {
        
        edge_data <- rbind(edge_data, data.frame(WeeksPreterm, Gender, AgeAtScan,
                                                 Handedness, Total_Bad_Volumes, Subject,
                                                 edge_value_standardized))
      }
    }
  }
  
  # Fit mixed model with standardized edge values
  if (nrow(edge_data) > 0 && length(unique(edge_data$edge_value_standardized)) > 1) {
    
    # Check for sufficient variation in standardized values
    edge_var <- var(edge_data$edge_value_standardized, na.rm = TRUE)
    
    if (edge_var > 0.01) {  # Minimum variance threshold
      # MODEL FROM EQUATION 3:
      # Edge = β₀ + β₁·GA + β₂·Age + β₃·Gender + β₄·Handedness + β₅·Motion + (1|Subject) + ε
      model <- lmer(edge_value_standardized ~ WeeksPreterm + AgeAtScan + Gender + Handedness +
                      Total_Bad_Volumes + (1 | Subject), data = edge_data)
      
      fixed_effects <- fixef(model)
      coefs <- summary(model)$coefficients
      
      # Extract p-values for variables of interest
      pval_WeeksPreterm <- coefs["WeeksPreterm", "Pr(>|t|)"]
      pval_AgeAtScan <- coefs["AgeAtScan", "Pr(>|t|)"]
      pval_Gender <- if("GenderMale" %in% rownames(coefs)) coefs["GenderMale", "Pr(>|t|)"] else NA
      pval_Handedness <- if("HandednessRight" %in% rownames(coefs)) coefs["HandednessRight", "Pr(>|t|)"] else NA
      pval_Total_Bad_Volumes <- coefs["Total_Bad_Volumes", "Pr(>|t|)"]
      
    } else {
      fixed_effects <- c(Intercept = 0, WeeksPreterm = 0, AgeAtScan = 0,
                         GenderMale = 0, HandednessRight = 0, Total_Bad_Volumes = 0)
      pval_WeeksPreterm <- pval_AgeAtScan <- pval_Gender <- pval_Handedness <- pval_Total_Bad_Volumes <- NA
    }
  } else {
    fixed_effects <- c(Intercept = 0, WeeksPreterm = 0, AgeAtScan = 0,
                       GenderMale = 0, HandednessRight = 0, Total_Bad_Volumes = 0)
    pval_WeeksPreterm <- pval_AgeAtScan <- pval_Gender <- pval_Handedness <- pval_Total_Bad_Volumes <- NA
  }
  
  return(c(i, j, fixed_effects, pval_WeeksPreterm, pval_AgeAtScan,
           pval_Gender, pval_Handedness, pval_Total_Bad_Volumes))
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Load matrices and standardize per-edge
print("Loading adjacency matrices...")
all_adjacency_matrices <- load_all_adjacency_matrices(matrix_directory)

if (length(all_adjacency_matrices) == 0) {
  stop("No adjacency matrices found")
}

# Standardize per-edge across subjects (matches published methods)
all_adjacency_matrices_standardized <- standardize_matrices_per_edge(all_adjacency_matrices)

# Get matrix dimensions
num_nodes <- nrow(all_adjacency_matrices_standardized[[1]])
print(paste("Matrix dimensions:", num_nodes, "x", num_nodes))

# Create edge indices (upper triangle only)
edge_indices <- expand.grid(1:(num_nodes - 1), 2:num_nodes)
edge_indices <- edge_indices[edge_indices$Var1 < edge_indices$Var2, ]
print(paste("Total edges to process:", nrow(edge_indices)))

# ==============================================================================
# PARALLEL PROCESSING
# ==============================================================================

num_cores <- detectCores() - 2
cl <- makeCluster(num_cores)
print(paste("Setting up parallel cluster with", num_cores, "cores"))

clusterExport(cl, c("all_adjacency_matrices_standardized", "complete_data",
                    "process_edge_per_edge_standardized", "edge_indices"))

clusterEvalQ(cl, {
  library(dplyr)
  library(lme4)
  library(lmerTest)
})

print("Starting parallel processing with per-edge standardized values...")

results <- parLapply(cl, 1:nrow(edge_indices), function(k) {
  process_edge_per_edge_standardized(k, edge_indices, all_adjacency_matrices_standardized, complete_data)
})

stopCluster(cl)
print("Processing completed")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

if (length(results) > 0) {
  results_matrix <- do.call(rbind, results)
  col_names <- c('Node_i', 'Node_j', 'Intercept', 'WeeksPreterm', 'AgeAtScan',
                 'GenderMale', 'HandednessRight', 'Total_Bad_Volumes',
                 'pval_WeeksPreterm', 'pval_AgeAtScan', 'pval_Gender',
                 'pval_Handedness', 'pval_Total_Bad_Volumes')
  
  if (ncol(results_matrix) == length(col_names)) {
    colnames(results_matrix) <- col_names
    results_df <- as.data.frame(results_matrix)
    
    output_file <- file.path(output_dir, 'Combined_All_Subjects_LMM_Results.csv')
    
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    write.csv(results_df, output_file, row.names = FALSE)
    
    print(paste("Results saved to:", output_file))
    print("")
    print("COEFFICIENT INTERPRETATION:")
    print("- WeeksPreterm: SD change in edge strength per 1 SD increase in gestational age")
    print("- AgeAtScan: SD change in edge strength per 1 SD increase in age at scan")
    print("- GenderMale: SD difference in edge strength between males and females")
    print("- HandednessRight: SD difference in edge strength between right and left handed")
    print("- Total_Bad_Volumes: SD change in edge strength per 1 SD increase in motion")
    
    # Summary statistics
    sig_WeeksPreterm <- sum(results_df$pval_WeeksPreterm < 0.05, na.rm = TRUE)
    sig_AgeAtScan <- sum(results_df$pval_AgeAtScan < 0.05, na.rm = TRUE)
    
    print("")
    print("SIGNIFICANT EFFECTS (p < 0.05, uncorrected):")
    print(paste("Gestational Age (WeeksPreterm):", sig_WeeksPreterm, "edges"))
    print(paste("Age at Scan:", sig_AgeAtScan, "edges"))
    
    # Direction of effects
    pos_gestational_effects <- sum(results_df$WeeksPreterm > 0, na.rm = TRUE)
    neg_gestational_effects <- sum(results_df$WeeksPreterm < 0, na.rm = TRUE)
    
    print("")
    print("GESTATIONAL AGE EFFECT DIRECTION:")
    print(paste("Positive associations:", pos_gestational_effects, "edges"))
    print(paste("Negative associations:", neg_gestational_effects, "edges"))
  }
}
