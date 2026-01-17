library(lme4)
library(lmerTest)
library(dplyr)
library(readr)
library(parallel)

complete_data <- read_csv('your_data')

# Recode Handedness and set 0 values to NA
complete_data <- complete_data %>%
  mutate(Handedness = ifelse(Handedness == 0, NA,
                             ifelse(Handedness %in% c(-2, -1), "Left", "Right")),
         Handedness = factor(Handedness, levels = c("Left", "Right")),
         WeeksPreterm = scale(WeeksPreterm),
         AgeAtScan = scale(AgeAtScan))

# List all directories in the main folder
main_directory <- 'your_directory'  # Replace with your main directory path
directories <- list.dirs(main_directory, full.names = TRUE, recursive = FALSE)

process_edge <- function(k, edge_indices, adjacency_matrices, complete_data) {  
  i <- edge_indices[k, 1]  
  j <- edge_indices[k, 2]  
  edge_data <- data.frame()  
  
  for (subject_session in names(adjacency_matrices)) {    
    new_ID <- subject_session  
    subject_data <- complete_data %>% filter(new_ID == !!new_ID)    
    
    if (nrow(subject_data) > 0) {      
      WeeksPreterm <- subject_data$WeeksPreterm     
      Gender <- subject_data$Gender     
      AgeAtScan <- subject_data$AgeAtScan     
      Handedness <- subject_data$Handedness      
      Subject <- subject_data$ID      
      edge_value <- adjacency_matrices[[subject_session]][i, j]    
      
      # Ensure lengths match before combining
      if (length(WeeksPreterm) == length(Gender) && length(Gender) == length(AgeAtScan) &&
          length(AgeAtScan) == length(Handedness) && length(Handedness) == length(Subject) &&
          length(Subject) == length(edge_value)) {
        edge_data <- rbind(edge_data, data.frame(WeeksPreterm, Gender, AgeAtScan, Handedness, Subject, edge_value))
      } else {
        print(paste("Mismatch in lengths: WeeksPreterm =", length(WeeksPreterm),
                    "Gender =", length(Gender), "AgeAtScan =", length(AgeAtScan),
                    "Handedness =", length(Handedness), "Subject =", length(Subject),
                    "edge_value =", length(edge_value)))
      }
    }
  }  
  
  if (nrow(edge_data) > 0) {    
    model <- try(lmer(edge_value ~ WeeksPreterm + AgeAtScan + Gender + Handedness + (1 | Subject), data = edge_data), silent = TRUE)
    
    if (inherits(model, "try-error") || !is.null(model@optinfo$conv$lme4$messages)) {
      fixed_effects <- c(Intercept = 0, WeeksPreterm = 0, AgeAtScan = 0, Gender = 0, Handedness = 0)
      pval_WeeksPreterm <- NA
    } else {
      fixed_effects <- fixef(model)    
      coefs <- summary(model)$coefficients    
      pval_WeeksPreterm <- coefs["WeeksPreterm", "Pr(>|t|)"]
    }
  } else {    
    fixed_effects <- c(Intercept = 0, WeeksPreterm = 0, AgeAtScan = 0, Gender = 0, Handedness = 0)    
    pval_WeeksPreterm <- NA  
  }  
  
  return(c(i, j, fixed_effects, pval_WeeksPreterm))
}

# Load adjacency matrices
load_adjacency_matrices <- function(directory) {  
  print(paste("Loading matrices for directory:", directory))
  files <- list.files(directory, pattern = "*.csv", recursive = TRUE, full.names = TRUE)  
  matrices <- list()  
  
  for (file in files) {    
    filename <- basename(file)    
    parts <- strsplit(filename, "_")[[1]]    
    subject_session <- paste(parts[2], parts[3], sep = "_")    
    matrices[[subject_session]] <- as.matrix(read.csv(file, header = FALSE))  
  }  
  
  return(matrices)
}

# Loop over each directory
for (directory_path in directories) {  
  print(paste("Processing directory:", directory_path))
  adjacency_matrices <- load_adjacency_matrices(directory_path)  
  if (length(adjacency_matrices) == 0) { 
    print("No adjacency matrices found, skipping directory")
    next 
  }  
  
  # Create a list of edge indices  
  num_nodes <- nrow(adjacency_matrices[[1]])  
  edge_indices <- expand.grid(1:(num_nodes - 1), 2:num_nodes)  
  edge_indices <- edge_indices[edge_indices$Var1 < edge_indices$Var2, ]  
  
  # Set up parallel cluster  
  num_cores <- detectCores() - 2  
  cl <- makeCluster(num_cores)  
  print("Parallel cluster set up")
  
  # Export necessary variables and functions to the cluster  
  clusterExport(cl, c("adjacency_matrices", "complete_data", "process_edge", "edge_indices"))  
  print("Variables exported to cluster")
  
  # Run the processing in parallel  
  results <- parLapply(cl, 1:nrow(edge_indices), function(k) {    
    library(dplyr)    
    library(lme4)    
    library(lmerTest)    
    process_edge(k, edge_indices, adjacency_matrices, complete_data)  
  })  
  print("Parallel processing completed")
  
  # Stop the cluster  
  stopCluster(cl)  
  print("Cluster stopped")
  
  # Check results
  print(paste("Type of results:", typeof(results)))
  print(paste("Length of results:", length(results)))
  print("Head of results:")
  print(head(results))
  
  # Convert results to a DataFrame for easier handling
  if (length(results) > 0 && all(sapply(results, length) == length(results[[1]]))) {
    results_df <- do.call(rbind, results)
    colnames(results_df) <- c('Node_i', 'Node_j', 'Intercept', 'WeeksPreterm', 'AgeAtScan', 'Gender', 'Handedness', 'pval_WeeksPreterm')  
    print("Results combined into a DataFrame")
    
    # Save the results with a filename including the directory name  
    output_dir <- 'your_output_directory'  # Replace with your output directory path
    output_file <- file.path(output_dir, paste0(basename(dirname(directory_path)), '_', basename(directory_path), '.csv'))  
    write.csv(results_df, output_file, row.names = FALSE)  
    print(paste("Results saved to:", output_file))
    
    message <- paste("Processing complete for directory:", directory_path, "Output saved to:", output_file)  
    print(message)
  } else {
    print("Results are not properly aligned")
  }
}