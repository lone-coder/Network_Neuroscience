#!/bin/bash

#################################
#     Resource Requirements     #
#################################
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=4gb
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH --array=1-53

# Email settings 
#SBATCH --mail-user=##############
#SBATCH --mail-type=ALL

bash bulk/Testing/Children_Connectome_Processing_Workflow.sh