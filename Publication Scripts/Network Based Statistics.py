#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:45:21 2025

@author: ragnarok
"""

#####################
#######
#working
import pandas as pd
import networkx as nx
import numpy as np
import glob
import os
import bct as bct
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement

# Define the directories
cross_sect_VPT_fold = '/your_files/Very_Preterm'
cross_sect_FT_fold = '/your_files/Full_Term'
parent_output_fold = '/your_file_path'
atlas_labels = '/your_files/node_labels_with_networks_5_removed.csv'

# Function to load the data
def load_data(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '**/*.csv'), recursive=True)
    population_tensor = []  # This will hold the matrices for each population
    
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        
        matrix = np.nan_to_num(matrix)
        
        # Add the matrix to the population tensor
        population_tensor.append(matrix)
    
    # Convert the list of matrices to a 3D numpy array
    population_tensor = np.array(population_tensor)
    
    # Transpose the axes to match the (N, N, S) format
    population_tensor = np.transpose(population_tensor, (1, 2, 0))
    
    return population_tensor

# Load the data for each population
VPT_population_tensor = load_data(cross_sect_VPT_fold)
FT_population_tensor = load_data(cross_sect_FT_fold)

# Load the node labels
node_labels = pd.read_csv(atlas_labels, header=None)

# Print the size of the tensors
print("Size of VPT population tensor: ", VPT_population_tensor.shape)
print("Size of FT population tensor: ", FT_population_tensor.shape)

# Perform the NBS analysis
#thresh = .76088  # Set your threshold here. Determines how many SD away from the mean difference of 0. this is Cohen's D ~ 0.1
#thresh = 1.5218 # this is .2
thresh = 2.2826 # this is .3
#thresh = 3.569  #  this is .5


k = 100 # number of permutations
tail = 'both'  # does not 
paired = False  # not paired data, different samples
verbose = True
#nbs_result = bct.nbs_bct(VPT_population_tensor,FT_population_tensor,  thresh, k, 'left',  paired, verbose) #X<Y  vpt < FT
#nbs_result = bct.nbs_bct(VPT_population_tensor,FT_population_tensor,  thresh, k, 'right',  paired, verbose) #X>Y FT< VPT
nbs_result = bct.nbs_bct(VPT_population_tensor,FT_population_tensor,  thresh, k, tail,  paired, verbose) #X<Y  vpt < FT


output_fold = os.path.join(parent_output_fold,f'threshold_{thresh}')
os.makedirs(output_fold, exist_ok =True)


# Get the components from the NBS result
components = nbs_result[1] 
p_values= nbs_result[0]

# Get the indices of the components with p-values less than or equal to 0.05
#significant_components_indices = np.argwhere(p_values <= 0.05).flatten() + 1
significant_components_indices = np.argwhere(p_values <= 0.025).flatten() + 1  ###for two sided


# Initialize an empty DataFrame to hold the edge indices
edges_df = pd.DataFrame(columns=['i', 'j'])

# Create an indices df for each significant component
for component_index in np.unique(components):
    # Skip if the component index is 0 (no component)
    if component_index == 0:
        continue

    # Check if the component is significant
    if component_index not in significant_components_indices:
        continue

    # Get the indices of the edges in this component. Avoids duplicates
    edge_indices = np.argwhere((components == component_index) & (np.triu(np.ones(components.shape), k=1).astype(bool)))

    # Create a temporary DataFrame with the edge indices
    temp_df = pd.DataFrame(edge_indices, columns=['i', 'j'])

    # Save the indices to a CSV file
    temp_df.to_csv(os.path.join(output_fold, f'Component_{component_index}_Indices.csv'), index=False)

    

# Create a labels df for each significant component
for component_index in np.unique(components):
    # Skip if the component index is 0 (no component)
    if component_index == 0:
        continue

    # Check if the component is significant
    if component_index not in significant_components_indices:
        continue

    # Get the indices of the edges in this component. Avoids duplicates
    edge_indices = np.argwhere((components == component_index) & (np.triu(np.ones(components.shape), k=1).astype(bool)))

    # Map the indices back to their labels
    edge_labels = node_labels.iloc[edge_indices[:, 0]], node_labels.iloc[edge_indices[:, 1]]

    # Create a DataFrame with the labels of the nodes that each edge connects
    edge_labels_df = pd.DataFrame(list(zip(edge_labels[0][0], edge_labels[1][0])))

    # Save the labels to a CSV file
    edge_labels_df.to_csv(os.path.join(output_fold, f'Component_{component_index}_labels.csv'), index=False)
    
# Save the output
for i, arr in enumerate(nbs_result):
    np.savetxt(os.path.join(output_fold, f'nbs_thresh_{thresh}_result_{i}.csv'), arr, delimiter=',')
    

##### 

#NBS based heatmap 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def process_nbs_heatmap(nbs_file_path):
    # Define the number of nodes
    num_nodes = 227
    
    # Load the NBS result matrix
    nbs_matrix = np.loadtxt(nbs_file_path, delimiter=',')
    
    # Create a symmetric matrix
    symmetric_nbs_matrix = (nbs_matrix + nbs_matrix.T)
    
    # Fill diagonal with self-connection values
    np.fill_diagonal(symmetric_nbs_matrix, nbs_matrix.diagonal())
    
    # Set insignificant values to 0 (white in the heatmap)
    symmetric_nbs_matrix[symmetric_nbs_matrix == 0] = np.nan
    
    # Create a custom colormap centered at white
   # custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "red"], N=256)
    # Create a custom colormap centered at white
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["blue", "white", "red"], N=256)


    plt.figure(figsize=(30, 30))
    sns.heatmap(symmetric_nbs_matrix, cmap=custom_cmap, center=0, annot=False, xticklabels=False, yticklabels=False, mask=np.isnan(symmetric_nbs_matrix))
    
    # Add labels and title to the plot
    plt.xlabel('')
    plt.ylabel('')
    
    # Extract filename without extension for title and save name
    filename_without_ext = os.path.splitext(os.path.basename(nbs_file_path))[0]
    
    plt.title(f'NBS Significant Components - {filename_without_ext}', fontsize=20)
    
    # Save the plot with an appropriate filename
    plt.savefig(f'{filename_without_ext}_nbs_heatmap.png')
    
    # Show the plot (optional)
    plt.show()


# Specify the file path for the NBS array output

nbs_file_path = '/your_file_path/threshold_0.76088/nbs_thresh_0.76088_result_1.csv'

if os.path.exists(nbs_file_path):
    process_nbs_heatmap(nbs_file_path)

