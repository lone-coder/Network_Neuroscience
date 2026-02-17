#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network-Based Statistics (NBS) for Supplementary Analysis
===========================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Perform Network-Based Statistics to identify connected components
         of edges showing group differences between Very Preterm and Full-Term
         groups. This supplementary analysis validates the L MM findings.

THRESHOLD CALCULATION:
    The NBS requires a primary t-statistic threshold. We convert effect sizes
    (Cohen's d) to t-statistics using the harmonic mean of group sizes:
    
    t = d * sqrt(harmonic_mean)
    
    Where harmonic_mean = 2*n1*n2 / (n1 + n2)
    For VPT (n=37) vs FT (n=133): harmonic_mean ≈ 57.89, sqrt ≈ 7.6088
    
    Thus:
    - d = 0.1 → t = 0.76088
    - d = 0.3 → t = 2.2826

OUTPUT:
    - Component CSV files with edge labels for significant components
    - NBS result matrices saved for downstream heatmap visualization
    
REFERENCE:
    Zalesky et al. (2010). Network-based statistic: Identifying differences
    in brain networks. NeuroImage, 53(4), 1197-1207.
"""

import glob
import numpy as np
import pandas as pd
import os
import bct as bct

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = '[PATH_TO_DATA]'

# Input directories for individual matrices
VPT_DIR = os.path.join(BASE_DIR, 'Matrices/Baseline/Groups_Baseline_filtered_weights/Very_Preterm')
FT_DIR = os.path.join(BASE_DIR, 'Matrices/Baseline/Groups_Baseline_filtered_weights/Full_Term')

# Atlas labels
ATLAS_PATH = os.path.join(BASE_DIR, 'node_labels_5_missing_with_networks.csv')

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/nbs')

# NBS parameters
COHENS_D_THRESHOLD = 0.1  # Effect size threshold
N_PERMUTATIONS = 1000     # Number of permutations
TAIL = 'both'             # Two-tailed test
PAIRED = False            # Independent samples
VERBOSE = True

# Group sizes for threshold calculation
N_VPT = 37
N_FT = 133

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_population_tensor(directory_path):
    """
    Load individual connectivity matrices and create population tensor.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing individual CSV matrices
        
    Returns
    -------
    tensor : ndarray
        3D array of shape (N_nodes, N_nodes, N_subjects)
    """
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    print(f"Loading {len(csv_files)} matrices from {os.path.basename(directory_path)}...")
    
    population_tensor = []
    
    for file_path in csv_files:
        try:
            matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
            matrix = np.nan_to_num(matrix)  # Replace NaN with 0
            population_tensor.append(matrix)
        except Exception as e:
            print(f"Warning: Error loading {os.path.basename(file_path)}: {e}")
    
    # Convert to 3D array and transpose to (N, N, S) format
    population_tensor = np.array(population_tensor)
    population_tensor = np.transpose(population_tensor, (1, 2, 0))
    
    return population_tensor


def calculate_t_threshold(cohens_d, n1, n2):
    """
    Convert Cohen's d to t-statistic threshold using harmonic mean.
    
    Parameters
    ----------
    cohens_d : float
        Effect size (Cohen's d)
    n1, n2 : int
        Group sample sizes
        
    Returns
    -------
    t_threshold : float
        Corresponding t-statistic threshold
    """
    harmonic_mean = (2 * n1 * n2) / (n1 + n2)
    t_threshold = cohens_d * np.sqrt(harmonic_mean)
    return t_threshold

# ==============================================================================
# MAIN NBS ANALYSIS
# ==============================================================================

def run_nbs_analysis():
    """Execute Network-Based Statistics analysis."""
    
    print("=" * 70)
    print("Network-Based Statistics Analysis")
    print("=" * 70)
    
    # Load data
    vpt_tensor = load_population_tensor(VPT_DIR)
    ft_tensor = load_population_tensor(FT_DIR)
    
    print(f"\nVPT tensor shape: {vpt_tensor.shape}")
    print(f"FT tensor shape: {ft_tensor.shape}")
    
    # Calculate t-statistic threshold
    t_threshold = calculate_t_threshold(COHENS_D_THRESHOLD, N_VPT, N_FT)
    
    print(f"\nNBS Parameters:")
    print(f"  Cohen's d threshold: {COHENS_D_THRESHOLD}")
    print(f"  t-statistic threshold: {t_threshold:.5f}")
    print(f"  Permutations: {N_PERMUTATIONS}")
    print(f"  Test type: {TAIL}")
    print(f"  Significance level (two-tailed): α = 0.025 per tail")
    
    # Run NBS
    print(f"\nRunning NBS...")
    nbs_result = bct.nbs_bct(vpt_tensor, ft_tensor, t_threshold, 
                             N_PERMUTATIONS, TAIL, PAIRED, VERBOSE)
    
    # Extract results
    p_values = nbs_result[0]
    components = nbs_result[1]
    
    # Identify significant components (two-tailed: α=0.025 per tail)
    significant_idx = np.argwhere(p_values <= 0.025).flatten() + 1
    
    print(f"\nResults:")
    print(f"  Total components found: {len(p_values)}")
    print(f"  Significant components (p ≤ 0.025): {len(significant_idx)}")
    
    if len(significant_idx) > 0:
        print(f"  Significant component IDs: {significant_idx.tolist()}")
    
    # Create output directory
    output_subdir = os.path.join(OUTPUT_DIR, f'threshold_{t_threshold:.5f}_d_{COHENS_D_THRESHOLD}')
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load atlas labels
    node_labels = pd.read_csv(ATLAS_PATH, header=None)
    
    # Save significant component edge labels
    for comp_idx in np.unique(components):
        if comp_idx == 0:
            continue
        
        if comp_idx not in significant_idx:
            continue
        
        # Get edge indices (upper triangle only to avoid duplicates)
        edge_indices = np.argwhere(
            (components == comp_idx) & 
            (np.triu(np.ones(components.shape), k=1).astype(bool))
        )
        
        # Map indices to labels
        edge_labels_i = node_labels.iloc[edge_indices[:, 0]]
        edge_labels_j = node_labels.iloc[edge_indices[:, 1]]
        
        # Create DataFrame
        edge_labels_df = pd.DataFrame({
            'Node_i': edge_labels_i.values.flatten(),
            'Node_j': edge_labels_j.values.flatten()
        })
        
        # Save
        output_file = os.path.join(output_subdir, f'Component_{comp_idx}_labels.csv')
        edge_labels_df.to_csv(output_file, index=False)
        print(f"  Saved: {os.path.basename(output_file)} ({len(edge_labels_df)} edges)")
    
    # Save NBS result matrices
    for i, arr in enumerate(nbs_result):
        output_file = os.path.join(output_subdir, f'nbs_result_{i}.csv')
        np.savetxt(output_file, arr, delimiter=',')
    
    print(f"\n✓ NBS analysis complete. Results saved to:")
    print(f"  {output_subdir}")
    
    return nbs_result, significant_idx


if __name__ == "__main__":
    run_nbs_analysis()
