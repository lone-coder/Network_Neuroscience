#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge Labeling and Categorization for Preterm Connectome Study
==============================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Process LMM edge results by:
    1. Adding anatomical labels to edges (from atlas)
    2. Categorizing edges by connection length (Short/Medium/Long)
    3. Categorizing edges by rich-club membership (RC/Feeder/Local)

CRITICAL METHODOLOGICAL DECISION:
    This script uses the FIXED filtering approach where edges must have:
    - Non-zero beta coefficient (WeeksPreterm OR AgeAtScan)
    - Non-zero weight in group-averaged matrix
    - Non-zero distance in group-averaged distance matrix
    
    This ensures CONSISTENCY between distance and rich-club categorization.
    Both analyses operate on the identical edge set.

DISTANCE CATEGORIZATION:
    Following Marebwa et al. (2018), edges are categorized by interquartile range:
    - Short: < 25th percentile of streamline lengths
    - Medium: 25th - 75th percentile
    - Long: > 75th percentile

RICH-CLUB CATEGORIZATION:
    Using the group-averaged weighted connectome:
    - Rich-club: Both endpoints are in top N% by degree
    - Feeder: One endpoint is RC, one is not
    - Local: Neither endpoint is RC
    
    Primary analysis uses top 10%; 15% and 20% provided as supplementary.
"""

import os
import pandas as pd
import numpy as np


# ==============================================================================
# FUNCTION: Add Anatomical Labels to Edges
# ==============================================================================

def create_edge_labels(lmm_results_path, atlas_labels_path, output_folder):
    """
    Map node indices to anatomical labels and network assignments.
    
    Parameters
    ----------
    lmm_results_path : str
        Path to LMM results CSV with columns: Node_i, Node_j, WeeksPreterm, etc.
    atlas_labels_path : str
        Path to atlas labels CSV with node names and network assignments.
    output_folder : str
        Path to save edge_labels output CSV.
    
    Returns
    -------
    pd.DataFrame
        Edge labels dataframe with anatomical information added.
    
    Output Columns
    --------------
    - Node_i, Node_j: Node indices (0-indexed)
    - Label_i, Label_j: Anatomical labels
    - Network_i, Network_j: Network assignments (e.g., "DMN", "Visual")
    - Intercept, WeeksPreterm, AgeAtScan, etc.: Beta coefficients
    - pval_WeeksPreterm: P-value for gestational age effect
    """
    
    # Load the atlas labels
    atlas_data = pd.read_csv(atlas_labels_path, header=None, usecols=[0, 1])
    node_to_network = atlas_data.set_index(0)[1].to_dict()
    
    # Load LMM results
    results_df = pd.read_csv(lmm_results_path)
    print(f"Loaded {len(results_df)} edges from LMM results")
    
    # Create edge labels dataframe
    edge_labels_list = []
    
    for _, row in results_df.iterrows():
        # Convert to 0-indexed (LMM output may be 1-indexed)
        node_i = int(row['Node_i']) - 1
        node_j = int(row['Node_j']) - 1
        
        # Get anatomical labels
        label_i = atlas_data.iloc[node_i, 0]
        label_j = atlas_data.iloc[node_j, 0]
        network_i = node_to_network[label_i]
        network_j = node_to_network[label_j]
        
        edge_labels_list.append({
            'Node_i': node_i,
            'Node_j': node_j,
            'Label_i': label_i,
            'Label_j': label_j,
            'Network_i': network_i,
            'Network_j': network_j,
            'Intercept': row['Intercept'],
            'WeeksPreterm': row['WeeksPreterm'],
            'AgeAtScan': row['AgeAtScan'],
            'Gender': row.get('GenderMale', row.get('Gender', np.nan)),
            'Handedness': row.get('HandednessRight', row.get('Handedness', np.nan)),
            'Total_Bad_Volumes': row.get('Total_Bad_Volumes', np.nan),
            'pval_WeeksPreterm': row['pval_WeeksPreterm']
        })
    
    edge_labels_df = pd.DataFrame(edge_labels_list)
    
    # Save output
    output_filename = 'edge_labels_' + os.path.basename(lmm_results_path)
    output_path = os.path.join(output_folder, output_filename)
    edge_labels_df.to_csv(output_path, index=False)
    print(f"Saved edge labels to: {output_path}")
    
    return edge_labels_df


# ==============================================================================
# FUNCTION: Distance Categorization (FIXED Version)
# ==============================================================================

def categorize_by_distance(edge_file_path, distance_matrix_path, weights_matrix_path, output_path):
    """
    Categorize edges by connection length using IQR-based thresholds.
    
    CRITICAL: Uses FIXED filtering that requires BOTH non-zero distance AND
    non-zero weight. This ensures consistency with rich-club categorization.
    
    Parameters
    ----------
    edge_file_path : str
        Path to edge labels CSV (output from create_edge_labels).
    distance_matrix_path : str
        Path to group-averaged distance matrix (streamline lengths).
    weights_matrix_path : str
        Path to group-averaged weights matrix.
    output_path : str
        Path to save distance categorization CSV.
    
    Returns
    -------
    pd.DataFrame
        Categorized edges with columns: Index_1, Index_2, Length, Category
    
    Category Definitions (Marebwa et al. 2018)
    ------------------------------------------
    - Short: < 25th percentile of valid edge lengths
    - Medium: 25th - 75th percentile
    - Long: > 75th percentile
    
    Notes
    -----
    Edges must satisfy ALL of:
    1. Non-zero WeeksPreterm OR non-zero AgeAtScan coefficient
    2. Non-zero weight in group-averaged matrix
    3. Non-zero distance in group-averaged distance matrix
    
    This filtering ensures the same edges are used in distance and RC analyses.
    """
    
    # Load edge file (LMM output with labels)
    try:
        edge_df = pd.read_csv(edge_file_path)
        print(f"\nLoaded {len(edge_df)} edges from edge file")
        
        # FILTER: Keep only edges with non-zero effect
        # Scientific rationale: Zero coefficients indicate no detectable effect
        filtered_edges = edge_df[
            (edge_df['WeeksPreterm'] != 0) | (edge_df['AgeAtScan'] != 0)
        ]
        print(f"Edges with non-zero WeeksPreterm OR AgeAtScan: {len(filtered_edges)}")
        
    except Exception as e:
        print(f"Error reading edge file: {e}")
        return None
    
    # Load distance and weights matrices
    try:
        distance_matrix = pd.read_csv(distance_matrix_path, header=None)
        weights_matrix = pd.read_csv(weights_matrix_path, header=None)
        print(f"Distance matrix shape: {distance_matrix.shape}")
        print(f"Weights matrix shape: {weights_matrix.shape}")
    except Exception as e:
        print(f"Error reading matrix files: {e}")
        return None
    
    # Match edges from LMM results to matrices
    # CRITICAL: Apply same filtering as RC categorization for consistency
    matched_edges = []
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # Check if this edge is in the filtered LMM results
            mask = (filtered_edges['Node_i'] == i) & (filtered_edges['Node_j'] == j)
            if mask.sum() > 0:
                length = distance_matrix.iat[i, j]
                weight = weights_matrix.iat[i, j]
                matched_edges.append((i, j, length, weight))
    
    matched_df = pd.DataFrame(matched_edges, columns=['Index_1', 'Index_2', 'Length', 'Weight'])
    print(f"Matched edges: {len(matched_df)}")
    
    # FIXED FILTERING: Require BOTH non-zero distance AND non-zero weight
    # Scientific rationale: Edges with zero weight don't exist in the connectome
    # Edges with zero distance are biologically implausible (preprocessing artifact)
    valid_edges = matched_df[(matched_df['Length'] > 0) & (matched_df['Weight'] > 0)]
    print(f"Valid edges (non-zero length AND weight): {len(valid_edges)}")
    
    # Calculate IQR-based thresholds
    lengths = valid_edges['Length'].values
    short_threshold = np.percentile(lengths, 25)  # 25th percentile
    long_threshold = np.percentile(lengths, 75)   # 75th percentile
    
    print(f"\nDistance thresholds (IQR-based):")
    print(f"  Short: < {short_threshold:.2f} mm")
    print(f"  Medium: {short_threshold:.2f} - {long_threshold:.2f} mm")
    print(f"  Long: > {long_threshold:.2f} mm")
    
    # Categorize edges
    categorized = []
    for _, row in valid_edges.iterrows():
        i, j, length = int(row['Index_1']), int(row['Index_2']), row['Length']
        
        if length < short_threshold:
            category = 'Short'
        elif length <= long_threshold:
            category = 'Medium'
        else:
            category = 'Long'
        
        categorized.append((i, j, length, category))
    
    result_df = pd.DataFrame(categorized, columns=['Index_1', 'Index_2', 'Length', 'Category'])
    
    # Print category counts
    print(f"\nDistance Category Counts:")
    for cat in ['Short', 'Medium', 'Long']:
        count = len(result_df[result_df['Category'] == cat])
        pct = count / len(result_df) * 100
        print(f"  {cat}: {count} edges ({pct:.1f}%)")
    
    # Save results
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved distance categorization to: {output_path}")
    
    return result_df


# ==============================================================================
# FUNCTION: Rich-Club Categorization (FIXED Version)
# ==============================================================================

def categorize_by_rich_club(edge_file_path, weights_matrix_path, distance_matrix_path, 
                            node_labels_path, output_path, percentages=[10, 15, 20]):
    """
    Categorize edges by rich-club membership.
    
    CRITICAL: Uses FIXED filtering that requires BOTH non-zero distance AND
    non-zero weight. This ensures consistency with distance categorization.
    
    Parameters
    ----------
    edge_file_path : str
        Path to edge labels CSV (output from create_edge_labels).
    weights_matrix_path : str
        Path to group-averaged weights matrix.
    distance_matrix_path : str
        Path to group-averaged distance matrix.
    node_labels_path : str
        Path to atlas labels CSV.
    output_path : str
        Directory to save RC categorization CSVs.
    percentages : list of int
        Percentages for top node analysis (default: [10, 15, 20]).
        Primary analysis uses 10%, supplementary uses 15% and 20%.
    
    Returns
    -------
    dict
        Dictionary mapping percentage to categorized edges dataframe.
    
    Category Definitions
    --------------------
    - Rich-club: Both endpoints are RC nodes (top N% by degree)
    - Feeder: One endpoint is RC, one is not
    - Local: Neither endpoint is RC
    
    Notes
    -----
    Degree is calculated using only valid edges (non-zero weight AND distance).
    This ensures the hub definition is based on the same edge set used for
    distance categorization.
    """
    
    print("=" * 60)
    print("RICH-CLUB CATEGORIZATION (FIXED VERSION)")
    print("=" * 60)
    
    # Load and filter LMM edges (same filtering as distance categorization)
    edge_df = pd.read_csv(edge_file_path)
    filtered_edges = edge_df[
        (edge_df['WeeksPreterm'] != 0) | (edge_df['AgeAtScan'] != 0)
    ]
    print(f"Filtered LMM edges: {len(filtered_edges)}")
    
    # Load matrices
    weights_matrix = pd.read_csv(weights_matrix_path, header=None)
    distance_matrix = pd.read_csv(distance_matrix_path, header=None)
    node_labels = np.loadtxt(node_labels_path, dtype=str, delimiter=',', usecols=0)
    
    # Match edges and apply FIXED filtering
    matched_edges = []
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            mask = (filtered_edges['Node_i'] == i) & (filtered_edges['Node_j'] == j)
            if mask.sum() > 0:
                weight = weights_matrix.iat[i, j]
                distance = distance_matrix.iat[i, j]
                matched_edges.append((i, j, weight, distance))
    
    matched_df = pd.DataFrame(matched_edges, columns=['Index_1', 'Index_2', 'Weight', 'Distance'])
    
    # FIXED FILTERING: Same as distance categorization
    valid_edges = matched_df[(matched_df['Weight'] > 0) & (matched_df['Distance'] > 0)]
    print(f"Valid edges (non-zero weight AND distance): {len(valid_edges)}")
    
    # Calculate degrees using ONLY valid edges
    # Scientific rationale: Hub definition should be based on the same edge set
    degrees = np.zeros(len(node_labels))
    for _, row in valid_edges.iterrows():
        i, j = int(row['Index_1']), int(row['Index_2'])
        degrees[i] += 1
        degrees[j] += 1
    
    print(f"Degree stats - Min: {degrees.min():.0f}, Max: {degrees.max():.0f}, Mean: {degrees.mean():.1f}")
    
    results = {}
    
    for percentage in percentages:
        print(f"\n--- Processing {percentage}% RC threshold ---")
        
        # Determine degree threshold for top N%
        threshold = np.percentile(degrees, 100 - percentage)
        print(f"Degree threshold for top {percentage}%: {threshold:.1f}")
        
        # Identify RC nodes
        rc_node_indices = np.where(degrees >= threshold)[0]
        print(f"Number of RC nodes: {len(rc_node_indices)}")
        
        # Categorize edges
        categorized = []
        for _, row in valid_edges.iterrows():
            i, j = int(row['Index_1']), int(row['Index_2'])
            weight = row['Weight']
            
            i_is_rc = i in rc_node_indices
            j_is_rc = j in rc_node_indices
            
            if i_is_rc and j_is_rc:
                category = 'Rich club'
            elif i_is_rc or j_is_rc:
                category = 'Feeder'
            else:
                category = 'Local'
            
            categorized.append({
                'Index_1': i,
                'Index_2': j,
                'Name_1': node_labels[i],
                'Name_2': node_labels[j],
                'Category': category,
                'Weight': weight
            })
        
        result_df = pd.DataFrame(categorized)
        
        # Print category counts
        print(f"\nRC Category Counts ({percentage}%):")
        for cat in ['Rich club', 'Feeder', 'Local']:
            count = len(result_df[result_df['Category'] == cat])
            pct = count / len(result_df) * 100
            print(f"  {cat}: {count} edges ({pct:.1f}%)")
        
        # Save results
        output_filename = f'RC_categorization_{percentage}_percent_FILTERED.csv'
        output_file = os.path.join(output_path, output_filename)
        result_df.to_csv(output_file, index=False)
        print(f"Saved to: {output_filename}")
        
        results[percentage] = result_df
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
    
    # Input paths
    lmm_results_path = '/path/to/Combined_All_Subjects_LMM_Results.csv'
    atlas_labels_path = '/path/to/node_labels_5_missing_with_networks.csv'
    weights_matrix_path = '/path/to/Baseline_filtered_weights_averaged_SCM.csv'
    distance_matrix_path = '/path/to/Baseline_filtered_distance_averaged_SCM.csv'
    
    # Output path
    output_folder = '/path/to/output_folder'
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Create edge labels
    print("=" * 60)
    print("STEP 1: CREATING EDGE LABELS")
    print("=" * 60)
    edge_labels_df = create_edge_labels(lmm_results_path, atlas_labels_path, output_folder)
    edge_labels_path = os.path.join(output_folder, 'edge_labels_' + os.path.basename(lmm_results_path))
    
    # Step 2: Distance categorization
    print("\n" + "=" * 60)
    print("STEP 2: DISTANCE CATEGORIZATION")
    print("=" * 60)
    distance_output = os.path.join(output_folder, 'distance_categorization.csv')
    distance_df = categorize_by_distance(
        edge_labels_path, distance_matrix_path, weights_matrix_path, distance_output
    )
    
    # Step 3: Rich-club categorization
    print("\n" + "=" * 60)
    print("STEP 3: RICH-CLUB CATEGORIZATION")
    print("=" * 60)
    rc_results = categorize_by_rich_club(
        edge_labels_path, weights_matrix_path, distance_matrix_path,
        atlas_labels_path, output_folder, percentages=[10, 15, 20]
    )
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    distance_count = len(distance_df) if distance_df is not None else 0
    rc_count = len(rc_results.get(10, [])) if rc_results else 0
    
    print(f"Distance categorization: {distance_count} edges")
    print(f"RC categorization (10%): {rc_count} edges")
    
    if distance_count == rc_count:
        print("✅ SUCCESS: Both categorizations use identical edge sets!")
    else:
        print(f"⚠️  Mismatch: {abs(distance_count - rc_count)} edges differ")
    
    print("\nProcessing complete.")
