#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Distance Categorization File
======================================

Creates distance_categorization.csv by categorizing edges based on
streamline lengths (Short, Medium, Long).
"""
import pandas as pd
import numpy as np

def categorize_lengths(edge_file_path, lengths_file, output_file):
    """
    Categorizes edge lengths into Short, Medium, Long based on quartiles.
    
    Args:
        edge_file_path (str): Path to edge labels CSV with LMM results
        lengths_file (str): Path to distance matrix CSV
        output_file (str): Path to save categorized edges CSV
    """
    print("="*80)
    print("GENERATING DISTANCE CATEGORIZATION FILE")
    print("="*80)
    
    # Load edge file
    edge_df = pd.read_csv(edge_file_path)
    print(f"\nEdges in LMM file: {edge_df.shape[0]}")
    
    # Filter for non-zero effects
    filtered_df = edge_df[(edge_df['WeeksPreterm'] != 0) | (edge_df['AgeAtScan'] != 0)]
    print(f"Edges with non-zero effects: {filtered_df.shape[0]}")
    
    # Load distance matrix
    lengths_df = pd.read_csv(lengths_file, header=None)
    print(f"Distance matrix size: {lengths_df.shape}")
    
    # Match edges and extract lengths
    matched_edges = []
    for _, row in filtered_df.iterrows():
        i = int(row['Node_i'])
        j = int(row['Node_j'])
        if i < lengths_df.shape[0] and j < lengths_df.shape[1]:
            length = lengths_df.iat[i, j]
            matched_edges.append((i, j, length))
    
    matched_df = pd.DataFrame(matched_edges, columns=['Index_1', 'Index_2', 'Length'])
    print(f"Matched edges: {matched_df.shape[0]}")
    
    # Extract non-zero lengths for quartile calculation
    values = matched_df['Length'].values
    values = values[values > 0]
    
    # Calculate quartiles
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    
    print(f"\nDistance thresholds:")
    print(f"  Short: < {q1:.2f} mm")
    print(f"  Medium: {q1:.2f} - {q3:.2f} mm")
    print(f"  Long: > {q3:.2f} mm")
    
    # Categorize edges
    categorized_edges = []
    for _, row in matched_df.iterrows():
        i, j, length = row
        if length > 0:
            if length < q1:
                category = 'Short'
            elif length <= q3:
                category = 'Medium'
            else:
                category = 'Long'
            categorized_edges.append((i, j, length, category))
    
    # Create output DataFrame
    output_df = pd.DataFrame(categorized_edges, columns=['Index_1', 'Index_2', 'Length', 'Category'])
    
    # Count by category
    print(f"\nEdge counts by category:")
    for cat in ['Short', 'Medium', 'Long']:
        count = len(output_df[output_df['Category'] == cat])
        pct = (count / len(output_df)) * 100 if len(output_df) > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Save to file
    output_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print("="*80)

if __name__ == "__main__":
    BASE_DIR = '[PATH_TO_DATA]'
    
    edge_file_path = f'{BASE_DIR}/edge_labels_Combined_All_Subjects_LMM_Results.csv'
    lengths_file = f'{BASE_DIR}/Matrices/AVERAGES/Baseline_filtered_distance_averaged_SCM.csv'
    output_file = f'{BASE_DIR}/distance_categorization.csv'
    
    categorize_lengths(edge_file_path, lengths_file, output_file)
