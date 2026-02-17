#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Rich Club Overlap Stats & Figure
==========================================

Purpose: Calculate the percentage overlap of rich-club nodes in individuals 
compared to the group-averaged connectome. Generate publication-ready figure
and perform statistical testing.

Logic:
1. Load group-averaged matrices and identify group-level RC nodes (Top 10%).
2. Load individual subject matrices.
3. Identify individual-level RC nodes (Top 10%) for each subject.
4. Calculate percentage of individual RC nodes that are also in group RC set.
5. Save individual data and run stats.
6. Plot Boxplot + Individual points.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Global Bold Settings
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

# CONFIGURATION
BASE_PATH = "[PATH_TO_DATA]"
GROUP_AVG_DIR = os.path.join(BASE_PATH, "Matrices/AVERAGES/baseline averages")
INDIVIDUAL_DIR = os.path.join(BASE_PATH, "Matrices/Baseline/Groups_Baseline_filtered_weights")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs/figures/supplementary")

# Dictionary mapping code names to file prefixes/folder names
GROUPS = {
    'Full_Term': {
        'avg_file': 'Baseline_Full_Term_averaged_SCM.csv',
        'sub_folder': 'Full_Term',
        'label': 'Full-term',
        'color': '#FF7F50' # Coral
    },
    'Preterm': {
        'avg_file': 'Baseline_Preterm_averaged_SCM.csv',
        'sub_folder': 'Preterm',
        'label': 'Moderate-to-late preterm',
        'color': '#20B2AA' # LightSeaGreen
    },
    'Very_Preterm': {
        'avg_file': 'Baseline_Very_Preterm_averaged_SCM.csv',
        'sub_folder': 'Very_Preterm',
        'label': 'Very preterm',
        'color': '#191970' # MidnightBlue
    }
}
GROUP_ORDER = ['Full_Term', 'Preterm', 'Very_Preterm']
RC_PERCENTILE = 10  # Top 10% of nodes

def calculate_degree(adj_matrix):
    """Calculate binary degree of each node."""
    binary_adj = (adj_matrix > 0).astype(int)
    degrees = np.sum(binary_adj, axis=1)
    return degrees

def get_rc_nodes(adj_matrix, top_percent):
    """Identify indices of top N% high-degree nodes."""
    degrees = calculate_degree(adj_matrix)
    
    # Threshold such that at least N% of nodes are above it
    cutoff = np.percentile(degrees, 100 - top_percent)
    
    # Identify nodes >= cutoff
    rc_indices = np.where(degrees >= cutoff)[0]
    
    return set(rc_indices)

def cliffs_delta(x, y):
    """Calculate Cliff's delta effect size."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0: return np.nan
    
    delta = 0
    for i in x:
        for j in y:
            if i > j: delta += 1
            elif i < j: delta -= 1
            
    return delta / (n1 * n2)

def interpret_cliffs_delta(d):
    d = abs(d)
    if d < 0.147: return 'negligible'
    if d < 0.33: return 'small'
    if d < 0.474: return 'medium'
    return 'large'

def holm_correction(p_values):
    """
    Perform Holm-Bonferroni correction on a list of p-values.
    Returns array of corrected p-values matching input order.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    if n == 0: return np.array([])
    
    # Get indices that would sort the p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    corrected_p = np.zeros(n)
    for i in range(n):
        # correction factor = n - rank + 1
        factor = n - i
        # p_corr = p * factor
        val = sorted_p[i] * factor
        
        # Enforce monotonicity: max of current and previous
        if i > 0:
            val = max(val, corrected_p[i-1])
            
        # Cap at 1.0
        corrected_p[i] = min(1.0, val)
        
    # Restore original order
    final_p = np.zeros(n)
    final_p[sorted_indices] = corrected_p
    
    return final_p


def main():
    print(f"--- Rich Club Overlap Analysis (Top {RC_PERCENTILE}%) ---\n")
    
    all_data = []
    
    # -------------------------------------------------------------------------
    # 1. GATHER DATA
    # -------------------------------------------------------------------------
    for group_key in GROUP_ORDER:
        info = GROUPS[group_key]
        
        # Load Group Average
        avg_path = os.path.join(GROUP_AVG_DIR, info['avg_file'])
        if not os.path.exists(avg_path):
            print(f"ERROR: Missing group average file: {avg_path}")
            continue
            
        try:
            group_matrix = pd.read_csv(avg_path, header=None).values
        except Exception as e:
            print(f"Error reading {avg_path}: {e}")
            continue
            
        # Get Group RC Set
        group_rc_nodes = get_rc_nodes(group_matrix, RC_PERCENTILE)
        
        # Load Individual Subjects
        subject_folder = os.path.join(INDIVIDUAL_DIR, info['sub_folder'])
        subject_files = glob.glob(os.path.join(subject_folder, "*.csv"))
        
        if not subject_files:
            print(f"WARNING: No subject files found in {subject_folder}")
            continue
            
        print(f"Processing {info['label']} (N={len(subject_files)})...")
        
        for sub_file in subject_files:
            try:
                sub_matrix = pd.read_csv(sub_file, header=None).values
                sub_rc_nodes = get_rc_nodes(sub_matrix, RC_PERCENTILE)
                
                # Intersection / Individual_Set_Size
                intersection = group_rc_nodes.intersection(sub_rc_nodes)
                if len(sub_rc_nodes) > 0:
                    percentage = (len(intersection) / len(sub_rc_nodes)) * 100
                else:
                    percentage = 0.0
                    
                all_data.append({
                    'Group': group_key,
                    'Label': info['label'],
                    'SubjectID': os.path.basename(sub_file).split('.')[0],
                    'Overlap_Percent': percentage
                })
                
            except Exception as e:
                print(f"Error processing {os.path.basename(sub_file)}: {e}")
    
    if not all_data:
        print("No data collected.")
        return

    df = pd.DataFrame(all_data)
    
    # Save Data
    csv_path = os.path.join(OUTPUT_DIR, "rich_club_overlap_individuals.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved individual stats to: {csv_path}")
    
    # -------------------------------------------------------------------------
    # 2. STATISTICS
    # -------------------------------------------------------------------------
    print("\n--- Statistical Analysis ---")
    
    # Descriptive
    desc = df.groupby('Label')['Overlap_Percent'].describe()
    print(desc)
    
    # Kruskal-Wallis
    groups_data = [df[df['Group'] == g]['Overlap_Percent'].values for g in GROUP_ORDER]
    h_stat, p_omnibus = stats.kruskal(*groups_data)
    print(f"\nKruskal-Wallis Omnibus: H={h_stat:.4f}, p={p_omnibus:.4e}")
    
    # Pairwise Mann-Whitney U with Holm Correction
    comparisons = []
    p_values = []
    
    for i in range(len(GROUP_ORDER)):
        for j in range(i + 1, len(GROUP_ORDER)):
            g1, g2 = GROUP_ORDER[i], GROUP_ORDER[j]
            d1 = df[df['Group'] == g1]['Overlap_Percent'].values
            d2 = df[df['Group'] == g2]['Overlap_Percent'].values
            
            u_stat, p_val = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            delta = cliffs_delta(d1, d2)
            
            comparisons.append({
                'Comparison': f"{GROUPS[g1]['label']} vs {GROUPS[g2]['label']}",
                'U_stat': u_stat,
                'p_uncorr': p_val,
                'Cliffs_d': delta,
                'Effect': interpret_cliffs_delta(delta)
            })
            p_values.append(p_val)
            
    # Correction
    p_corr = holm_correction(p_values)
    
    print("\nPairwise Comparisons (Holm-corrected):")
    for i, comp in enumerate(comparisons):
        comp['p_corr'] = p_corr[i]
        comp['Significant'] = '*' if p_corr[i] < 0.05 else ''
        print(f"  {comp['Comparison']}: p={comp['p_corr']:.4f}, d={comp['Cliffs_d']:.2f} ({comp['Effect']}) {comp['Significant']}")

    # Save Stats
    stats_df = pd.DataFrame(comparisons)
    stats_path = os.path.join(OUTPUT_DIR, "rich_club_overlap_stats_results.csv")
    stats_df.to_csv(stats_path, index=False)

    # -------------------------------------------------------------------------
    # 3. PLOTTING
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 7)) # Slightly wider for rotated labels
    sns.set_style("whitegrid")
    
    # Define colors
    palette = [GROUPS[g]['color'] for g in GROUP_ORDER]
    labels = [GROUPS[g]['label'] for g in GROUP_ORDER]
    
    # 1. Stripplot (Individual Points)
    # Added linewidth/edgecolor for clearer points
    ax = sns.stripplot(x='Group', y='Overlap_Percent', data=df, order=GROUP_ORDER,
                  palette=palette, hue='Group', legend=False,
                  dodge=False, size=6, alpha=0.7, jitter=0.20, zorder=1,
                   edgecolor='white', linewidth=0.5)
    
    # 2. Add Mean/Median Line
    width = 0.5
    
    for i, group_key in enumerate(GROUP_ORDER):
        group_data = df[df['Group'] == group_key]['Overlap_Percent']
        mean_val = group_data.mean()
        
        # Plot mean value
        plot_val = mean_val
        
        print(f"Group {group_key}: Mean={mean_val:.2f} -> Plotting at {plot_val:.2f}")
        
        # Plot horizontal line centered at i
        plt.plot([i - width/2, i + width/2], [plot_val, plot_val], 
                 color=GROUPS[group_key]['color'], linewidth=3, zorder=5)
                 
    # Customize axis
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', rotation=45, ha='right')
    plt.ylabel('Template overlap (%)', fontsize=14, fontweight='bold')
    plt.xlabel(' ', fontsize=14)
    plt.title('Rich-Club Consistency', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(65, 105) # Adjusted ylim to focus on top range
    
    # Add back black axis lines
    sns.despine(offset=10, trim=False) # Keep default (removes top/right)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)

    
    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    
    # Bold tick labels
    ax.tick_params(axis='y', labelsize=12, width=1.5)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()

    
    # Save Figure as Supplementary Figure S3
    fig_path = os.path.join(OUTPUT_DIR, "Supplementary_Figure_S3_Individual_Overlap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Figure to: {fig_path}")
    
if __name__ == "__main__":
    main()
