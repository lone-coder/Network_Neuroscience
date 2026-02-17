#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S4: Top 20 most frequent rich-club nodes
==============================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Horizontal bar chart showing the top 20 nodes that most frequently
         appear in individual participants' rich-club sets (top 10% by degree).

OUTPUT:
    - Supplementary_Figure_S4_RC_Node_Identity.png
"""

import matplotlib
matplotlib.use('Agg')

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Patch

# Global Bold Settings
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial', 'Helvetica'], 'weight': 'bold'})

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = '[PATH_TO_DATA]'
INDIVIDUAL_DIR = os.path.join(BASE_DIR, 'Matrices/Baseline/Groups_Baseline_filtered_weights')
ATLAS_PATH = os.path.join(BASE_DIR, 'node_labels_5_missing_with_networks.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/figures/supplementary')
RC_PERCENTILE = 10
TOP_N = 20

NETWORK_COLORS = {
    'Subcort': '#7f7f7f',    # Gray
    'Visual': '#9467bd',      # Purple
    'SomMot': '#1f77b4',      # Blue
    'DorsAttn': '#2ca02c',    # Green
    'VentAttn': '#d62728',    # Red
    'Limbic': '#8c564b',      # Brown
    'Control': '#e377c2',     # Pink
    'Default': '#ff7f0e'      # Orange
}


def calculate_degree(adj_matrix):
    """Calculate binary degree of each node."""
    binary_adj = (adj_matrix > 0).astype(int)
    return np.sum(binary_adj, axis=1)


def get_rc_nodes(adj_matrix, top_percent):
    """Identify indices of top N% high-degree nodes."""
    degrees = calculate_degree(adj_matrix)
    cutoff = np.percentile(degrees, 100 - top_percent)
    return set(np.where(degrees >= cutoff)[0])


def main():
    print("=" * 60)
    print("Figure S4: Top 20 Most Frequent Rich-Club Nodes")
    print("=" * 60)

    # Load atlas
    atlas = pd.read_csv(ATLAS_PATH, header=None, names=['node_name', 'network'])
    num_nodes = len(atlas)

    # Count RC membership across all subjects
    rc_count = np.zeros(num_nodes)
    total_subjects = 0

    groups = ['Full_Term', 'Preterm', 'Very_Preterm']
    for group in groups:
        subject_folder = os.path.join(INDIVIDUAL_DIR, group)
        if not os.path.isdir(subject_folder):
            print(f"  Warning: {group} folder not found, skipping")
            continue
        subject_files = glob.glob(os.path.join(subject_folder, '*.csv'))
        for sub_file in subject_files:
            try:
                matrix = pd.read_csv(sub_file, header=None).values
                rc_nodes = get_rc_nodes(matrix, RC_PERCENTILE)
                for node in rc_nodes:
                    rc_count[node] += 1
                total_subjects += 1
            except Exception as e:
                print(f"  Error processing {os.path.basename(sub_file)}: {e}")

    print(f"Processed {total_subjects} subjects")

    # Convert to percentage
    rc_percentage = (rc_count / total_subjects) * 100

    # Get top 20
    sorted_indices = np.argsort(-rc_percentage)[:TOP_N]
    top_names = [atlas.iloc[idx]['node_name'] for idx in sorted_indices]
    top_pcts = [rc_percentage[idx] for idx in sorted_indices]
    top_networks = [atlas.iloc[idx]['network'] for idx in sorted_indices]
    top_colors = [NETWORK_COLORS.get(net, '#4C8BF5') for net in top_networks]

    # Reverse for horizontal bar chart (highest at top)
    top_names = top_names[::-1]
    top_pcts = top_pcts[::-1]
    top_networks = top_networks[::-1]
    top_colors = top_colors[::-1]

    print(f"Top {TOP_N} nodes: {top_pcts[-1]:.1f}% – {top_pcts[0]:.1f}%")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8)) # Wider to accommodate legend

    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_pcts, color=top_colors, edgecolor='none', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10, fontweight='bold')
    ax.set_xlabel('Frequency (%)', fontsize=13, fontweight='bold')
    ax.set_title('Top 20 most frequent rich-club nodes',
                 fontsize=15, fontweight='bold', pad=12)

    ax.set_xlim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.tick_params(axis='x', labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    # Legend - only for networks present in top 20
    unique_networks = []
    # Preserve order of appearance or use a standard order
    for net in top_networks:
        if net not in unique_networks:
            unique_networks.append(net)
    
    # Sort them if needed, but here we just take them as they come
    legend_patches = [Patch(facecolor=NETWORK_COLORS[net], label=net) for net in unique_networks]
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5),
              title='Network', title_fontsize=12, frameon=True, fontsize=10, 
              edgecolor='black', prop={'weight': 'bold'})
    plt.setp(ax.get_legend().get_title(), fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'Supplementary_Figure_S4_RC_Node_Identity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
