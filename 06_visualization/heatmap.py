#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Ready Heatmap Visualization for LMM Beta Coefficients
==================================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Generate publication-ready heatmaps showing the spatial distribution
         of beta coefficients across the connectome WITH NETWORK ANNOTATIONS.

DATA SOURCE: Uses 'edge_labels_*.csv' files which contain Node_i, Node_j,
             Networks, and LMM results (WeeksPreterm, AgeAtScan).

FIGURES GENERATED:
    - Figure 1: WeeksPreterm (gestational age) beta coefficients
    - Supplementary Figure S1: WeeksPreterm filtered by NBS significance
    - Supplementary Figure S2: AgeAtScan beta coefficients

VISUALIZATION FEATURES:
    - Nodes in ORIGINAL atlas order (preserves hemisphere structure and diamond pattern)
    - Colored annotation bars on axes showing network membership
    - Alignment guarantees: Figure dimensions calculated to ensure square pixels
    - Network legend for reference
    - Clean colorbar labeled with β̂ (beta hat), centered under graph
    - Publication-quality formatting (300 DPI, proper font sizes)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

# ==============================================================================
# CONFIGURATION
# ==============================================================================

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

NETWORK_ORDER = ['Subcort', 'Visual', 'SomMot', 'DorsAttn', 
                 'VentAttn', 'Limbic', 'Control', 'Default']

# ==============================================================================
# HELPER: Load atlas
# ==============================================================================

def load_atlas(atlas_path):
    """Load atlas labels keeping original node order."""
    atlas_data = pd.read_csv(atlas_path, header=None, names=['node_name', 'network'])
    return {
        'labels': atlas_data['node_name'].tolist(),
        'networks': atlas_data['network'].tolist()
    }

# ==============================================================================
# MAIN HEATMAP GENERATOR
# ==============================================================================

def generate_heatmap(data_file, atlas_path, output_path, 
                     coefficient='WeeksPreterm', title_suffix=''):
    """
    Generate a heatmap from an edge_labels CSV file.
    
    Parameters
    ----------
    data_file : str
        Path to edge_labels CSV
    atlas_path : str
        Path to atlas CSV
    output_path : str
        Output filename
    coefficient : str
        Column to plot ('WeeksPreterm' or 'AgeAtScan')
    """
    print(f"Generating: {output_path}")
    
    # 1. Load Data
    df = pd.read_csv(data_file)
    
    # 2. Load Atlas (for network colors and node count)
    atlas_info = load_atlas(atlas_path)
    networks = atlas_info['networks']
    num_nodes = len(atlas_info['labels']) # Should be 227
    
    # 3. Create Matrix
    beta_matrix = np.zeros((num_nodes, num_nodes))
    
    # Check if we have data to plot
    if coefficient not in df.columns:
        print(f"Warning: Column {coefficient} not found in {data_file}")
        return

    # Populate matrix
    # Assuming 0-based indexing from file inspection (0 to 226)
    # But usually 'edge_labels' file might come from R which is 1-based?
    # Our inspection showed indices 0 to 225? And 1 to 226?
    # Let's handle both.
    
    max_idx = max(df['Node_i'].max(), df['Node_j'].max())
    zero_based = (df['Node_i'].min() == 0)
    
    for _, row in df.iterrows():
        i = int(row['Node_i'])
        j = int(row['Node_j'])
        
        if not zero_based:
            i -= 1
            j -= 1
            
        val = row[coefficient]
        
        if i < num_nodes and j < num_nodes:
            beta_matrix[i, j] = val
            beta_matrix[j, i] = val
            
    # 4. Setup Figure Dimensions for Perfect Alignment
    # We want the heatmap area to be physically square
    heatmap_size_in = 8.0
    
    # Margins (inches)
    left_margin = 0.5
    left_bar_w = 0.25
    gap = 0.05
    right_margin = 0.2 # gap before legend
    legend_w = 1.8
    
    bottom_margin = 0.5
    cbar_h = 0.25
    cbar_gap = 0.4
    
    top_bar_h = 0.25
    top_margin = 0.8 # for title
    
    # Total Dimensions
    total_width = left_margin + left_bar_w + gap + heatmap_size_in + right_margin + legend_w
    total_height = bottom_margin + cbar_h + cbar_gap + heatmap_size_in + gap + top_bar_h + top_margin
    
    fig = plt.figure(figsize=(total_width, total_height), dpi=300)
    
    # Calculate relative positions [left, bottom, width, height]
    def to_frac(inches, total_dim):
        return inches / total_dim
        
    hm_left = to_frac(left_margin + left_bar_w + gap, total_width)
    hm_bottom = to_frac(bottom_margin + cbar_h + cbar_gap, total_height)
    hm_w = to_frac(heatmap_size_in, total_width)
    hm_h = to_frac(heatmap_size_in, total_height)
    
    bar_top_bottom = to_frac(bottom_margin + cbar_h + cbar_gap + heatmap_size_in + gap, total_height)
    bar_top_h = to_frac(top_bar_h, total_height)
    
    bar_left_left = to_frac(left_margin, total_width)
    bar_left_w = to_frac(left_bar_w, total_width)
    
    # Create Axes
    ax_main = fig.add_axes([hm_left, hm_bottom, hm_w, hm_h])
    ax_top = fig.add_axes([hm_left, bar_top_bottom, hm_w, bar_top_h])
    ax_left = fig.add_axes([bar_left_left, hm_bottom, bar_left_w, hm_h])
    
    # 5. Plot Heatmap
    # Use aspect='auto' for everything. Since we forced physical dimensions to be square,
    # and the matrix is square, this results in square pixels AND perfect alignment.
    
    # Custom colormap - using pure white center for better contrast
    custom_cmap = LinearSegmentedColormap.from_list("beta_cmap", ["#2166AC", "#FFFFFF", "#B2182B"], N=256)
    vmax = np.max(np.abs(beta_matrix))
    if vmax == 0: vmax = 1
    
    im = ax_main.imshow(beta_matrix, cmap=custom_cmap, vmin=-vmax, vmax=vmax, 
                        aspect='auto', interpolation='none', origin='upper')
    
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    for spine in ax_main.spines.values():
        spine.set_linewidth(0.5)
        
    # 6. Plot Network Bars
    color_array = np.array([to_rgb(NETWORK_COLORS[net]) for net in networks])
    
    # Top Bar (1 row, N columns)
    ax_top.imshow(color_array.reshape(1, -1, 3), aspect='auto', interpolation='none', origin='upper')
    ax_top.axis('off')
    
    # Left Bar (N rows, 1 column)
    ax_left.imshow(color_array.reshape(-1, 1, 3), aspect='auto', interpolation='none', origin='upper')
    ax_left.axis('off')
    
    # 7. Colorbar
    cbar_w_in = heatmap_size_in * 0.6
    cbar_left_in = (left_margin + left_bar_w + gap) + (heatmap_size_in - cbar_w_in)/2
    
    cbar_ax = fig.add_axes([to_frac(cbar_left_in, total_width), 
                            to_frac(bottom_margin, total_height), 
                            to_frac(cbar_w_in, total_width), 
                            to_frac(cbar_h, total_height)])
                            
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\hat{\beta}$', fontsize=16, fontweight='bold')
    
    # Ticks at ends and 0
    ticks = [-vmax, -vmax/2, 0, vmax/2, vmax]
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f'{t:.2f}' for t in ticks], fontsize=12, fontweight='bold')
    
    # 8. Legend
    legend_left = to_frac(left_margin + left_bar_w + gap + heatmap_size_in + 0.5, total_width)
    legend_bottom = hm_bottom
    legend_w_frac = to_frac(legend_w, total_width)
    legend_h_frac = hm_h
    
    ax_legend = fig.add_axes([legend_left, legend_bottom, legend_w_frac, legend_h_frac])
    ax_legend.axis('off')
    
    patches = [Patch(facecolor=NETWORK_COLORS[net], edgecolor='black', linewidth=0.5, label=net) 
               for net in NETWORK_ORDER]
    legend = ax_legend.legend(handles=patches, loc='center left', fontsize=12, 
                     title='Network', title_fontsize=14, frameon=False)
    plt.setp(legend.get_title(), fontweight='bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    # 9. Title
    if 'WeeksPreterm' in coefficient:
        base_title = 'Gestational Age Effect on Edge Weight'
    else:
        base_title = 'Age at Scan Effect on Edge Weight'
        
    full_title = f"{base_title}{title_suffix}"
    fig.suptitle(full_title, fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR = '[PATH_TO_DATA]'
    ATLAS_PATH = os.path.join(BASE_DIR, 'node_labels_5_missing_with_networks.csv')
    OUTPUT_DIR_MAIN = os.path.join(BASE_DIR, 'outputs/figures/main')
    OUTPUT_DIR_SUPP = os.path.join(BASE_DIR, 'outputs/figures/supplementary')
    os.makedirs(OUTPUT_DIR_MAIN, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SUPP, exist_ok=True)

    # Figure 1: Gestational Age Beta Coefficients
    generate_heatmap(
        data_file=os.path.join(BASE_DIR, 'edge_labels_Combined_All_Subjects_LMM_Results.csv'),
        atlas_path=ATLAS_PATH,
        output_path=os.path.join(OUTPUT_DIR_MAIN, 'Figure_1_GA_Beta_Coefficients.png'),
        coefficient='WeeksPreterm'
    )
    
    # Supplementary Figure S1: NBS Filtered Beta Coefficients (WeeksPreterm)
    # Note: Combining NBS 0.76 and 2.28 into single supplementary figure reference
    generate_heatmap(
        data_file=os.path.join(BASE_DIR, 'edge_labels_NBS_0.76__Combined_All_Subjects_LMM_Results.csv'),
        atlas_path=ATLAS_PATH,
        output_path=os.path.join(OUTPUT_DIR_SUPP, 'Supplementary_Figure_S1_NBS_0.76.png'),
        coefficient='WeeksPreterm',
        title_suffix=' (NBS Threshold 0.76)'
    )
    
    generate_heatmap(
        data_file=os.path.join(BASE_DIR, 'edge_labels_NBS_2.28__Combined_All_Subjects_LMM_Results.csv'),
        atlas_path=ATLAS_PATH,
        output_path=os.path.join(OUTPUT_DIR_SUPP, 'Supplementary_Figure_S1_NBS_2.28.png'),
        coefficient='WeeksPreterm',
        title_suffix=' (NBS Threshold 2.28)'
    )
    
    # Supplementary Figure S2: Age at Scan Beta Coefficients
    generate_heatmap(
        data_file=os.path.join(BASE_DIR, 'edge_labels_Combined_All_Subjects_LMM_Results.csv'),
        atlas_path=ATLAS_PATH,
        output_path=os.path.join(OUTPUT_DIR_SUPP, 'Supplementary_Figure_S2_AgeAtScan.png'),
        coefficient='AgeAtScan'
    )
    
    print("✓ Done generating all heatmap figures.")
    print(f"  - Main figures: {OUTPUT_DIR_MAIN}")
    print(f"  - Supplementary figures: {OUTPUT_DIR_SUPP}")
