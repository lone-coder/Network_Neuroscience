#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S5: Edge category composition
===================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Three-panel stacked bar chart showing the percentage of edges with 
         positive vs negative β coefficients:
           A) Rich-club categories: Rich club, Feeder, Local
           B) Distance categories: Long, Medium, Short
           C) Rich-club × Distance crosstabulation

OUTPUT:
    - Supplementary_Figure_S5_Edge_Categorization.png
"""

import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Patch

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial', 'Helvetica'], 'weight': 'bold'})

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = '[PATH_TO_DATA]'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/figures/supplementary')

EDGE_FILE = os.path.join(BASE_DIR, 'edge_labels_Combined_All_Subjects_LMM_Results.csv')
RC_FILE = os.path.join(BASE_DIR, 'Baseline_filtered_weights_averaged_SCM_Edges_top_10_percent_RC_FILTERED.csv')
DISTANCE_FILE = os.path.join(BASE_DIR, 'distance_categorization.csv')

TARGET_COLUMN = 'WeeksPreterm'

RC_CATEGORIES = ['Rich club', 'Feeder', 'Local']
DIST_CATEGORIES = ['Long', 'Medium', 'Short']

RC_COLORS = ['#DC143C', '#4169E1', '#708090']

RC_COLORS = ['#DC143C', '#4169E1', '#708090']
DIST_COLORS = ['#9370DB', '#FFD700', '#228B22'] # For Panel B
# Red/Orange gradient for Panel C (Long=DarkRed, Medium=Orange, Short=Salmon)
# Order in DIST_CATEGORIES is ['Long', 'Medium', 'Short']
CROSSTAB_COLORS = ['#B22222', '#FFA500', '#FA8072']


def draw_stacked_panel(ax, proportions, categories, colors, panel_label):
    """Create stacked bar chart with +/- labels and percentages."""
    pos_vals = [proportions[c]['positive'] for c in categories]
    neg_vals = [proportions[c]['negative'] for c in categories]

    x = np.arange(len(categories))
    width = 0.6

    # Positive (bottom), Negative (top, lighter)
    ax.bar(x, pos_vals, width, color=colors, alpha=0.85, edgecolor='none')
    ax.bar(x, neg_vals, width, bottom=pos_vals, color=colors, alpha=0.4, edgecolor='none')

    # Labels inside bars
    for i, (pv, nv) in enumerate(zip(pos_vals, neg_vals)):
        if pv > 5:
            cy = pv / 2
            ax.text(i, cy + 4, '+', ha='center', va='center',
                    fontweight='bold', fontsize=18, color='black')
            ax.text(i, cy - 4, f'{pv:.0f}%', ha='center', va='center',
                    fontweight='bold', fontsize=11, color='black')
        if nv > 5:
            cy = pv + nv / 2
            ax.text(i, cy + 4, f'{nv:.0f}%', ha='center', va='center',
                    fontweight='bold', fontsize=11, color='black')
            ax.text(i, cy - 4, '−', ha='center', va='center',
                    fontweight='bold', fontsize=18, color='black')

    if panel_label == 'A':
        ax.set_ylabel('Percentage of edges', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('')
        
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.text(0.02, -0.12, f'{panel_label})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def draw_crosstab_panel(ax, rc_merged, rc_categories, dist_categories, dist_colors, panel_label):
    """Create crosstab showing distance breakdown within each RC category."""
    # Calculate proportions
    crosstab_data = {}
    for rc_cat in rc_categories:
        rc_cat_data = rc_merged[rc_merged['Category'] == rc_cat]
        total = len(rc_cat_data)
        if total > 0:
            dist_counts = {}
            for dist_cat in dist_categories:
                count = len(rc_cat_data[rc_cat_data['Category_y'] == dist_cat])
                dist_counts[dist_cat] = (count / total) * 100
            crosstab_data[rc_cat] = dist_counts

    # Create stacked bars
    x = np.arange(len(rc_categories))
    width = 0.6
    bottom = np.zeros(len(rc_categories))

    for i, dist_cat in enumerate(dist_categories):
        heights = [crosstab_data[rc_cat][dist_cat] for rc_cat in rc_categories]
        ax.bar(x, heights, width, bottom=bottom, color=dist_colors[i],
               alpha=0.85, edgecolor='none', label=dist_cat)
        
        # Add percentage labels
        for j, (rc_cat, height) in enumerate(zip(rc_categories, heights)):
            if height > 5:
                y_pos = bottom[j] + height / 2
                ax.text(j, y_pos, f'{height:.1f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='black')
        
        bottom += heights

    if panel_label == 'A':
        ax.set_ylabel('Percentage of edges', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.set_xticks(x)
    ax.set_xticklabels(rc_categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Legend - Outside Right
    # Reverse legend order to match stacked bars (visual consistency)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=10, frameon=True, fancybox=False,
              edgecolor='black', title='', prop={'weight': 'bold'})

    ax.text(0.02, -0.12, f'{panel_label})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def calculate_proportions(merged_data, categories, target_column):
    """Calculate percentage of positive/negative betas per category."""
    proportions = {}
    for cat in categories:
        cat_data = merged_data[merged_data['Category'] == cat]
        pos = len(cat_data[cat_data[target_column] > 0])
        neg = len(cat_data[cat_data[target_column] < 0])
        total = pos + neg
        if total > 0:
            proportions[cat] = {'positive': (pos / total) * 100,
                                'negative': (neg / total) * 100}
        else:
            proportions[cat] = {'positive': 0, 'negative': 0}
    return proportions


def main():
    print("=" * 60)
    print("Figure S5: Edge Category Composition (3 panels)")
    print("=" * 60)

    edge_data = pd.read_csv(EDGE_FILE)
    rc_data = pd.read_csv(RC_FILE)
    dist_data = pd.read_csv(DISTANCE_FILE)

    # RC merge
    rc_merged = pd.merge(edge_data, rc_data,
                         left_on=['Node_i', 'Node_j'],
                         right_on=['Index_1', 'Index_2'], how='inner')
    
    # Distance merge
    dist_merged = pd.merge(edge_data, dist_data,
                           left_on=['Node_i', 'Node_j'],
                           right_on=['Index_1', 'Index_2'], how='inner')
    
    # Combined merge for crosstab (use suffixes to distinguish category columns)
    combined = pd.merge(rc_data, dist_data,
                       on=['Index_1', 'Index_2'], how='inner',
                       suffixes=('_rc', '_dist'))
    combined_with_betas = pd.merge(edge_data, combined,
                                    left_on=['Node_i', 'Node_j'],
                                    right_on=['Index_1', 'Index_2'], how='inner')

    rc_proportions = calculate_proportions(rc_merged, RC_CATEGORIES, TARGET_COLUMN)
    dist_proportions = calculate_proportions(dist_merged, DIST_CATEGORIES, TARGET_COLUMN)

    for cat in RC_CATEGORIES:
        p = rc_proportions[cat]
        print(f"  RC  {cat}: +{p['positive']:.1f}%  −{p['negative']:.1f}%")
    for cat in DIST_CATEGORIES:
        p = dist_proportions[cat]
        print(f"  Dist {cat}: +{p['positive']:.1f}%  −{p['negative']:.1f}%")

    # Three-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    draw_stacked_panel(ax1, rc_proportions, RC_CATEGORIES, RC_COLORS, 'A')
    draw_stacked_panel(ax2, dist_proportions, DIST_CATEGORIES, DIST_COLORS, 'B')
    
    # Prepare data for crosstab - use only Category_rc and Category_dist
    combined_with_betas = combined_with_betas.rename(columns={'Category_rc': 'Category', 'Category_dist': 'Category_y'})
    draw_crosstab_panel(ax3, combined_with_betas, RC_CATEGORIES, DIST_CATEGORIES, CROSSTAB_COLORS, 'C')

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'Supplementary_Figure_S5_Edge_Categorization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
