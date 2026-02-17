#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Figure S6: Edge-wise β coefficients by rich-club membership
          at alternative thresholds (15% and 20%)
=========================================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Two side-by-side 2-row violin panels, each showing β coefficients
         split by positive (top) and negative (bottom) for RC categories
         (Rich club, Feeder, Local), at 15% and 20% RC thresholds.

OUTPUT:
    - Supplementary_Figure_S6_Beta_by_RC.png
"""

import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial', 'Helvetica'], 'weight': 'bold'})

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = '[PATH_TO_DATA]'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/figures/supplementary')

EDGE_FILE = os.path.join(BASE_DIR, 'edge_labels_Combined_All_Subjects_LMM_Results.csv')
RC_15_FILE = os.path.join(BASE_DIR, 'Baseline_filtered_weights_averaged_SCM_Edges_top_15_percent_RC_FILTERED.csv')
RC_20_FILE = os.path.join(BASE_DIR, 'Baseline_filtered_weights_averaged_SCM_Edges_top_20_percent_RC_FILTERED.csv')

TARGET_COLUMN = 'WeeksPreterm'
RC_CATEGORIES = ['Rich club', 'Feeder', 'Local']
RC_COLORS = ['#DC143C', '#4169E1', '#708090']  # Red, Blue, Gray


def add_median_annotations_updated(axes, positive_df_long, negative_df_long, categories):
    """Helper function to add median annotations with BIGGER, BOLDER fonts"""
    
    # BIGGER font for median annotations
    text_fontsize = 16
    text_color = 'black'
    y_offset_factor = 0.03

    # Annotate Positive Plot
    if not positive_df_long.empty:
        plot_range_pos = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        y_offset_pos = plot_range_pos * y_offset_factor
        for i, category in enumerate(categories):
            category_data = positive_df_long[positive_df_long['Category'] == category]
            if not category_data.empty:
                median_val = category_data['Value'].median()
                axes[0].text(x=i, y=median_val + y_offset_pos, s=f'{median_val:.3f}',
                           ha='center', va='bottom', fontsize=text_fontsize, 
                           color=text_color, weight='bold')

    # Annotate Negative Plot
    if not negative_df_long.empty:
        plot_range_neg = axes[1].get_ylim()[1] - axes[1].get_ylim()[0]
        y_offset_neg = plot_range_neg * y_offset_factor
        for i, category in enumerate(categories):
            category_data = negative_df_long[negative_df_long['Category'] == category]
            if not category_data.empty:
                median_val = category_data['Value'].median()
                axes[1].text(x=i, y=median_val + y_offset_neg, s=f'{median_val:.3f}',
                           ha='center', va='bottom', fontsize=text_fontsize, 
                           color=text_color, weight='bold')


def create_violin_subplot(edge_data, categorization_data, categories):
    """Create a 2-row violin subplot data (positive top, negative bottom)"""
    
    # Merge data
    merged_data = pd.merge(edge_data, categorization_data, 
                          left_on=['Node_i', 'Node_j'], 
                          right_on=['Index_1', 'Index_2'])

    # Separate data by category and sign
    positive_values = {category: [] for category in categories}
    negative_values = {category: [] for category in categories}

    for _, row in merged_data.iterrows():
        edge_type = row['Category']
        value = row[TARGET_COLUMN]

        if pd.isna(value): 
            continue

        if edge_type in categories:
            if value > 0:
                positive_values[edge_type].append(value)
            elif value < 0:
                negative_values[edge_type].append(value)

    # Prepare DataFrames
    pos_plot_data, pos_category_labels = [], []
    for category in categories:
        if category in positive_values and positive_values[category]:
            values = positive_values[category]
            pos_plot_data.extend(values)
            pos_category_labels.extend([category] * len(values))
    positive_df_long = pd.DataFrame({'Category': pos_category_labels, 'Value': pos_plot_data})

    neg_plot_data, neg_category_labels = [], []
    for category in categories:
        if category in negative_values and negative_values[category]:
            values = negative_values[category]
            neg_plot_data.extend(values)
            neg_category_labels.extend([category] * len(values))
    negative_df_long = pd.DataFrame({'Category': neg_category_labels, 'Value': neg_plot_data})

    return positive_df_long, negative_df_long


def main():
    print("=" * 60)
    print("Figure S6: Violin plots — 15% & 20% RC thresholds")
    print("=" * 60)

    # Load data
    edge_data = pd.read_csv(EDGE_FILE)
    rc_15 = pd.read_csv(RC_15_FILE)
    rc_20 = pd.read_csv(RC_20_FILE)

    # Create subplots for 15% and 20%
    pos_15, neg_15 = create_violin_subplot(edge_data, rc_15, RC_CATEGORIES)
    pos_20, neg_20 = create_violin_subplot(edge_data, rc_20, RC_CATEGORIES)

    print(f"15% RC merged edges: {len(pos_15) + len(neg_15)}")
    print(f"20% RC merged edges: {len(pos_20) + len(neg_20)}")

    # Create figure with 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex='col', sharey='row')
    
    # Add space between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # 15% RC PLOTS (left column)
    # Plot Positive values (top left)
    if not pos_15.empty:
        sns.violinplot(x='Category', y='Value', data=pos_15, ax=axes[0, 0],
                      order=RC_CATEGORIES, inner="quart", cut=0, palette=RC_COLORS)
        max_val = pos_15['Value'].max()
        padding = max(max_val * 0.05, 0.01)
        axes[0, 0].set_ylim(bottom=-0.01, top=max_val + padding)

    # Plot Negative values (bottom left)
    if not neg_15.empty:
        sns.violinplot(x='Category', y='Value', data=neg_15, ax=axes[1, 0],
                      order=RC_CATEGORIES, inner="quart", cut=0, palette=RC_COLORS)
        min_val = neg_15['Value'].min()
        padding = max(abs(min_val) * 0.05, 0.01)
        axes[1, 0].set_ylim(bottom=min_val - padding, top=0.01)

    # 20% RC PLOTS (right column)
    # Plot Positive values (top right)
    if not pos_20.empty:
        sns.violinplot(x='Category', y='Value', data=pos_20, ax=axes[0, 1],
                      order=RC_CATEGORIES, inner="quart", cut=0, palette=RC_COLORS)
        max_val = pos_20['Value'].max()
        padding = max(max_val * 0.05, 0.01)
        axes[0, 1].set_ylim(bottom=-0.01, top=max_val + padding)

    # Plot Negative values (bottom right)
    if not neg_20.empty:
        sns.violinplot(x='Category', y='Value', data=neg_20, ax=axes[1, 1],
                      order=RC_CATEGORIES, inner="quart", cut=0, palette=RC_COLORS)
        min_val = neg_20['Value'].min()
        padding = max(abs(min_val) * 0.05, 0.01)
        axes[1, 1].set_ylim(bottom=min_val - padding, top=0.01)

    # Enhanced formatting with BIGGER, BOLDER fonts
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # BIGGER, BOLDER tick labels
            if i == 1:  # Only show x-tick labels on bottom plots
                ax.set_xticklabels(RC_CATEGORIES, fontsize=16, weight='bold')
            else:
                ax.set_xticklabels([])
                
            # Set Y-axis tick labels to bold and consistent size
            y_labels = [label.get_text() for label in ax.get_yticklabels()]
            ax.set_yticklabels(y_labels, fontsize=14, weight='bold')

            # BIGGER Y-axis labels
            ax.tick_params(axis='y', labelsize=14, width=2, length=6)
            ax.tick_params(axis='x', labelsize=16, width=2, length=6)

            # Clean design
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.tick_params(top=False, right=False)

    # Add median annotations
    add_median_annotations_updated([axes[0, 0], axes[1, 0]], pos_15, neg_15, RC_CATEGORIES)
    add_median_annotations_updated([axes[0, 1], axes[1, 1]], pos_20, neg_20, RC_CATEGORIES)

    # Panel titles
    axes[0, 0].set_title('Top 15% rich club', fontsize=18, weight='bold', pad=12)
    axes[0, 1].set_title('Top 20% rich club', fontsize=18, weight='bold', pad=12)

    # Y-axis label (only on left column)
    fig.text(0.04, 0.5, 'β coefficient (standardized)', va='center', rotation=90, 
             fontsize=18, weight='bold')

    # Panel labels
    axes[1, 0].text(0.02, -0.18, 'A)', transform=axes[1, 0].transAxes,
                    fontsize=15, fontweight='bold', va='top', ha='left')
    axes[1, 1].text(0.02, -0.18, 'B)', transform=axes[1, 1].transAxes,
                    fontsize=15, fontweight='bold', va='top', ha='left')

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'Supplementary_Figure_S6_Beta_by_RC.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
