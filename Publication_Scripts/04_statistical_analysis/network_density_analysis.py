#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Density Metrics Analysis
================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Compare global network density metrics across gestational age groups:
    - Full-term (≥37 weeks)
    - Moderate-to-late preterm (32-36 weeks)  
    - Very preterm (<32 weeks)

METRICS CALCULATED:
    1. Unweighted Density: proportion of possible edges that exist
       Formula: number_of_edges / max_possible_edges
       
    2. Sum Weighted Density: total connection strength across network
       Formula: sum(all_edge_weights)
       
    3. Average Weighted Density: mean connection strength per existing edge
       Formula: sum(edge_weights) / number_of_edges

STATISTICAL APPROACH:
    - Omnibus test: Kruskal-Wallis H test
    - Pairwise comparisons: Mann-Whitney U test
    - Multiple comparison correction: Holm's method (within each metric)
    - Effect size: Cliff's delta

RATIONALE FOR HOLM CORRECTION WITHIN METRIC:
    Each density metric captures a distinct network property:
    - Unweighted density = network sparsity/topology
    - Sum weighted = overall connection strength
    - Average weighted = mean connection efficiency
    
    Therefore, corrections are applied separately within each metric's
    family of 3 pairwise comparisons (not across all 9 tests).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime


# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# ==============================================================================

# Directory containing group subfolders with connectivity matrices
data_directory = '[PATH_TO_DATA]/Matrices/Baseline/Groups_Baseline_filtered_weights'

# Output directory for results - updated to use centralized outputs folder
output_path = '[PATH_TO_DATA]/outputs/figures/main'

# Expected group subfolder names
GROUP_NAME_MAPPING = {
    'Full_Term': 'Full-term',
    'Preterm': 'Moderate-to-late preterm',
    'Very_Preterm': 'Very preterm'
}

# Color scheme for visualizations
COLOR_PALETTE = {
    'Full-term': '#FF7F50',                    # Coral
    'Moderate-to-late preterm': '#20B2AA',     # LightSeaGreen
    'Very preterm': '#191970'                  # MidnightBlue
}



# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_matrices_from_subfolders(directory_path):
    """
    Load all CSV connectivity matrices from group subfolders.
    
    Expected directory structure:
        directory_path/
        ├── Full_Term/
        │   ├── subject1.csv
        │   ├── subject2.csv
        │   └── ...
        ├── Preterm/
        │   └── ...
        └── Very_Preterm/
            └── ...
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing group subfolders
        
    Returns
    -------
    dict
        {group_name: list of (filepath, matrix) tuples}
    """
    matrices = {}
    
    # Get all subfolders
    subfolders = [d for d in os.listdir(directory_path) 
                  if os.path.isdir(os.path.join(directory_path, d))]
    
    print(f"Discovered group subfolders: {subfolders}")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory_path, subfolder)
        csv_files = glob.glob(os.path.join(subfolder_path, '*.csv'))
        
        matrices[subfolder] = []
        
        for csv_path in csv_files:
            try:
                # Load as pure numeric matrix (no headers/index)
                matrix = np.loadtxt(csv_path, delimiter=',')
                matrices[subfolder].append((csv_path, matrix))
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        
        print(f"  {subfolder}: loaded {len(matrices[subfolder])} matrices")
    
    return matrices


# ==============================================================================
# METRIC CALCULATION
# ==============================================================================

def calculate_density_metrics(matrix):
    """
    Calculate density metrics for a single connectivity matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Symmetric connectivity matrix (NxN)
        
    Returns
    -------
    dict
        Dictionary containing all calculated metrics
    
    Notes
    -----
    Only the upper triangle is used to avoid double-counting in
    undirected networks.
    """
    n = matrix.shape[0]
    
    # Extract upper triangle (excluding diagonal - no self-connections)
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_values = matrix[upper_tri_indices]
    
    # Maximum possible edges in undirected network without self-loops
    max_possible_edges = n * (n - 1) / 2
    
    # 1. Unweighted density: binarize (>0 becomes 1), count edges / max possible
    binary_edges = (upper_tri_values > 0).astype(int)
    num_edges = np.sum(binary_edges)
    unweighted_density = num_edges / max_possible_edges
    
    # 2. Sum weighted density: sum of all edge weights
    sum_weighted_density = np.sum(upper_tri_values)
    
    # 3. Average weighted density: sum of weights / number of actual edges
    if num_edges > 0:
        avg_weighted_density = sum_weighted_density / num_edges
    else:
        avg_weighted_density = 0.0
    
    return {
        'unweighted_density': unweighted_density,
        'sum_weighted_density': sum_weighted_density,
        'avg_weighted_density': avg_weighted_density,
        'num_edges': num_edges,
        'max_possible_edges': max_possible_edges,
        'num_nodes': n
    }


def compute_metrics_for_all_subjects(matrices_dict):
    """
    Compute density metrics for all subjects in all groups.
    
    Parameters
    ----------
    matrices_dict : dict
        Output from load_matrices_from_subfolders
        
    Returns
    -------
    dict
        {group_name: list of metric dicts}
    """
    all_metrics = {}
    
    for group_name, matrix_list in matrices_dict.items():
        all_metrics[group_name] = []
        
        for filepath, matrix in matrix_list:
            metrics = calculate_density_metrics(matrix)
            metrics['filepath'] = filepath
            metrics['subject_id'] = os.path.basename(filepath)
            all_metrics[group_name].append(metrics)
        
        print(f"{group_name}: computed metrics for {len(all_metrics[group_name])} subjects")
    
    return all_metrics


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size (non-parametric).
    
    Cliff's delta measures the degree of overlap between two distributions.
    
    Parameters
    ----------
    x, y : array-like
        Two groups of observations to compare
        
    Returns
    -------
    float
        Cliff's delta value between -1 and 1
        
    Interpretation
    --------------
    - |δ| < 0.15: Negligible effect
    - |δ| 0.15-0.33: Small effect
    - |δ| 0.34-0.47: Medium effect
    - |δ| ≥ 0.48: Large effect
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    delta = 0
    for i in x:
        for j in y:
            if i > j:
                delta += 1
            elif i < j:
                delta -= 1
    
    return delta / (n1 * n2)


def interpret_cliffs_delta(delta):
    """Interpret Cliff's delta magnitude."""
    if pd.isna(delta) or delta is None:
        return ""
    abs_delta = abs(delta)
    if abs_delta < 0.15:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.47:
        return "medium"
    else:
        return "large"


def perform_statistical_tests(all_metrics, metric_name):
    """
    Perform statistical tests for a single metric.
    
    Applies Holm's correction WITHIN this metric's pairwise comparisons.
    
    Parameters
    ----------
    all_metrics : dict
        Output from compute_metrics_for_all_subjects
    metric_name : str
        Name of metric to test (e.g., 'unweighted_density')
        
    Returns
    -------
    dict
        Results including omnibus test, pairwise comparisons, and descriptive stats
    """
    results = {}
    
    # Extract data for each group
    group_data = {}
    for group_name, metrics_list in all_metrics.items():
        group_data[group_name] = [m[metric_name] for m in metrics_list]
    
    group_names = list(group_data.keys())
    
    # Descriptive statistics
    results['descriptive_stats'] = {}
    for group_name, data in group_data.items():
        results['descriptive_stats'][group_name] = {
            'n': len(data),
            'mean': np.mean(data),
            'std': np.std(data, ddof=1),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }
    
    results['sample_sizes'] = {g: len(d) for g, d in group_data.items()}
    
    # Kruskal-Wallis omnibus test
    h_stat, p_kruskal = stats.kruskal(*group_data.values())
    results['kruskal_wallis'] = {
        'h_stat': h_stat,
        'p_value': p_kruskal,
        'df': len(group_names) - 1
    }
    
    # Levene's test for equality of variances
    levene_stat, levene_p = stats.levene(*group_data.values())
    results['levene_test'] = {
        'statistic': levene_stat,
        'p_value': levene_p
    }
    
    # Pairwise Mann-Whitney U tests
    results['pairwise_tests'] = {}
    pairwise_comparisons = []
    
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1 = group_names[i]
            group2 = group_names[j]
            
            data1 = group_data[group1]
            data2 = group_data[group2]
            
            # Mann-Whitney U test
            u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Cliff's delta effect size
            cliff_d = cliffs_delta(data1, data2)
            
            comparison_key = f"{group1} vs {group2}"
            results['pairwise_tests'][comparison_key] = {
                'u_stat': u_stat,
                'p_value_uncorrected': p_val,
                'cliffs_delta': cliff_d,
                'n1': len(data1),
                'n2': len(data2)
            }
            pairwise_comparisons.append((comparison_key, p_val))
    
    # Apply Holm's correction WITHIN this metric
    pairwise_comparisons.sort(key=lambda x: x[1])
    n_comparisons = len(pairwise_comparisons)
    
    for i, (comparison_key, p_val) in enumerate(pairwise_comparisons):
        # Holm's correction: p_corrected = p × (n - rank + 1)
        alpha_divisor = n_comparisons - i
        p_corrected = min(1.0, p_val * alpha_divisor)
        
        # Ensure monotonicity
        if i > 0:
            prev_corrected = pairwise_comparisons[i-1][2] if len(pairwise_comparisons[i-1]) > 2 else 0
            p_corrected = max(p_corrected, prev_corrected)
        
        results['pairwise_tests'][comparison_key]['p_value_corrected'] = p_corrected
        results['pairwise_tests'][comparison_key]['holm_rank'] = i + 1
        
        pairwise_comparisons[i] = (comparison_key, p_val, p_corrected)
    
    results['correction_info'] = {
        'method': 'Holm',
        'n_comparisons': n_comparisons,
        'scope': 'within_metric'
    }
    
    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_unweighted_density_figure(all_metrics, output_path):
    """
    Create publication-quality unweighted density violin plot with group differences.
    
    Figure shows:
        - Violin plots for each gestational age group
        - Individual subject data points (scatter)
        - Mean (solid red) and median (dashed blue) horizontal lines
        - Legend positioned outside plot to avoid data overlap
    
    Parameters
    ----------
    all_metrics : dict
        Output from compute_metrics_for_all_subjects
    output_path : str
        Directory to save the figure
        
    Returns
    -------
    str
        Path to saved figure
    """
    group_order = ['Full-term', 'Moderate-to-late preterm', 'Very preterm']

    
    # Prepare data
    plot_data = []
    for group_name, metrics_list_data in all_metrics.items():
        display_name = GROUP_NAME_MAPPING.get(group_name, group_name)
        for m in metrics_list_data:
            plot_data.append({
                'Group': display_name,
                'Density': m['unweighted_density']
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create figure with extra space on right for legend
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Violin plot
    violin_parts = ax.violinplot(
        [df_plot[df_plot['Group'] == g]['Density'].values for g in group_order],
        positions=range(len(group_order)),
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins
    colors = [COLOR_PALETTE[g] for g in group_order]
    for i, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.7)
        body.set_linewidth(1)
    
    # Add individual points with jitter
    np.random.seed(42)  # Reproducible jitter
    for i, group in enumerate(group_order):
        group_data = df_plot[df_plot['Group'] == group]['Density'].values
        jitter = np.random.uniform(-0.15, 0.15, len(group_data))
        ax.scatter(
            np.full_like(group_data, i) + jitter,
            group_data,
            c='black',
            s=25,
            alpha=0.6,
            edgecolor='none',
            zorder=3
        )
    
    # Add mean line
    mean_line = None
    for i, group in enumerate(group_order):
        group_data = df_plot[df_plot['Group'] == group]['Density'].values
        mean_val = np.mean(group_data)
        
        # Mean line (solid red)
        mean_line = ax.hlines(mean_val, i - 0.25, i + 0.25, colors='#E53935', 
                              linewidth=2.5, linestyle='-', zorder=4)
    
    # Axis formatting
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, fontsize=12, fontweight='bold')
    ax.set_ylabel('Unweighted Density', fontsize=14, fontweight='bold')
    ax.set_title('Thresholded Unweighted Network Density', fontsize=16, fontweight='bold', pad=15)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Y-axis formatting
    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Legend OUTSIDE plot area (upper right, outside)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#E53935', linewidth=2.5, linestyle='-', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=False, edgecolor='gray', fontsize=10)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.82)
    
    # Save figure as Figure 4 for manuscript
    fig_path = os.path.join(output_path, 'Figure_4_Network_Density.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {fig_path}")
    
    return fig_path


def create_distribution_plots(all_metrics, output_path):
    """
    Create box plots showing all three metrics across groups.
    
    Parameters
    ----------
    all_metrics : dict
        Output from compute_metrics_for_all_subjects
    output_path : str
        Directory to save plots
    """
    group_order = ['Full-term', 'Moderate-to-late preterm', 'Very preterm']

    
    metrics_list = ['unweighted_density', 'sum_weighted_density', 'avg_weighted_density']
    titles = ['Unweighted Density', 'Sum of Edge Weights', 'Average Edge Weight']
    ylabels = ['Density', 'Sum of Weights', 'Mean Weight']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (metric_name, title, ylabel) in enumerate(zip(metrics_list, titles, ylabels)):
        ax = axes[idx]
        
        # Prepare data
        plot_data = []
        for group_name, metrics_list_data in all_metrics.items():
            display_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            for m in metrics_list_data:
                plot_data.append({
                    'Group': display_name,
                    'Value': m[metric_name]
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Box plot
        sns.boxplot(data=df_plot, x='Group', y='Value', order=group_order,
                   palette=COLOR_PALETTE, ax=ax)
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        # Bold tick labels
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Network Density Metrics by Birth Group', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    fig_path = os.path.join(output_path, 'combined_density_metrics.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {fig_path}")


# ==============================================================================
# RESULTS OUTPUT
# ==============================================================================

def save_results(all_results, all_metrics, output_path):
    """
    Save all results to files.
    
    Parameters
    ----------
    all_results : dict
        Results from all statistical tests
    all_metrics : dict
        Raw metrics for all subjects
    output_path : str
        Directory to save results
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create combined results table
    all_tables = []
    for metric_name, results in all_results.items():
        table_data = []
        for comparison, test_results in results['pairwise_tests'].items():
            parts = comparison.split(' vs ')
            display_comparison = ' vs '.join([GROUP_NAME_MAPPING.get(p, p) for p in parts])
            
            table_data.append({
                'Metric': metric_name,
                'Comparison': display_comparison,
                'U Statistic': test_results['u_stat'],
                'p-value (uncorrected)': test_results['p_value_uncorrected'],
                'p-value (Holm)': test_results['p_value_corrected'],
                'Holm Rank': test_results['holm_rank'],
                'Cliff\'s Delta': test_results['cliffs_delta'],
                'Effect Size': interpret_cliffs_delta(test_results['cliffs_delta'])
            })
        all_tables.append(pd.DataFrame(table_data))
    
    combined_table = pd.concat(all_tables, ignore_index=True)
    csv_path = os.path.join(output_path, 'density_metrics_pairwise_comparisons.csv')
    combined_table.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save individual subject metrics
    raw_data = []
    for group_name, metrics_list in all_metrics.items():
        for m in metrics_list:
            raw_data.append({
                'Group': GROUP_NAME_MAPPING.get(group_name, group_name),
                'Subject_ID': m['subject_id'],
                'Num_Nodes': m['num_nodes'],
                'Num_Edges': m['num_edges'],
                'Max_Possible_Edges': m['max_possible_edges'],
                'Unweighted_Density': m['unweighted_density'],
                'Sum_Weighted_Density': m['sum_weighted_density'],
                'Avg_Weighted_Density': m['avg_weighted_density']
            })
    
    raw_df = pd.DataFrame(raw_data)
    raw_csv_path = os.path.join(output_path, 'individual_subject_metrics.csv')
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"✓ Saved: {raw_csv_path}")
    
    return combined_table


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_density_analysis(data_directory, output_path):
    """
    Main function to run the complete density analysis.
    
    Parameters
    ----------
    data_directory : str
        Path to directory containing group subfolders with CSV matrices
    output_path : str
        Path to save results
    """
    print("=" * 80)
    print("NETWORK DENSITY METRICS ANALYSIS")
    print("=" * 80)
    print(f"Data directory: {data_directory}")
    print(f"Output directory: {output_path}")
    print()
    
    # Step 1: Load matrices
    print("Step 1: Loading connectivity matrices...")
    matrices = load_matrices_from_subfolders(data_directory)
    print()
    
    # Step 2: Compute metrics
    print("Step 2: Computing density metrics for all subjects...")
    all_metrics = compute_metrics_for_all_subjects(matrices)
    print()
    
    # Step 3: Statistical tests
    print("Step 3: Performing statistical tests (Holm correction within each metric)...")
    metrics_to_test = ['unweighted_density', 'sum_weighted_density', 'avg_weighted_density']
    
    all_results = {}
    for metric_name in metrics_to_test:
        print(f"  Testing: {metric_name}")
        all_results[metric_name] = perform_statistical_tests(all_metrics, metric_name)
    print()
    
    # Step 4: Generate plots
    print("Step 4: Generating distribution plots...")
    os.makedirs(output_path, exist_ok=True)
    create_distribution_plots(all_metrics, output_path)
    create_unweighted_density_figure(all_metrics, output_path)  # Publication figure
    print()
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    results_table = save_results(all_results, all_metrics, output_path)
    print()
    
    # Print summary to console
    print("=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    
    for metric_name, results in all_results.items():
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        kw = results['kruskal_wallis']
        sig = "*" if kw['p_value'] < 0.05 else ""
        print(f"  Kruskal-Wallis: H({kw['df']}) = {kw['h_stat']:.3f}, p = {kw['p_value']:.4f} {sig}")
        
        print("  Pairwise (Holm-corrected):")
        sorted_tests = sorted(results['pairwise_tests'].items(), 
                            key=lambda x: x[1]['holm_rank'])
        for comparison, test_results in sorted_tests:
            parts = comparison.split(' vs ')
            display = ' vs '.join([GROUP_NAME_MAPPING.get(p, p) for p in parts])
            sig = "*" if test_results['p_value_corrected'] < 0.05 else ""
            print(f"    {display}: p = {test_results['p_value_corrected']:.4f} {sig}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return {
        'metrics': all_metrics,
        'results': all_results,
        'table': results_table
    }


if __name__ == "__main__":
    results = run_density_analysis(data_directory, output_path)
