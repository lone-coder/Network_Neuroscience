#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rich-Club Coefficient Curve Analysis
=====================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Analyze rich-club coefficient (Φ) curves across gestational age groups.
         This tests whether network hubs are more interconnected than expected
         by chance, and whether this differs between birth groups.

WHAT IS THE RICH-CLUB COEFFICIENT?
    The rich-club coefficient measures whether high-degree nodes (hubs) are
    more densely interconnected than would be expected if connections were
    random. A network exhibits a "rich-club" when Φ(k) > 1 for high degree k.
    
    - Φ(k) = fraction of edges among nodes with degree ≥ k
    - Normalized Φ(k) = Φ(k) / Φ_random(k) where Φ_random is from null models
    - Φ_norm > 1 indicates rich-club organization

METHODOLOGICAL APPROACH:
    1. Load pre-computed rich-club curves from pickle files
       (Curves are generated per-subject using BCT/NetworkX)
    2. Interpolate curves to common percentile axis for comparison
    3. Calculate group means and standard deviations
    4. Statistical comparison at selected percentile thresholds

CRITICAL METHODOLOGICAL DECISION - SINGLE PERCENTILE TESTING:
    This script uses the "single-percentile" approach where statistical tests
    are performed at ONE specific threshold (e.g., top 10% of nodes).
    
    Rationale: Each subject contributes exactly ONE observation, ensuring
    statistical validity. The alternative "aggregated" approach would use
    multiple data points per subject, violating independence assumptions.
    
    This follows recommendations for avoiding pseudo-replication in
    repeated-measures designs.

OUTPUT:
    - Publication-ready curve plots with error bars
    - Statistical test results at specified percentiles
    - Effect sizes (Cliff's delta) for group comparisons
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats


# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# ==============================================================================

# Directories containing pickle files organized by group
BASE_DIR = '[PATH_TO_DATA]'
weighted_directory = f'{BASE_DIR}/weights_pickle_subfolders'
unweighted_directory = f'{BASE_DIR}/unweighted_pickle_subfolders'

# Output directory
output_path = f'{BASE_DIR}/outputs/figures/main'

# Target percentiles for statistical testing
# These correspond to: top 20%, 15%, 10%, 5% of nodes
PERCENTILES_TO_TEST = [0.20, 0.15, 0.10, 0.05]

# Group display settings
GROUP_NAME_MAPPING = {
    'Full_Term': 'Full-term',
    'Preterm': 'Moderate-to-late preterm',
    'Very_Preterm': 'Very preterm'
}

COLOR_MAP = {
    'Full_Term': '#FF7F50',      # Coral
    'Preterm': '#20B2AA',        # LightSeaGreen
    'Very_Preterm': '#191970',   # MidnightBlue
}

LEGEND_ORDER = ['Full_Term', 'Preterm', 'Very_Preterm']


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_graphs_from_subfolders(directory_path):
    """
    Load all pickle files containing matplotlib figures from subfolders.
    
    These pickle files contain pre-computed rich-club coefficient curves
    for each subject, saved as matplotlib figure objects.
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing group subfolders
        
    Returns
    -------
    list
        List of (file_path, figure) tuples
    """
    pickle_files = glob.glob(os.path.join(directory_path, '**', '*.pkl'), recursive=True)
    figures = []

    for file_path in pickle_files:
        try:
            with open(file_path, 'rb') as f:
                fig = pickle.load(f)
                figures.append((file_path, fig))
                print(f"Loaded figure from {os.path.basename(file_path)}")

        except pickle.UnpicklingError as e:
            print(f"Error unpickling file {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing file {file_path}: {e}")

    print(f"Total figures loaded: {len(figures)}")
    return figures


def extract_line_data(figures, directory_path):
    """
    Extract rich-club coefficient data from figure objects.
    
    Converts degree-based x-axis to percentile-based axis for comparison
    across subjects (who may have different degree distributions).
    
    Parameters
    ----------
    figures : list
        Output from load_graphs_from_subfolders
    directory_path : str
        Base directory for determining subfolder membership
        
    Returns
    -------
    dict
        {subfolder: {percentiles, normalized_coefficients list}}
    """
    graphs = {}
    all_percentiles = np.linspace(0, 1, 100)  # Standard percentile axis

    for file_path, fig in figures:
        if not isinstance(fig, plt.Figure) or not fig.axes:
            print(f"Skipping non-figure object: {file_path}")
            continue

        ax = fig.axes[0]
        line = ax.lines[0]
        degrees = line.get_xdata()
        normalized_coefficients = line.get_ydata()

        # Convert degrees to percentiles
        sorted_degrees = sorted(degrees)
        degree_to_percentile = {}
        n_degrees = len(sorted_degrees)
        
        if n_degrees > 1:
            for i, degree in enumerate(sorted_degrees):
                percentile = i / (n_degrees - 1)
                degree_to_percentile[degree] = percentile
        elif n_degrees == 1:
            degree_to_percentile[sorted_degrees[0]] = 0

        percentiles_for_graph = np.array([degree_to_percentile[d] for d in degrees])

        # Interpolate to standard percentiles
        interp_func = interp1d(
            percentiles_for_graph, 
            normalized_coefficients, 
            kind='linear', 
            fill_value=np.nan, 
            bounds_error=False
        )
        interp_coefficients = interp_func(all_percentiles)

        # Determine group from file path
        subfolder = os.path.relpath(os.path.dirname(file_path), directory_path)
        
        if subfolder not in graphs:
            graphs[subfolder] = {
                'percentiles': all_percentiles, 
                'normalized_coefficients': []
            }

        graphs[subfolder]['normalized_coefficients'].append(interp_coefficients)

    return graphs


def calculate_mean_and_error_bars(graphs):
    """
    Calculate group-level means and standard deviations.
    
    Parameters
    ----------
    graphs : dict
        Output from extract_line_data
        
    Returns
    -------
    dict
        {subfolder: {percentiles, mean_coefficients, error_bars}}
    """
    result = {}

    for subfolder, data in graphs.items():
        mean_coefficients = []
        error_bars = []

        for i in range(len(data['percentiles'])):
            coefficients_at_percentile = [
                coefficients[i] for coefficients in data['normalized_coefficients']
            ]
            coefficients_at_percentile = np.array(coefficients_at_percentile)
            valid_coefficients = coefficients_at_percentile[~np.isnan(coefficients_at_percentile)]

            mean_coefficients.append(np.nanmean(valid_coefficients))
            error_bars.append(np.nanstd(valid_coefficients))

        result[subfolder] = {
            'percentiles': data['percentiles'],
            'mean_coefficients': np.array(mean_coefficients),
            'error_bars': np.array(error_bars)
        }

    return result


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_rich_club_curves(result, title, y_lim, output_path=None):
    """
    Create publication-ready rich-club coefficient plot.
    
    Parameters
    ----------
    result : dict
        Output from calculate_mean_and_error_bars
    title : str
        Plot title
    y_lim : tuple
        Y-axis limits (min, max)
    output_path : str, optional
        If provided, save figure to this path
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    plot_handles = []
    plot_labels = []
    
    for subfolder_key in LEGEND_ORDER:
        if subfolder_key in result:
            subfolder_data = result[subfolder_key]
            percentiles = subfolder_data['percentiles']
            mean_coefficients = subfolder_data['mean_coefficients']
            error_bars = subfolder_data['error_bars']

            color = COLOR_MAP.get(subfolder_key, 'black')
            display_name = GROUP_NAME_MAPPING.get(subfolder_key, subfolder_key)

            # Remove NaNs for continuous plotting
            valid_indices = ~np.isnan(mean_coefficients)
            valid_percentiles = percentiles[valid_indices]
            valid_mean_coefficients = mean_coefficients[valid_indices]
            valid_error_bars = error_bars[valid_indices]

            # Plot mean line
            line, = plt.plot(valid_percentiles, valid_mean_coefficients, 
                            color=color, linewidth=2.5)
            
            plot_handles.append(line)
            plot_labels.append(display_name)

            # Shaded error bars
            plt.fill_between(valid_percentiles, 
                            valid_mean_coefficients - valid_error_bars, 
                            valid_mean_coefficients + valid_error_bars, 
                            color=color, alpha=0.15)

    # Formatting
    plt.xlim(0, 1)
    plt.ylim(y_lim)

    # X-axis: convert to "Top X Percent" labels
    percent_ticks_positions = np.arange(0.1, 1.1, 0.1)
    percent_ticks_labels = [f'{100-(int(p*100))}' for p in percent_ticks_positions]
    plt.xticks(percent_ticks_positions, percent_ticks_labels, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel('Top X Percent of Nodes', fontsize=16, weight='bold')
    plt.ylabel('Φ(norm)', fontsize=16, weight='bold')
    plt.title(title, fontsize=18, weight='bold', pad=20)

    # Bold tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Legend
    legend_font = {'weight': 'bold', 'size': 14}
    plt.legend(plot_handles, plot_labels, loc='upper left', prop=legend_font, frameon=False)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.show()


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

def cliffs_delta(x, y):
    """
    Calculate Cliff's delta effect size (non-parametric).
    
    Interpretation:
    - |δ| < 0.15: Negligible
    - |δ| 0.15-0.33: Small
    - |δ| 0.34-0.47: Medium
    - |δ| ≥ 0.48: Large
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


def perform_single_percentile_tests(graphs, target_percentile):
    """
    Perform statistical tests at a single percentile threshold.
    
    CRITICAL: Each subject contributes exactly ONE data point, ensuring
    statistical validity and avoiding pseudo-replication.
    
    Parameters
    ----------
    graphs : dict
        Output from extract_line_data
    target_percentile : float
        Specific percentile to test (e.g., 0.10 for top 10%)
        
    Returns
    -------
    dict
        Statistical test results including Kruskal-Wallis and pairwise tests
    """
    results = {}
    subfolders = list(graphs.keys())
    single_point_data = {subfolder: [] for subfolder in subfolders}

    for subfolder in subfolders:
        data = graphs[subfolder]
        percentiles = data['percentiles']
        list_of_curves = data['normalized_coefficients']

        # Find closest percentile index
        percentile_index = np.argmin(np.abs(percentiles - target_percentile))
        actual_percentile = percentiles[percentile_index]
        
        print(f"  {subfolder}: Using percentile {actual_percentile:.3f} "
              f"(closest to {target_percentile:.3f})")

        # Extract ONE coefficient per subject
        for subject_coefficients in list_of_curves:
            coefficient = subject_coefficients[percentile_index]
            
            if np.isscalar(coefficient) and not np.isnan(coefficient):
                single_point_data[subfolder].append(coefficient)
            elif isinstance(coefficient, np.ndarray) and coefficient.size == 1:
                scalar = coefficient.item()
                if not np.isnan(scalar):
                    single_point_data[subfolder].append(scalar)

    # Check all groups have data
    if not all(len(data) > 0 for data in single_point_data.values()):
        print("Warning: Some groups have no data at this percentile")
        return results

    # Sample sizes
    results['sample_sizes'] = {s: len(d) for s, d in single_point_data.items()}
    
    # Descriptive statistics
    results['descriptive_stats'] = {}
    for subfolder, data in single_point_data.items():
        results['descriptive_stats'][subfolder] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'n': len(data)
        }

    # Kruskal-Wallis omnibus test
    k = len(single_point_data)
    df = k - 1
    h_val, p_val_kruskal = stats.kruskal(*single_point_data.values())
    results['kruskal_wallis'] = {'h_val': h_val, 'p_val': p_val_kruskal, 'df': df}
    
    # Levene's test for variance equality
    levene_stat, levene_p = stats.levene(*single_point_data.values())
    results['levene_test'] = {'statistic': levene_stat, 'p_value': levene_p}

    # Pairwise Mann-Whitney U tests
    results['mann_whitney_u_tests'] = {}
    pairwise_p_values = []

    for i in range(len(subfolders)):
        for j in range(i + 1, len(subfolders)):
            group1 = subfolders[i]
            group2 = subfolders[j]

            group1_data = [v for v in single_point_data[group1] if not np.isnan(v)]
            group2_data = [v for v in single_point_data[group2] if not np.isnan(v)]
            
            if len(group1_data) > 0 and len(group2_data) > 0:
                u_stat, p_val = stats.mannwhitneyu(
                    group1_data, group2_data, alternative='two-sided'
                )
                cliff_d = cliffs_delta(group1_data, group2_data)
                
                comparison_key = f"{group1} vs {group2}"
                results['mann_whitney_u_tests'][comparison_key] = {
                    'u_stat': u_stat,
                    'p_val': p_val,
                    'cliffs_delta': cliff_d,
                    'effect_size': interpret_cliffs_delta(cliff_d)
                }
                pairwise_p_values.append((comparison_key, p_val))

    # Apply Holm correction
    if pairwise_p_values:
        pairwise_p_values.sort(key=lambda x: x[1])
        n_comparisons = len(pairwise_p_values)
        
        for i, (comparison_key, p_val) in enumerate(pairwise_p_values):
            corrected_p = min(1.0, p_val * (n_comparisons - i))
            
            if i > 0:
                prev_corrected = results['mann_whitney_u_tests'][pairwise_p_values[i-1][0]].get('p_val_corrected', 0)
                corrected_p = max(corrected_p, prev_corrected)
            
            results['mann_whitney_u_tests'][comparison_key]['p_val_corrected'] = corrected_p

    return results


def run_analysis_at_multiple_percentiles(graphs, percentiles_to_test, output_path=None):
    """
    Run statistical analysis at multiple percentile thresholds.
    
    Parameters
    ----------
    graphs : dict
        Output from extract_line_data
    percentiles_to_test : list
        List of percentiles to test (e.g., [0.20, 0.15, 0.10, 0.05])
    output_path : str, optional
        Directory to save results
        
    Returns
    -------
    dict
        Results for all tested percentiles
    """
    all_results = {}
    
    print("=" * 80)
    print("RICH-CLUB COEFFICIENT STATISTICAL ANALYSIS")
    print("(Single-percentile approach: 1 observation per subject)")
    print("=" * 80)
    
    for percentile in percentiles_to_test:
        pct_label = f"{percentile*100:.0f}%"
        print(f"\n--- Testing at Top {pct_label} ---")
        
        results = perform_single_percentile_tests(graphs, percentile)
        all_results[pct_label] = results
        
        # Print summary
        if 'kruskal_wallis' in results:
            kw = results['kruskal_wallis']
            print(f"  Kruskal-Wallis: H({kw['df']}) = {kw['h_val']:.4f}, p = {kw['p_val']:.4f}")
        
        if 'mann_whitney_u_tests' in results:
            print("  Pairwise comparisons (Holm-corrected):")
            for comparison, test in results['mann_whitney_u_tests'].items():
                display_comp = ' vs '.join([GROUP_NAME_MAPPING.get(g, g) 
                                           for g in comparison.split(' vs ')])
                p_corr = test.get('p_val_corrected', test['p_val'])
                cliff_d = test.get('cliffs_delta', np.nan)
                effect = test.get('effect_size', '')
                print(f"    {display_comp}: p = {p_corr:.4f}, δ = {cliff_d:.3f} ({effect})")
    
    # Save results
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Compile into DataFrame
        table_data = []
        for pct_label, results in all_results.items():
            if 'mann_whitney_u_tests' in results:
                for comparison, test in results['mann_whitney_u_tests'].items():
                    display_comp = ' vs '.join([GROUP_NAME_MAPPING.get(g, g) 
                                               for g in comparison.split(' vs ')])
                    table_data.append({
                        'Percentile': pct_label,
                        'Comparison': display_comp,
                        'U Statistic': test['u_stat'],
                        'p-value (uncorrected)': test['p_val'],
                        'p-value (Holm)': test.get('p_val_corrected', np.nan),
                        'Cliff\'s Delta': test.get('cliffs_delta', np.nan),
                        'Effect Size': test.get('effect_size', '')
                    })
        
        df = pd.DataFrame(table_data)
        csv_path = os.path.join(output_path, 'rich_club_statistical_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved: {csv_path}")
    
    return all_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "=" * 80)
    print("RICH-CLUB COEFFICIENT CURVE ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # UNWEIGHTED ANALYSIS
    # -------------------------------------------------------------------------
    print("\n--- Loading UNWEIGHTED rich-club data ---")
    unweighted_figures = load_graphs_from_subfolders(unweighted_directory)
    unweighted_graphs = extract_line_data(unweighted_figures, unweighted_directory)
    unweighted_result = calculate_mean_and_error_bars(unweighted_graphs)
    
    # Plot
    plot_rich_club_curves(
        unweighted_result, 
        'Unweighted Rich Club', 
        (1, 1.2),
        output_path=os.path.join(output_path, 'Figure_2_Unweighted_Rich_Club.png')
    )
    
    # Statistics
    print("\n--- UNWEIGHTED Statistical Analysis ---")
    unweighted_stats = run_analysis_at_multiple_percentiles(
        unweighted_graphs, 
        PERCENTILES_TO_TEST,
        output_path=os.path.join(output_path, 'unweighted')
    )
    
    # -------------------------------------------------------------------------
    # WEIGHTED ANALYSIS
    # -------------------------------------------------------------------------
    print("\n--- Loading WEIGHTED rich-club data ---")
    weighted_figures = load_graphs_from_subfolders(weighted_directory)
    weighted_graphs = extract_line_data(weighted_figures, weighted_directory)
    weighted_result = calculate_mean_and_error_bars(weighted_graphs)
    
    # Plot
    plot_rich_club_curves(
        weighted_result, 
        'Weighted Rich Club', 
        (0, 20),
        output_path=os.path.join(output_path, 'weighted_rich_club.png')
    )
    
    # Statistics  
    print("\n--- WEIGHTED Statistical Analysis ---")
    weighted_stats = run_analysis_at_multiple_percentiles(
        weighted_graphs, 
        PERCENTILES_TO_TEST,
        output_path=os.path.join(output_path, 'weighted')
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return {
        'unweighted': {'graphs': unweighted_graphs, 'stats': unweighted_stats},
        'weighted': {'graphs': weighted_graphs, 'stats': weighted_stats}
    }


if __name__ == "__main__":
    results = main()
