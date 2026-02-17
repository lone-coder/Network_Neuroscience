#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis of LMM Edge Results by Category
=====================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Compare beta coefficients from LMM across edge categories:
    - Distance Categories: Short vs Medium vs Long
    - Rich-Club Categories: RC vs Feeder vs Local (at 10%, 15%, 20% thresholds)

METHODOLOGICAL DECISIONS:

1. STATISTICAL TESTS:
   - Kruskal-Wallis H test: Omnibus test for overall group differences
   - Mann-Whitney U test: Pairwise comparisons (non-parametric)
   - Rationale: Beta coefficients may not be normally distributed

2. MULTIPLE COMPARISON CORRECTION:
   - Method: Holm-Bonferroni (sequential Bonferroni)
   - Rationale: Less conservative than classical Bonferroni (as per directive)
     while still controlling family-wise error rate (FWER)
   
   - Distance Analysis: n=6 corrections
     (3 pairwise comparisons × 2 directions [positive/negative])
   
   - Rich-Club Analysis: n=18 corrections (pooled across levels)
     (3 pairwise comparisons × 2 directions × 3 RC thresholds)

3. EFFECT DIRECTION:
   - Beta coefficients are analyzed separately by direction:
     * Positive betas: Strengthened connectivity with higher GA
     * Negative betas: Weakened connectivity with higher GA
   - This follows the published analysis approach

OUTPUT:
    - Publication-ready tables with Holm-corrected p-values
    - CSV files with full results for each analysis
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal


# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SYSTEM
# ==============================================================================

# Edge file with LMM results
edge_file_path = '/path/to/edge_labels_Combined_All_Subjects_LMM_Results.csv'

# Distance categorization file
distance_categorization_file = '/path/to/distance_categorization.csv'

# RC categorization files for each level
rc_categorization_files = {
    '10%': '/path/to/RC_categorization_10_percent_FILTERED.csv',
    '15%': '/path/to/RC_categorization_15_percent_FILTERED.csv',
    '20%': '/path/to/RC_categorization_20_percent_FILTERED.csv'
}

# Output paths
output_path_distance = '/path/to/output_distance'
output_path_rc = '/path/to/output_rc'

# Target column for analysis
target_column = 'WeeksPreterm'

# Categories
distance_categories = ['Long', 'Medium', 'Short']
rc_categories = ['Rich club', 'Feeder', 'Local']


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def apply_holm_correction(results_list, n_corrections):
    """
    Apply Holm-Bonferroni correction with a specified number of corrections.
    
    Holm's method is a step-down procedure that is uniformly more powerful
    than Bonferroni while still controlling FWER.
    
    Procedure:
    1. Sort p-values from smallest to largest
    2. For rank i (1=smallest), corrected_p = p × (n - i + 1)
    3. Ensure monotonicity (corrected p-values can't decrease)
    4. Cap at 1.0
    
    Parameters
    ----------
    results_list : list of dicts
        Each dict must have 'mw_pvalue' field
    n_corrections : int
        Total number of corrections to apply (e.g., 6 for distance, 18 for RC)
    """
    # Filter valid results
    valid_results = [r for r in results_list if not pd.isna(r.get('mw_pvalue'))]
    
    if not valid_results:
        return
    
    # Sort by p-value (ascending)
    sorted_results = sorted(valid_results, key=lambda x: x['mw_pvalue'])
    
    # Apply Holm correction: p_corrected = p × (n - rank + 1)
    corrected_values = []
    for idx, res in enumerate(sorted_results):
        corrected_p = res['mw_pvalue'] * (n_corrections - idx)
        corrected_values.append(corrected_p)
    
    # Ensure monotonicity: corrected p-values must not decrease
    for i in range(1, len(corrected_values)):
        corrected_values[i] = max(corrected_values[i], corrected_values[i-1])
    
    # Apply corrections back, capping at 1.0
    for idx, res in enumerate(sorted_results):
        res['mw_corrected_pvalue'] = min(1.0, corrected_values[idx])
        res['holm_rank'] = idx + 1
    
    # Set NaN for invalid results
    for res in results_list:
        if pd.isna(res.get('mw_pvalue')):
            res['mw_corrected_pvalue'] = np.nan


def format_pvalue(p_val):
    """Format p-value for publication."""
    if pd.isna(p_val):
        return "—"
    elif p_val < 0.001:
        return "< 0.001"
    else:
        return f"{p_val:.3f}"


def format_statistic(stat):
    """Format statistic for publication."""
    if pd.isna(stat):
        return "—"
    else:
        return f"{stat:,.0f}"


# ==============================================================================
# DISTANCE ANALYSIS - HOLM CORRECTION OF 6
# ==============================================================================

def run_distance_analysis(edge_file_path, categorization_file, categories,
                          target_column='WeeksPreterm', output_path=None):
    """
    Run distance analysis with Holm correction of 6.
    
    6 tests = 3 pairwise comparisons × 2 directions (positive/negative)
    
    Parameters
    ----------
    edge_file_path : str
        Path to edge labels CSV
    categorization_file : str
        Path to distance categorization CSV
    categories : list
        Distance categories (e.g., ['Long', 'Medium', 'Short'])
    target_column : str
        Column containing beta coefficients (default: 'WeeksPreterm')
    output_path : str, optional
        Directory to save results
    
    Returns
    -------
    dict
        Dictionary containing all results
    """
    
    print("=" * 80)
    print("DISTANCE ANALYSIS WITH HOLM CORRECTION OF 6")
    print("(3 comparisons × 2 directions = 6 tests)")
    print("=" * 80)
    
    # Load and merge data
    categorization_data = pd.read_csv(categorization_file)
    edge_data = pd.read_csv(edge_file_path)
    
    merged_data = pd.merge(
        categorization_data, edge_data,
        left_on=['Index_1', 'Index_2'],
        right_on=['Node_i', 'Node_j']
    )
    
    # Filter for non-zero effects and separate by direction
    filtered_data = merged_data[merged_data[target_column] != 0]
    analysis_data = filtered_data[filtered_data['Category'].isin(categories)]
    positive_data = analysis_data[analysis_data[target_column] > 0]
    negative_data = analysis_data[analysis_data[target_column] < 0]
    
    print(f"\nData summary:")
    print(f"  Total edges with non-zero {target_column}: {len(filtered_data):,}")
    print(f"  Positive (strengthened): {len(positive_data):,}")
    print(f"  Negative (weakened): {len(negative_data):,}")
    
    # Print category breakdown
    print(f"\nCategory breakdown:")
    for direction, data in [("Positive", positive_data), ("Negative", negative_data)]:
        print(f"  {direction}:")
        for cat in categories:
            count = len(data[data['Category'] == cat])
            print(f"    {cat}: {count:,}")
    
    # Kruskal-Wallis omnibus tests
    print(f"\nOmnibus tests (Kruskal-Wallis):")
    for direction, data in [("Positive", positive_data), ("Negative", negative_data)]:
        groups = [data[data['Category'] == cat][target_column] for cat in categories]
        if all(len(g) > 0 for g in groups):
            h_stat, p_val = kruskal(*groups)
            print(f"  {direction}: H = {h_stat:.4f}, p = {p_val:.4f}")
    
    # Calculate pairwise Mann-Whitney tests
    def calculate_pairwise_stats(data, direction_label):
        results = []
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1 = categories[i]
                cat2 = categories[j]
                values1 = data[data['Category'] == cat1][target_column]
                values2 = data[data['Category'] == cat2][target_column]
                
                if len(values1) > 0 and len(values2) > 0:
                    mw_result = mannwhitneyu(values1, values2, alternative='two-sided')
                    result = {
                        'direction': direction_label,
                        'comparison': f"{cat1} vs {cat2}",
                        'n1': len(values1),
                        'n2': len(values2),
                        'mw_statistic': mw_result.statistic,
                        'mw_pvalue': mw_result.pvalue
                    }
                else:
                    result = {
                        'direction': direction_label,
                        'comparison': f"{cat1} vs {cat2}",
                        'n1': len(values1),
                        'n2': len(values2),
                        'mw_statistic': np.nan,
                        'mw_pvalue': np.nan
                    }
                results.append(result)
        return results
    
    pos_results = calculate_pairwise_stats(positive_data, "Positive")
    neg_results = calculate_pairwise_stats(negative_data, "Negative")
    all_results = pos_results + neg_results
    
    # Apply Holm correction with n=6
    print(f"\nApplying Holm correction with n=6...")
    apply_holm_correction(all_results, n_corrections=6)
    
    # Separate back for display
    pos_results_corrected = [r for r in all_results if r['direction'] == 'Positive']
    neg_results_corrected = [r for r in all_results if r['direction'] == 'Negative']
    
    # Print results table
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U TEST RESULTS (Distance - Holm-6)")
    print("=" * 80)
    
    print(f"\n{'':.<20} {'Strengthened':^25} {'Weakened':^25}")
    print(f"{'Comparison':<20} {'U-stat':>10} {'p-value':>12} {'U-stat':>10} {'p-value':>12}")
    print("-" * 80)
    
    for pos_res, neg_res in zip(pos_results_corrected, neg_results_corrected):
        print(f"{pos_res['comparison']:<20} "
              f"{format_statistic(pos_res['mw_statistic']):>10} "
              f"{format_pvalue(pos_res['mw_corrected_pvalue']):>12} "
              f"{format_statistic(neg_res['mw_statistic']):>10} "
              f"{format_pvalue(neg_res['mw_corrected_pvalue']):>12}")
    
    print("\nNote: All p-values corrected using Holm-Bonferroni method with n=6.")
    
    # Save results
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        output_data = []
        for pos_res, neg_res in zip(pos_results_corrected, neg_results_corrected):
            output_data.append({
                'Comparison': pos_res['comparison'],
                'Pos_n': pos_res['n1'],
                'Pos_U_Statistic': pos_res['mw_statistic'],
                'Pos_pvalue_uncorrected': pos_res['mw_pvalue'],
                'Pos_pvalue_Holm6': pos_res['mw_corrected_pvalue'],
                'Neg_n': neg_res['n1'],
                'Neg_U_Statistic': neg_res['mw_statistic'],
                'Neg_pvalue_uncorrected': neg_res['mw_pvalue'],
                'Neg_pvalue_Holm6': neg_res['mw_corrected_pvalue']
            })
        
        df_output = pd.DataFrame(output_data)
        output_file = os.path.join(output_path, f'Distance_MannWhitney_Holm6_{target_column}.csv')
        df_output.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    return {
        'all_results': all_results,
        'positive_results': pos_results_corrected,
        'negative_results': neg_results_corrected
    }


# ==============================================================================
# RICH-CLUB ANALYSIS - HOLM CORRECTION OF 18
# ==============================================================================

def run_rc_analysis(edge_file_path, rc_files_dict, categories,
                    target_column='WeeksPreterm', output_path=None):
    """
    Run RC analysis across multiple percentile levels with global Holm correction of 18.
    
    18 tests = 3 pairwise comparisons × 2 directions × 3 levels
    
    This pools all tests across RC thresholds (10%, 15%, 20%) for a single
    family-wise correction. This is more conservative but more appropriate
    when all thresholds are examining the same underlying question.
    
    Parameters
    ----------
    edge_file_path : str
        Path to edge labels CSV
    rc_files_dict : dict
        Dictionary mapping threshold labels to RC categorization file paths
    categories : list
        RC categories (e.g., ['Rich club', 'Feeder', 'Local'])
    target_column : str
        Column containing beta coefficients (default: 'WeeksPreterm')
    output_path : str, optional
        Directory to save results
    
    Returns
    -------
    dict
        Dictionary containing all results
    """
    
    print("=" * 80)
    print("RICH CLUB ANALYSIS WITH GLOBAL HOLM CORRECTION OF 18")
    print("(3 comparisons × 2 directions × 3 levels = 18 tests)")
    print("=" * 80)
    
    all_level_results = {}
    all_test_results = []  # Collect ALL tests for global correction
    
    # Step 1: Run tests at each percentile level WITHOUT correction
    for percentile_label, rc_file in rc_files_dict.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {percentile_label} Rich Club threshold")
        print(f"{'=' * 50}")
        
        # Load and merge data
        categorization_data = pd.read_csv(rc_file)
        edge_data = pd.read_csv(edge_file_path)
        
        merged_data = pd.merge(
            categorization_data, edge_data,
            left_on=['Index_1', 'Index_2'],
            right_on=['Node_i', 'Node_j']
        )
        
        # Filter and separate
        filtered_data = merged_data[merged_data[target_column] != 0]
        analysis_data = filtered_data[filtered_data['Category'].isin(categories)]
        positive_data = analysis_data[analysis_data[target_column] > 0]
        negative_data = analysis_data[analysis_data[target_column] < 0]
        
        print(f"  Positive (strengthened): {len(positive_data):,}")
        print(f"  Negative (weakened): {len(negative_data):,}")
        
        # Calculate pairwise tests
        def calculate_stats(data, direction_label, level_label):
            results = []
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    cat1 = categories[i]
                    cat2 = categories[j]
                    values1 = data[data['Category'] == cat1][target_column]
                    values2 = data[data['Category'] == cat2][target_column]
                    
                    if len(values1) > 0 and len(values2) > 0:
                        mw_result = mannwhitneyu(values1, values2, alternative='two-sided')
                        result = {
                            'level': level_label,
                            'direction': direction_label,
                            'comparison': f"{cat1} vs {cat2}",
                            'n1': len(values1),
                            'n2': len(values2),
                            'mw_statistic': mw_result.statistic,
                            'mw_pvalue': mw_result.pvalue
                        }
                    else:
                        result = {
                            'level': level_label,
                            'direction': direction_label,
                            'comparison': f"{cat1} vs {cat2}",
                            'n1': len(values1),
                            'n2': len(values2),
                            'mw_statistic': np.nan,
                            'mw_pvalue': np.nan
                        }
                    results.append(result)
            return results
        
        pos_results = calculate_stats(positive_data, "Positive", percentile_label)
        neg_results = calculate_stats(negative_data, "Negative", percentile_label)
        
        all_level_results[percentile_label] = {
            'positive_results': pos_results,
            'negative_results': neg_results
        }
        
        # Add to master list for global correction
        all_test_results.extend(pos_results)
        all_test_results.extend(neg_results)
    
    # Step 2: Apply GLOBAL Holm correction with n=18
    print(f"\n{'=' * 50}")
    print("Applying global Holm correction with n=18")
    print(f"{'=' * 50}")
    
    apply_holm_correction(all_test_results, n_corrections=18)
    
    valid_tests = sum(1 for r in all_test_results if not pd.isna(r.get('mw_pvalue')))
    print(f"  Total valid tests: {valid_tests} (corrected with n=18)")
    
    # Step 3: Print results for each percentile level
    for percentile_label in rc_files_dict.keys():
        print(f"\n{'=' * 80}")
        print(f"RESULTS FOR {percentile_label} RICH CLUB (Holm-18 corrected)")
        print("=" * 80)
        
        level_results = [r for r in all_test_results if r['level'] == percentile_label]
        pos_results = [r for r in level_results if r['direction'] == 'Positive']
        neg_results = [r for r in level_results if r['direction'] == 'Negative']
        
        print(f"\n{'':.<25} {'Strengthened':^20} {'Weakened':^20}")
        print(f"{'Comparison':<25} {'U-stat':>8} {'p-value':>10} {'U-stat':>8} {'p-value':>10}")
        print("-" * 75)
        
        for pos_res, neg_res in zip(pos_results, neg_results):
            print(f"{pos_res['comparison']:<25} "
                  f"{format_statistic(pos_res['mw_statistic']):>8} "
                  f"{format_pvalue(pos_res['mw_corrected_pvalue']):>10} "
                  f"{format_statistic(neg_res['mw_statistic']):>8} "
                  f"{format_pvalue(neg_res['mw_corrected_pvalue']):>10}")
    
    print("\nNote: All p-values corrected using Holm-Bonferroni method with n=18")
    print("      (pooled across all 3 RC levels).")
    
    # Save results
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Save combined results
        all_output_data = []
        for res in all_test_results:
            all_output_data.append({
                'Level': res['level'],
                'Direction': res['direction'],
                'Comparison': res['comparison'],
                'n1': res['n1'],
                'n2': res['n2'],
                'U_Statistic': res['mw_statistic'],
                'pvalue_uncorrected': res['mw_pvalue'],
                'pvalue_Holm18': res.get('mw_corrected_pvalue', np.nan)
            })
        
        df_all = pd.DataFrame(all_output_data)
        combined_file = os.path.join(output_path, f'RC_All_Levels_MannWhitney_Holm18_{target_column}.csv')
        df_all.to_csv(combined_file, index=False)
        print(f"\n✓ Combined results saved to: {combined_file}")
    
    print(f"\n{'=' * 50}")
    print("✓ Rich Club analysis complete!")
    print(f"{'=' * 50}")
    
    return {
        'all_results': all_test_results,
        'level_results': all_level_results
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 25 + "LMM STATISTICAL ANALYSIS" + " " * 29 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80 + "\n")
    
    # -------------------------------------------------------------------------
    # PART 1: DISTANCE ANALYSIS (Holm-6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: DISTANCE ANALYSIS")
    print("=" * 80)
    
    distance_results = run_distance_analysis(
        edge_file_path=edge_file_path,
        categorization_file=distance_categorization_file,
        categories=distance_categories,
        target_column=target_column,
        output_path=output_path_distance
    )
    
    # -------------------------------------------------------------------------
    # PART 2: RICH CLUB ANALYSIS (Holm-18)
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("PART 2: RICH CLUB ANALYSIS")
    print("=" * 80)
    
    rc_results = run_rc_analysis(
        edge_file_path=edge_file_path,
        rc_files_dict=rc_categorization_files,
        categories=rc_categories,
        target_column=target_column,
        output_path=output_path_rc
    )
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 30 + "ANALYSIS COMPLETE" + " " * 31 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print("\nSummary of corrections applied:")
    print("-" * 40)
    print("Distance Analysis:")
    print("  - 3 pairwise comparisons × 2 directions = 6 tests")
    print("  - Holm correction applied with n = 6")
    print("\nRich Club Analysis:")
    print("  - 3 pairwise comparisons × 2 directions × 3 levels = 18 tests")
    print("  - Holm correction applied with n = 18 (pooled across all levels)")
    print(f"\nOutput locations:")
    print(f"  Distance: {output_path_distance}")
    print(f"  Rich Club: {output_path_rc}")
