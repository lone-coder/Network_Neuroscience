#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:57:05 2025

@author: ragnarok
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_graphs_from_subfolders(directory_path):
    """Loads all pickle files from subfolders in the given directory and returns a list of figures."""
    pickle_files = glob.glob(os.path.join(directory_path, '**', '*.pkl'), recursive=True)
    figures = []

    for file_path in pickle_files:
        try:
            with open(file_path, 'rb') as f:
                fig = pickle.load(f)
                figures.append((file_path, fig))
                print(f"Loaded figure from {file_path}")

        except pickle.UnpicklingError as e:
            print(f"Error unpickling file {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing file {file_path}: {e}")

    return figures

def extract_line_data(figures, directory_path):
    """Extracts line data from figures and organizes it by subfolder."""
    graphs = {}
    all_percentiles = np.linspace(0, 1, 100)  # Define standard percentiles for interpolation

    for file_path, fig in figures:
        if not isinstance(fig, plt.Figure) or not fig.axes:
            print(f"Skipping non-figure object or figure with no axes: {file_path}")
            continue

        ax = fig.axes[0]
        line = ax.lines[0]
        degrees = line.get_xdata()
        normalized_coefficients = line.get_ydata()

        # Calculate percentiles for the degrees in this graph
        sorted_degrees = sorted(degrees)
        degree_to_percentile = {}
        n_degrees = len(sorted_degrees)
        if n_degrees > 1:
            for i, degree in enumerate(sorted_degrees):
                percentile = i / (n_degrees - 1)
                degree_to_percentile[degree] = percentile
        elif n_degrees == 1:
            degree_to_percentile[sorted_degrees[0]] = 0 # Handle case with only one degree

        percentiles_for_graph = np.array([degree_to_percentile[d] for d in degrees])

        # Interpolate normalized coefficients to standard percentiles
        interp_func = interp1d(percentiles_for_graph, normalized_coefficients, kind='linear', fill_value=np.nan, bounds_error=False)
        interp_coefficients = interp_func(all_percentiles)

        subfolder = os.path.relpath(os.path.dirname(file_path), directory_path)
        if subfolder not in graphs:
            graphs[subfolder] = {'percentiles': all_percentiles, 'normalized_coefficients': []}

        graphs[subfolder]['normalized_coefficients'].append(interp_coefficients)

    return graphs


def calculate_mean_and_error_bars(graphs):
    """Calculates mean and error bars (from individual normalized coefficients) for each subfolder"""
    result = {}

    for subfolder, data in graphs.items():
        mean_coefficients = []
        error_bars = []

        for i in range(len(data['percentiles'])):
            coefficients_at_percentile = [coefficients[i] for coefficients in data['normalized_coefficients']]
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

def plot_graphs(result, title):
    """Plots the mean coefficients with error bars for each subfolder."""
    plt.figure(figsize=(10, 6))

    for subfolder, data in result.items():
        plt.errorbar(data['percentiles'], data['mean_coefficients'], yerr=data['error_bars'], label=subfolder, capsize=5)

    plt.xlabel('Degree Percentile (0 to 1)')
    plt.ylabel('Normalized Rich Club Coefficient')
    plt.title(title + ' (Percentile X-axis)')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 20)  # Adjust ylim if needed based on your data
    plt.show()

def save_as_pickle(data, directory_path):
    parent_directory = os.path.dirname(directory_path)
    lowest_level_folder = os.path.basename(directory_path)
    filename = f"{lowest_level_folder}_combined_mean_coefficients_percentile_xaxis.pkl"
    filepath = os.path.join(parent_directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved as {filepath}")

# Example usage
weights_directory_path = 'your_path/weights'
topology_directory_path = '/your_path/topology'
unweighted_directory_path= '/your_path/unweighted'

# Load and process unweighted data
unweighted_figures = load_graphs_from_subfolders(unweighted_directory_path)
unweighted_graphs = extract_line_data(unweighted_figures, unweighted_directory_path)
unweighted_result = calculate_mean_and_error_bars(unweighted_graphs)

# Load and process weights data
weights_figures = load_graphs_from_subfolders(weights_directory_path)
weights_graphs = extract_line_data(weights_figures, weights_directory_path)
weights_result = calculate_mean_and_error_bars(weights_graphs)

# Load and process topology data
topology_figures = load_graphs_from_subfolders(topology_directory_path)
topology_graphs = extract_line_data(topology_figures, topology_directory_path)
topology_result = calculate_mean_and_error_bars(topology_graphs)


# Plot weights data
plot_graphs(weights_result, 'Mean Normalized Coefficients with Error Bars (Weights)')

# Plot topology data
plot_graphs(topology_result, 'Mean Normalized Coefficients with Error Bars (Topology)')

plot_graphs(unweighted_result, 'Mean Normalized Coefficients with Error Bars (Unweighted)')


# Save results as pickle files
save_as_pickle(unweighted_result, unweighted_directory_path)
save_as_pickle(weights_result, weights_directory_path)
save_as_pickle(topology_result, topology_directory_path)

#############


##### adjusting the graphs

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_percentile_from_pickle(pickle_file_path, title, y_lim):
    """
    Plots percentile data from a pickle file.

    Args:
        pickle_file_path (str): Path to the pickle file.
        title (str): Title of the plot.
        y_lim (tuple): Tuple specifying the y-axis limits (e.g., (0, 10)).
    """
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    color_map = {
        'Full_Term': 'green',
        'Very_Preterm': 'blue',
        'Preterm': 'orange',
    }

    plt.figure(figsize=(10, 6))

    for subfolder, subfolder_data in data.items():
        percentiles = subfolder_data['percentiles']
        mean_coefficients = subfolder_data['mean_coefficients']
        error_bars = subfolder_data['error_bars']

        color = color_map.get(subfolder, 'black')

        # Remove NaNs for plotting.
        valid_indices = ~np.isnan(percentiles)
        valid_percentiles = percentiles[valid_indices]
        valid_mean_coefficients = mean_coefficients[valid_indices]
        valid_error_bars = error_bars[valid_indices]

        # Ensure endpoints are 0 and 1
        if len(valid_percentiles) > 0:
            if valid_percentiles[0] != 0:
                valid_percentiles = np.insert(valid_percentiles, 0, 0)
                valid_mean_coefficients = np.insert(valid_mean_coefficients, 0, np.nan)
                valid_error_bars = np.insert(valid_error_bars, 0, np.nan)
            if valid_percentiles[-1] != 1:
                valid_percentiles = np.append(valid_percentiles, 1)
                valid_mean_coefficients = np.append(valid_mean_coefficients, np.nan)
                valid_error_bars = np.append(valid_error_bars, np.nan)

            # Remove NaNs again after adding 0 and 1
            valid_indices_final = ~np.isnan(valid_mean_coefficients)
            valid_percentiles = valid_percentiles[valid_indices_final]
            valid_mean_coefficients = valid_mean_coefficients[valid_indices_final]
            valid_error_bars = valid_error_bars[valid_indices_final]

            # Plot the mean coefficients
            plt.plot(valid_percentiles, valid_mean_coefficients, label=subfolder, color=color)

            # Plot the shaded error bars
            plt.fill_between(valid_percentiles, valid_mean_coefficients - valid_error_bars, valid_mean_coefficients + valid_error_bars, color=color, alpha=0.2)

    # Set new limits for the x-axis and y-axis
    plt.xlim(0, 1) #Changed to 0,1
    plt.ylim(y_lim)

    # Add labels and title
    percent_ticks_positions = np.arange(0.1, 1.1, 0.1) # Positions for 10%, 20%, ..., 100%

    percent_ticks_labels = [f'{100-(int(p*100))}' for p in percent_ticks_positions] # Format as "10", "20", etc.
    plt.xticks(percent_ticks_positions, percent_ticks_labels) # Set tick positions and labels
    
    
    plt.xlabel('Top X Percent of Nodes') #Changed label
    plt.ylabel('Φw(norm)')
    plt.title(title)

    # Ensure the legend is visible
    plt.legend(loc='upper left')

    # Show the plot
    plt.show()

# Example usage:
file_paths_and_titles = [
    ('your_path/topology_combined_mean_coefficients_percentile_xaxis.pkl', 'Weighted Rich Club: Shuffle Weights and Topology', (0, 20)),
    ('your_path/weights_combined_mean_coefficients_percentile_xaxis.pkl', 'Weighted Rich Club: Shuffle Weights', (0, 12)),
    ('your_path/unweighted_combined_mean_coefficients_percentile_xaxis.pkl', 'Unweighted Rich Club: Shuffle Topology', (.6, 1.4))
]

for file_path, title, y_lim in file_paths_and_titles:
    plot_percentile_from_pickle(file_path, title, y_lim)


############
import scipy.stats as stats
import numpy as np

def perform_nonparametric_tests_aggregated_percentiles(graphs, start_percentile, end_percentile):
    results = {}

    subfolders = list(graphs.keys())
    aggregated_data = {subfolder: [] for subfolder in subfolders}

    for subfolder in subfolders:
        data = graphs[subfolder]
        percentiles = data['percentiles']
        list_of_normalized_coefficients_curves = data['normalized_coefficients'] # Corrected variable name

        for normalized_coefficients in list_of_normalized_coefficients_curves: # Iterate through list of curves
            for i, percentile in enumerate(percentiles):
                if start_percentile <= percentile <= end_percentile:
                    coefficient = normalized_coefficients[i]

                    # Ensure coefficient is a scalar and not NaN
                    if np.isscalar(coefficient):
                        if not np.isnan(coefficient):
                            aggregated_data[subfolder].append(coefficient)
                    elif isinstance(coefficient, np.ndarray): # Handle numpy array case explicitly
                        if coefficient.size == 1: # If it's a single element array
                            scalar_coefficient = coefficient.item() # Extract the scalar
                            if not np.isnan(scalar_coefficient):
                                aggregated_data[subfolder].append(scalar_coefficient)
                        elif coefficient.size > 1: # If it's a multi-element array
                            print(f"Warning: Encountered multi-element array instead of scalar for coefficient at percentile {percentile}, subfolder {subfolder}. Skipping array.")
                        elif coefficient.size == 0: # If it's an empty array
                            print(f"Warning: Encountered empty array instead of scalar for coefficient at percentile {percentile}, subfolder {subfolder}. Skipping array.")

    if all(len(data) > 0 for data in aggregated_data.values()):  # Perform tests only if all groups have valid data
        # Perform Kruskal-Wallis test
        h_val, p_val_kruskal = stats.kruskal(*aggregated_data.values())
        results['kruskal_wallis'] = {'h_val': h_val, 'p_val': p_val_kruskal}

        results['mann_whitney_u_tests'] = {}
        pairwise_p_values = []

        for i in range(len(subfolders)):
            for j in range(i + 1, len(subfolders)):
                group1 = subfolders[i]
                group2 = subfolders[j]

                # Perform Mann-Whitney U test
                group1_data_mw = [val for val in aggregated_data[group1] if not np.isnan(val)] # Explicitly filter NaNs before test
                group2_data_mw = [val for val in aggregated_data[group2] if not np.isnan(val)] # Explicitly filter NaNs before test
                if len(group1_data_mw) > 0 and len(group2_data_mw) > 0: # Perform Mann-Whitney U test only if both groups have data after NaN removal
                    u_stat, p_val_mann_whitney = stats.mannwhitneyu(group1_data_mw, group2_data_mw, alternative='two-sided')
                    results['mann_whitney_u_tests'][f"{group1} vs {group2}"] = {
                        'u_stat': u_stat,
                        'p_val': p_val_mann_whitney,
                        'p_val_uncorrected': p_val_mann_whitney
                    }
                    pairwise_p_values.append((f"{group1} vs {group2}", p_val_mann_whitney))
                else:
                    results['mann_whitney_u_tests'][f"{group1} vs {group2}"] = "Insufficient data (NaNs or empty group after filtering)"
        
        # Apply Holm-Bonferroni correction to Mann-Whitney p-values
        if pairwise_p_values:
            # Sort p-values in ascending order
            sorted_p_values = sorted(pairwise_p_values, key=lambda x: x[1])
            n_comparisons = len(sorted_p_values)
            
            for idx, (comparison, p_val) in enumerate(sorted_p_values):
                # Holm-Bonferroni correction: multiply by (n_comparisons - rank + 1)
                corrected_p_val = min(1.0, p_val * (n_comparisons - idx))
                results['mann_whitney_u_tests'][comparison]['p_val_corrected'] = corrected_p_val
        
        levene_stat, levene_p_val = stats.levene(*aggregated_data.values())
        results['levene_test'] = {'statistic': levene_stat, 'p_value': levene_p_val}
        
        bartlett_stat, bartlett_p_val = stats.bartlett(*aggregated_data.values())
        results['bartlett_test'] = {'statistic': bartlett_stat, 'p_value': bartlett_p_val}
    return results



# Print results - Homogeneity of Variance Tests
#print("\nHomogeneity of Variance Tests:")
#print(f"Levene's Test Statistic={nonparametric_test_results_aggregated_percentiles['levene_test']['statistic']}, p-value={nonparametric_test_results_aggregated_percentiles['levene_test']['p_value']}")
#print(f"Bartlett's Test Statistic={nonparametric_test_results_aggregated_percentiles['bartlett_test']['statistic']}, p-value={nonparametric_test_results_aggregated_percentiles['bartlett_test']['p_value']}")


# Example usage with percentile range (e.g., 10% to 30% percentile range - corresponding to 0.1 to 0.3)
nonparametric_test_results_aggregated_percentiles = perform_nonparametric_tests_aggregated_percentiles(
    unweighted_graphs, start_percentile=0.8, end_percentile=1.0
)

# Print results
print(f"UN_WEIGHTED Kruskal-Wallis H-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['h_val']}, p-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['p_val']}")
for comparison, mw_test_result in nonparametric_test_results_aggregated_percentiles['mann_whitney_u_tests'].items():
    if isinstance(mw_test_result, dict):
        print(f"Mann-Whitney U test {comparison}: p-value (uncorrected)={mw_test_result['p_val_uncorrected']:.6f}, p-value (Holm-Bonferroni corrected)={mw_test_result['p_val_corrected']:.6f}")
    else:
        print(f"Mann-Whitney U test {comparison}: result={mw_test_result}")
##############



nonparametric_test_results_aggregated_percentiles = perform_nonparametric_tests_aggregated_percentiles(
    weights_graphs, start_percentile=0.8, end_percentile=1.0
)

# Print results
print(f"WEIGHTS Kruskal-Wallis H-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['h_val']}, p-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['p_val']}")
for comparison, mw_test_result in nonparametric_test_results_aggregated_percentiles['mann_whitney_u_tests'].items():
    if isinstance(mw_test_result, dict):
        print(f"Mann-Whitney U test {comparison}: p-value (uncorrected)={mw_test_result['p_val_uncorrected']:.6f}, p-value (Holm-Bonferroni corrected)={mw_test_result['p_val_corrected']:.6f}")
    else:
        print(f"Mann-Whitney U test {comparison}: result={mw_test_result}")


##################


nonparametric_test_results_aggregated_percentiles = perform_nonparametric_tests_aggregated_percentiles(
    topology_graphs, start_percentile=0.8, end_percentile=1.0
)

# Print results
print(f"TOPOLOGY AND Weights Kruskal-Wallis H-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['h_val']}, p-value={nonparametric_test_results_aggregated_percentiles['kruskal_wallis']['p_val']}")
for comparison, mw_test_result in nonparametric_test_results_aggregated_percentiles['mann_whitney_u_tests'].items():
    if isinstance(mw_test_result, dict):
        print(f"Mann-Whitney U test {comparison}: p-value (uncorrected)={mw_test_result['p_val_uncorrected']:.6f}, p-value (Holm-Bonferroni corrected)={mw_test_result['p_val_corrected']:.6f}")
    else:
        print(f"Mann-Whitney U test {comparison}: result={mw_test_result}")
    
    


#####


#####
def find_percentile_range_with_max_difference(results_dict, subfolders, rc_types): # rc_types is likely not needed for dict access now
    """
    Calculates the 10-percentile range where the curves show the maximum difference.

    Args:
        results_dict (dict): Output from calculate_mean_and_error_bars.
        subfolders (list): List of subfolder names (e.g., ['Full_Term', 'Preterm', 'Very_Preterm']).
        rc_types (list): List of rich club coefficient types (e.g., ['unweighted', 'topology', 'weights']). - likely not used for dict access

    Returns:
        dict: Dictionary containing the 10-percentile range with the maximum difference and the difference scores for all ranges.
    """
    print("Debugging inside find_percentile_range_with_max_difference:")
    print("results_dict keys:", results_dict.keys())
    print("rc_types:", rc_types)
    print("subfolders:", subfolders)


    percentiles = results_dict[subfolders[0]]['percentiles'] # Get percentiles array - CORRECTED DICT ACCESS
    difference_scores = {} # Store difference scores for each 10% range
    max_difference_range = None
    max_difference_score = -1 # Initialize with a very low value

    # Iterate through 10% percentile ranges
    for start_percentile_int in range(0, 91, 10): # 0, 10, 20, ..., 90
        end_percentile_int = start_percentile_int + 10
        start_percentile = start_percentile_int / 100.0
        end_percentile = end_percentile_int / 100.0
        range_label = f"{start_percentile_int}-{end_percentile_int}%"
        difference_scores[range_label] = 0 # Initialize score for this range

        # Calculate difference score for this range
        for i in range(len(percentiles)):
            if start_percentile <= percentiles[i] < end_percentile: # Check if percentile is within the current range
                range_percentile = percentiles[i] # For clarity

                # Sum of Squared Differences across group pairs
                group_pairs_diff_sq_sum = 0
                for i_group in range(len(subfolders)):
                    for j_group in range(i_group + 1, len(subfolders)):
                        group1_name = subfolders[i_group]
                        group2_name = subfolders[j_group]

                        # Get mean coefficients for current percentile and groups - CORRECTED DICT ACCESS
                        mean_coeff_group1 = results_dict[group1_name]['mean_coefficients'][i]
                        mean_coeff_group2 = results_dict[group2_name]['mean_coefficients'][i]

                        if not np.isnan(mean_coeff_group1) and not np.isnan(mean_coeff_group2): # Ensure both are not NaN
                            diff_sq = (mean_coeff_group1 - mean_coeff_group2)**2
                            group_pairs_diff_sq_sum += diff_sq

                difference_scores[range_label] += group_pairs_diff_sq_sum # Accumulate squared differences for this range


        # Check if current range has the maximum difference score so far
        current_range_score = difference_scores[range_label]
        if current_range_score > max_difference_score:
            max_difference_score = current_range_score
            max_difference_range = range_label


    return {
        'max_difference_range': max_difference_range,
        'max_difference_score': max_difference_score,
        'difference_scores_per_range': difference_scores
    }
rc_types_to_analyze = ['unweighted', 'topology', 'weights'] # Analyze all RC types
results_for_range_analysis = { # Use the pre-calculated results from calculate_mean_and_error_bars
    'unweighted': unweighted_result,
    'topology': topology_result,
    'weights': weights_result
}

subfolders_list = list(unweighted_result.keys()) # Get subfolders - assuming they are consistent across result types


for rc_type_to_analyze in rc_types_to_analyze: # Loop through each RC type
    # Corrected function call - pass results_for_range_analysis[rc_type_to_analyze], subfolders_list, and rc_types_to_analyze
    range_difference_results = find_percentile_range_with_max_difference(
        results_for_range_analysis[rc_type_to_analyze], # Pass the correct results_dict for the current rc_type
        subfolders_list,
        [rc_type_to_analyze] # Still pass rc_types as a list, even though it's not used for dict access anymore in the function
    )

    print(f"\n--- {rc_type_to_analyze.upper()} Rich Club Coefficient ---")
    print(f"10-Percentile Range with Maximum Curve Difference: {range_difference_results['max_difference_range']}")
    print(f"Maximum Difference Score: {range_difference_results['max_difference_score']:.4f}")
    print("Difference Scores for all 10% Ranges:")
    for range_label, score in range_difference_results['difference_scores_per_range'].items():
        print(f"  {range_label}: {score:.4f}")

