
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def categorize_lengths_with_indices_and_plot(edge_file_path, lengths_file, output_file):
    """
    Categorizes edge lengths into short medium, long, based on quartiles for edges with non-zero coefficients
    and generates plots.

    Args:
        edge_file_path (str): Path to the edge file (LMM output).
        lengths_file (str): Path to the lengths file.
        output_file (str): Path to save the categorized edges CSV.
    """
    # Read the edge file to compare (LMM output)
    try:
        edge_df = pd.read_csv(edge_file_path)
        print(f"\nNumber of edges in LMM graph (original): {edge_df.shape[0]}")

        # FILTER LMM EDGES: Keep only edges with non-zero WeeksPreterm or AgeAtScan
        filtered_lmm_edges_df = edge_df[(edge_df['WeeksPreterm'] != 0) | (edge_df['AgeAtScan'] != 0)] # OR condition
        print(f"Number of edges in LMM graph (non-zero WeeksPreterm OR AgeAtScan): {filtered_lmm_edges_df.shape[0]}")
        print(f"  Edges with non-zero WeeksPreterm: {edge_df[edge_df['WeeksPreterm'] != 0].shape[0]}, Edges with non-zero AgeatScan: {edge_df[edge_df['AgeAtScan'] != 0].shape[0]}") # Original counts for reference


    except FileNotFoundError:
        print(f"Error: Edge file not found at '{edge_file_path}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Edge file '{edge_file_path}' is empty.")
        return
    except Exception as e:
        print(f"Error reading edge file '{edge_file_path}': {e}")
        return

    # Read the lengths file
    try:
        lengths_df = pd.read_csv(lengths_file, header=None)
        print(f"\nNumber of edges in length graph: {lengths_df.shape[0]}")
    except FileNotFoundError:
        print(f"Error: Lengths file not found at '{lengths_file}'")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Lengths file '{lengths_file}' is empty.")
        return
    except Exception as e:
        print(f"Error reading lengths file '{lengths_file}': {e}")
        return

    # Match edges based on indices from *FILTERED* LMM edge file and extract lengths
    matched_edges = []
    for i in range(lengths_df.shape[0]):
        for j in range(i + 1, lengths_df.shape[1]):
            if filtered_lmm_edges_df[(filtered_lmm_edges_df['Node_i'] == i) & (filtered_lmm_edges_df['Node_j'] == j)].shape[0] > 0: # Use filtered_lmm_edges_df
                length = lengths_df.iat[i, j]
                matched_edges.append((i, j, length))

    matched_edges_df = pd.DataFrame(matched_edges, columns=['Index_1', 'Index_2', 'Length'])
    print(f"\nNumber of edges in matched graph: {matched_edges_df.shape[0]}, Edges with non-zero lengths: {matched_edges_df[matched_edges_df['Length'] > 0].shape[0]}")

    # Extract lengths and filter out zero values
    values = matched_edges_df['Length'].values
    values = values[values > 0] # Filters to ensure no zero values are included

    # Calculate quartiles for categorization
    first_quartile = np.percentile(values, 25)
    third_quartile = np.percentile(values, 75)
    short_threshold = first_quartile
    long_threshold = third_quartile

    # Categorize streamlines based on quartiles
    short_streamlines = values[values < short_threshold]
    medium_streamlines = values[(values >= short_threshold) & (values <= long_threshold)]
    long_streamlines = values[values > long_threshold]

    # Print category statistics and thresholds
    print(f'\nStreamline Category Counts (based on filtered LMM edges):') # Updated description
    print(f'Short Streamlines: {len(short_streamlines)} (Threshold: < {short_threshold:.2f} mm)')
    print(f'Medium Streamlines: {len(medium_streamlines)} (Threshold: {short_threshold:.2f} mm - {long_threshold:.2f} mm)')
    print(f'Long Streamlines: {len(long_streamlines)} (Threshold: > {long_threshold:.2f} mm)')
    print(f'\nShort Range Cutoff: < {short_threshold:.2f} mm')
    print(f'Long Range Cutoff: > {long_threshold:.2f} mm')

    # Categorize edges and create DataFrame for saving
    categorized_edges = []
    for _, row in matched_edges_df.iterrows():
        i, j, length = row
        if length > 0: # Ensure we are not processing zero length edges
            if length < short_threshold:
                category = 'Short'
            elif length <= long_threshold:
                category = 'Medium'
            else:
                category = 'Long'
            categorized_edges.append((i, j, length, category))

    categorized_edges_df = pd.DataFrame(categorized_edges, columns=['Index_1', 'Index_2', 'Length', 'Category'])
    print(f"\nNumber of edges in categorized graph (lengths > 0, based on filtered LMM edges): {categorized_edges_df.shape[0]}") # Updated description


    # Save the categorized edges to CSV
    try:
        categorized_edges_df.to_csv(output_file, index=False)
        print(f"Categorized edges saved to: {output_file}")
    except Exception as e:
        print(f"Error saving categorized edges to '{output_file}': {e}")

    # Create and display 2-panel plot: Boxplot and Histogram
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Panel 1: Box plot
    axs[0].boxplot(values, vert=True, patch_artist=True, showmeans=True)
    axs[0].set_title('Streamline Lengths Distribution (Filtered LMM Edges)') # Updated title
    axs[0].set_ylabel('Average Streamline Length (mm)')

    # Add median value label
    median = np.median(values)
    axs[0].text(1.1, median, f'Median: {median:.2f}', ha='center', va='center', color='orange')

    # Panel 2: Histogram
    axs[1].hist(values, bins=99, color='blue', edgecolor='black')
    axs[1].set_title('Streamline Lengths Histogram (Filtered LMM Edges)') # Updated title
    axs[1].set_xlabel('Average Streamline Length (mm)')
    axs[1].set_ylabel('Frequency (Edge Count)')
    axs[1].axvline(first_quartile, color='red', linestyle='--', linewidth=1, label=f'Q1: {short_threshold:.2f} mm')
    axs[1].axvline(third_quartile, color='red', linestyle='-', linewidth=1, label=f'Q3: {long_threshold:.2f} mm')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


################
import os
import numpy as np

def calculate_top_percent_RC_nodes_and_edges(data_path, output_path, percentages, node_labels_path):
    """
    Calculates top percentage nodes and categorizes edges (RC, Feeder, Local)
    for a SINGLE MATRIX CSV FILE, based on UNWEIGHTED node degree.
    This version does NOT subdivide the Rich Club.

    Args:
        data_path (str): Path to the SINGLE CSV file containing the weighted graph matrix.
        output_path (str): Path to the directory where output files will be saved.
        percentages (list of int): List of percentages for top node analysis (e.g., [10, 15, 20]).
        node_labels_path (str): Path to the CSV file containing node labels.
    """
    print(f"Starting function with:")
    print(f"  Data File Path: {data_path}")
    print(f"  Output Path: {output_path}")
    print(f"  Percentages: {percentages}")
    print(f"  Node Labels Path: {node_labels_path}")

    # Check if paths exist
    if not os.path.exists(data_path):
        print(f"Error: Data file path '{data_path}' does not exist.")
        return
    if not os.path.exists(output_path):
        print(f"Error: Output path '{output_path}' does not exist.")
        return
    if not os.path.exists(node_labels_path):
        print(f"Error: Node labels path '{node_labels_path}' does not exist.")
        return

    # Load node labels from the CSV file
    print(f"Loading node labels from: {node_labels_path}")
    try:
        node_labels = np.loadtxt(node_labels_path, dtype=str, delimiter=',', usecols=0)
        print(f"Node labels loaded successfully. Number of labels: {len(node_labels)}")
    except Exception as e:
        print(f"Error loading node labels: {e}")
        return

    file_path = data_path # Directly use data_path as file_path
    print(f"\nProcessing file: {file_path}") # Indicate the file being processed
    # Load the weighted graph from the CSV file
    try:
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        print(f"  Matrix loaded successfully. Shape: {matrix.shape}")
    except Exception as e:
        print(f"  Error loading matrix from '{file_path}': {e}")
        return # Exit if loading the matrix fails

    # Calculate the UNWEIGHTED degree for each node
    unweighted_degrees = np.sum(matrix > 0, axis=1) # Line for UNWEIGHTED degree
    print(f"  Unweighted degrees calculated. Example degrees: {unweighted_degrees[:5]}") # Updated print statement

    for percentage in percentages:
        print(f"  Processing percentage: {percentage}%")
        # Determine the threshold for the top percentage nodes by UNWEIGHTED degree
        threshold = np.percentile(unweighted_degrees, 100 - percentage) # Use unweighted degrees
        print(f"    Threshold for top {percentage}%: {threshold}") # Updated print statement

        # Identify the nodes that are in the top percentage by UNWEIGHTED degree
        top_percent_indices = np.where(unweighted_degrees >= threshold)[0] # Use unweighted degrees
        top_percent_nodes = node_labels[top_percent_indices]
        top_percent_degrees = unweighted_degrees[top_percent_indices] # Use unweighted_degrees
        print(f"    Found {len(top_percent_nodes)} top {percentage}% nodes.")

        # Combine node names and their unweighted degrees
        output_data = np.column_stack((top_percent_nodes, top_percent_degrees)) # Use top_percent_degrees

        # Save the list of top percentage nodes and their unweighted degrees to a new CSV file
        output_file_name = os.path.basename(file_path).replace('.csv', f'_Nodes_top_{percentage}_percent_RC.csv') # Updated filename
        output_file_path = os.path.join(output_path, output_file_name)

        try:
            np.savetxt(output_file_path, output_data, fmt='%s', delimiter=',', header='Node,Unweighted_Degree', comments='') # Updated header
            print(f"    Top {percentage}% nodes saved to: {output_file_path}")
        except Exception as e:
            print(f"    Error saving top nodes to '{output_file_path}': {e}")


        # Identify and categorize edges
        edges = []

        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i, j] > 0:  # Check if there is an edge
                    if i in top_percent_indices and j in top_percent_indices:
                        edge_type = 'RC'
                    elif i in top_percent_indices or j in top_percent_indices:
                        edge_type = 'Feeder'
                    else:
                        edge_type = 'Local'

                    edges.append((i, j, node_labels[i], node_labels[j], edge_type, matrix[i, j]))

        # Save the categorized edges to a new CSV file
        edge_output_file_name = os.path.basename(file_path).replace('.csv', f'_Edges_top_{percentage}_percent_RC.csv') # Updated filename
        edge_output_file_path = os.path.join(output_path, edge_output_file_name)

        try:
            np.savetxt(edge_output_file_path, edges, fmt='%s', delimiter=',', header='Index_1,Index_2,Name_1,Name_2,Type,Weight', comments='') # Header remains the same for edges
            print(f"    Edges saved to: {edge_output_file_path}")
        except Exception as e:
            print(f"    Error saving edges to '{edge_output_file_path}': {e}")
    print("Function finished.")



edge_file_path = '/Users/ragnarok/Desktop/PROJECTS/Control_data/Results/Weighted_Longitudinal/LMM_Edges_Rough_Pass_Pvalue/latest/edge_labels_LMM_Longitudinal_10_60_LMM_Longitudinal_10_60_BadVolumes_StandardizedEdges.csv'

######################
###short medium and long tests
###########

#categorization_file = 'your_path.csv'
#categories = ['Long', 'Medium', 'Short'] # Define categories here, before it's used

##################
#### Rich club tests
####################
categorization_file ='your_path.csv'
categories = ['RC', 'Feeder', 'Local']

# Choose the target column to analyze
target_column = 'WeeksPreterm'  # Change to 'AgeAtScan' if needed
#target_column = 'AgeAtScan'

### Output path for results
output_path = 'your_path' # Update if needed

# Load the categorization data
print("Loading categorization data...")
categorization_data = pd.read_csv(categorization_file)
print("Categorization data loaded.")

# Load the results data
print("Loading results data...")
results = pd.read_csv(edge_file_path)
print("Results data loaded.")

# Merge the categorization data with the results data on Index_1 and Index_2
print("Merging data...")
merged_data = pd.merge(categorization_data, results, left_on=['Index_1', 'Index_2'], right_on=['Node_i', 'Node_j'])
print("Data merged.")

# Filter out rows where target_column is 0
print("Filtering data...")
filtered_data = merged_data[merged_data[target_column] != 0]
print("Data filtered.")

# Filter to include only specified edge types
analysis_data = filtered_data[filtered_data['Category'].isin(categories)]

# Calculate and print category counts and percentages
category_counts = analysis_data['Category'].value_counts()
total_edges = len(analysis_data)
print(f"\n{target_column} - Counts and Percentages:")
for category in categories:
    if category in category_counts:
        count = category_counts[category]
        percentage = (count / total_edges) * 100
        print(f"  {category}: {count} edges ({percentage:.2f}%)")
    else:
        print(f"  {category}: 0 edges (0.00%)")

# Separate the data into positive and negative values
positive_data = analysis_data[analysis_data[target_column] > 0]
negative_data = analysis_data[analysis_data[target_column] < 0]

# Calculate proportions of positive and negative values for each category
proportions = {category: {'positive': 0, 'negative': 0} for category in categories}

for category in categories:
    category_data = analysis_data[analysis_data['Category'] == category]
    positive_count = len(category_data[category_data[target_column] > 0])
    negative_count = len(category_data[category_data[target_column] < 0])
    total_count = positive_count + negative_count
    if total_count > 0:
        proportions[category]['positive'] = positive_count / total_count
        proportions[category]['negative'] = negative_count / total_count
    else:
        proportions[category]['positive'] = 0
        proportions[category]['negative'] = 0

# Print the proportions
print(f"\n{target_column} Proportions:")
for category in categories:
    print(f"Category: {category}")
    print(f"Percent positive values: {proportions[category]['positive']:.2%}")

# Function to perform statistical tests and print results with corrected p-values
def perform_tests(data, data_type, target_col, categories):
    
    print(f"\n{data_type} {target_col} Tests")
    for category in categories:
        category_data = data[data['Category'] == category]
        mean_value = category_data[target_col].mean()
        median_value = category_data[target_col].median()
        print(f"  {data_type} {category} - Mean: {mean_value:.4f}, Median: {median_value:.4f}")

    # Extract values for Kruskal-Wallis test for each category
    category_values = [data[data['Category'] == cat][target_col] for cat in categories]

    # Perform Kruskal-Wallis H test
    kruskal_result = kruskal(*category_values)
    print(f"  {data_type} Kruskal-Wallis result: H={kruskal_result.statistic:.4f}, p={kruskal_result.pvalue:.4f}")

    # Perform pairwise Mann-Whitney U tests for all pairs of categories
    pairwise_results = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1 = categories[i]
            cat2 = categories[j]
            values1 = data[data['Category'] == cat1][target_col]
            values2 = data[data['Category'] == cat2][target_col]
            mannwhitney_result = mannwhitneyu(values1, values2)
            pairwise_results.append({
                'comparison': f"{cat1} vs {cat2}",
                'statistic': mannwhitney_result.statistic,
                'pvalue': mannwhitney_result.pvalue
            })
    
    # Apply Holm-Bonferroni correction
    p_values = [result['pvalue'] for result in pairwise_results]
    n_comparisons = len(p_values)
    
    # Sort p-values and apply Holm-Bonferroni correction
    sorted_indices = np.argsort(p_values)
    corrected_p_values = [0] * n_comparisons
    
    for idx, sorted_idx in enumerate(sorted_indices):
        alpha_level = 0.05 / (n_comparisons - idx)
        corrected_p = min(p_values[sorted_idx] * (n_comparisons - idx), 1.0)
        corrected_p_values[sorted_idx] = corrected_p
    
    # Print results with corrected p-values
    print(f"  {data_type} Pairwise Mann-Whitney U tests (Holm-Bonferroni corrected):")
    for i, result in enumerate(pairwise_results):
        print(f"    {result['comparison']}: U={result['statistic']:.4f}, p_raw={result['pvalue']:.4f}, p_corrected={corrected_p_values[i]:.4f}")

# Perform tests for positive values
perform_tests(positive_data, "Positive", target_column, categories)

# Perform tests for negative values
perform_tests(negative_data, "Negative", target_column, categories)



#######


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import kruskal, mannwhitneyu



def create_vertical_violin_plots_with_medians_nogrid(edge_files_path, categorization_file, output_path, target_column, categories):
    """
    Loads edge data and categorization, merges them, and creates vertically arranged
    violin plots (Positive over Negative) of raw values for a specified target column,
    grouped by specified categories. Uses inner="quart" for violins (shows median/quartiles).
    Removes axis labels but keeps Y-axis tick marks. Removes grid lines.
    Adjusts Y-axis limits individually for each plot to minimize whitespace around zero,
    adding a small gap beyond the zero line.
    Adds MEDIAN value text annotation next to each violin.

    Args:
        edge_files_path (str): Path to the CSV file containing edge data
                               (must include 'Node_i', 'Node_j', and target_column).
        categorization_file (str): Path to the CSV file containing edge categories
                                   (must include 'Index_1', 'Index_2', 'Category').
        output_path (str): Directory path to save the output plot.
        target_column (str): The name of the column in edge_data to analyze
                             (e.g., 'WeeksPreterm', 'AgeAtScan').
        categories (list): A list of strings representing the categories to plot
                           (e.g., ['Local', 'Feeder', 'RC']).
    """
    print(f"--- Creating Vertical Violin Plots with Median Annotations (No Grid) for Column: {target_column} ---") # Updated print message
    print(f"--- Using Categories: {categories} ---")

    # --- 1. Load and Merge Data ---
    try:
        # *** Check if the placeholder path is still being used ***
        if 'your_actual_edge_data.csv' in edge_files_path:
             print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("!!! ERROR: 'edge_file_path' is not set correctly.               !!!")
             print(f"!!!       Current value: {edge_files_path}")
             print("!!!       Please update it to your actual edge data file path. !!!")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
             # return

        edge_data = pd.read_csv(edge_files_path)
        categorization_data = pd.read_csv(categorization_file)

        # --- Input Data Validation ---
        if target_column not in edge_data.columns:
            print(f"Error: Target column '{target_column}' not found in edge file: {edge_files_path}")
            return
        required_edge_cols = ['Node_i', 'Node_j', target_column]
        required_cat_cols = ['Index_1', 'Index_2', 'Category']
        if not all(col in edge_data.columns for col in required_edge_cols):
            print(f"Error: Edge file {edge_files_path} must contain columns: {required_edge_cols}")
            return
        if not all(col in categorization_data.columns for col in required_cat_cols):
            print(f"Error: Categorization file {categorization_file} must contain columns: {required_cat_cols}")
            return
        # --- End Validation ---

        merged_data = pd.merge(edge_data, categorization_data, left_on=['Node_i', 'Node_j'], right_on=['Index_1', 'Index_2'])
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        if 'your_actual_edge_data.csv' in str(e):
             print(">>> Hint: Did you forget to set the 'edge_file_path' variable? <<<")
        return
    except Exception as e:
        print(f"An error occurred during data loading/merging: {e}")
        return

    # --- 2. Separate Data by Category and Sign ---
    positive_values = {category: [] for category in categories}
    negative_values = {category: [] for category in categories}

    for _, row in merged_data.iterrows():
        edge_type = row['Category']
        value = row[target_column]

        if pd.isna(value): continue # Skip missing values

        if edge_type in categories:
            if value > 0:
                positive_values[edge_type].append(value)
            elif value < 0:
                negative_values[edge_type].append(value)
            # Values exactly equal to 0 are ignored

    # --- 3. Prepare Long-Form DataFrames ---
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

    if positive_df_long.empty and negative_df_long.empty:
        print("\nNo non-zero data found for any specified category. Cannot generate plot.")
        return

    # --- 4. Create Violin Plots (Vertical 2x1 Layout) ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True, sharey=False)
    chosen_palette = ['violet', 'gold', 'red']

    # --- 5. Plot and Set Individual Y-Limits ---
    # Plot Positive values
    if not positive_df_long.empty:
        # Using inner="quart" which shows median/quartiles visually
        sns.violinplot(x='Category', y='Value', data=positive_df_long, ax=axes[0],
                       order=categories, inner="quart", cut=0, palette=chosen_palette)
        max_val = positive_df_long['Value'].max()
        padding = max(max_val * 0.05, 0.01)
        lower_lim = -0.01
        upper_lim = max_val + padding
        axes[0].set_ylim(bottom=lower_lim, top=upper_lim)
        print(f"Positive plot Y-limits set to: ({lower_lim:.3f}, {upper_lim:.3f})")
    else:
        axes[0].text(0.5, 0.5, 'Positive Values (No Data)', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_ylim(-0.01, 0.1)

    # Plot Negative values
    if not negative_df_long.empty:
         # Using inner="quart" which shows median/quartiles visually
        sns.violinplot(x='Category', y='Value', data=negative_df_long, ax=axes[1],
                       order=categories, inner="quart", cut=0, palette=chosen_palette)
        min_val = negative_df_long['Value'].min()
        padding = max(abs(min_val) * 0.05, 0.01)
        lower_lim = min_val - padding
        upper_lim = 0.01
        axes[1].set_ylim(bottom=lower_lim, top=upper_lim)
        print(f"Negative plot Y-limits set to: ({lower_lim:.3f}, {upper_lim:.3f})")
        
        

    else:
        axes[1].text(0.5, 0.5, 'Negative Values (No Data)', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_ylim(-0.1, 0.01)
   


    # --- 5b. Add MEDIAN Annotations --- <--- REVERTED TO MEDIAN
    print("--- Adding Median Annotations ---") # Back to Median
    text_fontsize = 8
    text_ha = 'center'
    text_va = 'bottom'
    text_color = 'black'
    y_offset_factor = 0.02

    # Annotate Positive Plot with MEDIAN
    if not positive_df_long.empty:
        plot_range_pos = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        y_offset_pos = plot_range_pos * y_offset_factor
        for i, category in enumerate(categories):
            category_data = positive_df_long[positive_df_long['Category'] == category]
            if not category_data.empty:
                median_val = category_data['Value'].median() # Calculate MEDIAN
                axes[0].text(x=i,
                             y=median_val + y_offset_pos, # Position near median + offset
                             s=f'{median_val:.2f}', # Display MEDIAN value
                             ha=text_ha,
                             va=text_va,
                             fontsize=text_fontsize,
                             color=text_color)
                print(f"  Positive - Category '{category}': Median={median_val:.2f}") # Back to Median
            else:
                 print(f"  Positive - Category '{category}': No data to calculate median.") # Back to Median


    # Annotate Negative Plot with MEDIAN
    if not negative_df_long.empty:
        plot_range_neg = axes[1].get_ylim()[1] - axes[1].get_ylim()[0]
        y_offset_neg = plot_range_neg * y_offset_factor
        for i, category in enumerate(categories):
            category_data = negative_df_long[negative_df_long['Category'] == category]
            if not category_data.empty:
                median_val = category_data['Value'].median() # Calculate MEDIAN
                axes[1].text(x=i,
                             y=median_val + y_offset_neg, # Position near median + offset
                             s=f'{median_val:.2f}', # Display MEDIAN value
                             ha=text_ha,
                             va=text_va,
                             fontsize=text_fontsize,
                             color=text_color)
                print(f"  Negative - Category '{category}': Median={median_val:.2f}") # Back to Median
            else:
                print(f"  Negative - Category '{category}': No data to calculate median.") # Back to Median
    # --- End Median Annotations ---


    # --- 6. Set Common Properties & Formatting ---
    for i, ax in enumerate(axes):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)

        # Remove grid lines <--- CHANGE HERE
        ax.grid(False) # Turn off grid lines for both x and y axes

    # plt.subplots_adjust(hspace=0.1)
    plt.tight_layout() # Call *before* saving

    # --- 7. Save and Show Plot ---
    os.makedirs(output_path, exist_ok=True)
    categories_label = "_".join(categories)
    # Update filename
    output_file_name = f'violinplots_vertical_{target_column}_{categories_label}_adj_ylim_gap_medians_nogrid.png' # Added _nogrid
    output_file_path = os.path.join(output_path, output_file_name)

    try:
        plt.savefig(output_file_path)
        print(f"\nPlot saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()



# === Execution ===
if __name__ == "__main__":
    # (Keep the path checking logic as before)
    path_error = False
    if 'your_actual_edge_data.csv' in edge_file_path:
        print("!!!!!!!!!!!!!!!!!!! Path Error !!!!!!!!!!!!!!!!!!!") # Shortened error
        path_error = True
    if '/path/to/' in categorization_file:
         print("!!!!!!!!!!!!!!!!!!! Path Warning !!!!!!!!!!!!!!!!!!!") # Shortened warning
         # path_error = True # Decide if this is fatal

    if not path_error:
        # Call the updated function
        create_vertical_violin_plots_with_medians_nogrid(edge_file_path, categorization_file, output_path, # Call new function name
                                                         target_column=target_column_to_analyze,
                                                         categories=categories_to_use)
    else:
        print("\nScript halted due to path errors. Please correct the paths and rerun.")


