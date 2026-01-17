

#########
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import numpy as np
import os

def analyze_family_composition(file_path, output_csv_name="family_composition_final.csv"):
    """
    Calculates counts of singletons, non-multiple siblings, and twins for each group.
    Performs two separate sets of pairwise statistical tests with Holm-Bonferroni correction:
    1. Compares the proportion of twins across groups.
    2. Compares the proportion of non-multiple siblings across groups.
    Summarizes all results in a new CSV file.

    Args:
        file_path (str): Path to the CSV file with 'ID', 'Group', 'Family', 'AgeAtScan'.
        output_csv_name (str): Name for the output CSV file.
    """
    try:
        df_all_scans = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

    required_cols = ['ID', 'Group', 'Family', 'AgeAtScan']
    if not all(col in df_all_scans.columns for col in required_cols):
        print(f"Error: Input CSV must contain the columns: {', '.join(required_cols)}")
        return None

    df_unique_individuals = df_all_scans.drop_duplicates(subset=['ID'], keep='first')[['ID', 'Group', 'Family']].copy()
    groups = sorted(df_unique_individuals['Group'].unique())

    if len(groups) < 2:
        print("Analysis requires at least two groups.")
        return None
        
    results_data = {} 

    for group_name in groups:
        current_group_unique_ids_df = df_unique_individuals[df_unique_individuals['Group'] == group_name]
        all_unique_ids_in_this_group = set(current_group_unique_ids_df['ID'])
        total_individuals_in_group = len(all_unique_ids_in_this_group)

        if total_individuals_in_group == 0:
            results_data[group_name] = {
                "Number of Singletons": 0,
                "Number of Non-Multiple Siblings": 0,
                "Number of Twins": 0,
            }
            continue

        df_scans_for_current_group_ids = df_all_scans[df_all_scans['ID'].isin(all_unique_ids_in_this_group)]

        # --- Identify Twins  ---
        twin_pairs = []
        grouped_by_family_age = df_scans_for_current_group_ids.groupby(['Family', 'AgeAtScan'])
        for _, subgroup in grouped_by_family_age:
            unique_ids_in_subgroup = set(subgroup['ID'].unique())
            if len(unique_ids_in_subgroup) == 2:
                twin_pairs.append(unique_ids_in_subgroup)
        
        final_twin_ids = {id_ for twin_pair in twin_pairs for id_ in twin_pair}
        num_twins = len(final_twin_ids)
        
        # --- Identify Singletons and Non-Multiple Siblings ---
        family_counts = current_group_unique_ids_df['Family'].value_counts()
        multi_member_families = family_counts[family_counts > 1].index.tolist()
        ids_from_multi_member_families = set(
            current_group_unique_ids_df[current_group_unique_ids_df['Family'].isin(multi_member_families)]['ID']
        )
        
        num_non_multiple_siblings = len({
            id_ for id_ in ids_from_multi_member_families if id_ not in final_twin_ids
        })
        num_singletons = total_individuals_in_group - len(ids_from_multi_member_families)

        results_data[group_name] = {
            "Number of Singletons": num_singletons,
            "Number of Non-Multiple Siblings": num_non_multiple_siblings,
            "Number of Twins": num_twins,
        }

    summary_df = pd.DataFrame.from_dict(results_data, orient='index')
    
    # --- Helper Function for Statistics ---
    def perform_pairwise_tests(summary_df, category_name):
        raw_p_values = {}
        for g1, g2 in combinations(groups, 2):
            label = f"{category_name} ({g1} vs {g2})"
            
            n_category_g1 = summary_df.loc[g1, category_name]
            n_total_g1 = summary_df.loc[g1].sum()
            
            n_category_g2 = summary_df.loc[g2, category_name]
            n_total_g2 = summary_df.loc[g2].sum()
            
            table = [[n_category_g1, n_total_g1 - n_category_g1], [n_category_g2, n_total_g2 - n_category_g2]]
            try:
                if np.any(np.array(table) < 0): raise ValueError("Negative values in table")
                raw_p_values[label] = chi2_contingency(table)[1]
            except ValueError:
                raw_p_values[label] = np.nan
        
        valid_labels = [lbl for lbl, p in raw_p_values.items() if not pd.isna(p)]
        p_to_correct = [raw_p_values[lbl] for lbl in valid_labels]
        
        corrected_p_dict = {}
        if p_to_correct:
            _, pvals_corrected, _, _ = multipletests(p_to_correct, alpha=0.05, method='holm')
            for i, label in enumerate(valid_labels):
                corrected_p_dict[label] = pvals_corrected[i]
        return corrected_p_dict

    # --- Run Both Sets of Tests ---
    corrected_p_twin = perform_pairwise_tests(summary_df, "Number of Twins")
    corrected_p_sibling = perform_pairwise_tests(summary_df, "Number of Non-Multiple Siblings")
    
    # --- Format Final Table ---
    output_table_df = summary_df.T.reindex(columns=groups).copy()
    
    def add_pvalue_rows(table_df, corrected_p_dict):
        p_value_row_template = {group: "" for group in table_df.columns}
        for label, p_val in corrected_p_dict.items():
            p_str = f"{p_val:.4g}" if p_val is not None else "N/A"
            p_value_row = p_value_row_template.copy()
            p_value_row[table_df.columns[0]] = p_str
            table_label = f"Corrected p-value ({label})"
            table_df.loc[table_label] = pd.Series(p_value_row)

    add_pvalue_rows(output_table_df, corrected_p_twin)
    add_pvalue_rows(output_table_df, corrected_p_sibling)
    
    # --- Save to CSV ---
    output_dir = os.path.dirname(file_path) if file_path and os.path.dirname(file_path) else '.'
    full_output_path = os.path.join(output_dir, output_csv_name)
    try:
        output_table_df.to_csv(full_output_path, index=True)
        print(f"✅ Family composition analysis saved to: {full_output_path}")
    except Exception as e:
        print(f"❌ An error occurred while saving the output CSV: {e}")

    return output_table_df


import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def analyze_mri_data_with_std(file_path):
    """
    Loads MRI scan data and calculates group-wise statistics including averages and standard deviations.
    Recodes Handedness into Left/Right/Ambidextrous and tests the distribution.
    Performs pairwise non-parametric tests with per-variable Holm-Bonferroni correction.

    Args:
        file_path (str): The path to the CSV file containing the MRI data.

    Returns:
        pandas.DataFrame: DataFrame with detailed statistics and corrected p-values.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return pd.DataFrame()

    # --- 1. Pre-process Data ---
    # Recode Handedness into categories
    def recode_handedness(h):
        if h in [-2, -1]:
            return "Left"
        elif h == 0:
            return "Ambidextrous"
        elif h in [1, 2]:
            return "Right"
        else:
            return np.nan # Or handle other values as needed

    df['Handedness_Category'] = df['Handedness'].apply(recode_handedness)
    
    unique_subjects = df.drop_duplicates(subset=['ID']).copy()
    individual_averages = df.groupby('ID').agg({
        'AgeAtScan': 'mean',
        'WeeksPreterm': 'mean',
        'Group': 'first'
    }).reset_index()

    groups = sorted(df['Group'].unique())
    all_metrics_data = {}
    group_raw_data = {group_name: {} for group_name in groups}
    
    all_genders = sorted(unique_subjects['Gender'].dropna().unique())
    all_handedness_categories = ["Left", "Right", "Ambidextrous"]

    # --- 2. Calculate Group-wise Statistics (including SD) ---
    for group_name in groups:
        # Data subsets for the current group
        current_individual_averages = individual_averages[individual_averages['Group'] == group_name]
        current_group_df = df[df['Group'] == group_name]
        current_unique_subjects = unique_subjects[unique_subjects['Group'] == group_name]

        # Prepare raw data distributions for statistical tests
        group_raw_data[group_name]['AgeAtScan'] = current_individual_averages['AgeAtScan'].dropna().tolist()
        group_raw_data[group_name]['WeeksPreterm'] = current_individual_averages['WeeksPreterm'].dropna().tolist()
        group_raw_data[group_name]['Total_Bad_Volumes'] = current_group_df['Total_Bad_Volumes'].dropna().tolist()
        
        # Prepare categorical counts for tests
        group_raw_data[group_name]['Gender_Counts'] = current_unique_subjects['Gender'].value_counts().reindex(all_genders, fill_value=0)
        group_raw_data[group_name]['Handedness_Counts'] = current_unique_subjects['Handedness_Category'].value_counts().reindex(all_handedness_categories, fill_value=0)

        # Store summary statistics (Avg and SD) for the final table
        all_metrics_data[group_name] = {
            'AgeAtScan_Avg': np.mean(group_raw_data[group_name]['AgeAtScan']),
            'AgeAtScan_SD': np.std(group_raw_data[group_name]['AgeAtScan']),
            'WeeksPreterm_Avg': np.mean(group_raw_data[group_name]['WeeksPreterm']),
            'WeeksPreterm_SD': np.std(group_raw_data[group_name]['WeeksPreterm']),
            'Total_Bad_Volumes_Avg': np.mean(group_raw_data[group_name]['Total_Bad_Volumes']),
            'Total_Bad_Volumes_SD': np.std(group_raw_data[group_name]['Total_Bad_Volumes']),
            'Gender_Counts': group_raw_data[group_name]['Gender_Counts'].to_dict(),
            'Handedness_Counts': group_raw_data[group_name]['Handedness_Counts'].to_dict(),
        }

    # --- 3. Perform Pairwise Statistical Tests ---
    numerical_vars = ['AgeAtScan', 'WeeksPreterm', 'Total_Bad_Volumes']
    categorical_vars = {'Gender_Counts': all_genders, 'Handedness_Counts': all_handedness_categories}
    
    variable_to_labels_map = {key: [] for key in numerical_vars + list(categorical_vars.keys())}
    raw_p_values_dict = {}
    test_labels = []
    group_pairs = list(combinations(groups, 2))

    for var_name in numerical_vars:
        for g1, g2 in group_pairs:
            label = f'{var_name}_Avg ({g1} vs {g2})' # Label reflects we tested the distribution of the avg metric
            test_labels.append(label)
            variable_to_labels_map[var_name].append(label)
            data1 = group_raw_data[g1][var_name]
            data2 = group_raw_data[g2][var_name]
            if len(data1) > 0 and len(data2) > 0:
                try:
                    raw_p_values_dict[label] = mannwhitneyu(data1, data2, alternative='two-sided')[1]
                except ValueError: raw_p_values_dict[label] = np.nan
            else:
                raw_p_values_dict[label] = np.nan

    for var_name, categories in categorical_vars.items():
        for g1, g2 in group_pairs:
            label = f'{var_name} ({g1} vs {g2})'
            test_labels.append(label)
            variable_to_labels_map[var_name].append(label)
            counts1 = group_raw_data[g1][var_name]
            counts2 = group_raw_data[g2][var_name]
            table = pd.DataFrame([counts1, counts2]).reindex(categories, axis=1, fill_value=0).values
            if table.sum() > 0:
                try:
                    raw_p_values_dict[label] = chi2_contingency(table)[1]
                except ValueError: raw_p_values_dict[label] = np.nan
            else:
                raw_p_values_dict[label] = np.nan

    # --- 4. Apply Holm-Bonferroni Correction (Per Variable) ---
    corrected_p_values_map = {}
    all_vars_for_correction = numerical_vars + list(categorical_vars.keys())
    
    for var_name in all_vars_for_correction:
        labels_for_this_var = variable_to_labels_map[var_name]
        valid_labels = [lbl for lbl in labels_for_this_var if not pd.isna(raw_p_values_dict.get(lbl))]
        p_to_correct = [raw_p_values_dict[lbl] for lbl in valid_labels]
        
        if p_to_correct:
            _, pvals_corrected, _, _ = multipletests(p_to_correct, alpha=0.05, method='holm')
            for i, label in enumerate(valid_labels):
                corrected_p_values_map[label] = pvals_corrected[i]

    # --- 5. Format and Save Output Table ---
    output_df = pd.DataFrame(all_metrics_data).T.sort_index()

    # Restructure table to have Avg and SD as sub-columns
    final_table = pd.DataFrame()
    for group_name in groups:
        for var in numerical_vars:
            final_table.loc[f'{var}: Average', group_name] = output_df.loc[group_name, f'{var}_Avg']
            final_table.loc[f'{var}: SD', group_name] = output_df.loc[group_name, f'{var}_SD']
        for var, cats in categorical_vars.items():
            for cat in cats:
                final_table.loc[f'{var}: {cat}', group_name] = output_df.loc[group_name, var][cat]
    
    # Add p-values
    p_value_row_template = {group: "" for group in final_table.columns}
    for label in test_labels:
        p_val = corrected_p_values_map.get(label)
        p_str = f"{p_val:.4g}" if p_val is not None else "N/A"
        p_value_row = p_value_row_template.copy()
        p_value_row[final_table.columns[0]] = p_str
        final_table.loc[f"Corrected p-value ({label})"] = pd.Series(p_value_row)
        
    output_directory = os.path.dirname(file_path) if file_path and os.path.dirname(file_path) else '.'
    output_file_name = 'mri_group_statistics_detailed_std.csv'
    output_file_path = os.path.join(output_directory, output_file_name)

    try:
        final_table.to_csv(output_file_path, index=True)
        print(f"✅ Detailed analysis saved to: {output_file_path}")
    except Exception as e:
        print(f"❌ An error occurred while saving the output CSV: {e}")

    return final_table
