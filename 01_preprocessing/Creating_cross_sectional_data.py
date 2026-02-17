#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Sectional Data Creation for Preterm Connectome Study
===========================================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Prepare cross-sectional datasets from longitudinal imaging data by:
    1. Averaging multiple sessions per subject (for longitudinal analysis)
    2. Selecting baseline (earliest) sessions (for cross-sectional analysis)
    3. Organizing data into group subfolders

USAGE:
    - create_subject_averaged_dataset(): Average all sessions per subject
    - create_baseline_dataset(): Select only the earliest session per subject
    - copy_and_check_files(): Copy files to group-organized subfolders
"""

import os
import numpy as np
import re
import shutil


# ==============================================================================
# FUNCTION: Create Subject-Averaged Cross-Sectional Dataset
# ==============================================================================

def create_subject_averaged_dataset(source_folder, output_folder):
    """
    Creates a cross-sectional dataset by averaging multiple sessions at the subject level.
    
    This is useful when you want to use all available data per subject while
    reducing to one observation per subject for cross-sectional analyses.
    
    Parameters
    ----------
    source_folder : str
        Path to folder containing session-level connectome matrices.
        Files should follow pattern: 'filtered_sub-SUBJECTID_ses-SESSIONID_*.csv'
    output_folder : str
        Path to save averaged matrices.
        Output files follow pattern: 'averaged_sub-SUBJECTID.csv'
    
    Notes
    -----
    - Assumes no subfolders in source_folder
    - Only processes files matching the expected naming pattern
    - Averaging is element-wise across all sessions for each subject
    """
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing files directly in folder: {source_folder}")

    # 1. Group files by subject and calculate average matrix per subject
    subject_matrices = {}  # Dictionary to store matrices per subject
    
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            
            # Validate filename pattern
            if (len(parts) >= 3 and 
                parts[0] == 'filtered' and 
                parts[1].startswith('sub-') and 
                parts[2].startswith('ses-')):
                
                subject_id = parts[1]  # e.g., 'sub-0876001'
                file_path = os.path.join(source_folder, file_name)
                
                try:
                    matrix = np.loadtxt(file_path, delimiter=',', dtype=float)
                    if subject_id not in subject_matrices:
                        subject_matrices[subject_id] = []
                    subject_matrices[subject_id].append(matrix)
                except Exception as e:
                    print(f"  Error loading matrix file: {file_name}, error: {e}")
            else:
                print(f"  Warning: Filename '{file_name}' does not match expected pattern. Skipping.")

    # 2. Average matrices for each subject
    averaged_subject_matrices = {}
    for subject_id, matrix_list in subject_matrices.items():
        if matrix_list:
            averaged_matrix = np.mean(matrix_list, axis=0)
            averaged_subject_matrices[subject_id] = averaged_matrix
            print(f"  Subject {subject_id}: Averaged {len(matrix_list)} matrices.")
        else:
            print(f"  Warning: No valid matrices found for subject {subject_id}")

    # 3. Save the averaged subject matrices to the output folder
    for subject_id, averaged_matrix in averaged_subject_matrices.items():
        output_filename = f'averaged_{subject_id}.csv'
        output_file_path = os.path.join(output_folder, output_filename)
        np.savetxt(output_file_path, averaged_matrix, delimiter=',', fmt='%.6f')
        print(f"  Subject {subject_id}: Saved averaged matrix to {output_file_path}")

    print("\nSubject-level averaging and cross-sectional dataset creation completed")


# ==============================================================================
# FUNCTION: Create Baseline (Earliest Session) Dataset
# ==============================================================================

def create_baseline_dataset(source_folder, output_folder):
    """
    Creates a cross-sectional dataset by selecting the earliest timepoint for each subject.
    
    This is useful for analyses that require true baseline data (e.g., comparing
    groups at study entry) rather than averaged data.
    
    Parameters
    ----------
    source_folder : str
        Path to folder containing session-level connectome matrices.
        Files should follow pattern: 'filtered_sub-SUBJECTID_ses-SESSIONID_*.csv'
    output_folder : str
        Path to copy baseline matrices.
        Original filenames are preserved.
    
    Notes
    -----
    - Session numbers are extracted and compared numerically
    - Only the lowest session number per subject is retained
    - Files are copied (not moved) to preserve the original data
    """
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist")
        return
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing files directly in folder: {source_folder}")
    
    # 1. Group files by subject and find the earliest session for each
    subject_files = {}  # {subject_id: [(session_num, filename), ...]}
    
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            parts = file_name.split('_')
            
            if (len(parts) >= 3 and 
                parts[0] == 'filtered' and 
                parts[1].startswith('sub-') and 
                parts[2].startswith('ses-')):
                
                subject_id = parts[1]
                session_part = parts[2]  # e.g., 'ses-00' or 'ses-06'
                
                try:
                    session_num = int(session_part.split('-')[1])
                    
                    if subject_id not in subject_files:
                        subject_files[subject_id] = []
                    subject_files[subject_id].append((session_num, file_name))
                    
                except (IndexError, ValueError) as e:
                    print(f"  Warning: Could not extract session number from '{session_part}'. Skipping.")
            else:
                print(f"  Warning: Filename '{file_name}' does not match expected pattern. Skipping.")
    
    # 2. For each subject, find the file with the lowest session number
    baseline_files = {}  # {subject_id: filename}
    
    for subject_id, session_files in subject_files.items():
        if session_files:
            # Sort by session number and take the first (lowest)
            session_files.sort(key=lambda x: x[0])
            lowest_session_num, baseline_filename = session_files[0]
            baseline_files[subject_id] = baseline_filename
            print(f"  Subject {subject_id}: Selected session {lowest_session_num:02d} (file: {baseline_filename})")
        else:
            print(f"  Warning: No valid sessions found for subject {subject_id}")
    
    # 3. Copy the baseline files to the output folder
    for subject_id, filename in baseline_files.items():
        source_path = os.path.join(source_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            shutil.copy2(source_path, output_path)
            print(f"  Subject {subject_id}: Copied {filename} to output folder")
        except Exception as e:
            print(f"  Error copying file {filename}: {e}")
    
    print(f"\nBaseline dataset creation completed. {len(baseline_files)} files copied to {output_folder}")


# ==============================================================================
# FUNCTION: Copy and Organize Files by Group
# ==============================================================================

def copy_and_check_files(source_folder, destination_folder, cross_sectional_folder, subfolders):
    """
    Copies averaged files to group-organized subfolders based on subject membership.
    
    This function matches averaged files to group assignments from a reference folder
    and organizes them accordingly (e.g., Full_Term, Preterm, Very_Preterm).
    
    Parameters
    ----------
    source_folder : str
        Path to folder containing averaged matrices.
    destination_folder : str
        Path to create group subfolders and copy files to.
    cross_sectional_folder : str
        Reference folder containing group subfolders with subject assignments.
    subfolders : list
        List of group subfolder names (e.g., ['Full_Term', 'Preterm', 'Very_Preterm'])
    
    Notes
    -----
    - Uses regex for robust subject ID extraction
    - Reports any subjects that couldn't be matched
    """
    
    # Create destination subfolders if they don't exist
    for subfolder in subfolders:
        os.makedirs(os.path.join(destination_folder, subfolder), exist_ok=True)

    # Iterate over each subfolder in the cross_sectional_folder
    for subfolder in subfolders:
        cross_sectional_subfolder = os.path.join(cross_sectional_folder, subfolder)

        # Iterate over each file in the cross_sectional_subfolder (used for subject ID)
        for file_name in os.listdir(cross_sectional_subfolder):
            if file_name.endswith('.csv'):
                # Extract the subject ID using regex
                match_cs = re.search(r'_(sub-\d+)_', file_name)
                
                if match_cs:
                    subject_id_cs = match_cs.group(1)

                    # Find matching files in the source folder
                    matching_files = []
                    for source_file in os.listdir(source_folder):
                        match_source = re.search(r'averaged_(sub-\d+)', source_file)
                        if match_source:
                            subject_id_source = match_source.group(1)
                            if subject_id_source == subject_id_cs:
                                matching_files.append(source_file)

                    # Check for errors - No matching files found
                    if not matching_files:
                        print(f"Error: No matching file found for subject ID: {subject_id_cs}")
                        continue

                    # Copy each matching file to the corresponding destination subfolder
                    for matching_file in matching_files:
                        source_file_path = os.path.join(source_folder, matching_file)
                        destination_file_path = os.path.join(destination_folder, subfolder, matching_file)
                        shutil.copy2(source_file_path, destination_file_path)
                else:
                    print(f"Warning: Could not extract subject ID from filename: {file_name}. Skipping.")

        print(f'File copying for subfolder {subfolder} completed.')

    print('File copying for all subfolders completed.')


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Example paths - UPDATE THESE FOR YOUR SYSTEM
    source_folder = 'your_source_folder'
    output_folder = 'your_output_folder'
    
    # Create averaged dataset
    # create_subject_averaged_dataset(source_folder, output_folder)
    
    # OR create baseline dataset
    # create_baseline_dataset(source_folder, output_folder)
    
    print("Update the paths and uncomment the function you want to run.")
