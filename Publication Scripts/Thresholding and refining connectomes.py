#Connectome Post-Processing


import os
import glob
import numpy as np
import networkx as nx
import warnings

def load_and_filter_matrices(directory_path, output_dir, identifiers):
    """Indentifies nodes in the the streamline count connectivity graphs that are not connected, and removes the same nodes from all graphs. In the final analyses, 
5 disconnected nodes were removed from all brain graphs"""
    # Only load specific .csv file types
    csv_files = glob.glob(os.path.join(directory_path, '**/*tckcount_SIFT2.csv'), recursive=True)
    matrix_lists = {identifier: [] for identifier in identifiers}
    disconnected_nodes_dict = {identifier: set() for identifier in identifiers}  # Change to a dictionary of sets
    
    for file_path in csv_files:
        for identifier in identifiers:
            if identifier in os.path.dirname(file_path):
                # Specify the correct delimiter
                matrix = np.loadtxt(file_path, dtype=float, delimiter=' ')
                G = nx.from_numpy_array(matrix)
                if not nx.is_connected(G):
                    warnings.warn(f"The graph in {file_path} from directory {identifier} was not connected before thresholding.")
                    largest_cc = max(nx.connected_components(G), key=len)
                    not_connected_nodes = [node for node in G.nodes if node not in largest_cc]
                    graph_name = os.path.basename(file_path)
                    disconnected_nodes_dict[identifier].update(not_connected_nodes)  # Add the nodes to the set
                    
                    # Save the disconnected nodes to a separate file only if there are any
                    if not_connected_nodes:
                        # Increase node number by 1
                        not_connected_nodes = [node+1 for node in not_connected_nodes]
                        # Include the subject name in the name of the disconnected nodes text file
                        subject_name = os.path.dirname(file_path).split('/')[-2]
                        disconnected_nodes_file = os.path.join(output_dir, f'disconnected_nodes_{identifier}_{subject_name}_{graph_name}.txt')
                        np.savetxt(disconnected_nodes_file, not_connected_nodes, fmt='%d')
                    
                matrix_lists[identifier].append((file_path, matrix))
    
    for identifier, matrices in matrix_lists.items():
        if matrices:
            # Create a new directory for each identifier
            identifier_dir = os.path.join(output_dir, identifier)
            os.makedirs(identifier_dir, exist_ok=True)
            for file_path, matrix in matrices:
                # Remove the corresponding rows and columns from the matrix
                matrix = np.delete(matrix, list(disconnected_nodes_dict[identifier]), axis=0)
                matrix = np.delete(matrix, list(disconnected_nodes_dict[identifier]), axis=1)
                # Save the filtered matrices in the newly created directories
                subject_name = os.path.dirname(file_path).split('/')[-2]
                atlas_name = os.path.basename(file_path).split('_')[0]
                study_name = file_path.split('/')[-4]
                study_dir = os.path.join(identifier_dir, study_name)
                os.makedirs(study_dir, exist_ok=True)
                output_file_path = os.path.join(study_dir, f'filtered_{subject_name}_{atlas_name}.csv')
                np.savetxt(output_file_path, matrix, delimiter=',')



   

#############
#thresholding

def threshold_by_streamline_length(lengths_file, output_file=None):
    """
    Sets values to 0 when they are >0 but <5 in a matrix file.
    
    Args:
        lengths_file (str): Path to input CSV file
        output_file (str): Path to save modified file (optional)
    
    Returns:
        pd.DataFrame: Modified DataFrame
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(lengths_file, header=None)
        
        # Initialize counter for modified values
        count_modified = 0
        matrix_dimension = 227
        
        # Iterate through the matrix and set values >0 and <5 to 0
        for i in range(min(matrix_dimension, df.shape[0])):
            for j in range(min(matrix_dimension, df.shape[1])):
                value = df.iloc[i, j]
                if 0 < value < 5:  # Condition: Value is greater than 0 and less than 5
                    df.iloc[i, j] = 0
                    count_modified += 1
        
        print(f"Modified {count_modified} values (set to 0) that were >0 and <5 in '{lengths_file}'")
        
        # Save to output file if specified
        if output_file:
            df.to_csv(output_file, index=False, header=False)
            print(f"Modified matrix saved to '{output_file}'")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at '{lengths_file}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
##############

"""This script applies absolute thresholds to edge weights in adjacency matrices stored in CSV files.
It filters edges based on user-defined thresholds and saves the modified matrices to new CSV files."""
def threshold_by_edge_weight(directory_path, output_path, threshold_streamlines):
    # Convert the threshold percentages string to a list of integers
    thresholds = [int(streamline) for streamline in threshold_streamlines.split()]
    
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '*/*.csv'), recursive=True)
    
    # Process each threshold percentage
    for threshold in thresholds:
   
        # Create output directory for the threshold percentage
        threshold_dir = os.path.join(output_path, f"absolute_threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        # Process each file again for thresholding
        for file in csv_files:
            # Load the adjacency matrix and create a graph
            matrix = np.loadtxt(file, dtype=float, delimiter=',')
            G = nx.from_numpy_array(matrix)
            
            # Create a new matrix with original weights, applying the threshold
            thresholded_matrix = np.copy(matrix)
            for u, v, data in G.edges(data=True):
                if data['weight'] < threshold:
                    thresholded_matrix[u, v] = 0
                    thresholded_matrix[v, u] = 0  # For undirected graphs
            
            # Save the thresholded graph
            filename = os.path.splitext(os.path.basename(file))[0]  # Get the filename without extension
            new_filename = f"{filename}_absolute_{threshold}.csv"  # Append the threshold level to the filename
            np.savetxt(os.path.join(threshold_dir, new_filename), thresholded_matrix, delimiter=',')
        
        # Print a progress message
        print(f"Threshold {threshold} streamlines completed for all files")




##########

def threshold_by_consensus(directory_path, output_path, threshold_percentage):
    """Thresholds edges in adjacency matrices based on their occurrence across multiple graphs (by consensus)"""
    # Initialize a dictionary to store the count of each edge across all graphs
    edge_counts = {}

    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'), recursive=True)
    total_files = len(csv_files)

    # Iterate through each CSV file to count edges and store original edge values
    for file in csv_files:
        # Load the adjacency matrix and create a graph
        matrix = np.loadtxt(file, dtype=float, delimiter=',')

        G = nx.from_numpy_array(matrix)

        # Update edge counts
        for u, v, data in G.edges(data=True):
            edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

    # Create output directory for the threshold percentage - AUTOMATICALLY NAMED
    base_directory_name = os.path.basename(directory_path) # Get the base name of the input directory
    threshold_dir_name = f"{base_directory_name}_consensus_{threshold_percentage}" # Construct the threshold directory name
    threshold_dir = os.path.join(output_path, threshold_dir_name) # Create the full path in output_path
    os.makedirs(threshold_dir, exist_ok=True)

    # Threshold the graph based on the percentage
    threshold_count = int(total_files * threshold_percentage / 100)

    for file in csv_files:
        # Load the adjacency matrix and create a graph
        matrix = np.loadtxt(file, dtype=float, delimiter=',')
        G = nx.from_numpy_array(matrix)

        # Zero out edges that do not meet the threshold
        for u, v, data in G.edges(data=True):
            if edge_counts.get((u, v), 0) < threshold_count:
                data['weight'] = 0

        # Create a new adjacency matrix from the modified graph
        thresholded_matrix = nx.to_numpy_array(G)

        # Save the thresholded graph
        filename = os.path.splitext(os.path.basename(file))[0]  # Get the filename without extension
        new_filename = f"{filename}_consensus_{threshold_percentage}.csv"  # Append the threshold level to the filename
        np.savetxt(os.path.join(threshold_dir, new_filename), thresholded_matrix, delimiter=',')

    # Print a progress message
    print(f"Threshold {threshold_percentage}% completed for all files. Output saved to: {threshold_dir}")



############

def match_and_filter_matrices(source_folder, threshold_folder, output_folder):
    """
    Matches averaged distance and weight matrices. Filters out streamlines with too low weights from distance matrices based on threshold matrices.
    Assumes no subfolders, files are directly in source_folder and threshold_folder.
    Assumes filenames are matched by subject ID.
    """
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return

    # Check if the threshold folder exists
    if not os.path.exists(threshold_folder):
        print(f"Error: Threshold folder '{threshold_folder}' does not exist.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing files in source folder: {source_folder}")

    # Iterate through files in the source folder (averaged distance matrices)
    for source_file_name in os.listdir(source_folder):
        if source_file_name.endswith('.csv') and source_file_name.startswith('averaged_sub-'): # Target averaged distance matrices
            # Extract subject ID from the averaged distance matrix filename
            parts_source = source_file_name.split('_')
            if len(parts_source) >= 2 and parts_source[0] == 'averaged' and parts_source[1].startswith('sub-'):
                subject_id = parts_source[1] # e.g., 'sub-0876001'

                # Construct the expected threshold filename (averaged weight matrix)
                threshold_file_name = f'averaged_{subject_id}_consensus_60.csv' # Correct filename format
                threshold_file_path = os.path.join(threshold_folder, threshold_file_name)

                # Check if the threshold file exists
                if not os.path.exists(threshold_file_path):
                    print(f"Warning: No matching threshold file found for subject: {subject_id} (Expected: {threshold_file_name} in {threshold_folder}). Skipping {source_file_name}")
                    continue # Skip to the next source file if no threshold file is found

                source_file_path = os.path.join(source_folder, source_file_name)

                # Load the source and threshold data using pandas
                try:
                    source_data = pd.read_csv(source_file_path, header=None)
                    threshold_data = pd.read_csv(threshold_file_path, header=None)
                except Exception as e:
                    print(f"Error loading data files for subject {subject_id}: {e}. Skipping {source_file_name}")
                    continue

                # Ensure both matrices are 227 x 227
                if source_data.shape != (227, 227):
                    print(f"Skipping {source_file_name}: Source data shape is {source_data.shape}, expected (227, 227)")
                    continue
                if threshold_data.shape != (227, 227):
                    print(f"Skipping {threshold_file_name}: Threshold data shape is {threshold_data.shape}, expected (227, 227)")
                    continue

                # Set values to 0 in the source data where threshold data is 0
                filtered_data = source_data.where(threshold_data != 0, 0)

                # Save the filtered data to the output folder as CSV
                output_file_name = source_file_name # Keep the same filename as source for clarity
                output_file_path = os.path.join(output_folder, output_file_name)
                filtered_data.to_csv(output_file_path, index=False, header=False)
                print(f"  Filtered and saved: {source_file_name} to {output_file_path}")

    print("\nFiltering of averaged distance matrices completed.")

########

def process_matrices_against_reference(reference_file, input_folder, output_folder):
    """
    Processes matrices in a folder based on a reference matrix.
    Filters values in input matrices to 0 where the reference matrix value is 0.
    Prints the number of values changed to 0 for each input matrix.
    """
    try:
        # Load the reference matrix
        reference_matrix = np.loadtxt(reference_file, delimiter=',', dtype=np.float64)
        print(f"Reference matrix loaded. Shape: {reference_matrix.shape}")

        # Find indices where the reference matrix value is equal to 0
        indices_to_filter = np.where(reference_matrix == 0)

        num_filter_indices = len(indices_to_filter[0])
        print(f"Total indices in reference matrix with values = 0: {num_filter_indices}")

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith(".csv"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Load the input matrix
                input_matrix = np.loadtxt(input_path, delimiter=',', dtype=np.float64)
                #print(f"\nProcessing File: {filename}. Input matrix shape: {input_matrix.shape}")

                values_changed_count = 0  # Initialize counter for each input matrix

                # Iterate through the indices that should be filtered and count changes
                for row_idx, col_idx in zip(indices_to_filter[0], indices_to_filter[1]):
                    if row_idx < input_matrix.shape[0] and col_idx < input_matrix.shape[1]:  # Check index bounds
                        original_value = input_matrix[row_idx, col_idx]
                        if not np.isclose(original_value, 0): # Check if original value is NOT already zero (using np.isclose for float comparison)
                            input_matrix[row_idx, col_idx] = 0
                            values_changed_count += 1

                print(f"  File: {filename} - Values changed to 0: {values_changed_count}")


                # Save the modified matrix
                np.savetxt(output_path, input_matrix, delimiter=',', fmt='%.6f')
                #print(f"  Filtered and saved: {filename}") # Removed redundant print here, count is more informative

    except FileNotFoundError:
        print(f"Error: File not found: {reference_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    reference_file = 'your_file.csv'
    input_folder = 'your_folder'
    output_folder = 'your_output_folder'
    process_matrices_against_reference(reference_file, input_folder, output_folder)
    
    