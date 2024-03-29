#!/usr/bin/env python
# coding: utf-8

# In[18]:


""" Indentifies nodes in the the connectivity matrixes from MRI that are not connected,
and removes those nodes from the matrixes"""

import os
import glob
import numpy as np
import networkx as nx
import warnings

def load_and_filter_matrices(directory_path, output_dir, identifiers):
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
   


# In[20]:


"""Threshold all filtered connectivity matrices to the highest minimal edge density across all the graphs"""

import os
import glob
import numpy as np
import networkx as nx
import warnings

def threshold_network(directory_path, output_dir):
    csv_files = glob.glob(os.path.join(directory_path, '**', '*filtered*connectome*'), recursive=True)
    min_edge_density = 0
    individual_min_edge_densities = []  # List to store individual minimum edge densities

    # First pass: Determine the minimal edge density to remain connected across all graphs
    for file_path in csv_files:
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        G = nx.from_numpy_array(matrix)
        
        atlas_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(file_path)))
        os.makedirs(atlas_output_dir, exist_ok=True)
        
        if not nx.is_connected(G):
            warnings.warn(f"The graph in {file_path} was not connected before thresholding.")
            largest_cc = max(nx.connected_components(G), key=len)
            not_connected_nodes = [node for node in G.nodes if node not in largest_cc]
            
            unconnected_file_path = os.path.join(atlas_output_dir, os.path.basename(file_path).replace('.csv', '_INITIAL_UNconnected_nodes.txt'))
            np.savetxt(unconnected_file_path, not_connected_nodes, fmt='%d')
            
        threshold = 0
        while nx.is_connected(G):
            threshold += 1
            matrix[matrix < threshold] = 0
            G = nx.from_numpy_array(matrix)

            if not nx.is_connected(G):
                break

        threshold -= 1
        edge_density = nx.density(G)
        min_edge_density = max(min_edge_density, edge_density) #this needs to be max to avoid disconnection

        # Save individual minimum edge density and corresponding graph name
        individual_min_edge_densities.append([os.path.basename(file_path), edge_density])

    # Save the final min_edge_density to a separate CSV file
    directory_name = os.path.basename(directory_path)
    np.savetxt(os.path.join(output_dir, f"{directory_name}_min_edge_density.csv"), [min_edge_density], delimiter=',')

    # Save individual minimum edge densities to a separate CSV file
    np.savetxt(os.path.join(output_dir, f"{directory_name}_individual_min_edge_densities.csv"), individual_min_edge_densities, delimiter=',', fmt='%s')


    # Second pass: Threshold all the graphs to the highest minimal edge density across all the graphs
    for file_path in csv_files:
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        G = nx.from_numpy_array(matrix)
        
        atlas_name = os.path.basename(os.path.dirname(file_path))
        atlas_output_dir = os.path.join(output_dir, atlas_name)
        os.makedirs(atlas_output_dir, exist_ok=True)

        
        threshold = 0
        while nx.is_connected(G) and nx.density(G) > (1.05 * min_edge_density): # this needs to be larger than 1 to avoid disconnection
            threshold += 1
            matrix[matrix < threshold] = 0
            temp_G = nx.from_numpy_array(matrix)

            if not nx.is_connected(temp_G):
                print(f"Error: The graph in {file_path} became disconnected when thresholded to edge density {nx.density(G)}.")
                break

            G = temp_G
    

         # Save the thresholded matrix to a new CSV file in the output directory
        SCM_filename = os.path.splitext(os.path.basename(file_path))[0] + "_threshold_SCM.csv"
        np.savetxt(os.path.join(atlas_output_dir, SCM_filename), matrix, delimiter=',')

    


# In[24]:


"""Create an averaged connectivity matrix across subjects"""
def average_matrices(path, averaged_data_path):
    files = glob.glob(os.path.join(path, '**/*SCM.csv'), recursive=True)
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            matrix_list = []
            for file_path in files:
                if subfolder in os.path.dirname(file_path):
                    matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
                    # Add this line after reading the matrix from the file
                    if matrix.shape == (445, 445):
                        print(f"The matrix in {file_path} has the dimensions 445 x 445")
                    matrix_list.append(matrix)
            if matrix_list:
                averaged_matrix = np.mean(matrix_list, axis=0)
                averaged_data_subfolder_path = os.path.join(averaged_data_path, subfolder)
                os.makedirs(averaged_data_subfolder_path, exist_ok=True)
                np.savetxt(os.path.join(averaged_data_subfolder_path, f'averaged_Schaefer_{subfolder}_SCM.csv'), 
                           averaged_matrix,
                           delimiter=',')


# In[26]:


"""saves edge density of all averaged SCM matrices to a file"""

def check_edge_density(directory_path):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*.csv'), recursive=True)

    # Define the output file path
    output_file = os.path.join(directory_path, 'Averaged_edge_densities.txt')

    with open(output_file, 'w') as f:
        for file_path in csv_files:
            # Load the matrix
            matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
            
            # Create a graph from the matrix
            G = nx.from_numpy_array(matrix)
            
            # Calculate the edge density
            edge_density = nx.density(G)
            
            # Write the file path and edge density to the output file
            f.write(f"{file_path}: {edge_density}\n")



# In[19]:


""" Average the individual thresholded matrices"""
import numpy as np

def average_matrices(path, averaged_data_path, *identifiers): #specify the number of node in each parcellation 
    files = glob.glob(os.path.join(path, '**/*SCM.csv'), recursive=True)
    
    matrix_lists = {identifier: [] for identifier in identifiers}
    averaged_data_paths = {identifier: os.path.join(averaged_data_path, identifier) for identifier in identifiers}
    
    for file_path in files:
        for identifier in identifiers:
            if identifier in os.path.dirname(file_path):
                matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
                matrix_lists[identifier].append(matrix)

    for identifier in identifiers:
        os.makedirs(averaged_data_paths[identifier], exist_ok=True)
        if matrix_lists[identifier]:
            averaged_matrix = sum(matrix_lists[identifier]) / len(matrix_lists[identifier])
            np.savetxt(os.path.join(averaged_data_paths[identifier], f'averaged_Schaefer_{identifier}_SCM.csv'), 
                       averaged_matrix,
                       delimiter=',')


# In[ ]:


"""Plot the unweighted degree distribution"""
import os
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def calculate_nodal_degrees_and_plot(directory_path, output_dir):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*SCM.csv'), recursive=True)

    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        
        # Binarize the matrix
        matrix[matrix > 0] = 1

        # Create a graph from the matrix
        G = nx.from_numpy_array(matrix)

        # Calculate the degrees of each node
        degrees = [deg for node, deg in G.degree()]

        # Calculate the mean and standard deviation
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)

        # Separate degrees into three lists: one for degrees less than one standard deviation above the mean, one for degrees one standard deviation or more above the mean but less than two standard deviations, and one for degrees two standard deviations or more above the mean
        degrees_less = [deg for deg in degrees if deg < mean_deg + std_deg]
        degrees_more = [deg for deg in degrees if mean_deg + std_deg <= deg < mean_deg + 2*std_deg]
        degrees_most = [deg for deg in degrees if deg >= mean_deg + 2*std_deg]

        # Get the directory name (up two levels) from the file path
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

        # Get the 'Schaefer_Tian_450_atlas' or other atlas name from the file path
        atlas_name = os.path.basename(os.path.dirname(file_path))

        # Create a subdirectory for each atlas
        atlas_output_dir = os.path.join(output_dir, atlas_name)
        os.makedirs(atlas_output_dir, exist_ok=True)

        # Save the degrees to a new CSV file in the output directory
        degrees_filename = os.path.splitext(os.path.basename(file_path))[0] + "_degrees.csv"
        np.savetxt(os.path.join(atlas_output_dir, dir_name + "_" + degrees_filename), degrees, delimiter=',')

        # Save the degrees one standard deviation or more above the mean to a new CSV file in the output directory
        degrees_more_filename = os.path.splitext(os.path.basename(file_path))[0] + "_degrees_more.csv"
   

        df = pd.DataFrame({
            'Node': [i for i, deg in enumerate(degrees) if deg >= mean_deg + std_deg],
            'Degree': [deg for deg in degrees if deg >= mean_deg + std_deg],
            'Standard Deviations Above Mean': [(deg - mean_deg) / std_deg for deg in degrees if deg >= mean_deg + std_deg]
        })
        df.to_csv(os.path.join(atlas_output_dir, dir_name + "_" + degrees_more_filename), index=False)

        # Define bins
        bins = np.linspace(min(degrees), max(degrees), len(G.nodes()) // 12 + 1)

        # Plot histogram
        plt.hist(degrees_less, bins=bins, align='mid', rwidth=0.8, color='blue')
        plt.hist(degrees_more, bins=bins, align='mid', rwidth=0.8, color='red')
        plt.hist(degrees_most, bins=bins, align='mid', rwidth=0.8, color='green')
        plt.title('Histogram of Unweighted Nodal Degrees')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')

        # Save the plot to a new PNG file in the output directory
        plot_filename = os.path.splitext(os.path.basename(file_path))[0] + "_histogram.png"
        plt.savefig(os.path.join(atlas_output_dir, dir_name + "_" + plot_filename))
        plt.close()


# In[ ]:


"""plot the unweighted rich club curve individually"""

import matplotlib.pyplot as plt
import warnings
from scipy import stats
import pickle  # Import the pickle module

def rich_club_curve(directory_path, output_dir):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*SCM.csv'), recursive=True)
    
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')

        # Binarize the matrix
        matrix[matrix > 0] = 1

        # Create an unweighted, undirected graph
        G = nx.from_numpy_array(matrix)

        # Calculate the rich club coefficient for the original graph
        rc = nx.rich_club_coefficient(G, normalized=False)

        # Generate a randomized version of the original graph
        rc_rand_avg = {}
        rc_rand_std = {}  # For storing the standard deviation
        num_iterations = 100   ### change this to speed up 
        for _ in range(num_iterations):
            G_rand = G.copy()  # Start with a copy of the original graph

            # Initialize a counter for the number of successful edge swaps
            num_swaps = 0

            # Try a double edge swap, but keep trying until we get a connected graph
            while num_swaps < int(.5*len(G.nodes)):  # Perform X edge swaps
                try:
                    # Perform a double edge swap
                    G_rand = nx.double_edge_swap(G_rand, nswap=int(.5*len(G.nodes)), max_tries=10000)
                    num_swaps += 1  # Increment the counter
                except nx.NetworkXError:
                    continue  # If the swap is not successful, try again

                # Remove self-loops from the randomized graph
                G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

                # Check if the degree sequence is preserved and the graph is connected
                if nx.is_connected(G_rand) and sorted(d for n, d in G.degree()) == sorted(d for n, d in G_rand.degree()):
                    continue  # Continue the loop if G_rand is connected and the degree sequence is preserved
                else:
                    # If the graph is not connected or the degree sequence is not preserved, reset the counter and the graph
                    num_swaps = 0
                    G_rand = G.copy()

            print(f"Iteration {_+1}: Number of successful edge swaps: {num_swaps}")

            # Calculate the rich club coefficient for the randomized graph
            rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)

            # Add the coefficients to the average
            for k, v in rc_rand.items():
                if k in rc_rand_avg:
                    rc_rand_avg[k] += v
                    rc_rand_std[k].append(v)  # Append the value to the list for this degree
                else:
                    rc_rand_avg[k] = v
                    rc_rand_std[k] = [v]  # Start a new list for this degree

        # Divide by the number of iterations to get the average
        for k in rc_rand_avg.keys():
            rc_rand_avg[k] /= num_iterations

        # Calculate the standard deviation for every 5th degree
        rc_rand_std = {k: np.std(v) for k, v in rc_rand_std.items() if k % 2 == 0}
        # Save the rich club calculations to a file
        with open(os.path.join(output_dir, 'rich_club_calculations.pkl'), 'wb') as f:
            pickle.dump((rc, rc_rand_avg, rc_rand_std), f)

        # Plot the rich club curve
        plt.figure()
        plt.plot(list(rc.keys()), list(rc.values()), 'r-', label='Original')
        plt.errorbar(list(rc_rand_std.keys()), [rc_rand_avg[k] for k in rc_rand_std.keys()], yerr=list(rc_rand_std.values()), fmt='k:', label='Random')
        plt.xlabel('Degree')
        plt.ylabel('Rich Club Coefficient')
        plt.title('Rich Club Curve')
        plt.legend(loc='upper left')
        
              # Save the plot as a PNG file in the output directory
        output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_rich_club_curve.png'))
        plt.savefig(output_file_path)
        plt.close()



# In[ ]:


"""Compare rich club curve of different groups on the same graph"""

def rich_club_curve(directory_path, output_dir):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*SCM.csv'), recursive=True)
    
    plt.figure()
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')

        # Binarize the matrix
        matrix[matrix > 0] = 1

        # Create an unweighted, undirected graph
        G = nx.from_numpy_array(matrix)

        # Calculate the rich club coefficient for the original graph
        rc = nx.rich_club_coefficient(G, normalized=False)

        # Generate a randomized version of the original graph
        rc_rand_avg = {}
        rc_rand_std = {}  # For storing the standard deviation
        num_iterations = 100   ### change this to speed up 
        for _ in range(num_iterations):
            G_rand = G.copy()  # Start with a copy of the original graph

            # Initialize a counter for the number of successful edge swaps
            num_swaps = 0

            # Try a double edge swap, but keep trying until we get a connected graph
            while num_swaps < int(.5*len(G.nodes)):  # Perform X edge swaps
                try:
                    # Perform a double edge swap
                    G_rand = nx.double_edge_swap(G_rand, nswap=int(.5*len(G.nodes)), max_tries=10000)
                    num_swaps += 1  # Increment the counter
                except nx.NetworkXError:
                    continue  # If the swap is not successful, try again

                # Remove self-loops from the randomized graph
                G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

                # Check if the degree sequence is preserved and the graph is connected
                if nx.is_connected(G_rand) and sorted(d for n, d in G.degree()) == sorted(d for n, d in G_rand.degree()):
                    continue  # Continue the loop if G_rand is connected and the degree sequence is preserved
                else:
                    # If the graph is not connected or the degree sequence is not preserved, reset the counter and the graph
                    num_swaps = 0
                    G_rand = G.copy()

            print(f"Iteration {_+1}: Number of successful edge swaps: {num_swaps}")

            # Calculate the rich club coefficient for the randomized graph
            rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)

            # Add the coefficients to the average
            for k, v in rc_rand.items():
                if k in rc_rand_avg:
                    rc_rand_avg[k] += v
                    rc_rand_std[k].append(v)  # Append the value to the list for this degree
                else:
                    rc_rand_avg[k] = v
                    rc_rand_std[k] = [v]  # Start a new list for this degree

        for k in rc_rand_avg.keys():
            rc_rand_avg[k] /= num_iterations

        rc_rand_std = {k: np.std(v) for k, v in rc_rand_std.items() if k % 2 == 0}

        # Calculate the normalized rich club coefficient
        rc_norm = {k: rc[k] / rc_rand_avg[k] for k in rc.keys()}

        # Plot the normalized rich club coefficient
        plt.plot(list(rc_norm.keys()), list(rc_norm.values()), label=os.path.basename(file_path))

    plt.xlabel('Degree')
    plt.ylabel('Normalized Rich Club Coefficient')
    plt.title('Rich Club Curve')
    plt.legend(loc='upper left')
    plt.show()
        
    output_file_path = os.path.join(output_dir, 'comparison_rich_club_curve.png')
    plt.savefig(output_file_path)
    plt.close()

