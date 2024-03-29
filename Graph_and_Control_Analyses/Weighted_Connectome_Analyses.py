#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import bct
import os

def find_degree_range(directory_path):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*.csv'), recursive=True)
    
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(matrix)
        
        # Calculate the degree of each node
        degrees = [d for n, d in G.degree()]
        
        # Find the lowest and highest degree
        min_degree = min(degrees)
        max_degree = max(degrees)
        
        print(f"For the dataset {os.path.basename(file_path)}, the lowest degree is {min_degree} and the highest degree is {max_degree}.")


# In[7]:


def plot_strength_distribution(directory_path, output_dir):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*.csv'), recursive=True)
    
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(matrix)
        
        # Calculate the strength of each node (sum of the weights of its edges)
        strengths = [sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes()]
        
        # Plot the strength distribution
        plt.figure()
        plt.hist(strengths, bins=np.logspace(np.log10(min(strengths)), np.log10(max(strengths)), 20), edgecolor='black')
        plt.title(f'Strength Distribution ({os.path.splitext(os.path.basename(file_path))[0]})')
        plt.xlabel('Strength (Log scale)')
        plt.ylabel('Number of Nodes')
        plt.xscale('log')
        
        # Adjust the x-axis limits to fit the data
        plt.xlim(min(strengths), max(strengths))
        
        # Save the plot
        filename = os.path.splitext(os.path.basename(file_path))[0] + '_strength_distribution.png'
        plt.savefig(os.path.join(output_dir, filename))
        
        # Display the plot
        plt.show()


# In[11]:


"""Defines the weighted rich club coefficient as per Alstott et al 2014"""
def weighted_rich_club_coefficient(G):
    # Calculate the rich club coefficients for every degree
    degrees = range(1, max(dict(G.degree()).values()) + 1)
    degrees= range(1, 226) #excludes 226
    coefficients = []
    for k in degrees:
        if k == 226:
            coefficients.append(np.nan)
            continue
            
        # Identify the rich club nodes
        rich_club_nodes = [node for node, degree in G.degree() if degree > k]

        # Calculate W>k, the sum of the weights of the connections between the rich club nodes
        W_k = sum(data['weight'] for u, v, data in G.edges(data=True) if u in rich_club_nodes and v in rich_club_nodes)

        # Get all edges and their weights
        edges_weights = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]

        # Sort by weight in descending order
        edges_weights.sort(key=lambda x: x[2], reverse=True)

        # Determine E_k, the number of edges that connect the rich club nodes to each other
        E_k = len([edge for edge in G.edges() if edge[0] in rich_club_nodes and edge[1] in rich_club_nodes])

        # If E_k is zero, skip this degree
        if E_k == 0:
            coefficients.append(np.nan)
            continue

        # Get the E_k strongest edges
        E_k_strongest_edges = edges_weights[:E_k]

        # Calculate the sum of the weights of the E_k strongest edges
        sum_strongest_E_k = sum(weight for u, v, weight in E_k_strongest_edges)

        # Calculate the weighted rich club coefficient
        phi_w_k = W_k / sum_strongest_E_k

        coefficients.append(phi_w_k)
    
    return degrees, coefficients


# In[16]:


"""Plots the prominence of the weighted rich club with SD Error bars. 
Uses randomization from Serrano et al. 2008"""
def weighted_rich_club_curve(directory_path, output_dir):
    # Get all .csv files in the directory and its subdirectories
    csv_files = glob.glob(os.path.join(directory_path, '**/*SCM.csv'), recursive=True)
    
    # Initialize a figure for the plot
    plt.figure()
    
    for file_path in csv_files:
        # Load the matrix from the file
        matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
        
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(matrix)
        
        # Calculate the rich club coefficients for every degree
        degrees, orig_coefficients = weighted_rich_club_coefficient(G)
        
        # Generate m random networks and calculate their rich club coefficients
        m = 100
        random_coeffs_list = []
        normalized_coeffs_list = []
        for i in range(m):
            # Generate a random network with the same degree sequence as G
            W_rand = bct.null_model_und_sign(matrix, bin_swaps=0, wei_freq=0.3)[0]#adjust weight frequency here
            G_rand = nx.from_numpy_array(W_rand)
            
            # Calculate the rich club coefficients for the random network
            _, rand_coeffs = weighted_rich_club_coefficient(G_rand)
            random_coeffs_list.append(rand_coeffs)
            
            # Normalize the coefficients of the random network
            normalized_rand_coeffs = [orig_coeff / rand_coeff if rand_coeff != 0 else np.nan for orig_coeff, rand_coeff in zip(orig_coefficients, rand_coeffs)]
            normalized_coeffs_list.append(normalized_rand_coeffs)
            
            # Print the completion of this iteration
            print(f"Completed random graph iteration {i+1}/{m}")

        # Calculate the average and standard deviation of the rich club coefficient for the random networks
        avg_random_coeffs = np.nanmean(random_coeffs_list, axis=0)
        std_normalized_coeffs = np.nanstd(normalized_coeffs_list, axis=0)
        
        # Normalize the coefficients of the original network
        normalized_orig_coefficients = [orig_coeff / avg_rand_coeff if avg_rand_coeff != 0 else np.nan for orig_coeff, avg_rand_coeff in zip(orig_coefficients, avg_random_coeffs)]
        
        # Plot the normalized rich club curve for all degrees
        line, = plt.plot(degrees, normalized_orig_coefficients, label=os.path.splitext(os.path.basename(file_path))[0])

        # Add error bars to the plot for every third degree
        for degree, coeff, yerr in zip(degrees, normalized_orig_coefficients, std_normalized_coeffs):
            if degree % 3 == 0:  # Only add error bars for every third degree
                plt.errorbar(degree, coeff, yerr=yerr, fmt='none', capsize=3, color=line.get_color())  # Match the color with the line

    plt.xlim(175, 225)  # to limit the plot to set width on x axis

    # Add labels and title to the plot
    plt.xlabel('Rich Club Level k')
    plt.ylabel('Φw(k)norm')
    plt.title('Weighted Rich Club Organization')
    plt.legend()    
    
    
    # Save the plot to the output directory
    output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_weighted_rich_club_curve.png'))
    plt.savefig(output_file_path)
    plt.show()

