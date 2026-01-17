import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import pickle
import random
import bct


def unweighted_rich_club_curve(file_path, output_dir):
    """Generate the unweighted rich club curve from a connectivity matrix file."""
    # Load the matrix from the file
    matrix = np.loadtxt(file_path, dtype=float, delimiter=',')

    # Binarize the matrix
    matrix[matrix > 0] = 1

    # Create an unweighted, undirected graph for original rich club calculation
    G = nx.from_numpy_array(matrix)

    # Calculate the rich club coefficient for the original graph
    rc = nx.rich_club_coefficient(G, normalized=False)

    # Generate randomized versions using BCT's randmio_und_connected
    rc_rand_avg = {} 
    rc_rand_std = {}  
    num_iterations = 1000   
    num_rewires = 10
    
    for iteration in range(num_iterations):
        # Use BCT's randmio_und_connected - more efficient and standard
        matrix_rand = bct.randmio_und_connected(matrix, num_rewires)[0]
        
        # Convert back to NetworkX graph for rich club calculation
        G_rand = nx.from_numpy_array(matrix_rand)
        
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
        
        print(f"Completed iteration {iteration+1}/{num_iterations}")

    # Calculate averages and standard deviations
    for k in rc_rand_avg.keys():
        rc_rand_avg[k] /= num_iterations

    rc_rand_std = {k: np.std(v) for k, v in rc_rand_std.items()}

    # Calculate the normalized rich club coefficient
    rc_norm = {k: rc[k] / rc_rand_avg[k] for k in rc.keys()}

    # Plot the normalized rich club coefficient
    fig, ax = plt.subplots()
    line, = plt.plot(list(rc_norm.keys()), list(rc_norm.values()), label=os.path.basename(file_path))
    
    # Add error bars to the plot for every third degree
    for degree, coeff, yerr in zip(list(rc_norm.keys()), list(rc_norm.values()), list(rc_rand_std.values())):
        if degree % 3 == 0:  # Only add error bars for every third degree
            plt.errorbar(degree, coeff, yerr=yerr, fmt='none', capsize=3, color=line.get_color())  # Match the color with the line
    
    plt.xlim(150, 226)  # to limit the plot to set width on x axis
    plt.ylim(.99, 1.03)
    plt.xlabel('Degree')
    plt.ylabel('Unweighted Rich Club Coefficient')
    plt.title('Rich Club Curve')
    plt.legend(loc='lower left')

    output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_unweighted_rcc.pkl'))
    with open(output_file_path, 'wb') as f:
        pickle.dump(fig, f)
    
    plt.close(fig)

def weighted_rich_club_coefficient(G):
    """Calculate the weighted rich club coefficient for a given graph G."""
    degrees = range(1, max(dict(G.degree()).values()) + 1)
    coefficients = []
    for k in degrees:
        rich_club_nodes = [node for node, degree in G.degree() if degree >= k]
        W_k = sum(data['weight'] for u, v, data in G.edges(data=True) if u in rich_club_nodes and v in rich_club_nodes)
        edges_weights = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
        edges_weights.sort(key=lambda x: x[2], reverse=True)
        E_k = len([edge for edge in G.edges() if edge[0] in rich_club_nodes and edge[1] in rich_club_nodes])
        if E_k == 0:
            coefficients.append(np.nan)
            continue
        E_k_strongest_edges = edges_weights[:E_k]
        sum_strongest_E_k = sum(weight for u, v, weight in E_k_strongest_edges)
        phi_w_k = W_k / sum_strongest_E_k
        coefficients.append(phi_w_k)
    return degrees, coefficients




def shuffle_weights_rcc(file_path, output_dir):
    """Generate the rich club curve from a connectivity matrix file by shuffling the weights."""
    matrix = np.loadtxt(file_path, dtype=float, delimiter=',')
    G = nx.from_numpy_array(matrix)
    degrees, orig_coefficients = weighted_rich_club_coefficient(G)
    m = 1000
    random_coeffs_list = []
    normalized_coeffs_list = []
    for i in range(m):
        W_rand = bct.null_model_und_sign(matrix, bin_swaps=0, wei_freq=0.3)[0]  #brain connectivity toolbox. Shuffles weights, preserving the degree and strength distributions
        G_rand = nx.from_numpy_array(W_rand)
        _, rand_coeffs = weighted_rich_club_coefficient(G_rand)
        random_coeffs_list.append(rand_coeffs)
        normalized_rand_coeffs = [orig_coeff / rand_coeff if rand_coeff != 0 else np.nan for orig_coeff, rand_coeff in zip(orig_coefficients, rand_coeffs)]
        normalized_coeffs_list.append(normalized_rand_coeffs)
        print("Completed random graph iteration {}/{}".format(i+1, m))
    avg_random_coeffs = np.nanmean(random_coeffs_list, axis=0)
    std_normalized_coeffs = np.nanstd(normalized_coeffs_list, axis=0)
    normalized_orig_coefficients = [orig_coeff / avg_rand_coeff if avg_rand_coeff != 0 else np.nan for orig_coeff, avg_rand_coeff in zip(orig_coefficients, avg_random_coeffs)]
    fig, ax = plt.subplots()
    line, = plt.plot(degrees, normalized_orig_coefficients, label=os.path.splitext(os.path.basename(file_path))[0])
    for degree, coeff, yerr in zip(degrees, normalized_orig_coefficients, std_normalized_coeffs):
        if degree % 3 == 0:
            plt.errorbar(degree, coeff, yerr=yerr, fmt='none', capsize=3, color=line.get_color())
    plt.xlim(120, 226)
    plt.ylim(1, 30)
    plt.xlabel('Rich Club Level k')
    plt.ylabel('Φw(k)norm')
    plt.title('Shuffle weights WRC')
    plt.legend()
    output_file_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_weights.pkl'))
    with open(output_file_path, 'wb') as f:
        pickle.dump(fig, f)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate rich club curve from connectivity matrix.')
    parser.add_argument('input', type=str, help=' Path to the input CSV file.')
    parser.add_argument('output', type=str, help='Path to the output directory.')
    args = parser.parse_args()
    
    weights_output_dir = os.path.join(args.output, 'weights')
    unweighted_output_dir = os.path.join(args.output, 'unweighted')

    
    os.makedirs(weights_output_dir, exist_ok=True)
    os.makedirs(unweighted_output_dir, exist_ok=True)

    
    shuffle_weights_rcc(args.input, weights_output_dir)
    unweighted_rich_club_curve(args.input, unweighted_output_dir)
    