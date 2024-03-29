#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import numpy as np
import control 
import sys
import scipy.linalg as la
import pandas as pd

def avg_control(A):
    """
    Average controllability based on code from Basset
    https://complexsystemsupenn.com/codedata
    :param A: numpy array connectome
    :return: average controllability for each node
    """
    normA = la.svdvals(A)[0] + 1
    A = A / normA
    T, U = la.schur(A, output='real')
    eVals = np.diag(T)
    eVals.shape = [len(A[0,:]), 1]
    midMat = (U**2).transpose() 
    P = np.tile(1-eVals**2, (1, len(A[0,:])))
    res = sum(midMat/P)
    return res 


# In[41]:


def modal_control(G):
    """
    # Modal controllability based on code from Basset
    # https://complexsystemsupenn.com/codedata
    :param G: numpy array connectome
    :param stabilize: stabilize by ensuring largest e-value is 1.
    :return: modal controllability for each node
    """
    A = G
    # stabilize, then calculate Matrix norm
    normA = la.svdvals(A)[0] + 1
    A = A / normA
    # Schur decomp - stability
    T, U = la.schur(A, output='real')
    eVals = np.diag(T)
    N = len(eVals)
    phi = np.empty([N, ])
    for i in range(0, N):
        phi[i] = np.dot(U[i, :]**2, (1 - eVals**2))
    
    return phi


# In[48]:


### Function to run average control
def process_AVG_cont(conn_file, out_dir, atlas_file):
    # Load the data
    conn_mat = np.loadtxt(conn_file, dtype=float, delimiter=',')
    
    res_B = avg_control(conn_mat)
    
    # Load the atlas labels
    atlas_labels = pd.read_csv(atlas_file, header=None)
    
    # Create a DataFrame for the results with atlas labels as index
    res_df = pd.DataFrame(res_B, index=atlas_labels[0], columns=['Average Controllability'])
    # Extract the conn_file name from the file path
    conn_file_name = os.path.basename(conn_file)
    # Save the result as a CSV file with the same name as the loaded file
    csv_name = os.path.splitext(os.path.basename(conn_file))[0] + '_ac.csv'
    res_df.to_csv(os.path.join(out_dir, csv_name), header=False, float_format='%1.8f')


# In[49]:


def process_MOD_cont(conn_file, out_dir, atlas_file):
    # Load the data
    conn_mat = np.loadtxt(conn_file, dtype=float, delimiter=',')
    
    res_B = modal_control(conn_mat)
    
    # Load the atlas labels
    atlas_labels = pd.read_csv(atlas_file, header=None)
    
    # Create a DataFrame for the results with atlas labels as index
    res_df = pd.DataFrame(res_B, index=atlas_labels[0], columns=['Modal Controllability'])
    
    # Save the result as a CSV file with the same name as the loaded file
    # Extract the conn_file name from the file path
    conn_file_name = os.path.basename(conn_file)
    # Include the conn_file name in the output file name
    csv_name = os.path.splitext(os.path.basename(conn_file))[0] + '_mc.csv'
    res_df.to_csv(os.path.join(out_dir, csv_name), header=False, float_format='%1.8f')

