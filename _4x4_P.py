# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:11:22 2024

@author: ruiui
"""

import subprocess
import ltspice
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# Define the rotate_matrix_90 function
def rotate_matrix_90(matrix):
    transpose_matrix = np.transpose(matrix)
    rotated_matrix = np.flip(transpose_matrix, axis=1)
    return rotated_matrix

# Define the ALL function
def ALL(matrix, matrix_id):
    transformations = [
        # (0.1, 1, 1, 1, 'U'),    # Transformation set U
        # (0.1, 0.6, 0.6, 1, 'US'), # Transformation set US
        # (0.1, 0.6, 1, 1, 'USD'),   # Transformation set USD
        (0, 1, 1, 1, 'Z')        # Transformation set Z
    ]
    
    results = []
    
    for params in transformations:
        transformed_matrix = np.where(matrix == 1, params[0], matrix)
        transformed_matrix = np.where(matrix == 2, params[1], transformed_matrix)
        transformed_matrix = np.where(matrix == 3, params[2], transformed_matrix)
        transformed_matrix = np.where(matrix == 0, params[3], transformed_matrix)
        
        rotated_matrices = [(transformed_matrix, f"{params[4]}_{matrix_id}_0")]
        current_matrix = transformed_matrix
        
        for i in range(1, 4):
            rotated_matrix = rotate_matrix_90(current_matrix)
            rotated_matrices.append((rotated_matrix, f"{params[4]}_{matrix_id}_{i}"))
            current_matrix = rotated_matrix  # Rotate for the next iteration
        
        results.extend(rotated_matrices)
    
    # Remove duplicates by using numpy array equality
    unique_matrices = []
    for matrix, name in results:
        if not any(np.array_equal(matrix, unique_matrix) for unique_matrix, _ in unique_matrices):
            unique_matrices.append((matrix, name))
    
    return unique_matrices

# Matrices
matrices = {
    "AA": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AB": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AC": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AD": np.array([[1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]]),
    
    "AE": np.array([[1, 1, 2, 0], [1, 2, 3, 0], [2, 3, 0, 0], [0, 0, 0, 0]]),
    "AF": np.array([[1, 1, 1, 2], [1, 2, 2, 3], [1, 2, 0, 0], [2, 3, 0, 0]]),
    "AG": np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 0, 0], [1, 2, 0, 0]]),
    "AH": np.array([[1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 0]]),
    "AI": np.array([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 3, 0], [1, 2, 0, 0]]),
    "AJ": np.array([[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 0]]),
    
    "AK": np.array([[1, 2, 3, 0], [2, 1, 2, 0], [3, 2, 3, 0], [0, 0, 0, 0]]),
    "AL": np.array([[1, 2, 3, 0], [2, 1, 2, 3], [3, 2, 1, 2], [0, 3, 2, 3]]),
    "AM": np.array([[1, 2, 3, 0], [2, 1, 2, 3], [3, 2, 1, 2], [0, 3, 2, 1]]),
    
    "AN": np.array([[2, 3, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0]]),
    "AO": np.array([[2, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0]]),
    "AP": np.array([[2, 2, 2, 3], [1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 3]]),
    "AQ": np.array([[2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]),
    
    "AR": np.array([[1, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0], [1, 2, 3, 0]]),
    "AS": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 2, 3]]),
    "AT": np.array([[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 2]]),
    "AU": np.array([[2, 1, 1, 2], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 2]]),
    "AV": np.array([[1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2]]),
    
    "AW": np.array([[3, 2, 2, 3], [2, 1, 1, 2], [2, 1, 1, 2], [3, 2, 2, 3]]),
    "AX": np.array([[1, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]]),
    "AY": np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]),
    
    "AZ": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [2, 3, 0, 0], [1, 2, 0, 0]]),
    "BA": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [2, 2, 3, 0], [1, 1, 2, 0]]),
    "BB": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [2, 2, 2, 3], [1, 1, 1, 2]]),
    "BC": np.array([[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [1, 1, 1, 1]]),
    
    "BD": np.array([[1, 1, 2, 0], [1, 2, 3, 0], [1, 2, 3, 0], [1, 1, 2, 0]]),
    "BE": np.array([[1, 1, 1, 2], [1, 1, 2, 3], [1, 1, 2, 3], [1, 1, 1, 2]]),
    "BF": np.array([[2, 1, 1, 1], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1]]),
    "BG": np.array([[1, 1, 1, 2], [1, 2, 2, 3], [1, 2, 2, 3], [1, 1, 1, 2]]),
    "BH": np.array([[2, 1, 1, 1], [2, 1, 2, 2], [2, 1, 2, 2], [2, 1, 2, 2]]),
    "BI": np.array([[1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 1, 1]]),
    "BJ": np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2], [1, 1, 1, 1]]),
    
    "BK": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [0, 0, 3, 2], [0, 0, 2, 1]]),
    "BL": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [0, 3, 2, 2], [0, 2, 1, 1]]),
    "BM": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [3, 2, 2, 2], [2, 1, 1, 1]]),
    "BN": np.array([[1, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1]]),
    
    "BO": np.array([[1, 1, 2, 0], [1, 2, 3, 2], [2, 3, 2, 1], [0, 2, 1, 1]]),
    "BP": np.array([[1, 1, 1, 2], [1, 2, 2, 1], [1, 2, 2, 1], [2, 1, 1, 1]]),
    "BQ": np.array([[1, 1, 2, 0], [1, 1, 2, 2], [2, 2, 1, 1], [0, 2, 1, 1]]),
    
    "BR": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0], [0, 0, 0, 0]]),
    "BS": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0]]),
    "BT": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0]]),
    "BU": np.array([[1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 3]]),
    
    "BV": np.array([[1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 2, 3], [2, 2, 3, 0]]),
    "BW": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [2, 2, 3, 0]]),
    "BX": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 2, 0]]),
    
    "BY": np.array([[1, 2, 0, 0], [1, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0]]),
    "BZ": np.array([[2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1], [2, 1, 1, 1]]),
    
    "CA": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 2, 3], [3, 2, 3, 0]]),
    "CB": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 1, 2], [3, 2, 2, 3]]),
    "CC": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 1, 1], [3, 2, 1, 1]]),
    }

# Apply ALL function to each matrix and store results in a list
all_matrices = [ALL(matrix, matrix_id) for matrix_id, matrix in matrices.items()]

# Calculate the total number of arrays
total_arrays = sum(len(outer_list) for outer_list in all_matrices)

print(total_arrays)

# Flatten and check for duplicate matrices
scenarios = []
duplicate_matrices = []

# Flatten the list of lists
all_matrices_flat = [item for sublist in all_matrices for item in sublist]

for matrix, name in all_matrices_flat:
    if any(np.array_equal(matrix, existing_matrix) for existing_matrix, _ in scenarios):
        duplicate_matrices.append((matrix, name))
    else:
        scenarios.append((matrix, name))

# List to store the result
means_with_names = []

# Iterate through each tuple in the set
for matrix, name in scenarios:
    # Calculate the mean of the matrix
    mean_value = round(np.mean(matrix), 3)
    # Append the result to the list
    means_with_names.append((mean_value, name))

# Sort the list alphabetically by the name
means_with_names.sort(key=lambda x: x[1])


# List to store the result
medians_with_names = []

# Iterate through each tuple in the set
for matrix, name in scenarios:
    # Calculate the median of the matrix
    median_value = round(np.median(matrix), 3)
    # Append the result to the list
    medians_with_names.append((median_value, name))

# Sort the list alphabetically by the name
medians_with_names.sort(key=lambda x: x[1])

    
# ---------------------------------------------------------------------------

param_keys = ['a', 'b', 'c', 'd', 
              'e', 'f', 'g', 'h',
              'i', 'j', 'k', 'l', 
              'm', 'n', 'o', 'p']

# Define the folder where files will be saved
folder_path_output = "C:/Users/ruiui/Desktop/iteration data/z_4x4/new"

# Define variables for file paths and other parameters
ltspice_executable = "C:/Users/ruiui/AppData/Local/Programs/ADI/LTspice/LTspice.exe"

circuits = {
    "SP": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/VIgraph.raw",
        "data_file_prefix": "SP_data_iteration_",
        "line_number_to_replace": 779
    },
    "TCT": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/VIgraph.raw",
        "data_file_prefix": "TCT_data_iteration_",
        "line_number_to_replace": 779
    },
    "R1": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/VIgraph.raw",
        "data_file_prefix": "R1_data_iteration_",
        "line_number_to_replace": 822
    },
    "R2/R3": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/VIgraph.raw",
        "data_file_prefix": "R2R3_data_iteration_",
        "line_number_to_replace": 819
    },
    "PER": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/VIgraph.raw",
        "data_file_prefix": "PER_data_iteration_",
        "line_number_to_replace": 857
    },
    "L": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/VIgraph.raw",
        "data_file_prefix": "L_data_iteration_",
        "line_number_to_replace": 860
    },
    "SP_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "SP_D_data_iteration_",
        "line_number_to_replace_a": 853,
        "line_number_to_replace_b": 613,
        "line_number_to_replace_c": 587,
        "line_number_to_replace_d": 561,
        "line_number_to_replace_e": 827,
        "line_number_to_replace_f": 535,
        "line_number_to_replace_g": 509,
        "line_number_to_replace_h": 483,
        "line_number_to_replace_i": 801,
        "line_number_to_replace_j": 457,
        "line_number_to_replace_k": 431,
        "line_number_to_replace_l": 405,
        "line_number_to_replace_m": 900,
        "line_number_to_replace_n": 754,
        "line_number_to_replace_o": 728,
        "line_number_to_replace_p": 702
    },
    "TCT_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "TCT_D_data_iteration_",
        "line_number_to_replace_a": 871,
        "line_number_to_replace_b": 631,
        "line_number_to_replace_c": 605,
        "line_number_to_replace_d": 579,
        "line_number_to_replace_e": 845,
        "line_number_to_replace_f": 553,
        "line_number_to_replace_g": 527,
        "line_number_to_replace_h": 501,
        "line_number_to_replace_i": 819,
        "line_number_to_replace_j": 475,
        "line_number_to_replace_k": 449,
        "line_number_to_replace_l": 423,
        "line_number_to_replace_m": 918,
        "line_number_to_replace_n": 772,
        "line_number_to_replace_o": 746,
        "line_number_to_replace_p": 720
    },
    "R1_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "R1_D_data_iteration_",
        "line_number_to_replace_a": 914,
        "line_number_to_replace_b": 674,
        "line_number_to_replace_c": 648,
        "line_number_to_replace_d": 622,
        "line_number_to_replace_e": 888,
        "line_number_to_replace_f": 596,
        "line_number_to_replace_g": 570,
        "line_number_to_replace_h": 544,
        "line_number_to_replace_i": 862,
        "line_number_to_replace_j": 518,
        "line_number_to_replace_k": 492,
        "line_number_to_replace_l": 466,
        "line_number_to_replace_m": 947,
        "line_number_to_replace_n": 759,
        "line_number_to_replace_o": 733,
        "line_number_to_replace_p": 707
    },
    "R2/R3_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "R2R3_D_data_iteration_",
        "line_number_to_replace_a": 911,
        "line_number_to_replace_b": 671,
        "line_number_to_replace_c": 645,
        "line_number_to_replace_d": 619,
        "line_number_to_replace_e": 885,
        "line_number_to_replace_f": 593,
        "line_number_to_replace_g": 567,
        "line_number_to_replace_h": 541,
        "line_number_to_replace_i": 859,
        "line_number_to_replace_j": 515,
        "line_number_to_replace_k": 489,
        "line_number_to_replace_l": 463,
        "line_number_to_replace_m": 958,
        "line_number_to_replace_n": 812,
        "line_number_to_replace_o": 786,
        "line_number_to_replace_p": 760
    },
    "PER_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "PER_D_data_iteration_",
        "line_number_to_replace_a": 634,
        "line_number_to_replace_b": 609,
        "line_number_to_replace_c": 583,
        "line_number_to_replace_d": 849,
        "line_number_to_replace_e": 557,
        "line_number_to_replace_f": 531,
        "line_number_to_replace_g": 505,
        "line_number_to_replace_h": 823,
        "line_number_to_replace_i": 797,
        "line_number_to_replace_j": 771,
        "line_number_to_replace_k": 745,
        "line_number_to_replace_l": 901,
        "line_number_to_replace_m": 719,
        "line_number_to_replace_n": 693,
        "line_number_to_replace_o": 667,
        "line_number_to_replace_p": 875
    },
    "L_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4/DIODE/VIgraph.raw",
        "data_file_prefix": "L_D_data_iteration_",
        "line_number_to_replace_a": 868,
        "line_number_to_replace_b": 712,
        "line_number_to_replace_c": 686,
        "line_number_to_replace_d": 660,
        "line_number_to_replace_e": 842,
        "line_number_to_replace_f": 634,
        "line_number_to_replace_g": 608,
        "line_number_to_replace_h": 582,
        "line_number_to_replace_i": 816,
        "line_number_to_replace_j": 556,
        "line_number_to_replace_k": 530,
        "line_number_to_replace_l": 504,
        "line_number_to_replace_m": 894,
        "line_number_to_replace_n": 790,
        "line_number_to_replace_o": 764,
        "line_number_to_replace_p": 738
    }   
    
    # "SP": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "SP_data_iteration_",
    #     "line_number_to_replace": 779
    # },
    # "TCT": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "TCT_data_iteration_",
    #     "line_number_to_replace": 779
    # },
    # "R1": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "R1_data_iteration_",
    #     "line_number_to_replace": 822
    # },
    # "R2/R3": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "R2R3_data_iteration_",
    #     "line_number_to_replace": 819
    # },
    # "PER": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "PER_data_iteration_",
    #     "line_number_to_replace": 857
    # },
    # "L": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/VIgraph.raw",
    #     "data_file_prefix": "L_data_iteration_",
    #     "line_number_to_replace": 860
    # },
    # "SP_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "SP_D_data_iteration_",
    #     "line_number_to_replace_a": 853,
    #     "line_number_to_replace_b": 613,
    #     "line_number_to_replace_c": 587,
    #     "line_number_to_replace_d": 561,
    #     "line_number_to_replace_e": 827,
    #     "line_number_to_replace_f": 535,
    #     "line_number_to_replace_g": 509,
    #     "line_number_to_replace_h": 483,
    #     "line_number_to_replace_i": 801,
    #     "line_number_to_replace_j": 457,
    #     "line_number_to_replace_k": 431,
    #     "line_number_to_replace_l": 405,
    #     "line_number_to_replace_m": 900,
    #     "line_number_to_replace_n": 754,
    #     "line_number_to_replace_o": 728,
    #     "line_number_to_replace_p": 702
    # },
    # "TCT_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "TCT_D_data_iteration_",
    #     "line_number_to_replace_a": 871,
    #     "line_number_to_replace_b": 631,
    #     "line_number_to_replace_c": 605,
    #     "line_number_to_replace_d": 579,
    #     "line_number_to_replace_e": 845,
    #     "line_number_to_replace_f": 553,
    #     "line_number_to_replace_g": 527,
    #     "line_number_to_replace_h": 501,
    #     "line_number_to_replace_i": 819,
    #     "line_number_to_replace_j": 475,
    #     "line_number_to_replace_k": 449,
    #     "line_number_to_replace_l": 423,
    #     "line_number_to_replace_m": 918,
    #     "line_number_to_replace_n": 772,
    #     "line_number_to_replace_o": 746,
    #     "line_number_to_replace_p": 720
    # },
    # "R1_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R1_D_data_iteration_",
    #     "line_number_to_replace_a": 914,
    #     "line_number_to_replace_b": 674,
    #     "line_number_to_replace_c": 648,
    #     "line_number_to_replace_d": 622,
    #     "line_number_to_replace_e": 888,
    #     "line_number_to_replace_f": 596,
    #     "line_number_to_replace_g": 570,
    #     "line_number_to_replace_h": 544,
    #     "line_number_to_replace_i": 862,
    #     "line_number_to_replace_j": 518,
    #     "line_number_to_replace_k": 492,
    #     "line_number_to_replace_l": 466,
    #     "line_number_to_replace_m": 947,
    #     "line_number_to_replace_n": 759,
    #     "line_number_to_replace_o": 733,
    #     "line_number_to_replace_p": 707
    # },
    # "R2/R3_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R2R3_D_data_iteration_",
    #     "line_number_to_replace_a": 911,
    #     "line_number_to_replace_b": 671,
    #     "line_number_to_replace_c": 645,
    #     "line_number_to_replace_d": 619,
    #     "line_number_to_replace_e": 885,
    #     "line_number_to_replace_f": 593,
    #     "line_number_to_replace_g": 567,
    #     "line_number_to_replace_h": 541,
    #     "line_number_to_replace_i": 859,
    #     "line_number_to_replace_j": 515,
    #     "line_number_to_replace_k": 489,
    #     "line_number_to_replace_l": 463,
    #     "line_number_to_replace_m": 958,
    #     "line_number_to_replace_n": 812,
    #     "line_number_to_replace_o": 786,
    #     "line_number_to_replace_p": 760
    # },
    # "PER_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "PER_D_data_iteration_",
    #     "line_number_to_replace_a": 634,
    #     "line_number_to_replace_b": 609,
    #     "line_number_to_replace_c": 583,
    #     "line_number_to_replace_d": 849,
    #     "line_number_to_replace_e": 557,
    #     "line_number_to_replace_f": 531,
    #     "line_number_to_replace_g": 505,
    #     "line_number_to_replace_h": 823,
    #     "line_number_to_replace_i": 797,
    #     "line_number_to_replace_j": 771,
    #     "line_number_to_replace_k": 745,
    #     "line_number_to_replace_l": 901,
    #     "line_number_to_replace_m": 719,
    #     "line_number_to_replace_n": 693,
    #     "line_number_to_replace_o": 667,
    #     "line_number_to_replace_p": 875
    # },
    # "L_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/4x4_special/DIODE/VIgraph.raw",
    #     "data_file_prefix": "L_D_data_iteration_",
    #     "line_number_to_replace_a": 868,
    #     "line_number_to_replace_b": 712,
    #     "line_number_to_replace_c": 686,
    #     "line_number_to_replace_d": 660,
    #     "line_number_to_replace_e": 842,
    #     "line_number_to_replace_f": 634,
    #     "line_number_to_replace_g": 608,
    #     "line_number_to_replace_h": 582,
    #     "line_number_to_replace_i": 816,
    #     "line_number_to_replace_j": 556,
    #     "line_number_to_replace_k": 530,
    #     "line_number_to_replace_l": 504,
    #     "line_number_to_replace_m": 894,
    #     "line_number_to_replace_n": 790,
    #     "line_number_to_replace_o": 764,
    #     "line_number_to_replace_p": 738
    # }    
}


# Iterate over the circuit configurations
for circuit_name, config in circuits.items():
    
    # Read the file once
    with open(config["asc_file_path"], 'r') as file:
        lines = file.readlines()

    # Iterate over the unique matrices in scenarios
    for matrix_tuple in scenarios:
        matrix, name = matrix_tuple
        matrix_values = matrix.flatten()
        if len(matrix_values) != len(param_keys):
            print("Error: The number of matrix values does not match the number of parameter keys.")
            continue
        
        # Map matrix values to parameter keys
        params = {param_keys[i]: matrix_values[i] for i in range(len(param_keys))}
       
        if '_D' in circuit_name:
            for key in param_keys:
                line_number_key = f"line_number_to_replace_{key}"
                if line_number_key in config:
                    line_number = config[line_number_key]
                    old_line = lines[line_number]
                    new_line = f"SYMATTR Value {{imax*{params[key]}}}"
                    lines[line_number] = old_line.replace(old_line, new_line + '\n')
        else:
            old_line = lines[config["line_number_to_replace"]]
            new_line = f"TEXT 80 400 Left 0 !.param Is=3.25e-11 Imax=0.98722 Vmax=3.378 n=1.08*5 Rseries=0.2 Rshunt=205 " + ' '.join([f"{k}={v}" for k, v in params.items()])
            lines[config["line_number_to_replace"]] = old_line.replace(old_line, new_line + '\n')
        
        # Write the modified content back to the file
        with open(config["output_asc_file_path"], 'w') as file:
            file.writelines(lines)
        
        # Run LTspice simulation
        command = [ltspice_executable, "-Run", "-b", config["ltspice_run_file"]]
        subprocess.run(command)
        
        # Load and parse LTspice raw data
        ltspice_data = ltspice.Ltspice(config["ltspice_raw_file"])
        ltspice_data.parse()
        
        # Extract and plot data
        V = ltspice_data.get_data('Vpv')
        I = ltspice_data.get_data('I(D1)')
        P = V * I
        
        # plt.figure()
        # plt.plot(V, P)
        # plt.xlabel('Voltage (V)')
        # plt.ylabel('Power (W)')
        # plt.grid(True)
        # plt.xlim(0, max(V))
        # plt.ylim(0, max(P))
        # plt.show()
        
        # Ensure the folder exists
        os.makedirs(folder_path_output, exist_ok=True)
        
        # Loop to save data to a CSV file
        data_file_path = os.path.join(folder_path_output, f"{config['data_file_prefix']}{name}.csv")
        with open(data_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Voltage (V)', 'Current (I)', 'Power (P)'])
            for v, i, p in zip(V, I, P):
                csvwriter.writerow([v, i, p])
        
        print(f"Iteration {name} in {circuit_name} completed and data saved.")