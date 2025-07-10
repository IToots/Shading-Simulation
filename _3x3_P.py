# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:09:52 2024

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

# # Define the rotate_matrix_90 function
# def rotate_matrix_90(matrix):
#     transpose_matrix = np.transpose(matrix)
#     rotated_matrix = np.flip(transpose_matrix, axis=1)
#     return rotated_matrix

# # Define the ALL function
# def ALL(matrix, matrix_id):
#     transformations = [
#         # (0.1, 1, 1, 1, 'U'),    # Transformation set U
#         # (0.1, 0.6, 0.6, 1, 'US'), # Transformation set US
#         # (0.1, 0.6, 1, 1, 'USD'),   # Transformation set USD
#         (0, 1, 1, 1, 'Z')        # Transformation set Z
#     ]
    
#     results = []
    
#     for params in transformations:
#         transformed_matrix = np.where(matrix == 1, params[0], matrix)
#         transformed_matrix = np.where(matrix == 2, params[1], transformed_matrix)
#         transformed_matrix = np.where(matrix == 3, params[2], transformed_matrix)
#         transformed_matrix = np.where(matrix == 0, params[3], transformed_matrix)
        
#         rotated_matrices = [(transformed_matrix, f"{params[4]}_{matrix_id}_0")]
#         current_matrix = transformed_matrix
        
#         for i in range(1, 4):
#             rotated_matrix = rotate_matrix_90(current_matrix)
#             rotated_matrices.append((rotated_matrix, f"{params[4]}_{matrix_id}_{i}"))
#             current_matrix = rotated_matrix  # Rotate for the next iteration
        
#         results.extend(rotated_matrices)
    
#     # Remove duplicates by using numpy array equality and apply additional transformation
#     unique_matrices = []
#     for matrix, name in results:
#         # Apply the transformation before adding to the unique_matrices
#         # # PERL
#         # transformed_matrix = np.array([
#         #     [matrix[0][0], matrix[1][1], matrix[2][2]],
#         #     [matrix[1][0], matrix[2][1], matrix[0][2]],
#         #     [matrix[2][0], matrix[0][1], matrix[1][2]]
#         # ])
#         # # R1
#         # transformed_matrix = np.array([
#         #     [matrix[0][0], matrix[0][1], matrix[2][2]],
#         #     [matrix[1][0], matrix[2][1], matrix[0][2]],
#         #     [matrix[2][0], matrix[1][1], matrix[1][2]]
#         # ])
#         # # R2R3
#         # transformed_matrix = np.array([
#         #     [matrix[0][0], matrix[0][1], matrix[2][2]],
#         #     [matrix[1][0], matrix[1][1], matrix[0][2]],
#         #     [matrix[2][0], matrix[2][1], matrix[1][2]]
#         # ])
#         # Check for uniqueness based on the transformed matrix
#         if not any(np.array_equal(transformed_matrix, unique_matrix) for unique_matrix, _ in unique_matrices):
#             unique_matrices.append((transformed_matrix, name))
    
#     return unique_matrices


# Matrices with identifiers
matrices = {
    "AA": np.array([[1, 2, 0], [2, 3, 0], [0, 0, 0]]),
    "AB": np.array([[1, 1, 2], [2, 2, 3], [0, 0, 0]]),
    "AC": np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]]),
    
    "AD": np.array([[1, 1, 2], [1, 2, 3], [2, 3, 0]]),
    "AE": np.array([[1, 1, 1], [1, 2, 2], [1, 2, 0]]),
    "AF": np.array([[1, 1, 1], [1, 1, 2], [1, 2, 3]]),
    
    "AG": np.array([[1, 2, 3], [2, 1, 2], [3, 2, 3]]),
    "AH": np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]]), ##
    
    "AI": np.array([[2, 3, 0], [1, 2, 0], [2, 3, 0]]),
    "AJ": np.array([[2, 2, 3], [1, 1, 2], [2, 2, 3]]),
    "AK": np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]]), ##
    
    "AL": np.array([[1, 2, 3], [1, 1, 2], [1, 2, 3]]),
    "AM": np.array([[2, 1, 2], [1, 1, 1], [2, 1, 2]]), #
    "AN": np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]]),
    "AO": np.array([[1, 1, 2], [1, 1, 1], [1, 1, 2]]),
    
    "AP": np.array([[3, 2, 3], [2, 1, 2], [3, 2, 3]]), #
    "AQ": np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]), #
    "AR": np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]), #
    
    "AS": np.array([[1, 2, 0], [2, 3, 0], [1, 2, 0]]),
    "AT": np.array([[1, 1, 2], [1, 2, 3], [1, 1, 2]]),
    "AU": np.array([[1, 1, 1], [1, 2, 2], [1, 1, 1]]),
    
    "AV": np.array([[1, 1, 2], [2, 2, 3], [1, 1, 2]]),
    "AW": np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]]), ##
    
    "AX": np.array([[1, 2, 0], [2, 3, 2], [0, 2, 1]]), ##
    "AY": np.array([[1, 1, 2], [2, 2, 2], [2, 1, 1]]), ##
    "AZ": np.array([[1, 1, 2], [2, 1, 2], [2, 1, 1]]), ##
    
    "BA": np.array([[1, 1, 2], [1, 1, 2], [2, 2, 3]]),
    "BB": np.array([[1, 1, 2], [1, 1, 2], [1, 2, 3]]),
    "BC": np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]]),
    
    "BD": np.array([[1, 2, 0], [1, 2, 3], [1, 1, 2]]),
    "BE": np.array([[2, 1, 2], [2, 1, 2], [2, 1, 1]])
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

# # Print the means with names
# for mean_value, name in means_with_names:
#     print(f"Matrix name: {name}, Mean value: {mean_value}")
    
# ---------------------------------------------------------------------------

param_keys = ['a', 'b', 'c', 
              'd', 'e', 'f', 
              'g', 'h', 'i']

# Define the folder where files will be saved
folder_path_output = "C:/Users/ruiui/Desktop/iteration data/z_3x3/dvp_corr"

# Define variables for file paths and other parameters
ltspice_executable = "C:/Users/ruiui/AppData/Local/Programs/ADI/LTspice/LTspice.exe"


circuits = {
    "SP": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/VIgraph.raw",
        "data_file_prefix": "SP_data_iteration_",
        "line_number_to_replace": 435,
    },
    "SP_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "SP_D_data_iteration_",
        "line_number_to_replace": 522,
    },
    "TCT": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/VIgraph.raw",
        "data_file_prefix": "TCT_data_iteration_",
        "line_number_to_replace": 435
    },
    "TCT_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "TCT_D_data_iteration_",
        "line_number_to_replace": 530
    },
    
    "R1": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/VIgraph.raw",
        "data_file_prefix": "R1_data_iteration_",
        "line_number_to_replace": 468,
    },
    "R1_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "R1_D_data_iteration_",
        "line_number_to_replace": 559,
    },
    "R2/R3": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/VIgraph.raw",
        "data_file_prefix": "R2R3_data_iteration_",
        "line_number_to_replace": 462,
    },
    "R2/R3_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "R2R3_D_data_iteration_",
        "line_number_to_replace": 557,
    },
    "PER/L": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/VIgraph.raw",
        "data_file_prefix": "PERL_data_iteration_",
        "line_number_to_replace": 448
    },
    "PER/L_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "PERL_D_data_iteration_",
        "line_number_to_replace": 554
    }
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
        
        # Modify the specified line
        old_line = lines[config["line_number_to_replace"]]
        new_line = 'TEXT 80 400 Left 0 !.param ' + ' '.join([f'{k}={v}' for k, v in params.items()])

        lines[config["line_number_to_replace"]] = new_line + '\n'

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
        
        plt.figure()
        plt.plot(V, P)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power (W)')
        plt.grid(True)
        plt.xlim(0, max(V))
        plt.ylim(0, max(P))
        plt.show()
        
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
    
