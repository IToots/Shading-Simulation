# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:10:21 2024

@author: ruiui
"""


import subprocess
import ltspice
import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# Define the ALL function
def ALL(matrix, matrix_id):
    transformations = [(i / 1000, str(i)) for i in 
                       [111,222,333,444,556,667,778,889,1000]
                       # range(1, 1001)
                       ]
    results = []
    
    for params in transformations:
        transformed_matrix = np.where(matrix == 1, params[0], matrix)
        rotated_matrices = [(transformed_matrix, f"{matrix_id}_{params[1]}")]
        
        results.extend(rotated_matrices)
    
    # Convert results to a set to remove duplicates and then back to a list
    unique_matrices = list(set((tuple(matrix.flatten()), name) for matrix, name in results))
    
    # Convert the unique matrices back to numpy arrays
    unique_matrices = [(np.array(matrix).reshape(results[0][0].shape), name) for matrix, name in unique_matrices]
    
    return unique_matrices

# Matrices with identifiers
matrices = {
    "Irr": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
}

# Apply ALL function to each matrix and store results in a list
all_matrices = [ALL(matrix, matrix_id) for matrix_id, matrix in matrices.items()]

# Flatten and convert all_matrices to a set of tuples for comparison
scenarios = set()
duplicate_matrices = []

for matrices_set in all_matrices:
    for matrix, name in matrices_set:
        matrix_tuple = (tuple(matrix.flatten()), name)
        if matrix_tuple in scenarios:
            duplicate_matrices.append((matrix, name))
        else:
            scenarios.add(matrix_tuple)

# ---------------------------------------------------------------------------

param_keys = ['a', 'b', 'c', 
              'd', 'e', 'f', 
              'g', 'h', 'i']

# Define the folder where files will be saved
folder_path_output = "C:/Users/ruiui/Desktop/iteration data/z_3x3_irr/TCT_D/mod_corr"

# Define variables for file paths and other parameters
ltspice_executable = "C:/Users/ruiui/AppData/Local/Programs/ADI/LTspice/LTspice.exe"


circuits = {
    # "SP": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/VIgraph.raw",
    #     "data_file_prefix": "SP_data_iteration_",
    #     "line_number_to_replace": 435,
    # },
    # "SP": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/VIgraph.raw",
    #     "data_file_prefix": "SP_data_iteration_",
    #     "line_number_to_replace": 435,
    # },
    # "SP_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3/DIODE/VIgraph.raw",
    #     "data_file_prefix": "SP_D_data_iteration_",
    #     "line_number_to_replace": 522,
    # },
    # "SP_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/3x3_mod/DIODE/VIgraph.raw",
    #     "data_file_prefix": "SP_D_data_iteration_",
    #     "line_number_to_replace": 522,
    # },
    # "TCT": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/VIgraph.raw",
    #     "data_file_prefix": "TCT_data_iteration_",
    #     "line_number_to_replace": 435
    # },
    # "TCT": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/VIgraph.raw",
    #     "data_file_prefix": "TCT_data_iteration_",
    #     "line_number_to_replace": 435
    # },
    # "TCT_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3/DIODE/VIgraph.raw",
    #     "data_file_prefix": "TCT_D_data_iteration_",
    #     "line_number_to_replace": 530
    # },
    "TCT_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/3x3_mod/DIODE/VIgraph.raw",
        "data_file_prefix": "TCT_D_data_iteration_",
        "line_number_to_replace": 530
    },
    # "R1": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/VIgraph.raw",
    #     "data_file_prefix": "R1_data_iteration_",
    #     "line_number_to_replace": 468,
    # },
    # "R1_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/3x3/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R1_D_data_iteration_",
    #     "line_number_to_replace": 559,
    # },
    # "R2/R3": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/VIgraph.raw",
    #     "data_file_prefix": "R2R3_data_iteration_",
    #     "line_number_to_replace": 462,
    # },
    # "R2/R3_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/3x3/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R2R3_D_data_iteration_",
    #     "line_number_to_replace": 557,
    # },
    # "PER/L": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/VIgraph.raw",
    #     "data_file_prefix": "PERL_data_iteration_",
    #     "line_number_to_replace": 448
    # },
    # "PER/L_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/3x3/DIODE/VIgraph.raw",
    #     "data_file_prefix": "PERL_D_data_iteration_",
    #     "line_number_to_replace": 554
    # }
}

# Iterate over the circuit configurations
for circuit_name, config in circuits.items():
    # Store used matrices for each circuit
    used_matrices = set()
    
    # Read the file once
    with open(config["asc_file_path"], 'r') as file:
        lines = file.readlines()
    
    # Ensure the specified line exists
    if config["line_number_to_replace"] >= len(lines):
        print(f"Error: Line {config['line_number_to_replace']} does not exist in the file for circuit {circuit_name}.")
        continue

    # Iterate over the unique matrices in scenarios
    for matrix_tuple in scenarios:
        matrix, name = matrix_tuple
        matrix_values = list(matrix)
        if len(matrix_values) != len(param_keys):
            print(f"Error: The number of matrix values does not match the number of parameter keys.")
            continue
        
        # Map matrix values to parameter keys
        params = {param_keys[i]: matrix_values[i] for i in range(len(param_keys))}
        
        # Modify the specified line
        old_line = lines[config["line_number_to_replace"]]
        # new_line = f'TEXT 80 400 Left 0 !.param Is=1e-10 Imax=0.989 Vmax=3.15 n=1.06*5 Rseries=0.25 Rshunt=150 ' + ' '.join([f'{k}={v}' for k, v in params.items()])
        new_line = 'TEXT 80 400 Left 0 !.param Vmax=3.36 ' + ' '.join([f'{k}={v}' for k, v in params.items()])
        lines[config["line_number_to_replace"]] = old_line.replace(old_line, new_line + '\n')
        
        # Add the matrix to the set of used matrices
        used_matrices.add(matrix_tuple)

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
    
