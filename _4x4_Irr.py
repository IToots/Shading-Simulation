# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:34:02 2024

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
    # transformations = [(i / 1000, str(i)) for i in range(1, 1001)]
    transformations = [(i / 1000, str(i)) for i in 
                       [1,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]]
    
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
    "Irr": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
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

param_keys = ['a', 'b', 'c', 'd', 
              'e', 'f', 'g', 'h',
              'i', 'j', 'k', 'l', 
              'm', 'n', 'o', 'p']

# Define the folder where files will be saved
folder_path_output = "C:/Users/ruiui/Desktop/iteration data/z_4x4_irr"

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
    # "TCT": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/4x4/VIgraph.raw",
    #     "data_file_prefix": "TCT_data_iteration_",
    #     "line_number_to_replace": 780
    # },
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
}

# Iterate over the circuit configurations
for circuit_name, config in circuits.items():
    # Store used matrices for each circuit
    used_matrices = set()
    
    # Read the file once
    with open(config["asc_file_path"], 'r') as file:
        lines = file.readlines()

    # Iterate over the unique matrices in scenarios
    for matrix_tuple in scenarios:
        matrix, name = matrix_tuple
        matrix_values = list(matrix)
        if len(matrix_values) != len(param_keys):
            print(f"Error: The number of matrix values does not match the number of parameter keys.")
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
            new_line = f"TEXT 80 400 Left 0 !.param Is=1e-10 Imax=0.989 Vmax=3.15 n=1.06*5 Rseries=0.25 Rshunt=150 " + ' '.join([f"{k}={v}" for k, v in params.items()])
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