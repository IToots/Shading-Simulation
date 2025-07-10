# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:57:10 2024

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
    "Irr": np.array([[1, 1, 1, 1, 1], 
                     [1, 1, 1, 1, 1], 
                     [1, 1, 1, 1, 1], 
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])
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

param_keys = ['a', 'b', 'c', 'd', 'e', 
              'f', 'g', 'h', 'i', 'j', 
              'k', 'l', 'm', 'n', 'o', 
              'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y']

# Define the folder where files will be saved
folder_path_output = "C:/Users/ruiui/Desktop/iteration data/z_5x5_irr"

# Define variables for file paths and other parameters
ltspice_executable = "C:/Users/ruiui/AppData/Local/Programs/ADI/LTspice/LTspice.exe"


circuits = {
    # "SP": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/VIgraph.raw",
    #     "data_file_prefix": "SP_data_iteration_",
    #     "line_number_to_replace": 1217
    # },
    # "TCT": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/VIgraph.raw",
    #     "data_file_prefix": "TCT_data_iteration_",
    #     "line_number_to_replace": 1217
    # },
    # "R1": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/VIgraph.raw",
    #     "data_file_prefix": "R1_data_iteration_",
    #     "line_number_to_replace": 1280
    # },
    # "R2": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/VIgraph.raw",
    #     "data_file_prefix": "R2_data_iteration_",
    #     "line_number_to_replace": 1277
    # },
    # "R3": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/VIgraph.raw",
    #     "data_file_prefix": "R3_data_iteration_",
    #     "line_number_to_replace": 1296
    # },
    # "PER": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/VIgraph.raw",
    #     "data_file_prefix": "PER_data_iteration_",
    #     "line_number_to_replace": 1343
    # },
    # "L": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/VIgraph.raw",
    #     "data_file_prefix": "L_data_iteration_",
    #     "line_number_to_replace": 1353
    # },
    
    # "SP_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_SP/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "SP_D_data_iteration_",
    #     "line_number_to_replace_a": 1334,
    #     "line_number_to_replace_b": 1070,
    #     "line_number_to_replace_c": 830,
    #     "line_number_to_replace_d": 804,
    #     "line_number_to_replace_e": 778,
    #     "line_number_to_replace_f": 1308,
    #     "line_number_to_replace_g": 1044,
    #     "line_number_to_replace_h": 752,
    #     "line_number_to_replace_i": 726,
    #     "line_number_to_replace_j": 700,
    #     "line_number_to_replace_k": 1282,
    #     "line_number_to_replace_l": 1018,
    #     "line_number_to_replace_m": 674,
    #     "line_number_to_replace_n": 648,
    #     "line_number_to_replace_o": 622,
    #     "line_number_to_replace_p": 1381,
    #     "line_number_to_replace_q": 1117,
    #     "line_number_to_replace_r": 971,
    #     "line_number_to_replace_s": 945,
    #     "line_number_to_replace_t": 919,
    #     "line_number_to_replace_u": 1414,
    #     "line_number_to_replace_v": 1249,
    #     "line_number_to_replace_w": 1202,
    #     "line_number_to_replace_x": 1176,
    #     "line_number_to_replace_y": 1150
    # },
    "TCT_D": {
        "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/DIODE/PVcell.asc",
        "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/DIODE/PVcell.asc",
        "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/DIODE/VIgraph.asc",
        "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_TCT/5x5/DIODE/VIgraph.raw",
        "data_file_prefix": "TCT_D_data_iteration_",
        "line_number_to_replace_a": 1366,
        "line_number_to_replace_b": 1102,
        "line_number_to_replace_c": 862,
        "line_number_to_replace_d": 836,
        "line_number_to_replace_e": 810,
        "line_number_to_replace_f": 1340,
        "line_number_to_replace_g": 1076,
        "line_number_to_replace_h": 784,
        "line_number_to_replace_i": 758,
        "line_number_to_replace_j": 732,
        "line_number_to_replace_k": 1314,
        "line_number_to_replace_l": 1050,
        "line_number_to_replace_m": 706,
        "line_number_to_replace_n": 680,
        "line_number_to_replace_o": 654,
        "line_number_to_replace_p": 1413,
        "line_number_to_replace_q": 1149,
        "line_number_to_replace_r": 1003,
        "line_number_to_replace_s": 977,
        "line_number_to_replace_t": 951,
        "line_number_to_replace_u": 1446,
        "line_number_to_replace_v": 1281,
        "line_number_to_replace_w": 1234,
        "line_number_to_replace_x": 1208,
        "line_number_to_replace_y": 1182
    },
    # "R1_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R1/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R1_D_data_iteration_",
    #     "line_number_to_replace_a": 1429,
    #     "line_number_to_replace_b": 1165,
    #     "line_number_to_replace_c": 925,
    #     "line_number_to_replace_d": 899,
    #     "line_number_to_replace_e": 873,
    #     "line_number_to_replace_f": 1403,
    #     "line_number_to_replace_g": 1139,
    #     "line_number_to_replace_h": 847,
    #     "line_number_to_replace_i": 821,
    #     "line_number_to_replace_j": 795,
    #     "line_number_to_replace_k": 1377,
    #     "line_number_to_replace_l": 1113,
    #     "line_number_to_replace_m": 769,
    #     "line_number_to_replace_n": 743,
    #     "line_number_to_replace_o": 717,
    #     "line_number_to_replace_p": 1462,
    #     "line_number_to_replace_q": 1198,
    #     "line_number_to_replace_r": 1010,
    #     "line_number_to_replace_s": 984,
    #     "line_number_to_replace_t": 958,
    #     "line_number_to_replace_u": 1509,
    #     "line_number_to_replace_v": 1344,
    #     "line_number_to_replace_w": 1297,
    #     "line_number_to_replace_x": 1271,
    #     "line_number_to_replace_y": 1245
    # },
    # "R2_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R2/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R2_D_data_iteration_",
    #     "line_number_to_replace_a": 1426,
    #     "line_number_to_replace_b": 1162,
    #     "line_number_to_replace_c": 922,
    #     "line_number_to_replace_d": 896,
    #     "line_number_to_replace_e": 870,
    #     "line_number_to_replace_f": 1400,
    #     "line_number_to_replace_g": 1136,
    #     "line_number_to_replace_h": 844,
    #     "line_number_to_replace_i": 818,
    #     "line_number_to_replace_j": 792,
    #     "line_number_to_replace_k": 1374,
    #     "line_number_to_replace_l": 1110,
    #     "line_number_to_replace_m": 766,
    #     "line_number_to_replace_n": 740,
    #     "line_number_to_replace_o": 714,
    #     "line_number_to_replace_p": 1473,
    #     "line_number_to_replace_q": 1209,
    #     "line_number_to_replace_r": 1063,
    #     "line_number_to_replace_s": 1037,
    #     "line_number_to_replace_t": 1011,
    #     "line_number_to_replace_u": 1506,
    #     "line_number_to_replace_v": 1341,
    #     "line_number_to_replace_w": 1294,
    #     "line_number_to_replace_x": 1268,
    #     "line_number_to_replace_y": 1242
    # },
    # "R3_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_R3/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "R3_D_data_iteration_",
    #     "line_number_to_replace_a": 1445,
    #     "line_number_to_replace_b": 1181,
    #     "line_number_to_replace_c": 941,
    #     "line_number_to_replace_d": 915,
    #     "line_number_to_replace_e": 889,
    #     "line_number_to_replace_f": 1419,
    #     "line_number_to_replace_g": 1155,
    #     "line_number_to_replace_h": 863,
    #     "line_number_to_replace_i": 837,
    #     "line_number_to_replace_j": 811,
    #     "line_number_to_replace_k": 1393,
    #     "line_number_to_replace_l": 1129,
    #     "line_number_to_replace_m": 785,
    #     "line_number_to_replace_n": 759,
    #     "line_number_to_replace_o": 733,
    #     "line_number_to_replace_p": 1492,
    #     "line_number_to_replace_q": 1228,
    #     "line_number_to_replace_r": 1082,
    #     "line_number_to_replace_s": 1056,
    #     "line_number_to_replace_t": 1030,
    #     "line_number_to_replace_u": 1525,
    #     "line_number_to_replace_v": 1360,
    #     "line_number_to_replace_w": 1313,
    #     "line_number_to_replace_x": 1287,
    #     "line_number_to_replace_y": 1261
    # },
    # "PER_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_PER/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "PER_D_data_iteration_",
    #     "line_number_to_replace_a": 1466,
    #     "line_number_to_replace_b": 911,
    #     "line_number_to_replace_c": 886,
    #     "line_number_to_replace_d": 860,
    #     "line_number_to_replace_e": 1203,
    #     "line_number_to_replace_f": 1441,
    #     "line_number_to_replace_g": 834,
    #     "line_number_to_replace_h": 808,
    #     "line_number_to_replace_i": 782,
    #     "line_number_to_replace_j": 1177,
    #     "line_number_to_replace_k": 1539,
    #     "line_number_to_replace_l": 1130,
    #     "line_number_to_replace_m": 1104,
    #     "line_number_to_replace_n": 1078,
    #     "line_number_to_replace_o": 1276,
    #     "line_number_to_replace_p": 1506,
    #     "line_number_to_replace_q": 1031,
    #     "line_number_to_replace_r": 1005,
    #     "line_number_to_replace_s": 979,
    #     "line_number_to_replace_t": 1243,
    #     "line_number_to_replace_u": 1572,
    #     "line_number_to_replace_v": 1361,
    #     "line_number_to_replace_w": 1335,
    #     "line_number_to_replace_x": 1309,
    #     "line_number_to_replace_y": 1408
    # },
    # "L_D": {
    #     "asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/DIODE/PVcell.asc",
    #     "output_asc_file_path": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/DIODE/PVcell.asc",
    #     "ltspice_run_file": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/DIODE/VIgraph.asc",
    #     "ltspice_raw_file": "C:/Users/ruiui/Desktop/LTspice/_L/5x5/DIODE/VIgraph.raw",
    #     "data_file_prefix": "L_D_data_iteration_",
    #     "line_number_to_replace_a": 1362,
    #     "line_number_to_replace_b": 1154,
    #     "line_number_to_replace_c": 998,
    #     "line_number_to_replace_d": 972,
    #     "line_number_to_replace_e": 946,
    #     "line_number_to_replace_f": 1336,
    #     "line_number_to_replace_g": 1128,
    #     "line_number_to_replace_h": 920,
    #     "line_number_to_replace_i": 894,
    #     "line_number_to_replace_j": 868,
    #     "line_number_to_replace_k": 1310,
    #     "line_number_to_replace_l": 1102,
    #     "line_number_to_replace_m": 842,
    #     "line_number_to_replace_n": 816,
    #     "line_number_to_replace_o": 790,
    #     "line_number_to_replace_p": 1388,
    #     "line_number_to_replace_q": 1180,
    #     "line_number_to_replace_r": 1076,
    #     "line_number_to_replace_s": 1050,
    #     "line_number_to_replace_t": 1024,
    #     "line_number_to_replace_u": 1414,
    #     "line_number_to_replace_v": 1284,
    #     "line_number_to_replace_w": 1258,
    #     "line_number_to_replace_x": 1232,
    #     "line_number_to_replace_y": 1206
    # }    
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
    