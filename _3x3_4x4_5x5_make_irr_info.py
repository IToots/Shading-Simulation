# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:56:24 2024

@author: ruiui
"""

import os
import pandas as pd
from collections import defaultdict

# Path to the folder
folder_path = "C:/Users/ruiui/Desktop/iteration data/z_3x3_irr/SP_D/mod_corr"

# Specify the file path to save the sorted results
output_file = "z_3x3_irr_SP_D.csv"

# List all files in the folder
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def read_csv_file(file_path):
    """Read CSV file and convert its contents to float."""
    try:
        data = pd.read_csv(file_path).values.tolist()
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def extract_parameters(files_by_iteration):
    """Perform calculations for each CSV file grouped by iteration."""
    results = []
    for iteration, csv_files in files_by_iteration.items():
        for csv_file in csv_files:
            data = read_csv_file(csv_file)
            if not data:
                continue
            
            V = [row[0] for row in data]
            I = [row[1] for row in data]
            P = [row[2] for row in data]
            
            try:
                MPP_index = P.index(max(P))
                Vmp = V[MPP_index]
                Imp = I[MPP_index]
                MPP = max(P)
                results.append((iteration, Vmp, Imp, MPP))
            except Exception as e:
                print(f"Error extracting parameters from {csv_file}: {e}")
    
    return results

# Create a dictionary to map circuit names to iterations
files_by_iteration = defaultdict(list)

# Extract circuit names and iterations from file names
for file in files:
    file_name = os.path.basename(file)
    iteration_name = file_name.split('_data_iteration_Irr_')[1].split('.')[0]
    files_by_iteration[iteration_name].append(file)

# Perform calculations
calculation_results = extract_parameters(files_by_iteration)

# Sort results by MPP (index 3 in the tuple) from lowest to highest
sorted_results = sorted(calculation_results, key=lambda x: x[3])

# Convert sorted results to a DataFrame
columns = ["Irr", "Vmp", "Imp", "MPP"]
df = pd.DataFrame(sorted_results, columns=columns)

# Save DataFrame to CSV
df.to_csv(output_file, index=False)

print(f"Sorted results saved to {output_file}")

