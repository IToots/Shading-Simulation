# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:09:50 2024

@author: ruiui
"""

import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt

# # Define the directory where your files are stored
directory = "C:/Users/ruiui/Desktop/iteration data/z_3x3/dvp_corr"
column_order = ['it','base_code','SP','SP_D','TCT','TCT_D','R2R3','R2R3_D','R1','R1_D','PERL','PERL_D']

# directory = "C:/Users/ruiui/Desktop/iteration data/z_4x4/dvp_corr"
# column_order = ['it','base_code','SP','SP_D','TCT','TCT_D','R2R3','R2R3_D','R1','R1_D','L','L_D','PER','PER_D']

# directory = "C:/Users/ruiui/Desktop/iteration data/z_5x5/dvp_corr"
# column_order = ['it','base_code','SP','SP_D','TCT','TCT_D','R3','R3_D','R2','R2_D','R1','R1_D','L','L_D','PER','PER_D']

# Find all CSV files in the directory
file_pattern = os.path.join(directory, '*.csv')
all_files = glob.glob(file_pattern)

# Filter files that have '_Z_' in their names
filtered_files = [file for file in all_files if '_Z_' in os.path.basename(file)]

# Create an empty list to store results
results = []

# Process each filtered file
for file in filtered_files:
    # Read the file into a DataFrame
    df = pd.read_csv(file)
    
    # Find the maximum value in column 'Power (P)'
    max_p = df['Power (P)'].max()
    
    # Get the filename without the directory path
    filename = os.path.basename(file)
    
    # Append the result to the list
    results.append({'Filename': filename, 'Max Power (P)': max_p})

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

df=results_df

# Extract prefix (e.g., PERL, R1) - only up to the first underscore
df['circ'] = df['Filename'].str.extract(r'^(?P<circuit>\w+)_data_iteration_')

# Extract middle part (e.g., Z_AA_0, Z_AB_0) - part after the `Z_` and before `.csv`
df['it'] = df['Filename'].str.extract(r'_(Z_\w+_\d+)\.csv')

# Extract last part (e.g., AA, AB) - part after `Z_` and before the next underscore
df['base_code'] = df['Filename'].str.extract(r'_Z_(\w+)_')

df=df.drop(columns='Filename')

# Pivot the DataFrame
pivot_df = df.pivot_table(index=['it', 'base_code'], columns='circ', values='Max Power (P)', aggfunc='mean')

# Reset index to flatten the DataFrame
a_df_ltspice = pivot_df.reset_index()

a_df_ltspice = a_df_ltspice[column_order]

# Save the final merged DataFrame to a CSV file
output_file_path = "C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/ltspice_3x3_results.csv"
a_df_ltspice.to_csv(output_file_path, index=False)

# Optionally, print a message confirming the save
print(f"DataFrame saved to {output_file_path}")
