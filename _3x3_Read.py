# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:50:29 2024

@author: ruiui
"""

import matplotlib.pyplot as plt
import numpy as np

import os
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from collections import defaultdict
from statistics import mean, median, stdev
import matplotlib.patches as patches

Vals = [(0.811, 'USD_AA_0'), (0.811, 'USD_AA_1'), (0.811, 'USD_AA_2'), (0.811, 'USD_AA_3'), 
        (0.667, 'USD_AB_0'), (0.667, 'USD_AB_1'), (0.667, 'USD_AB_2'), (0.667, 'USD_AB_3'), 
        (0.567, 'USD_AD_0'), (0.567, 'USD_AD_1'), (0.567, 'USD_AD_2'), (0.567, 'USD_AD_3'), 
        (0.311, 'USD_AF_0'), (0.311, 'USD_AF_1'), (0.311, 'USD_AF_2'), (0.311, 'USD_AF_3'), 
        (0.622, 'USD_AG_0'), (0.622, 'USD_AG_1'), (0.622, 'USD_AG_2'), (0.622, 'USD_AG_3'), 
        (0.522, 'USD_AH_0'), (0.522, 'USD_AH_1'), 
        (0.767, 'USD_AI_0'), (0.767, 'USD_AI_1'), (0.767, 'USD_AI_2'), (0.767, 'USD_AI_3'),
        (0.578, 'USD_AJ_0'), (0.578, 'USD_AJ_1'), (0.578, 'USD_AJ_2'), (0.578, 'USD_AJ_3'), 
        (0.467, 'USD_AL_0'), (0.467, 'USD_AL_1'), (0.467, 'USD_AL_2'), (0.467, 'USD_AL_3'), 
        (0.722, 'USD_AP_0'), (0.422, 'USD_AQ_0'),
        (0.667, 'USD_AS_0'), (0.667, 'USD_AS_1'), (0.667, 'USD_AS_2'), (0.667, 'USD_AS_3'), 
        (0.367, 'USD_AT_0'), (0.367, 'USD_AT_1'), (0.367, 'USD_AT_2'), (0.367, 'USD_AT_3'),
        (0.422, 'USD_AV_0'), (0.422, 'USD_AV_1'), (0.422, 'USD_AV_2'), (0.422, 'USD_AV_3'), 
        (0.622, 'USD_AX_0'), (0.622, 'USD_AX_1'), 
        (0.422, 'USD_BA_0'), (0.422, 'USD_BA_1'), (0.422, 'USD_BA_2'), (0.422, 'USD_BA_3'), 
        (0.367, 'USD_BB_0'), (0.367, 'USD_BB_1'), (0.367, 'USD_BB_2'), (0.367, 'USD_BB_3'), 
        (0.467, 'USD_BD_0'), (0.467, 'USD_BD_1'), (0.467, 'USD_BD_2'), (0.467, 'USD_BD_3'),
        
        (0.767, 'US_AA_0'), (0.767, 'US_AA_1'), (0.767, 'US_AA_2'), (0.767, 'US_AA_3'), 
        (0.622, 'US_AB_0'), (0.622, 'US_AB_1'), (0.622, 'US_AB_2'), (0.622, 'US_AB_3'), 
        (0.567, 'US_AC_0'), (0.567, 'US_AC_1'), (0.567, 'US_AC_2'), (0.567, 'US_AC_3'),
        (0.478, 'US_AD_0'), (0.478, 'US_AD_1'), (0.478, 'US_AD_2'), (0.478, 'US_AD_3'), 
        (0.367, 'US_AE_0'), (0.367, 'US_AE_1'), (0.367, 'US_AE_2'), (0.367, 'US_AE_3'), 
        (0.267, 'US_AF_0'), (0.267, 'US_AF_1'), (0.267, 'US_AF_2'), (0.267, 'US_AF_3'),
        (0.489, 'US_AG_0'), (0.489, 'US_AG_1'), (0.489, 'US_AG_2'), (0.489, 'US_AG_3'),
        (0.433, 'US_AH_0'), (0.433, 'US_AH_1'), 
        (0.678, 'US_AI_0'), (0.678, 'US_AI_1'), (0.678, 'US_AI_2'), (0.678, 'US_AI_3'),
        (0.489, 'US_AJ_0'), (0.489, 'US_AJ_1'), (0.489, 'US_AJ_2'), (0.489, 'US_AJ_3'),
        (0.433, 'US_AK_0'), (0.433, 'US_AK_1'), 
        (0.378, 'US_AL_0'), (0.378, 'US_AL_1'), (0.378, 'US_AL_2'), (0.378, 'US_AL_3'), 
        (0.322, 'US_AM_0'), 
        (0.378, 'US_AN_0'), (0.378, 'US_AN_1'), (0.378, 'US_AN_2'), (0.378, 'US_AN_3'), 
        (0.211, 'US_AO_0'), (0.211, 'US_AO_1'), (0.211, 'US_AO_2'), (0.211, 'US_AO_3'),
        (0.544, 'US_AP_0'), 
        (0.378, 'US_AQ_0'),
        (0.156, 'US_AR_0'),
        (0.622, 'US_AS_0'), (0.622, 'US_AS_1'), (0.622, 'US_AS_2'), (0.622, 'US_AS_3'), 
        (0.322, 'US_AT_0'), (0.322, 'US_AT_1'), (0.322, 'US_AT_2'), (0.322, 'US_AT_3'), 
        (0.211, 'US_AU_0'), (0.211, 'US_AU_1'), (0.211, 'US_AU_2'), (0.211, 'US_AU_3'), 
        (0.378, 'US_AV_0'), (0.378, 'US_AV_1'), (0.378, 'US_AV_2'), (0.378, 'US_AV_3'), 
        (0.267, 'US_AW_0'), (0.267, 'US_AW_1'), 
        (0.578, 'US_AX_0'), (0.578, 'US_AX_1'), 
        (0.378, 'US_AY_0'), (0.378, 'US_AY_1'),
        (0.322, 'US_AZ_0'), (0.322, 'US_AZ_1'),
        (0.378, 'US_BA_0'), (0.378, 'US_BA_1'), (0.378, 'US_BA_2'), (0.378, 'US_BA_3'), 
        (0.322, 'US_BB_0'), (0.322, 'US_BB_1'), (0.322, 'US_BB_2'), (0.322, 'US_BB_3'), 
        (0.267, 'US_BC_0'), (0.267, 'US_BC_1'), (0.267, 'US_BC_2'), (0.267, 'US_BC_3'), 
        (0.422, 'US_BD_0'), (0.422, 'US_BD_1'), (0.422, 'US_BD_2'), (0.422, 'US_BD_3'), 
        (0.378, 'US_BE_0'), (0.378, 'US_BE_1'), (0.378, 'US_BE_2'), (0.378, 'US_BE_3'), 
        
        (0.900, 'U_AA_0'), (0.900, 'U_AA_1'), (0.900, 'U_AA_2'), (0.900, 'U_AA_3'), 
        (0.800, 'U_AB_0'), (0.800, 'U_AB_1'), (0.800, 'U_AB_2'), (0.800, 'U_AB_3'), 
        (0.700, 'U_AC_0'), (0.700, 'U_AC_1'), (0.700, 'U_AC_2'), (0.700, 'U_AC_3'), 
        (0.700, 'U_AD_0'), (0.700, 'U_AD_1'), (0.700, 'U_AD_2'), (0.700, 'U_AD_3'), 
        (0.500, 'U_AE_0'), (0.500, 'U_AE_1'), (0.500, 'U_AE_2'), (0.500, 'U_AE_3'),
        (0.400, 'U_AF_0'), (0.400, 'U_AF_1'), (0.400, 'U_AF_2'), (0.400, 'U_AF_3'),
        (0.800, 'U_AG_0'), (0.800, 'U_AG_1'), (0.800, 'U_AG_2'), (0.800, 'U_AG_3'), 
        (0.700, 'U_AH_0'), (0.700, 'U_AH_1'),
        (0.900, 'U_AI_0'), (0.900, 'U_AI_1'), (0.900, 'U_AI_2'), (0.900, 'U_AI_3'), 
        (0.800, 'U_AJ_0'), (0.800, 'U_AJ_1'), (0.800, 'U_AJ_2'), (0.800, 'U_AJ_3'), 
        (0.700, 'U_AK_0'), (0.700, 'U_AK_1'),
        (0.600, 'U_AL_0'), (0.600, 'U_AL_1'), (0.600, 'U_AL_2'), (0.600, 'U_AL_3'), 
        (0.500, 'U_AM_0'), 
        (0.600, 'U_AN_0'), (0.600, 'U_AN_1'), (0.600, 'U_AN_2'), (0.600, 'U_AN_3'), 
        (0.300, 'U_AO_0'), (0.300, 'U_AO_1'), (0.300, 'U_AO_2'), (0.300, 'U_AO_3'), 
        (0.900, 'U_AP_0'),
        (0.600, 'U_AQ_0'), 
        (0.200, 'U_AR_0'), 
        (0.800, 'U_AS_0'), (0.800, 'U_AS_1'), (0.800, 'U_AS_2'), (0.800, 'U_AS_3'),
        (0.500, 'U_AT_0'), (0.500, 'U_AT_1'), (0.500, 'U_AT_2'), (0.500, 'U_AT_3'),
        (0.300, 'U_AU_0'), (0.300, 'U_AU_1'), (0.300, 'U_AU_2'), (0.300, 'U_AU_3'),
        (0.600, 'U_AV_0'), (0.600, 'U_AV_1'), (0.600, 'U_AV_2'), (0.600, 'U_AV_3'),
        (0.400, 'U_AW_0'), (0.400, 'U_AW_1'),
        (0.800, 'U_AX_0'), (0.800, 'U_AX_1'),
        (0.600, 'U_AY_0'), (0.600, 'U_AY_1'),
        (0.500, 'U_AZ_0'), (0.500, 'U_AZ_1'), 
        (0.600, 'U_BA_0'), (0.600, 'U_BA_1'), (0.600, 'U_BA_2'), (0.600, 'U_BA_3'), 
        (0.500, 'U_BB_0'), (0.500, 'U_BB_1'), (0.500, 'U_BB_2'), (0.500, 'U_BB_3'), 
        (0.400, 'U_BC_0'), (0.400, 'U_BC_1'), (0.400, 'U_BC_2'), (0.400, 'U_BC_3'), 
        (0.600, 'U_BD_0'), (0.600, 'U_BD_1'), (0.600, 'U_BD_2'), (0.600, 'U_BD_3'), 
        (0.600, 'U_BE_0'), (0.600, 'U_BE_1'), (0.600, 'U_BE_2'), (0.600, 'U_BE_3'), 
        
        (0.889, 'Z_AA_0'), (0.889, 'Z_AA_1'), (0.889, 'Z_AA_2'), (0.889, 'Z_AA_3'), 
        (0.778, 'Z_AB_0'), (0.778, 'Z_AB_1'), (0.778, 'Z_AB_2'), (0.778, 'Z_AB_3'), 
        (0.667, 'Z_AC_0'), (0.667, 'Z_AC_1'), (0.667, 'Z_AC_2'), (0.667, 'Z_AC_3'), 
        (0.667, 'Z_AD_0'), (0.667, 'Z_AD_1'), (0.667, 'Z_AD_2'), (0.667, 'Z_AD_3'), 
        (0.444, 'Z_AE_0'), (0.444, 'Z_AE_1'), (0.444, 'Z_AE_2'), (0.444, 'Z_AE_3'), 
        (0.333, 'Z_AF_0'), (0.333, 'Z_AF_1'), (0.333, 'Z_AF_2'), (0.333, 'Z_AF_3'), 
        (0.778, 'Z_AG_0'), (0.778, 'Z_AG_1'), (0.778, 'Z_AG_2'), (0.778, 'Z_AG_3'), 
        (0.667, 'Z_AH_0'), (0.667, 'Z_AH_1'),
        (0.889, 'Z_AI_0'), (0.889, 'Z_AI_1'), (0.889, 'Z_AI_2'), (0.889, 'Z_AI_3'), 
        (0.778, 'Z_AJ_0'), (0.778, 'Z_AJ_1'), (0.778, 'Z_AJ_2'), (0.778, 'Z_AJ_3'), 
        (0.667, 'Z_AK_0'), (0.667, 'Z_AK_1'), 
        (0.556, 'Z_AL_0'), (0.556, 'Z_AL_1'), (0.556, 'Z_AL_2'), (0.556, 'Z_AL_3'), 
        (0.444, 'Z_AM_0'),
        (0.556, 'Z_AN_0'), (0.556, 'Z_AN_1'), (0.556, 'Z_AN_2'), (0.556, 'Z_AN_3'),
        (0.222, 'Z_AO_0'), (0.222, 'Z_AO_1'), (0.222, 'Z_AO_2'), (0.222, 'Z_AO_3'),
        (0.889, 'Z_AP_0'), 
        (0.556, 'Z_AQ_0'), 
        (0.111, 'Z_AR_0'), 
        (0.778, 'Z_AS_0'), (0.778, 'Z_AS_1'), (0.778, 'Z_AS_2'), (0.778, 'Z_AS_3'), 
        (0.444, 'Z_AT_0'), (0.444, 'Z_AT_1'), (0.444, 'Z_AT_2'), (0.444, 'Z_AT_3'), 
        (0.222, 'Z_AU_0'), (0.222, 'Z_AU_1'), (0.222, 'Z_AU_2'), (0.222, 'Z_AU_3'), 
        (0.556, 'Z_AV_0'), (0.556, 'Z_AV_1'), (0.556, 'Z_AV_2'), (0.556, 'Z_AV_3'), 
        (0.333, 'Z_AW_0'), (0.333, 'Z_AW_1'),
        (0.778, 'Z_AX_0'), (0.778, 'Z_AX_1'), 
        (0.556, 'Z_AY_0'), (0.556, 'Z_AY_1'), 
        (0.444, 'Z_AZ_0'), (0.444, 'Z_AZ_1'), 
        (0.556, 'Z_BA_0'), (0.556, 'Z_BA_1'), (0.556, 'Z_BA_2'), (0.556, 'Z_BA_3'), 
        (0.444, 'Z_BB_0'), (0.444, 'Z_BB_1'), (0.444, 'Z_BB_2'), (0.444, 'Z_BB_3'), 
        (0.333, 'Z_BC_0'), (0.333, 'Z_BC_1'), (0.333, 'Z_BC_2'), (0.333, 'Z_BC_3'), 
        (0.556, 'Z_BD_0'), (0.556, 'Z_BD_1'), (0.556, 'Z_BD_2'), (0.556, 'Z_BD_3'), 
        (0.556, 'Z_BE_0'), (0.556, 'Z_BE_1'), (0.556, 'Z_BE_2'), (0.556, 'Z_BE_3')
        ]

W = 12.5*12.5* 9 * 0.1

csv_directory = "C:/Users/ruiui/Desktop/iteration data/z_3x3"
circuit_irr_directory = "C:/Users/ruiui/Desktop/iteration data/_Irr_teste/3x3"

t1 = defaultdict(list)
percentage_diffs_pos = defaultdict(list)
percentage_diffs_val_mpp = defaultdict(list)
percentage_diffs = defaultdict(list)
mpp_sums = defaultdict(float)

# Adjust the values by multiplying by 1000 and converting to integers
adjusted_vals = {name: int(val * 1000) for val, name in Vals}

# Load CSV for each circuit
circuit_dfs = {}
for circuit in ['PERL_D', 'PERL', 'R1_D', 'R1', 'R2R3_D', 'R2R3', 'TCT_D', 'TCT', 'SP_D', 'SP']:
    circuit_file_path = os.path.join(circuit_irr_directory, f'z_3x3_irr_{circuit}.csv')
    circuit_dfs[circuit] = pd.read_csv(circuit_file_path)

# Function to get Max Power (P) for a given number and circuit
def get_max_power_for_number(circuit, number):
    df = circuit_dfs.get(circuit)
    if df is not None:
        row = df[df['Irr'] == number]
        if len(row) == 1:
            return row.iloc[0]['MPP']
    return None

# Create a dictionary with the Max Power (P) values for each circuit
max_power_dict = defaultdict(dict)
for name, number in adjusted_vals.items():
    for circuit in circuit_dfs.keys():
        max_power = get_max_power_for_number(circuit, number)
        if max_power is not None:
            max_power_dict[circuit][name] = max_power
        else:
            print(f"Warning: No unique match found for {name} with iteration {number} in circuit {circuit}")

# Read CSV file function
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path).values.tolist()
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# Extract parameters from CSV data
def extract_parameters(data):
    try:
        V = [row[0] for row in data]
        I = [row[1] for row in data]
        P = [row[2] for row in data]
        Voc = next(V[i] for i in range(len(I)) if I[i] < 0.0005)
        Isc = max(I)
        MPP_index = P.index(max(P))
        Vmp = V[MPP_index]
        Imp = I[MPP_index]
        MPP = max(P)
        FF = (MPP / (Isc * Voc)) * 100
        Ef = (MPP / W) * 100
        return Voc, Isc, Vmp, Imp, MPP, FF, Ef
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return None, None, None, None, None, None, None

# Perform calculations for each CSV file grouped by iteration
def calculations(files_by_iteration):
    for iteration, csv_files in files_by_iteration.items():
        for idx, csv_file in enumerate(csv_files):
            data = read_csv_file(csv_file)
            if not data:
                continue
            circuit_name = os.path.basename(csv_file).split('_data_iteration_')[0]
            params = extract_parameters(data)
            if None in params:
                continue
            Voc, Isc, Vmp, Imp, MPP, FF, Ef = params
            reference_mpp = max_power_dict[circuit_name].get(iteration, None)
            if reference_mpp is None:
                print(f"Warning: No reference MPP found for {circuit_name}")
                continue
            # percentage_diff = ((MPP - reference_mpp) / reference_mpp)
            percentage_diff = MPP - reference_mpp
            percentage_diffs[circuit_name].append(percentage_diff)
            mpp_sums[circuit_name] += percentage_diff
            t1[circuit_name].append((MPP, reference_mpp))
            if 0 < percentage_diff:
                percentage_diffs_pos[circuit_name].append((percentage_diff, iteration, circuit_name))
            if 0 < percentage_diff < 0.05:
                percentage_diff = 0
            percentage_diffs_val_mpp[circuit_name].append((percentage_diff, iteration, circuit_name))

# Create a dictionary to map circuit names to iterations
files_by_iteration = defaultdict(list)

# List CSV files
csv_files = [os.path.join(csv_directory, filename) for filename in os.listdir(csv_directory) if filename.endswith(".csv")]

# Extract circuit names and iterations from file names
for file in csv_files:
    file_name = os.path.basename(file)
    iteration_name = file_name.split('_data_iteration_')[1]
    iteration = iteration_name.split('.')[0]
    files_by_iteration[iteration].append(file)

# Perform calculations
calculations(files_by_iteration)

# Convert percentage_diffs_val_mpp to DataFrame
data_for_df = [
    {
        'Circuit': circuit,
        'Iteration': iteration,
        'PercentageDiff': value
    } for circuit_name, values in percentage_diffs_val_mpp.items() for value, iteration, circuit in values
]

percentage_diffs_df = pd.DataFrame(data_for_df)

# Plotting results
styles = {
    'PERL_D': {'color': '#80C080', 'marker': '|', 'phase': -0.3, 'size': 2},
    'PERL': {'color': '#008000', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'R1_D': {'color': '#FFD580', 'marker': '+', 'phase': 0.1, 'size': 2},
    'R1': {'color': '#FFA500', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'R2R3_D': {'color': '#C080C0', 'marker': '+', 'phase': 0.1, 'size': 2},
    'R2R3': {'color': '#800080', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'TCT_D': {'color': '#8080FF', 'marker': 'o', 'phase': -0.2, 'size': 2},
    'TCT': {'color': '#0000FF', 'marker': 's', 'phase': -0.1, 'size': 2},
    
    'SP_D': {'color': '#808080', 'marker': '_', 'phase': 0.3, 'size': 2},
    'SP': {'color': '#000000', 'marker': 'D', 'phase': 0.2, 'size': 2}
}

# Convert Vals to a dictionary
Vals_dict = {value: key for key, value in Vals}

# Map the iteration values to the DataFrame
percentage_diffs_df['IterationValue'] = percentage_diffs_df['Iteration'].map(Vals_dict)

# Create custom order list from dictionary keys
iteration_order = list(Vals_dict.keys())

# Add a temporary column for the custom iteration order
percentage_diffs_df['IterationOrder'] = percentage_diffs_df['Iteration'].apply(lambda x: iteration_order.index(x) if x in iteration_order else -1)

# Reset the index
percentage_diffs_df.reset_index(drop=True, inplace=True)

# Drop the temporary column
percentage_diffs_df.drop(columns=['IterationOrder'], inplace=True)

# Extract the iteration type (C_A, C_B, etc.)
percentage_diffs_df['IterationType'] = percentage_diffs_df['Iteration'].str.extract(r'([A-Z]+_[A-Z]+)_')

# Sort the DataFrame first by IterationValue in descending order, then by the custom IterationOrder
percentage_diffs_df = percentage_diffs_df.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

median_values = percentage_diffs_df.groupby(['Circuit', 'IterationType']).agg({
    'PercentageDiff': 'median',
    'IterationValue': 'first'
}).reset_index()

median_values = median_values.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

max_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmax()

result_df_max = percentage_diffs_df.loc[max_percentages].reset_index(drop=True)
result_df_max = result_df_max.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

min_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmin()

result_df_min = percentage_diffs_df.loc[min_percentages].reset_index(drop=True)
result_df_min = result_df_min.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

wiwu_U = result_df_min[result_df_min['Iteration'].str.startswith('U')]
wiwu_Z = result_df_min[result_df_min['Iteration'].str.startswith('Z')]

data = defaultdict(list)# Convert Vals to a dictionary
Vals_dict = {value: key for key, value in Vals}

# Map the iteration values to the DataFrame
percentage_diffs_df['IterationValue'] = percentage_diffs_df['Iteration'].map(Vals_dict)

# Create custom order list from dictionary keys
iteration_order = list(Vals_dict.keys())

# Add a temporary column for the custom iteration order
percentage_diffs_df['IterationOrder'] = percentage_diffs_df['Iteration'].apply(lambda x: iteration_order.index(x) if x in iteration_order else -1)

# Reset the index
percentage_diffs_df.reset_index(drop=True, inplace=True)

# Drop the temporary column
percentage_diffs_df.drop(columns=['IterationOrder'], inplace=True)

# Extract the iteration type (C_A, C_B, etc.)
percentage_diffs_df['IterationType'] = percentage_diffs_df['Iteration'].str.extract(r'([A-Z]+_[A-Z]+)_')

# Sort the DataFrame first by IterationValue in descending order, then by the custom IterationOrder
percentage_diffs_df = percentage_diffs_df.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

median_values = percentage_diffs_df.groupby(['Circuit', 'IterationType']).agg({
    'PercentageDiff': 'median',
    'IterationValue': 'first'
}).reset_index()

median_values = median_values.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

max_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmax()

result_df_max = percentage_diffs_df.loc[max_percentages].reset_index(drop=True)
result_df_max = result_df_max.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

min_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmin()

result_df_min = percentage_diffs_df.loc[min_percentages].reset_index(drop=True)
result_df_min = result_df_min.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

wiwu_U = result_df_min[result_df_min['Iteration'].str.startswith('U')]
wiwu_Z = result_df_min[result_df_min['Iteration'].str.startswith('Z')]


T_df_max_TCT = result_df_max[result_df_max['Iteration'].str.startswith('Z')]
T_df_min_TCT = result_df_min[result_df_min['Iteration'].str.startswith('Z')]
T_median_values_TCT =  median_values[median_values['IterationType'].str.startswith('Z')]

T_circuit_data_max_TCT = T_df_max_TCT[T_df_max_TCT['Circuit'] == 'TCT']
T_circuit_data_min_TCT = T_df_min_TCT[T_df_min_TCT['Circuit'] == 'TCT']

T_circuit_data_median_TCT = T_median_values_TCT[T_median_values_TCT['Circuit'] == 'TCT']
T_diffs_median_TCT = T_circuit_data_median_TCT['PercentageDiff']

data = defaultdict(list)

def plot_iterations_cumulative_2(df_max1, df_min1, styles, selected_circuits, it):
    # Filter the DataFrame based on the 'Iteration' column values starting with 'it'
    df_max = df_max1[df_max1['Iteration'].str.startswith(it)]
    df_min = df_min1[df_min1['Iteration'].str.startswith(it)]
    median_values1 =  median_values[median_values['IterationType'].str.startswith(it)]
    
    if df_max.empty or df_min.empty:
        return
    
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot each circuit separately with provided styles
    for circuit in selected_circuits:
        if circuit in styles:  # Ensure the circuit is in styles
            size = styles[circuit]['size']
            circuit_data_max = df_max[df_max['Circuit'] == circuit]
            circuit_data_min = df_min[df_min['Circuit'] == circuit]
            
            if circuit_data_max.empty or circuit_data_min.empty:
                print(f"No data for circuit {circuit}")
                continue
            
            iterations_max = circuit_data_max['Iteration']
            diffs_max = circuit_data_max['PercentageDiff']
            iterations_min = circuit_data_min['Iteration']
            diffs_min = circuit_data_min['PercentageDiff']
            
            # Calculate cumulative differences
            cumulative_diff_max = diffs_max.cumsum().tolist()
            cumulative_diff_min = diffs_min.cumsum().tolist()
            
            circuit_data_median = median_values1[median_values1['Circuit'] == circuit]
            diffs_median = circuit_data_median['PercentageDiff']
            cumulative_diff_median = diffs_median.cumsum().tolist()
                
            # Truncate the longer list to match the length of the shorter one
            min_len = min(len(cumulative_diff_max), len(cumulative_diff_min))
            cumulative_diff_max = cumulative_diff_max[:min_len]
            cumulative_diff_min = cumulative_diff_min[:min_len]
            
            if len(cumulative_diff_max) != len(cumulative_diff_min):
                print(f"Truncated cumulative differences for circuit {circuit}")
            
            # Print cumulative differences
            print(f"Circuit: {circuit}")
            print(f"Cumulative Max: {round(cumulative_diff_max[-1],3)}")
            print(f"Cumulative Min: {round(cumulative_diff_min[-1],3)}")
            print(f"Cumulative Median: {round(cumulative_diff_median[-1],3)}\n")
            
            cummax = cumulative_diff_max[-1] / len(cumulative_diff_max)
            cummin = cumulative_diff_min[-1] / len(cumulative_diff_min)
            cummedian = cumulative_diff_median[-1] / len(cumulative_diff_median)
            
            data[circuit].append((cummax, cummin, cummedian))
            
            # Plot cumulative differences
            line_max, = ax.plot(range(len(cumulative_diff_max)), cumulative_diff_max, 
                                color=styles[circuit]['color'], 
                                label=f"{circuit} max", 
                                linewidth=size)
            
            line_min, = ax.plot(range(len(cumulative_diff_min)), cumulative_diff_min, 
                                color=styles[circuit]['color'],
                                label=f"{circuit} min", 
                                linewidth=size)
           # Fill between max and min lines
            ax.fill_between(range(len(cumulative_diff_min)), cumulative_diff_max, cumulative_diff_min, 
                            color=styles[circuit]['color'], alpha=0.1)
            
            
            line_median, = ax.plot(range(len(cumulative_diff_median)), cumulative_diff_median, 
                                color='black',
                                label=f"{circuit} median", 
                                linewidth=size,
                                linestyle='--')
            
            # Determine the position for text alignment
            x_position = (len(cumulative_diff_median))*1.02
            y_position = cumulative_diff_median[-1]
            
            # Calculate text box dimensions
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
            
            # Add text to the plot aligned with the line
            ax.text(x_position, y_position, f"Circuit: {circuit}",
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transData,
                    fontsize=12, bbox=bbox_props)
            
    if it == 'Z':
        line_positions = [2.5, 7.5, 11.5, 19.5, 24.5, 27.5, 29.5]
        for pos in line_positions:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
    
    # Determine the range of values for x-axis labels
    max_iteration = df_max['IterationValue'].astype(float).max()
    min_iteration = df_max['IterationValue'].astype(float).min()
    
    # Generate 10 evenly spaced labels between min and max iteration values
    x_labels = np.linspace(max_iteration, min_iteration, num=10)
    x_pos = np.linspace(0, len(circuit_data_max)-1, num=10)
    
    # Set x ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{label:.2f}' for label in x_labels])
    
    ax.set_xlabel('Irradiation')
    ax.set_ylabel('Cumulative % Difference')
    ax.set_title(f'Cumulative Percentage Difference 3x3 for Iterations {it}')
    plt.legend()
    plt.tight_layout()
    plt.show()


selected_circuits = [
# 'PERL_D',
# 'PERL',
# 'R1_D',
# 'R1',
# 'R2R3_D',
# 'R2R3',
# 'TCT_D',
'TCT',
# 'SP_D',
# 'SP'
]

# Possible iterations: 
plot_iterations_cumulative_2(result_df_max, result_df_min, styles, selected_circuits,
# 'U_'
# 'US_'
# 'USD_' 
# 'U'
'Z'
)


# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Define the output folder
output_folder = "C:/Users/ruiui/Desktop/TABELAS/3"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_folder, 'Ud.csv')

# Save DataFrame to CSV
# df.to_csv(output_file, index=False)