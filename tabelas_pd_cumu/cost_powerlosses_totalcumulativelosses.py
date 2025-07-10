# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:07:11 2025

@author: ruiui
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns

comps_3x3 = {
    'SP': 80, 
    'TCT': 120, 
    'ST': 161, 
    'DG': 168, 
    'K': 202}
res_3x3 = {
    'SP': 12.500, 
    'TCT': 12.500, 
    'ST': 19.631, 
    'DG': 18.499, 
    'K': 18.948}

comps_4x4 = {
    'SP': 180, 
    'TCT': 240, 
    'ST': 318,
    'DG': 332, 
    'KH': 464,
    'KV': 484}
res_4x4 = {
    'SP': 16.549, 
    'TCT': 16.161,
    'ST': 24.428, 
    'DG': 25.781, 
    'KH': 24.643,
    'KV': 29.290}

comps_5x5 = {
    'SP': 280, 
    'TCT': 400,
    'ST': 620, 
    'DG': 546, 
    'KH': 948,
    'KV': 940}
res_5x5 = {
    'SP': 20.059, 
    'TCT': 19.010,
    'ST': 31.586, 
    'DG': 27.460, 
    'KH': 33.610,
    'KV': 33.610}

bushbar = 0.3616
diode = 0.531
panel = 3.51

def losses(L, I):
    L = L * 10e-2
    # width (m)
    W = 20e-3 
    # thickness (m)
    T = 0.3e-3 
    # resistivity (ohm/m)
    p = 1.68e-8 
    
    # surface area
    A = T*W 
    # resistance
    R = (p*L)/A 
    # drop
    Vd = I*R 
    Pd = I*Vd*R
    
    Vd = Vd * 1000
    Pd = Pd * 1000
    
    return Vd, Pd

# List to collect data
data = []

# Specify the folder containing your CSV files
folder_path = "C:/Users/ruiui/Desktop/solar plug and play old stuff/iteration data/tabelas_pd_cumu/min_cumu_dataframes"

# Create a dictionary to hold DataFrames
dataframes = {
    filename[:-4]: pd.read_csv(os.path.join(folder_path, filename))
    for filename in os.listdir(folder_path) if filename.endswith('.csv')
}

cumulative_values = {}

for key, df in dataframes.items():
    cumulative_cols = df.columns[df.columns.str.contains('cumulative')]
    last_values = df[cumulative_cols].iloc[-1]
    cumulative_values[key] = last_values


def get_last_cumulative_values(dataframes_dict):
    # Dictionary to store the results
    last_cumulative_values = {}
    
    # Iterate over each dataframe in the dictionary
    for df_name, df in dataframes_dict.items():
        # Find columns with 'cumulative' in their name
        cumulative_columns = [col for col in df.columns if 'cumulative' in col]
        
        # Get the last value of each cumulative column
        last_values = {}
        for col in cumulative_columns:
            last_values[col] = df[col].iloc[-1] * (-1)  # Get the last value of the column
        
        # Store the results for this dataframe
        last_cumulative_values[df_name] = last_values
    
    return last_cumulative_values

last_values_dict = get_last_cumulative_values(dataframes)

def find_ij_from_string(s):
    # Find the pattern of a number followed by 'x' and another number
    match = re.search(r'(\d+)x\1', s)
    if match:
        i = int(match.group(1))  # First number
        j = i ** 2  # Square of the first number
        return i, j
    else:
        return None, None  # If no match is found

# Iterate over all_data_3x3
for circuit, measures in last_values_dict.items():
    i, j = find_ij_from_string(circuit)
    
    if i == 3:
        comps = comps_3x3
        res = res_3x3
    elif i == 4:
        comps = comps_4x4
        res = res_4x4
    elif i == 5:
        comps = comps_5x5
        res = res_5x5
        
    for measure, values in measures.items():
        # Check if any key in comps_3x3 is a substring of the measure
        for key in comps:
            if key in measure:
                circ = measure[21:]
                # Extract median value and compute optloss
                optloss = values
                
                cost_ND = (comps[key] * 0.01 * bushbar + panel * j)
                cost_D = (cost_ND + diode * j)
                
                # Determine the cost based on the presence of '_D' in the measure
                cost = cost_D if measure.endswith('_D') else cost_ND
                
                L = res[key]
                
                Vd, Pd = losses(L, i)
                
                # Append to data list
                data.append([circuit, circ, comps[key], cost, optloss, Vd, Pd])

# Create DataFrame
df = pd.DataFrame(data, columns=['iteration', 'circuit', 'Complexity', 
                                      'Cost', 'Total cumulative losses', 'Voltage losses (mV)', 'Power losses (mW)'])

# Sort DataFrame by iteration and complexity
df.sort_values(by=['iteration', 'Complexity'], ascending=[True, True], na_position='first', inplace=True)

# Reset index to remove ambiguity
df.reset_index(drop=True, inplace=True)


# Use seaborn for better aesthetics
sns.set(style="whitegrid")

# Create a function to plot the data with improved aesthetics
def plot_by_iteration_type(df, iteration_type, column_to_plot):
    # Filter data for the iteration type (3x3, 4x4, 5x5) and only for 'ltspice'
    df_filtered = df[(df['iteration'].str.contains(iteration_type)) & 
                     (df['iteration'].str.contains('ltspice'))]
    
    # Create a new figure for each plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot for circuits without '_D'
    df_without_D = df_filtered[~df_filtered['circuit'].str.endswith('_D')]
    for _, row in df_without_D.iterrows():
        ax1.plot(df_without_D['Complexity'], df_without_D['Cost'], marker='D',
                 markerfacecolor='red', markeredgewidth=2, markersize=8,
                 linestyle='-', color='red', alpha=1.0, label='Cost' if _ == df_without_D.index[0] else "")

    # Plot for circuits with '_D' in lighter color and hollow symbols
    df_with_D = df_filtered[df_filtered['circuit'].str.endswith('_D')]
    for _, row in df_with_D.iterrows():
        ax1.plot(df_with_D['Complexity'], df_with_D['Cost'], marker='D',
                 markerfacecolor='none', markeredgecolor='red', markersize=8, markeredgewidth=2,
                 linestyle='-', color='red', alpha=0.1)

    # Create a second y-axis (right) for the dynamic column (e.g., 'optlosses', 'vlosses', etc.)
    ax2 = ax1.twinx()

    # Plot for circuits without '_D' for the selected column (e.g., optlosses)
    for _, row in df_without_D.iterrows():
        ax2.plot(df_without_D['Complexity'], df_without_D[column_to_plot], marker='D',
                 markerfacecolor='blue', markeredgewidth=2, markersize=8,
                 linestyle='-', color='blue', alpha=1.0, label=f'{column_to_plot}' if _ == df_without_D.index[0] else "")

    # Plot for circuits with '_D' for the selected column in lighter color and hollow symbols
    for _, row in df_with_D.iterrows():
        ax2.plot(df_with_D['Complexity'], df_with_D[column_to_plot], marker='D',
                 markerfacecolor='none', markeredgecolor='blue', markersize=8, markeredgewidth=2,
                 linestyle='-', color='blue', alpha=0.1)

    # Set the titles and labels with better styling
    plt.title(f'{iteration_type} Iteration for ltspice (with and without _D)', fontsize=16, fontweight='bold', color='black')
    ax1.set_xlabel('Complexity', fontsize=12)
    ax1.set_ylabel('Cost', color='red', fontsize=12)
    ax1.set_ylim(bottom=0, top=ax1.get_ylim()[1] * 1.05)  # Increase max of cost by 5%
    ax2.set_ylabel(f'{column_to_plot}', color='blue', fontsize=12)
    ax2.set_ylim(bottom=0, top=ax2.get_ylim()[1] * 1.3)  # Increase max of the selected column by 5%

    # Customize grid and background for a clean look
    ax1.grid(False)
    ax2.grid(False)

    # Add a tight layout to avoid clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()

# # List of iteration types to generate separate plots
# for iteration_type in ['3x3', '4x4', '5x5']:
#     # Here, you can specify the column to plot, e.g., 'Total cumulative losses', 'Voltage losses (mV)', or 'Power losses (mW)'
#     plot_by_iteration_type(df, iteration_type, 'Total cumulative losses')
#     plot_by_iteration_type(df, iteration_type, 'Power losses (mW)')
#     plot_by_iteration_type(df, iteration_type, 'Voltage losses (mV)')


# Create a function to plot the data with improved aesthetics
def plot_by_iteration_type(df, iteration_type):
    # Filter data for the iteration type (3x3, 4x4, 5x5) and only for 'ltspice'
    df_filtered = df[(df['iteration'].str.contains(iteration_type)) & 
                     (df['iteration'].str.contains('ltspice'))]
    
    # Create a new figure for each plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot for circuits without '_D'
    df_without_D = df_filtered[~df_filtered['circuit'].str.endswith('_D')]
    for _, row in df_without_D.iterrows():
        ax1.plot(df_without_D['Complexity'], df_without_D['Cost'], marker='D',
                 markerfacecolor='red', markeredgewidth=2, markersize=8,
                 linestyle='-', color='red', alpha=1.0, label='Cost' if _ == df_without_D.index[0] else "")

    # Plot for circuits with '_D' in lighter color and hollow symbols
    df_with_D = df_filtered[df_filtered['circuit'].str.endswith('_D')]
    for _, row in df_with_D.iterrows():
        ax1.plot(df_with_D['Complexity'], df_with_D['Cost'], marker='D',
                 markerfacecolor='none', markeredgecolor='red', markersize=8, markeredgewidth=2,
                 linestyle='-', color='red', alpha=0.1)

    # Create a second y-axis (right) for Power losses
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 60))  # Offset the second y-axis to the right

    # Plot Power losses
    for _, row in df_without_D.iterrows():
        ax2.plot(df_without_D['Complexity'], df_without_D['Power losses (mW)'], marker='D',
                 markerfacecolor='blue', markeredgewidth=2, markersize=8,
                 linestyle='-', color='blue', alpha=1.0, label='Power losses (mW)' if _ == df_without_D.index[0] else "")

    for _, row in df_with_D.iterrows():
        ax2.plot(df_with_D['Complexity'], df_with_D['Power losses (mW)'], marker='D',
                 markerfacecolor='none', markeredgecolor='blue', markersize=8, markeredgewidth=2,
                 linestyle='-', color='blue', alpha=0.1)

    # Create a third y-axis (right) for Total cumulative losses
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))  # Offset the third y-axis to the right

    # Plot Total cumulative losses
    for _, row in df_without_D.iterrows():
        ax3.plot(df_without_D['Complexity'], df_without_D['Total cumulative losses'], marker='D',
                 markerfacecolor='green', markeredgewidth=2, markersize=8,
                 linestyle='-', color='green', alpha=1.0, label='Total cumulative losses' if _ == df_without_D.index[0] else "")

    for _, row in df_with_D.iterrows():
        ax3.plot(df_with_D['Complexity'], df_with_D['Total cumulative losses'], marker='D',
                 markerfacecolor='none', markeredgecolor='green', markersize=8, markeredgewidth=2,
                 linestyle='-', color='green', alpha=0.1)

    # Set the titles and labels with better styling
    plt.title(f'{iteration_type} Iteration for ltspice (with and without _D)', fontsize=16, fontweight='bold', color='black')
    ax1.set_xlabel('Complexity', fontsize=12)
    ax1.set_ylabel('Cost', color='red', fontsize=12)
    ax1.set_ylim(bottom=0, top=ax1.get_ylim()[1] * 1.05)  # Increase max of cost by 5%
    ax2.set_ylabel('Power losses (mW)', color='blue', fontsize=12)
    ax2.set_ylim(bottom=0, top=ax2.get_ylim()[1] * 1.3)  # Increase max of Power losses by 30%
    ax3.set_ylabel('Total cumulative losses', color='green', fontsize=12)
    ax3.set_ylim(bottom=0, top=ax3.get_ylim()[1] * 1.3)  # Increase max of Total cumulative losses by 30%

    # Customize grid and background for a clean look
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    # Add a tight layout to avoid clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()

# List of iteration types to generate separate plots
for iteration_type in ['3x3', '4x4', '5x5']:
    plot_by_iteration_type(df, iteration_type)