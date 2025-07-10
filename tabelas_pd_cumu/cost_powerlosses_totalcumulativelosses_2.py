# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:35:20 2025

@author: ruiui
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import matplotlib.lines as mlines
import numpy as np
import matplotlib.ticker as ticker

# Use seaborn for better aesthetics
sns.set(style="whitegrid")

# Function to load data from a folder
def load_data_from_folder(folder_path):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Load each CSV file into a DataFrame and concatenate them
    dataframes = {filename[:-4]: pd.read_csv(os.path.join(folder_path, filename)) for filename in csv_files}
    return dataframes

# Function to get the last cumulative values from each DataFrame
def get_last_cumulative_values(dataframes_dict):
    last_cumulative_values = {}
    for df_name, df in dataframes_dict.items():
        cumulative_columns = [col for col in df.columns if 'cumulative' in col]
        last_values = {col: df[col].iloc[-1] * (-1) for col in cumulative_columns}
        last_cumulative_values[df_name] = last_values
    return last_cumulative_values

# Function to find i and j from the string
def find_ij_from_string(s):
    match = re.search(r'(\d+)x\1', s)
    if match:
        i = int(match.group(1))
        j = i ** 2
        return i, j
    else:
        return None, None

# Function to filter dictionary keys containing 'ltspice'
def filter_ltspice_dataframes(dataframes_dict):
    return {key: df for key, df in dataframes_dict.items() if 'ltspice' in key.lower()}

# Function to drop columns containing 'KH' in the name
def drop_KH_columns(dataframes_dict):
    for key in dataframes_dict:
        dataframes_dict[key] = dataframes_dict[key].drop(
            columns=[col for col in dataframes_dict[key].columns if 'KH' in col]
        )



comps_3x3 = {'SP': 20, 'TCT': 24, 'ST': 35, 'DG': 38, 'K': 46}
res_3x3 = {'SP': 12.500, 'TCT': 12.500, 'ST': 22.328, 'DG': 21.060, 'K': 21.772}

comps_4x4 = {'SP': 36, 'TCT': 48, 'ST': 73,'DG': 76, 'KV': 108}
res_4x4 = {'SP': 16.548, 'TCT': 16.161, 'ST': 28.026, 'DG': 29.886, 'KV': 30.384}

comps_5x5 = {'SP': 56, 'TCT': 80, 'ST': 143, 'DG': 123, 'KV': 220}
res_5x5 = {'SP': 20.060, 'TCT': 19.016, 'ST': 29.364, 'DG': 36.932, 'KV': 37.648}

bushbar = 0.3616
diode = 0.531
panel = 3.51

# Define a function to calculate losses
def losses(L, i):
    Vd = L * i  # Example calculation
    Pd = Vd * i  # Example calculation
    return Vd, Pd

# Load data from multiple folders
min_cumu_dataframes = load_data_from_folder("C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/min_cumu_dataframes")
max_cumu_dataframes = load_data_from_folder("C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/max_cumu_dataframes")
median_cumu_dataframes = load_data_from_folder("C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/median_cumu_dataframes")

# Apply the function to each dictionary
min_cumu_dataframes = filter_ltspice_dataframes(min_cumu_dataframes)
max_cumu_dataframes = filter_ltspice_dataframes(max_cumu_dataframes)
median_cumu_dataframes = filter_ltspice_dataframes(median_cumu_dataframes)

# Apply the function to each dictionary
drop_KH_columns(min_cumu_dataframes)
drop_KH_columns(max_cumu_dataframes)
drop_KH_columns(median_cumu_dataframes)

# Get the last cumulative values
min_cumu_values = get_last_cumulative_values(min_cumu_dataframes)
max_cumu_values = get_last_cumulative_values(max_cumu_dataframes)
median_cumu_values = get_last_cumulative_values(median_cumu_dataframes)

# Prepare the data for plotting
data = []

for circuit, measures in min_cumu_values.items():
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
        for key in comps:
            if key in measure:
                circ = measure[21:]  # Extract circuit name
                optloss = values
                
                cost_ND = (comps[key] * 5 * 0.01 * bushbar + panel * j)
                cost_D = (cost_ND + diode * j)
                
                cost = cost_D if measure.endswith('_D') else cost_ND
                
                L = res[key]
                
                Vd, Pd = losses(L, i)
                
                # Extract TCL values
                tcl_min = min_cumu_values[circuit].get(measure, None)
                tcl_max = max_cumu_values[circuit].get(measure, None)
                tcl_median = median_cumu_values[circuit].get(measure, None)
                
                data.append([circuit, circ, comps[key], cost, Vd, Pd, tcl_min, tcl_max, tcl_median])
                
# Create DataFrame
df = pd.DataFrame(data, columns=['iteration', 'circuit', 'Complexity', 
                                      'Cost', 'Voltage losses (mV)', 'Power losses (mW)', 'tclmin', 'tclmax', 'Total cumulative losses'])

# Sort DataFrame by iteration and complexity
df.sort_values(by=['iteration', 'Complexity'], ascending=[True, True], na_position='first', inplace=True)

# Reset index to remove ambiguity
df.reset_index(drop=True, inplace=True)

df = df[df['iteration'].str.contains('ltspice')]

# Create a new column for Cost difference and initialize it with NaN
df['Cost diff'] = None  

# Iterate over each unique iteration category (3x3, 4x4, 5x5)
for size in ['3x3', '4x4', '5x5']:
    # Filter rows for the current size
    subset = df[df['iteration'].str.contains(size)]
    
    # Get the reference cost where circuit == 'SP'
    reference_cost = subset.loc[subset['circuit'] == 'SP', 'Cost'].values
    
    # If reference cost exists, compute the percentage difference
    if len(reference_cost) > 0:
        reference_cost = reference_cost[0]  # Get the first (should be only) value
        df.loc[df['iteration'].str.contains(size), 'Cost diff'] = (
            (df.loc[df['iteration'].str.contains(size), 'Cost'] - reference_cost) / reference_cost
        ) * 100  # Convert to percentage

# Convert the new column to numeric (in case of NaN values)
df['Cost diff'] = pd.to_numeric(df['Cost diff'])

df['circuit'] = df['circuit'].replace({'KV': 'K', 'KV_D': 'K_D'})

# Define custom colors for each case
colorzz = plt.cm.tab20([6, 7, 0, 1, 4, 5])

# Define colors for different iteration types
iteration_colors = {'3x3': (colorzz[0],colorzz[1]), 
                    '4x4': (colorzz[2],colorzz[3]), 
                    '5x5': (colorzz[4],colorzz[5])}



# ------------------- Power losses (mW) Plot -------------------
tut = 'Power losses (mW)'

# Create a new figure for the combined Power Losses plot
fig, ax = plt.subplots(figsize=(6, 5))

# Loop through iteration types and plot them on the same axis
for iteration_type in ['3x3', '4x4', '5x5']:
    df_filtered = df[(df['iteration'].str.contains(iteration_type)) & 
                      (df['iteration'].str.contains('ltspice'))]
    
    df_without_D = df_filtered[~df_filtered['circuit'].str.endswith('_D')]
    # Plot for without diode
    ax.plot(df_without_D['Complexity'], df_without_D[tut], marker='o', 
            linestyle='-', linewidth=2, color=iteration_colors[iteration_type][0], label=f'{iteration_type} (no diode)')
    
    # Add circuit labels next to points
    for idx, row in df_without_D.iterrows():
        ax.text(row['Complexity'], row[tut] * 1.05, row['circuit'], 
                fontsize=12, ha='left', va='center', color=iteration_colors[iteration_type][0])

# Configure axes for Power Losses
ax.set_ylabel(tut, fontsize=16)
ax.set_ylim(50,1020)
ax.tick_params(axis='x', labelsize=13)
ax.set_xlabel('Complexity (a.u.)', fontsize=16)
ax.tick_params(axis='y', labelsize=13)
ax.grid(True, alpha=0.3, axis='both')
ax.legend(fontsize=12)
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

# Save and show the plot for Power Losses
plt.tight_layout()
plt.savefig(f"{tut}_comparison.svg", format="svg")
plt.show()


# ------------------- Cost increase (%) Plot with Broken Y-axis -------------------
tut = 'Cost diff'

# Create two subplots with shared x-axis
fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True, figsize=(6, 5), gridspec_kw={'height_ratios': [1, 1]})

# Loop through iteration types and plot them on both axes
for iteration_type in ['3x3', '4x4', '5x5']:
    df_filtered = df[(df['iteration'].str.contains(iteration_type)) & 
                      (df['iteration'].str.contains('ltspice'))]
    
    df_without_D = df_filtered[~df_filtered['circuit'].str.endswith('_D')]
    df_with_D = df_filtered[df_filtered['circuit'].str.endswith('_D')]

    # Plot for with diode (y between 15 and 20) -> Now on TOP subplot (ax2)
    ax2.plot(df_with_D['Complexity'], df_with_D[tut], marker='o', markerfacecolor='white',
             linestyle='--', linewidth=2, color=iteration_colors[iteration_type][0], label=f'{iteration_type} (with diode)')

    # Plot for without diode (y between 0 and 5) -> Now on BOTTOM subplot (ax1)
    ax1.plot(df_without_D['Complexity'], df_without_D[tut], marker='o', 
             linestyle='-', linewidth=2, color=iteration_colors[iteration_type][0], label=f'{iteration_type} (no diode)')

    # Add circuit labels next to points in both subplots
    for idx, row in df_without_D.iterrows():
        ax1.text(row['Complexity'], row[tut] * 1.05, row['circuit'], 
                 fontsize=12, ha='left', va='center', color=iteration_colors[iteration_type][0])

# Configure the broken y-axis
ax2.set_ylim(14.25, 19.2)  # Upper part (with diode)
ax1.set_ylim(-0.75, 4.25)    # Lower part (without diode)

# Hide spines between subplots
ax1.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)
ax1.xaxis.tick_bottom()

# Add diagonal lines to indicate the break
d = 0.015  # Line size
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Top-left diagonal
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top-right diagonal

kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d, +d), (-d, +d), **kwargs)  # Bottom-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom-right diagonal

# Set labels
ax1.set_xlabel('Complexity (a.u.)', fontsize=16)
ax1.set_ylabel('Cost increase (%)', fontsize=16)

# Add grid and legend
ax1.grid(True, alpha=0.3, axis='both')
ax2.grid(True, alpha=0.3, axis='both')
ax2.legend(fontsize=12)

# Save and show
plt.tight_layout()
plt.savefig(f"{tut}_comparison.svg", format="svg")
plt.show()


# ------------------- Total cumulative shading losses (%) Plot -------------------
# Create subplots (3 columns, 1 row)
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=False, 
                          gridspec_kw={'width_ratios': [2, 2, 2]})

# Define iteration types and their corresponding colors
iteration_types = ['3x3', '4x4', '5x5']
colors = { 
    '3x3': (colorzz[0], colorzz[1]),
    '4x4': (colorzz[2], colorzz[3]),  
    '5x5': (colorzz[4], colorzz[5])
}

# Loop through iteration types and plot on respective subplot
for i, iteration_type in enumerate(iteration_types):
    df_filtered = df[(df['iteration'].str.contains(iteration_type)) & 
                      (df['iteration'].str.contains('ltspice'))]

    df_without_D = df_filtered[~df_filtered['circuit'].str.endswith('_D')]
    df_with_D = df_filtered[df_filtered['circuit'].str.endswith('_D')]

    ax = axes[i]  # Select corresponding subplot

    # Assign colors
    color_without_D = colors[iteration_type][0]
    color_with_D = colors[iteration_type][0]

    # Fill range for total cumulative losses
    ax.fill_between(df_without_D['Complexity'], df_without_D['tclmin'], df_without_D['tclmax'],
                      color=color_without_D, alpha=0.15, label='TCL range')
    ax.plot(df_without_D['Complexity'], df_without_D['Total cumulative losses'], marker='o',
            linestyle='-', linewidth=2, color=color_without_D, label=f'{iteration_type} w/o diode')

    ax.fill_between(df_with_D['Complexity'], df_with_D['tclmin'], df_with_D['tclmax'],
                      color=color_with_D, alpha=0.2)
    ax.plot(df_with_D['Complexity'], df_with_D['Total cumulative losses'], marker='o', markerfacecolor='white',
            linestyle='--', linewidth=2, color=color_with_D, label=f'{iteration_type} w/ diode')
    
    # Add circuit labels next to points
    for idx, row in df_without_D.iterrows():
        ax.text(row['Complexity'], row['tclmin'], row['circuit'], 
                fontsize=12, ha='left', va='center', color=iteration_colors[iteration_type][0])
    
    # ax.set_xlim(right=df_without_D['tclmin'].max()*1.05)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, alpha=0.3, axis='both')
    ax.legend(fontsize=12)

# Set common Y label for all subplots
axes[0].set_ylabel('Total cumulative shading losses (%)', fontsize=16)
axes[1].set_xlabel('Complexity (a.u.)', fontsize=16)

axes[0].set_ylim(top=27)
axes[1].set_ylim(top=48.5)
axes[2].set_ylim(top=67)

axes[0].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
axes[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
axes[2].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

# Adjust layout and save
plt.tight_layout()
plt.savefig("tcl_comparison.svg", format="svg")
plt.show()
