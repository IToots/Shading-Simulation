# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:29:24 2025

@author: ruiui
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
from scipy.interpolate import interp1d
from tabulate import tabulate
import seaborn as sns
import matplotlib.lines as mlines

# Constants
W = 12.5 * 12.5 * 0.1 * 4  # Area * Irradiance

# Function to calculate percentage difference
def percentage_difference(value1, value2):
    try:
        return ((value1 - value2) / value2) * 100
    except ZeroDivisionError:
        return None

# Set folder path (Change this to your folder path)
folder_path = "C:/Users/ruiui/Desktop/solar plug and play old stuff/MP RESULTS VS N VS IDEAL MP"

# Get all files containing '_MP_'
files = [f for f in os.listdir(folder_path) if '_MP_' in f]

# Function to extract prefix before '_MP_'
def get_prefix(filename):
    match = re.match(r"([A-Za-z0-9]+)_MP_", filename)
    return match.group(1) if match else None

# Group files by prefix
file_groups = defaultdict(list)
for file in files:
    prefix = get_prefix(file)
    if prefix:
        file_groups[prefix].append(file)

# Store results
data = {}

# Process files
for prefix, file_list in file_groups.items():
    plt.figure(figsize=(10, 5))

    for file in file_list:
        file_path = os.path.join(folder_path, file)

        # Determine file type and load accordingly
        if file.endswith(".csv") or file.endswith(".cvs"):
            df = pd.read_csv(file_path)
            x_col, y_col = "Voltage", "Current"

        elif file.endswith(".txt"):
            df = pd.read_csv(file_path, delimiter="\t")
            x_col, y_col = "vpv", "I(D1)"

        else:
            continue  # Skip unknown file types

        # Filter data where Voltage and Current are positive
        df_filtered = df[(df[x_col] > 0) & (df[y_col] > 0)]
        
        # Extract values
        V = df_filtered[x_col].values
        I = df_filtered[y_col].values
        P = V * I  # Calculate Power

        # Interpolation
        interp_func_I = interp1d(V, I, kind='linear', fill_value='extrapolate')
        interp_func_P = interp1d(V, P, kind='linear', fill_value='extrapolate')
        voltage_axis = np.linspace(V.min(), V.max(), num=100)
        interpolated_current = interp_func_I(voltage_axis)
        interpolated_power = interp_func_P(voltage_axis)

        # Create a new DataFrame
        df_interp = pd.DataFrame({
            'Voltage (V)': voltage_axis,
            'Current (A)': interpolated_current,
            'Power (W)': interpolated_power
        })

        # Extract main parameters
        Voc = np.max(df_interp['Voltage (V)'])  # Open-circuit voltage
        Isc = np.max(df_interp['Current (A)'])  # Short-circuit current
        Pmax = np.max(df_interp['Power (W)'])  # Max power point
        Vmp = df_interp['Voltage (V)'][df_interp['Power (W)'].idxmax()]  # Voltage at max power
        Imp = df_interp['Current (A)'][df_interp['Power (W)'].idxmax()]  # Current at max power

        # Fill Factor (FF) and Efficiency (Ef)
        FF = (Pmax / (Isc * Voc)) * 100
        Ef = (Pmax / W) * 100

        # Store results
        if prefix not in data:
            data[prefix] = {}
        data[prefix][file] = [Voc, Isc, Vmp, Imp, Pmax, FF, Ef]

        # Plot the data
        plt.plot(df_interp['Voltage (V)'], df_interp['Current (A)'], label=f"{file} (I)")
        plt.plot(df_interp['Voltage (V)'], df_interp['Power (W)'], label=f"{file} (P)", linestyle="dashed")

    # Customize plot
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A) / Power (W)")
    plt.title(f"Plots for {prefix}_MP_ Files")
    plt.legend()
    plt.grid(True)
    plt.show()

# Compute percentage differences
percentage_differences = []

# Compare datasets within each prefix group
for prefix, datasets in data.items():
    keys = list(datasets.keys())
    if len(keys) >= 2:  # Only compare if at least two files exist
        for i in range(len(keys) - 1):
            file1, file2 = keys[i], keys[i + 1]
            values1, values2 = datasets[file1], datasets[file2]

            # Calculate percentage differences for each parameter
            Voc_diff = percentage_difference(values1[0], values2[0])
            Isc_diff = percentage_difference(values1[1], values2[1])
            Vmp_diff = percentage_difference(values1[2], values2[2])
            Imp_diff = percentage_difference(values1[3], values2[3])
            MPP_diff = percentage_difference(values1[4], values2[4])
            FF_diff = percentage_difference(values1[5], values2[5])
            Ef_diff = percentage_difference(values1[6], values2[6])

            percentage_differences.append([prefix, file1, file2, Voc_diff, Isc_diff, Vmp_diff, Imp_diff, MPP_diff, FF_diff, Ef_diff])

# Create DataFrame for percentage differences
columns_diff = ['Prefix', 'File1', 'File2', 'Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency']
percentage_diff_df = pd.DataFrame(percentage_differences, columns=columns_diff)

# Show the results
print("Percentage Differences Between Files:")
# ----------------------------------------------------------------------------
# Display the percentage differences as a table using tabulate
print(tabulate(percentage_diff_df, headers='keys', tablefmt='grid'))
# ----------------------------------------------------------------------------

# Reshape the DataFrame to long format for Seaborn
df_melted = pd.melt(percentage_diff_df, id_vars=['Prefix', 'File1', 'File2'], 
                    var_name='Parameter', value_name='Percentage Difference')

# Set the plot size
plt.figure(figsize=(12, 6))

# Add the gray box from -0.2% to +0.2% to indicate negligible change
plt.axhspan(-0.2, 0.2, color='gray', alpha=0.2, zorder=-1)

# Add a horizontal line at y=0 for reference
plt.axhline(0, color='black', linewidth=1.5)

# Create a grouped bar plot
ax = sns.barplot(x='Parameter', y='Percentage Difference', hue='Prefix', data=df_melted, palette="tab10")

# Add vertical dashed lines to separate parameter groups
num_params = len(percentage_diff_df.columns) - 3  # Subtract 3 for 'Prefix', 'File1', 'File2'
for i in range(1, num_params):
    plt.axvline(i - 0.5, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)

# Add stripes to the bars
for bar in ax.patches:
    # Add stripes inside the bar
    bar.set_hatch('//')  # Set hatch pattern to diagonal stripes

# Customize plot
plt.title('Percentage Difference per Parameter - different connections', fontsize=16)
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('Percentage Difference (%)', fontsize=12)
plt.legend(title='Connections', bbox_to_anchor=(1.005, 1), loc='upper left')
plt.tight_layout()
plt.ylim(-5, 5)  # Adjust limits as needed

# Show the plot
plt.show()



# Convert extracted data into a DataFrame for plotting
plot_data = []

for prefix, file_results in data.items():
    for file, values in file_results.items():
        plot_data.append([prefix, file] + values)

# Define column names
columns = ['Prefix', 'File', 'Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency']
plot_df = pd.DataFrame(plot_data, columns=columns)

# Define parameters to plot
parameters = ['Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency']
num_params = len(parameters)

fig, axs = plt.subplots(2, 4, figsize=(15, 8))  # Create a 2-row grid of subplots
axs = axs.ravel()  # Flatten the subplot array

# Assign colors based on unique files
unique_files = plot_df['File'].unique()
colors = plt.cm.tab10.colors  # Get color map
color_map = {file: colors[i % len(colors)] for i, file in enumerate(unique_files)}

# Plot each parameter
for i, param in enumerate(parameters):
    ax = axs[i]

    # Group data by Prefix
    grouped = plot_df.groupby('Prefix')

    for prefix, group in grouped:
        bottom = np.zeros(len(group))  # Initialize bottom values for stacking
        x_labels = group['Prefix']  # Use Prefix for x-axis

        for _, row in group.iterrows():
            file = row['File']
            value = row[param]

            # If the file contains '_NOPCB', make it transparent
            if '_NOPCB' in file:
                ax.bar(prefix, value, bottom=bottom[group.index.get_loc(_)], color='none', edgecolor='black', alpha=1.0, label=file)
            else:
                ax.bar(prefix, value, bottom=bottom[group.index.get_loc(_)], color=color_map[file], alpha=0.75, label=file)
                
            bottom[group.index.get_loc(_)] += value  # Update bottom for stacking

    ax.set_title(param)
    ax.set_xlabel("Prefix")
    ax.set_ylabel(param)
    ax.tick_params(axis='x', rotation=30)

# Remove empty subplot if not used
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

# Add a legend
fig.legend(unique_files, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)

# Save the figure as an SVG
plt.savefig(f'wiwu.svg', format='svg', bbox_inches='tight')

# Adjust layout
plt.tight_layout()
plt.show()