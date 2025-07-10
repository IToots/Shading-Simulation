# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:22:29 2024

@author: ruiui
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import re
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

comps_3x3 = {
    'SP': 80, 
    'TCT': 120, 
    'R1': 168, 
    'R3': 161,
    'R2': 161, 
    'L': 202,
    'PER': 202}
res_3x3 = {
    'SP': 12.500, 
    'TCT': 12.500, 
    'R1': 18.499, 
    'R3': 19.631,
    'R2': 19.631, 
    'L': 18.948,
    'PER': 18.948}

comps_4x4 = {
    'SP': 180, 
    'TCT': 240, 
    'R1': 332, 
    'R3': 318,
    'R2': 318, 
    'L': 484,
    'PER': 464}
res_4x4 = {
    'SP': 16.549, 
    'TCT': 16.161, 
    'R1': 25.781, 
    'R3': 24.428,
    'R2': 24.428, 
    'L': 29.290,
    'PER': 24.643}

comps_5x5 = {
    'SP': 280, 
    'TCT': 400,
    'R1': 546, 
    'R3': 620, 
    'R2': 525, 
    'L': 948,
    'PER': 940}
res_5x5 = {
    'SP': 20.059, 
    'TCT': 19.010,
    'R1': 31.586, 
    'R3': 27.460, 
    'R2': 30.831, 
    'L': 33.610,
    'PER': 33.610}

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
df = pd.DataFrame(data, columns=['iteration', 'circuit', 'complexity', 
                                      'cost', 'optlosses', 'vlosses', 'plosses'])

# Sort DataFrame by iteration and complexity
df.sort_values(by=['iteration', 'complexity'], ascending=[True, True], na_position='first', inplace=True)

# Reset index to remove ambiguity
df.reset_index(drop=True, inplace=True)

# Step 1: Get the cost for 'SP' circuit for each iteration
sp_cost = df[df['circuit'] == 'SP'].set_index('iteration')['cost']
sp_plosses = df[df['circuit'] == 'TCT'].set_index('iteration')['plosses']

# Step 2: Calculate the difference for each circuit against the SP cost
df['sp_cost_diff'] = df.apply(lambda row: row['cost'] - sp_cost[row['iteration']], axis=1)
df['sp_plosses_diff'] = df.apply(lambda row: row['plosses'] - sp_plosses[row['iteration']], axis=1)

# Step 3: Calculate the percentage difference
df['sp_cost_percent_diff'] = df.apply(
    lambda row: (row['sp_cost_diff'] / sp_cost[row['iteration']]) * 100 if sp_cost[row['iteration']] != 0 else 0,
    axis=1
)
df['sp_plosses_percent_diff'] = df.apply(
    lambda row: (row['sp_plosses_diff'] / sp_plosses[row['iteration']]) * 100 if sp_plosses[row['iteration']] != 0 else 0,
    axis=1
)

# Step 4: Create a new DataFrame with the required columns
result_df = df[['iteration', 'circuit', 'complexity', 'optlosses',
                 'sp_cost_percent_diff', 'sp_plosses_percent_diff']]

# Drop duplicate rows based on 'iteration' and 'circuit'
df_unique = result_df.drop_duplicates(subset=['iteration', 'circuit'])



def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name
        
def plot(iteration, df, losses, pvlosses):
    # Filter the DataFrame for the chosen iteration
    df_iteration = df[df['iteration'] == iteration]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot cost
    color_cost = 'tab:red'
    ax1.set_xlabel('Complexity')
    ax1.set_ylabel('Cost (€)', color=color_cost)
    
    # Scatter plot for cost
    for _, row in df_iteration.iterrows():
        marker = 'o' if '_D' not in row['circuit'] else '^'
        ax1.scatter(row['complexity'], row['cost'], color=color_cost, marker=marker, label='Cost' if '_D' not in row['circuit'] else 'Cost_D')
    
    # Fit a linear regression model to cost for non-_D points
    non_D = df_iteration[df_iteration['circuit'].str.contains('_D') == False]
    if not non_D.empty:
        X_cost = non_D['complexity'].values.reshape(-1, 1)
        y_cost = non_D['cost'].values.reshape(-1, 1)
        model_cost = LinearRegression().fit(X_cost, y_cost)
        ax1.plot(X_cost, model_cost.predict(X_cost), color=color_cost, label='Trend Line (Cost)')

    # Fit a linear regression model to cost for _D points
    D = df_iteration[df_iteration['circuit'].str.contains('_D') == True]
    if not D.empty:
        X_cost_D = D['complexity'].values.reshape(-1, 1)
        y_cost_D = D['cost'].values.reshape(-1, 1)
        model_cost_D = LinearRegression().fit(X_cost_D, y_cost_D)
        ax1.plot(X_cost_D, model_cost_D.predict(X_cost_D), color='lightcoral', linestyle='--', label='Trend Line (Cost _D)')
    
    ax1.tick_params(axis='y', labelcolor=color_cost)
    
    # Create a second y-axis to plot optlosses
    ax2 = ax1.twinx()
    color_losses = 'tab:blue'
    ax2.set_ylabel('Shading (%)', color=color_losses)
    
    # Scatter plot for optlosses
    for _, row in df_iteration.iterrows():
        marker = 'o' if '_D' not in row['circuit'] else '^'
        ax2.scatter(row['complexity'], row[losses], color=color_losses, marker=marker, label='Optlosses' if '_D' not in row['circuit'] else 'Optlosses_D')
    
    # Fit a linear regression model to optlosses for non-_D points
    if not non_D.empty:
        X_losses = non_D['complexity'].values.reshape(-1, 1)
        y_losses = non_D[losses].values.reshape(-1, 1)
        model_losses = LinearRegression().fit(X_losses, y_losses)
        ax2.plot(X_losses, model_losses.predict(X_losses), color=color_losses, label='Trend Line (Optlosses)')

        # Calculate R2 score for optlosses without _D
        r2_losses = r2_score(y_losses, model_losses.predict(X_losses))
        print(f'R2 score (Optlosses without _D): {r2_losses:.2f}')

    # Fit a linear regression model to optlosses for _D points
    if not D.empty:
        X_losses_D = D['complexity'].values.reshape(-1, 1)
        y_losses_D = D[losses].values.reshape(-1, 1)
        model_losses_D = LinearRegression().fit(X_losses_D, y_losses_D)
        ax2.plot(X_losses_D, model_losses_D.predict(X_losses_D), color='lightblue', linestyle='--', label='Trend Line (Optlosses _D)')
        
        # Calculate R2 score for optlosses with _D
        r2_losses_D = r2_score(y_losses_D, model_losses_D.predict(X_losses_D))
        print(f'R2 score (Optlosses with _D): {r2_losses_D:.2f}')
    
    ax2.tick_params(axis='y', labelcolor=color_losses)
    
    # Create a third y-axis to plot vlosses
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
    color_pvlosses = 'tab:green'
    ax3.set_ylabel('Wire length (mW)', color=color_pvlosses)
    
    # Scatter plot for vlosses
    for _, row in df_iteration.iterrows():
        if '_D' not in row['circuit']:
            marker = '*' 
            ax3.scatter(row['complexity'], row[pvlosses], color=color_pvlosses, marker=marker, label=pvlosses, s=200)
            ax3.annotate(row['circuit'], (row['complexity'], row[pvlosses]), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Get current y-limits for ax1 (Cost axis)
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ymin3, ymax3 = ax3.get_ylim()
    
    # Increase the upper limit by 5%
    ymax1_new = ymax1 * 1.05
    ymax2_new = ymax2 * 1.3
    ymax3_new = ymax3 * 2.6
    
    ax1.set_ylim(bottom=0, top=ymax1_new)
    ax2.set_ylim(bottom=0, top=ymax2_new)
    ax3.set_ylim(bottom=0, top=ymax3_new)
    
    ax3.tick_params(axis='y', labelcolor=color_pvlosses)

    name = get_var_name(df)
    
    # Adding title and showing the plot
    plt.title(f'Cost and Losses for {iteration} - {name}')
    fig.tight_layout()  # Adjust layout to fit both y-axis labels
    plt.show()

# Usage example
plot('ltspice_3x3_results', df, 'optlosses', 'plosses')
plot('ltspice_4x4_results', df, 'optlosses', 'plosses')
plot('ltspice_5x5_results', df, 'optlosses', 'plosses')

# ----------------------------------------------------------------------------

def plot_iteration_results(df, iteration_string, bar_width=0.4, bar_location_offset=4, line_length=0.3, line_width=2, figw=12):
    # Filter the DataFrame for the specified iteration
    filtered_df = df[df['iteration'] == iteration_string]
    
    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        print(f"No data found for iteration: {iteration_string}")
        return
    
    fig, ax1 = plt.subplots(figsize=(figw, 6))  # Set figure size
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis

    circuit_level = 0
    odd_circuits = []  # List to store circuit values from odd rows
    total_rows = len(filtered_df)

    for index, (i, row) in enumerate(filtered_df.iterrows()):
        # Determine the alpha based on whether the index is odd or even
        bar_alpha = 0.7 if (index + 1) % 2 == 1 else 0.3

        # Calculate x positions for bars
        x_optlosses = circuit_level
        x_sp_cost_percent_diff = circuit_level + 1

        # Plot optlosses and sp_cost_percent_diff on ax1 (primary y-axis)
        ax1.bar(x_optlosses, row['optlosses'], color='tab:green', alpha=bar_alpha, width=bar_width, label='optlosses' if index == 0 else "")
        ax1.bar(x_sp_cost_percent_diff, row['sp_cost_percent_diff'], color='tab:red', alpha=bar_alpha, width=bar_width, label='sp_cost_percent_diff' if index == 0 else "")
        
        # Add horizontal black lines at the top of the bars for optlosses and sp_cost_percent_diff
        ax1.plot([x_optlosses - line_length, x_optlosses + line_length], [row['optlosses'], row['optlosses']], color='black', lw=line_width)
        ax1.plot([x_sp_cost_percent_diff - line_length, x_sp_cost_percent_diff + line_length], [row['sp_cost_percent_diff'], row['sp_cost_percent_diff']], color='black', lw=line_width)

        # Plot sp_plosses_percent_diff on ax2 (secondary y-axis)
        x_sp_plosses_percent_diff = circuit_level + 2
        ax2.bar(x_sp_plosses_percent_diff, row['sp_plosses_percent_diff'], color='tab:blue', alpha=bar_alpha, width=bar_width, label='sp_plosses_percent_diff' if index == 0 else "")
        ax2.plot([x_sp_plosses_percent_diff - line_length, x_sp_plosses_percent_diff + line_length], [row['sp_plosses_percent_diff'], row['sp_plosses_percent_diff']], color='black', lw=line_width)

        # Save the circuit value for odd rows
        if (index + 1) % 2 == 1:  # If odd index
            odd_circuits.append(row['circuit'])

        # For even indexes, compare with the previous odd-index row and apply text placement logic for optlosses
        if (index + 1) % 2 == 0 and index > 0:  # Ensure there's a previous odd row to compare with
            prev_row = filtered_df.iloc[index - 1]

            # Calculate the difference for optlosses
            diff_optlosses = abs(row['optlosses'] - prev_row['optlosses'])

            # Place text for optlosses
            if diff_optlosses < 5:
                # Place the higher value above the black line and the lower value below it
                if row['optlosses'] > prev_row['optlosses']:
                    ax1.text(x_optlosses, row['optlosses'] + 1, f'{row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)  # Above for current
                    ax1.text(x_optlosses, prev_row['optlosses'] - 2, f'{prev_row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)  # Below for previous
                else:
                    ax1.text(x_optlosses, prev_row['optlosses'] + 1, f'{prev_row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)  # Above for previous
                    ax1.text(x_optlosses, row['optlosses'] - 2, f'{row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)  # Below for current
            else:
                ax1.text(x_optlosses, row['optlosses'] + 1, f'{row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)
                ax1.text(x_optlosses, prev_row['optlosses'] + 1, f'{prev_row["optlosses"]:.2f}', ha='center', color='black', fontsize=10)  # Above for previous

            # Place text for sp_cost_percent_diff (always above the black line)
            ax1.text(x_sp_cost_percent_diff, row['sp_cost_percent_diff'] + 1, f'{row["sp_cost_percent_diff"]:.2f}', ha='center', color='black', fontsize=10)

            # Place text for sp_plosses_percent_diff (always above the black line)
            ax2.text(x_sp_plosses_percent_diff, row['sp_plosses_percent_diff'] + 5, f'{row["sp_plosses_percent_diff"]:.2f}', ha='center', color='black', fontsize=10)

        else:
            # For the first (odd) row, place the text above the black line as usual
            ax1.text(x_sp_cost_percent_diff, row['sp_cost_percent_diff'] + 1, f'{row["sp_cost_percent_diff"]:.2f}', ha='center', color='black', fontsize=10)
            ax2.text(x_sp_plosses_percent_diff, row['sp_plosses_percent_diff'] + 5, f'{row["sp_plosses_percent_diff"]:.2f}', ha='center', color='black', fontsize=10)

        if (index + 1) % 2 == 0:  # Move to next circuit_level after plotting both odd and even
            circuit_level += bar_location_offset  # Move to the next set of bars

    # Set the maximum y-values to 1.2 times the current maximum
    max_y1 = ax1.get_ylim()[1]
    ax1.set_ylim(0, max_y1 * 1.3)

    max_y2 = ax2.get_ylim()[1]
    ax2.set_ylim(0, max_y2 * 1.3)

    # Create a list starting from 1 and adding 4 for each subsequent value
    step_values = [1 + 4 * i for i in range(len(odd_circuits))]

    # Set the x-axis labels using odd_circuits
    ax1.set_xticks(step_values)  # Set x-tick positions
    ax1.set_xticklabels(odd_circuits)  # Set x-tick labels

    # Set labels for the y axes
    ax1.set_xlabel('Circuit')
    ax1.set_ylabel('Optlosses / Cost Percent Diff', color='black')
    ax2.set_ylabel('Power Losses Percent Diff', color='black')

    # Title and legends
    plt.title(f'Iteration: {iteration_string}')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()

    print("Odd Circuit Values:", odd_circuits)
    print("Step Values:", step_values)


# Example of how to use the function:
plot_iteration_results(df_unique, 'ltspice_3x3_results', bar_width=0.9, bar_location_offset=4, line_length=0.4, line_width=2, figw=13)
plot_iteration_results(df_unique, 'ltspice_4x4_results', bar_width=0.9, bar_location_offset=4, line_length=0.4, line_width=2, figw=16)
plot_iteration_results(df_unique, 'ltspice_5x5_results', bar_width=0.9, bar_location_offset=4, line_length=0.4, line_width=2, figw=17)
