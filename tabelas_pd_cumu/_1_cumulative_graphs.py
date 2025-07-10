# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:25:32 2024

@author: ruiui
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

def lighten_color(color, amount=0.5):
    """Lightens a given color by the specified amount."""
    c = mcolors.to_rgba(color)
    return mcolors.to_hex((c[0] + amount * (1 - c[0]), c[1] + amount * (1 - c[1]), c[2] + amount * (1 - c[2]), c[3]))

order_3x3 = ['AA', 'AI', 'AP', 
             'AB', 'AG', 'AJ', 'AS', 'AX', 
             'AC', 'AD', 'AH', 'AK', 
             'AL', 'AN', 'AQ', 'AV', 'AY', 'BA', 'BD', 'BE', 
             'AE', 'AM', 'AT', 'AZ', 'BB', 
             'AF', 'AW', 'BC', 
             'AO', 'AU', 
             'AR']

order_4x4 = ['AA', 
             'AB', 'AK', 'AN', 'AZ', 'BK', 
             'AC', 'AE', 'AL', 
             'AD', 'AM', 'AO', 'AW', 'AX', 'BA', 'BL', 'BR', 
             'AF', 
             'AH', 'AP', 'AR', 'BB', 'BD', 'BH', 'BM', 'BO', 'BS', 'BY', 'CA', 
             'AG', 'CB', 
             'AI', 'AQ', 'BC', 'BG', 'BQ', 'BT', 'BV', 
             'BU', 
             'AJ', 'AS', 'AU', 'BE', 'BF', 'BJ', 'BN', 'BP', 'BW', 'BZ', 'CC',
             'AT', 'AY', 'BI', 'BX', 
             'AV']

order_5x5 = ['AA', 'AW', 'BA', 
             'AB', 'AP', 'BB', 'BP', 'BZ', 
             'AC', 'AF', 'AQ', 'BC', 'BF', 
             'AD', 'AR', 'BD', 'BK', 'BQ', 'CA', 'CT', 
             'AE', 'AG', 'AS', 'BE', 
             'AJ', 'BG', 'BR', 'CB', 'CO', 'CU', 
             'AH', 'AT', 'CD', 
             'AK', 'BS', 'BU', 'CC', 'CM', 'CV', 
             'AI', 'AX', 'BH', 'BL', 'CE', 'CH', 'CJ', 'CX', 
             'AL', 'AM', 'AU', 'BT', 'CP', 'CW', 
             'CF', 'DA', 
             'AN', 'BI', 'CI', 'CQ', 'CY', 
             'AV', 'BO', 'BV', 'BX', 'CG', 
             'BM', 'CK', 'CS', 'DB', 
             'AO', 'BJ', 'CZ', 
             'AZ', 'BY', 'CR', 'CN', 
             'BW', 'BN', 
             'CL', 
             'AY']

# Specify the folder containing your CSV files
folder_path = "C:/Users/ruiui/Desktop/solar plug and play old stuff/iteration data/tabelas_pd_cumu/cumu_graphs"

u_diff = {
    filename[:-4]: pd.read_csv(os.path.join(folder_path, filename))
    for filename in os.listdir(folder_path) if filename.endswith('.csv')
}

# Create a dictionary to hold DataFrames
dataframes = {
    filename[:-4]: pd.read_csv(os.path.join(folder_path, filename))
    for filename in os.listdir(folder_path) if filename.endswith('.csv')
}

def modify_all_below_threshold(dim):
    ltspice_df = dataframes[f'ltspice_{dim}_results']
    python_df = dataframes[f'python_{dim}_results']
    
    # Only assign ss_df if dim is '3x3', otherwise set it to None
    ss_df = dataframes[f'sun_simulator_{dim}_results'] if dim == '3x3' else None
    
    # Get all column names except 'it' and 'base_code'
    ltspice_cols = set(ltspice_df.columns) - {'it', 'base_code'}
    python_cols = set(python_df.columns) - {'it', 'base_code'}
    
    # If ss_df exists, get its columns as well
    ss_cols = set(ss_df.columns) - {'it', 'base_code'} if ss_df is not None else set()

    # Modify values for all columns in each dataframe
    for col in ltspice_cols:
        if pd.api.types.is_numeric_dtype(ltspice_df[col]):
            ltspice_df[col] = ltspice_df[col].mask(ltspice_df[col] < 2, 0)
    
    for col in python_cols:
        if pd.api.types.is_numeric_dtype(python_df[col]):
            python_df[col] = python_df[col].mask(python_df[col] < 2, 0)
    
    if ss_df is not None:
        for col in ss_cols:
            if pd.api.types.is_numeric_dtype(ss_df[col]):
                ss_df[col] = ss_df[col].mask(ss_df[col] < 2, 0)

# Apply changes to all dimensions
for dim in ['3x3', '4x4', '5x5']:
    modify_all_below_threshold(dim)
    
def analyze_dataframes(dim):
    ltspice_df = dataframes[f'ltspice_{dim}_results']
    python_df = dataframes[f'python_{dim}_results']
    
    # Find common columns excluding 'it' and 'base_code'
    common_columns = set(ltspice_df.columns) & set(python_df.columns) - {'it', 'base_code'}
    
    unique_it_values = set()
    
    if dim == '3x3':
        threshold = 20.85*0.98
    elif dim == '4x4':
        threshold = 37.06*0.98
    elif dim == '5x5':
        threshold = 57.90*0.98
    
    # Loop over the common columns
    for col in common_columns:
        # Ensure both columns are numeric
        if pd.api.types.is_numeric_dtype(ltspice_df[col]) and pd.api.types.is_numeric_dtype(python_df[col]):
            
            # Condition 1: Values less than 1 in one DataFrame and greater than 1 in the other
            indices_diff = ltspice_df[(ltspice_df[col] < 2) & (python_df[col] > 2)].index.union(
                python_df[(ltspice_df[col] > 2) & (python_df[col] < 2)].index
            )
            
            # Condition 2: Values above the threshold
            indices_above_threshold = ltspice_df[ltspice_df[col] > threshold].index.union(
                python_df[python_df[col] > threshold].index
            )
            
            # Update the unique_it_values set with 'it' values from both conditions
            unique_it_values.update(ltspice_df.loc[indices_diff, 'it'].tolist() + python_df.loc[indices_diff, 'it'].tolist() +
                                    ltspice_df.loc[indices_above_threshold, 'it'].tolist() + python_df.loc[indices_above_threshold, 'it'].tolist())
    
    # Return a sorted list of unique 'it' values
    return sorted(unique_it_values)

# Analyze DataFrames for each dimension
unique_it_lists = {dim: analyze_dataframes(dim) for dim in ['3x3', '4x4', '5x5']}

# Filter out rows for DataFrames based on substrings in the 'it' column
for name, df in dataframes.items():
    for dim in ['3x3', '4x4', '5x5']:
        if dim in name:  # Check if the dimension string is in the DataFrame name
            substrings = unique_it_lists[dim]  # Get the list of substrings for this dimension
            
            # Skip filtering if the substrings list is empty
            if not substrings:
                continue  # Move to the next DataFrame without filtering
            
            # Create a regex pattern from the substrings list
            pattern = '|'.join(substrings)
            
            # Filter the DataFrame by excluding rows where 'it' contains any of the substrings
            df_filtered = df[~df['it'].str.contains(pattern, na=False)]  # Keep rows that don't match
            
            # Update the DataFrame in the dictionary
            dataframes[name] = df_filtered
            
            break  # Break out of the loop since we found the matching dimension

def modify_below_threshold(dim):
    ltspice_df = dataframes[f'ltspice_{dim}_results']
    python_df = dataframes[f'python_{dim}_results']
    
    # Find common columns excluding 'it' and 'base_code'
    common_columns = set(ltspice_df.columns) & set(python_df.columns) - {'it', 'base_code'}
    
    for col in common_columns:
        # Ensure both columns are numeric before modifying
        if pd.api.types.is_numeric_dtype(ltspice_df[col]) and pd.api.types.is_numeric_dtype(python_df[col]):
            ltspice_df.loc[ltspice_df[col] < 2, col] = 0
            python_df.loc[python_df[col] < 2, col] = 0

# Modify values for each dimension
for dim in ['3x3', '4x4', '5x5']:
    modify_below_threshold(dim)
    
# # # ----------------------------------------------------------------------------
# dataframes = u_diff
# # # ----------------------------------------------------------------------------

# Reorder DataFrames based on orders
for df_name, df in dataframes.items():
    dim = next((d for d in ['3x3', '4x4', '5x5'] if d in df_name), None)
    if dim:
        valid_order = [code for code in globals()[f'order_{dim}'] if code in df['base_code'].values]
        dataframes[df_name] = df.set_index('base_code').loc[valid_order].reset_index()

# Function to sort DataFrame by custom order of base_code
def sort_by_custom_order(df, order):
    df['base_code'] = pd.Categorical(df['base_code'], categories=order, ordered=True)
    df = df.sort_values('base_code').reset_index(drop=True)
    return df

# Create min, median, and max DataFrames
def create_aggregated_dfs(df_dict):
    min_df, median_df, max_df = {}, {}, {}
    
    for df_name, df in df_dict.items():
        # Ensure only numeric columns are aggregated
        numeric_columns = df.select_dtypes(include='number').columns
        
        # Group by 'base_code' and calculate the min, median, and max for each numeric column
        max_rows = df.groupby('base_code')[numeric_columns].max().reset_index()
        min_rows = df.groupby('base_code')[numeric_columns].min().reset_index()
        median_rows = df.groupby('base_code')[numeric_columns].median().reset_index()
        
        # Apply custom ordering based on the dataframe name
        if '3x3' in df_name:
            order = order_3x3
        elif '4x4' in df_name:
            order = order_4x4
        elif '5x5' in df_name:
            order = order_5x5
        else:
            continue  # Skip DataFrames that don't match '3x3', '4x4', or '5x5'
        
        # Sort the rows based on the custom order for 'base_code'
        max_rows = sort_by_custom_order(max_rows, order)
        min_rows = sort_by_custom_order(min_rows, order)
        median_rows = sort_by_custom_order(median_rows, order)
        
        # Store results in the dictionaries
        max_df[df_name] = max_rows
        min_df[df_name] = min_rows
        median_df[df_name] = median_rows

    return min_df, median_df, max_df

# Example usage:
min_df, median_df, max_df = create_aggregated_dfs(dataframes)

# Reference values for different categories
reference_values_python = {'3x3': 21, 
                            '4x4': 37., 
                            '5x5': 58}

reference_values_ltspice = {'3x3': 21, 
                            '4x4': 37*1.05, 
                            '5x5': 58*1.05}

reference_values_sunsim = {'3x3': 23}

# Calculate fractional differences
def apply_fractional_difference(df_dict):
    for df_name, df in df_dict.items():
        # Determine which reference values to use based on the DataFrame name
        if 'ltspice' in df_name:
            reference_values = reference_values_ltspice
        elif 'python' in df_name:
            reference_values = reference_values_python
        else:
            reference_values = reference_values_sunsim
        
        # Get the dimension (e.g., '3x3', '4x4', '5x5') from the DataFrame name
        dimension = next(dim for dim in reference_values if dim in df_name)
        reference = reference_values[dimension]
        
        # Apply the fractional difference calculation to numeric columns
        for column in df.select_dtypes(include='number').columns:
            df[f'frac_diff_{column}'] = (df[column] - reference) / reference
            df[f'cumulative_frac_diff_{column}'] = df[f'frac_diff_{column}'].cumsum()

# Apply this function to max_df, min_df, and median_df
for df_dict in [max_df, min_df, median_df]:
    apply_fractional_difference(df_dict)
    
# ----------------------------------------------------------------------------

# specify the output folder
output_folder = "C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/min_cumu_dataframes"

# create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for key, df in min_df.items():
    filename = f"{key}.csv"
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)
    
# ----------------------------------------------------------------------------

# specify the output folder
output_folder = "C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/median_cumu_dataframes"

# create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for key, df in median_df.items():
    filename = f"{key}.csv"
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)

# ----------------------------------------------------------------------------

# specify the output folder
output_folder = "C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/max_cumu_dataframes"

# create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for key, df in max_df.items():
    filename = f"{key}.csv"
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)
    
# ----------------------------------------------------------------------------

def graph_cumu_result(array, offsets, vline_positions, g=None):
    # Extract the relevant cumulative data, markers, and colors from cumulative_dict
    cumulative_data = cumulative_dict[array]['cumulative_data']

    # Prepare for plotting
    plt.figure(figsize=(14, 7))

    xticklabels = []  # For storing the labels corresponding to xticks
    x_pos_tracker = []  # For storing the x positions of the points to place xticks

    # Get all DataFrame names and sort them based on the required criteria
    df_names = list(max_df.keys())
    sorted_df_names = sorted(df_names, key=lambda name: (
        0 if 'sun_simulator' in name else
        1 if 'python' in name else
        2 if 'ltspice' in name else
        3  # Keep other DataFrames at the end
    ))

    # Prepare a set for custom legend entries to avoid duplicates
    legend_entries = set()

    # Loop through each column in the cumulative data
    for i, (column, info) in enumerate(cumulative_data.items()):
        # Prepare lists to hold values for the current column
        median_values = []
        x_labels = []
        yerr = []

        # Loop through sorted DataFrame names
        for df_name in sorted_df_names:
            if array in df_name:
                # Check if the column exists in the median, max, and min DataFrames
                if column in median_df[f'{df_name}'].columns and \
                    column in max_df[f'{df_name}'].columns and \
                    column in min_df[f'{df_name}'].columns:
                    
                    # Extract values for the current column
                    median_val = median_df[f'{df_name}'][column].iloc[-1]
                    max_val = max_df[f'{df_name}'][column].iloc[-1]
                    min_val = min_df[f'{df_name}'][column].iloc[-1]
                    
                    # Calculate the error for the whiskers
                    lower_error = abs(median_val - min_val)
                    upper_error = abs(max_val - median_val)
                    
                    # Store the values
                    median_values.append(median_val)
                    x_labels.append(df_name)
                    yerr.append([lower_error, upper_error])

                    # Set x positions for this cluster, incorporating offsets
                    x_pos = i + offsets.get(df_name, 0)  # Use the offset for the current dataframe
                    
                    # Determine the marker based on the DataFrame name
                    if 'sun_simulator' in df_name:
                        marker = 'o'  # Circle
                        legend_label = 'Sun Simulator'
                    elif 'python' in df_name:
                        marker = 'P'  # Square
                        legend_label = 'Python'
                    elif 'ltspice' in df_name:
                        marker = 'D'  # Diamond
                        legend_label = 'LTSpice'
                    else:
                        marker = 'x'  # Default to circle if none match (just in case)
                        legend_label = 'Other'
                    
                    if 'KH' not in column:  # Ignore columns containing 'KH'
                        if column.endswith('_D'):
                            # Plotting the median points with whiskers for this column directly
                            plt.errorbar(x_pos, median_val, yerr=[[lower_error], [upper_error]], 
                                          fmt=marker, color=info['color'], 
                                          capsize=5, markersize=10, fillstyle='none', elinewidth=2)
                        else:
                            # Plotting the median points with whiskers for this column directly
                            plt.errorbar(x_pos, median_val, yerr=[[lower_error], [upper_error]], 
                                          fmt=marker, color=info['color'], 
                                          capsize=5, markersize=10, elinewidth=2)
                    
                        # Append xticklabels only for columns that do not end with '_D' and are from 'ltspice' DataFrames
                        if not column.endswith('_D') and 'ltspice' in df_name:
                            xticklabels.append(column[21:])
                            x_pos_tracker.append(x_pos)
        
                    # Add to legend_entries set to avoid duplicates
                    legend_entries.add((marker, legend_label))

    # Plot vertical lines at specified positions if provided
    if vline_positions is not None:
        for vline in vline_positions:
            plt.axvline(x=vline, color='gray', linestyle='-', linewidth=1)
    
    # Set labels and title
    plt.ylabel('Total cumulative losses', fontsize=16)
    plt.xlabel('Circuits', fontsize=16)
    plt.title(f'Total cumulative losses for {array} array', fontsize=20)
    
    # Set xticks and labels using the xticklabels list and the tracked x positions
    plt.xticks([pos - 1 for pos in vline_positions], xticklabels, fontsize=14)  # Rotate for better readability
    
    # Create custom legend
    for marker, label in legend_entries:
        plt.plot([], [], marker=marker, markersize=10, color='black', label=label)  # Empty plot to create legend entry

    # Place legend outside of the plot
    plt.legend(loc='lower right', fontsize=14)

    # Tight layout for better fitting and adjust padding
    plt.tight_layout(pad=3)
    
    if g == 'zoom out':
        plt.ylim(top=0)
        
    plt.yticks(fontsize=14)
    
    plt.xlim(left=-0.5, right=vline_positions[-1])
    
    plt.grid(axis='x')
    
    # Get current axis
    ax = plt.gca()
    
    # Set major locator for x-axis (6 major ticks, integer values)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    
    
    plt.savefig(f"{array}_{g}_tcl.svg", format="svg")

    # Show plot
    plt.show()
    
# Define the dictionary for cumulative columns, markers, and colors
cumulative_dict = {
    '3x3': {
        'cumulative_data': {
            'cumulative_frac_diff_SP': {'color': '#000000'},
            'cumulative_frac_diff_SP_D': {'color': '#000000'},
            'cumulative_frac_diff_TCT': {'color': '#832aa9'},
            'cumulative_frac_diff_TCT_D': {'color': '#832aa9'},
            'cumulative_frac_diff_ST': {'color': '#248f24'},
            'cumulative_frac_diff_ST_D': {'color': '#248f24'},
            'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            'cumulative_frac_diff_K': {'color': '#cc3333'},
            'cumulative_frac_diff_K_D': {'color': '#cc3333'},
        }
    },
    '4x4': {
        'cumulative_data': {
            'cumulative_frac_diff_SP': {'color': '#000000'},
            'cumulative_frac_diff_SP_D': {'color': '#000000'},
            'cumulative_frac_diff_TCT': {'color': '#832aa9'},
            'cumulative_frac_diff_TCT_D': {'color': '#832aa9'},
            'cumulative_frac_diff_ST': {'color': '#248f24'},
            'cumulative_frac_diff_ST_D': {'color': '#248f24'},
            'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            # 'cumulative_frac_diff_KH': {'color': '#cc7f33'},
            # 'cumulative_frac_diff_KH_D': {'color': '#cc7f33'},
            'cumulative_frac_diff_KV': {'color': '#cc3333'},
            'cumulative_frac_diff_KV_D': {'color': '#cc3333'},
        }
    },
    '5x5': {
        'cumulative_data': {
            'cumulative_frac_diff_SP': {'color': '#000000'},
            'cumulative_frac_diff_SP_D': {'color': '#000000'},
            'cumulative_frac_diff_TCT': {'color': '#832aa9'},
            'cumulative_frac_diff_TCT_D': {'color': '#832aa9'},
            'cumulative_frac_diff_ST': {'color': '#248f24'},
            'cumulative_frac_diff_ST_D': {'color': '#248f24'},
            'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            # 'cumulative_frac_diff_KH': {'color': '#cc7f33'},
            # 'cumulative_frac_diff_KH_D': {'color': '#cc7f33'},
            'cumulative_frac_diff_KV': {'color': '#cc3333'},
            'cumulative_frac_diff_KV_D': {'color': '#cc3333'},
        }
    }
}

# Define the offset for each dataframe group
offsets_3 = {
            'sun_simulator_3x3_results': -0.2,  # Shift left
            'python_3x3_results': 0.0,           # No shift
            'ltspice_3x3_results': 0.2          # Shift right
            }
v_3 = [1.5, 3.6, 5.6, 7.6, 9.6]

# Define the offset for each dataframe group
offsets_4 = {
            'python_4x4_results': -0.2,           # No shift
            'ltspice_4x4_results': 0.2          # Shift right
            }
v_4 = [1.5, 3.5, 5.5, 7.5, 9.5]

# Define the offset for each dataframe group
offsets_5 = {
            'python_5x5_results': -0.2,           # No shift
            'ltspice_5x5_results': 0.2          # Shift right
            }
v_5 = [1.5, 3.5, 5.5, 7.5, 9.5]


# # ----------------------------------------------------------------------------
graph_cumu_result('3x3', offsets_3, v_3)
graph_cumu_result('3x3', offsets_3, v_3,'zoom out')
graph_cumu_result('4x4', offsets_4, v_4)
graph_cumu_result('4x4', offsets_4, v_4,'zoom out')
graph_cumu_result('5x5', offsets_5, v_5)
graph_cumu_result('5x5', offsets_5, v_5,'zoom out')
# # ----------------------------------------------------------------------------


def graph_cumu_progress(array, vline_positions, c, vline_2=None):
    plt.figure(figsize=(9, 7))
    
    # Extract the relevant cumulative data
    cumulative_data = cumulative_dict[array]['cumulative_data']
    
    # Loop through each cumulative column
    for column, props in cumulative_data.items():
        all_medians = []
        all_maxes = []
        all_mins = []
        x_labels = []

        # Loop through all relevant DataFrames
        for df_name in max_df.keys():
            # Check if any string in 'c' is a substring of df_name
            if any(substring.lower() in df_name.lower() for substring in c):
                if array in df_name:
                    try:
                        # Check if the column exists in the DataFrames
                        if column in median_df[df_name].columns and \
                            column in max_df[df_name].columns and \
                            column in min_df[df_name].columns:
                            
                            # Extract entire column values
                            median_val = median_df[df_name][column]
                            max_val = max_df[df_name][column]
                            min_val = min_df[df_name][column]
                            
                            x = len(median_val)
                            
                            # Store the values for plotting
                            all_medians.append(median_val)
                            all_maxes.append(max_val)
                            all_mins.append(min_val)
                            x_labels.append(df_name)
                        else:
                            print(f"Column '{column}' not found in DataFrame '{df_name}'")
                    except KeyError as e:
                        print(f"KeyError: {e} for DataFrame {df_name}")
        
        if not all_medians:  # Check if there are any medians to plot
            print(f"No data found for column: {column}")
            continue  # Skip to the next column if there's no data

        # Determine the base color and plot max, median, and min for each DataFrame
        base_color = props['color']
        adjusted_color = lighten_color(base_color, 0.5)
        
        for j in range(len(all_medians)):
            df_name = x_labels[j]

            # Adjust the color based on the DataFrame name
            adjusted_color = base_color
            if 'sun_simulator' in df_name.lower():
                marker = 'o'
                plt.plot(all_medians[j].index, all_medians[j], 
                        color=base_color, 
                        marker=marker,
                        markersize=4)
            elif 'python' in df_name.lower():
                marker = 'P'
                plt.plot(all_medians[j].index, all_medians[j], 
                        color=base_color, 
                        marker=marker,
                        markersize=4)
            elif 'ltspice' in df_name.lower():
                marker = 'D'
                # Plot median
                plt.plot(all_medians[j].index, all_medians[j], 
                        color=base_color, 
                        label=f'{column[21:]}', 
                        marker=marker,
                        markersize=4)
            else:
                marker = 'x'
            
            # Plot max with a dashed line
            plt.plot(all_medians[j].index, all_maxes[j], 
                      color=adjusted_color)

            # Plot min with a dotted line
            plt.plot(all_medians[j].index, all_mins[j], 
                      color=adjusted_color)

        # Set x-ticks and labels for each column
        plt.ylabel('Cumulative losses', fontsize=20)
        plt.title(f'Cumulative losses progression for {array} array', fontsize=20)
        plt.legend(loc='upper right', fontsize=18)
        plt.xlabel('Scenarios', fontsize=20)
    
    # Plot vertical lines at specified positions if provided
    if vline_positions is not None:
        for vline in vline_positions:
            plt.axvline(x=vline, color='gray', linestyle='--', linewidth=1)
    
    if vline_2 is not None:
        for vline in vline_2:
            plt.axvline(x=vline, color='red', linestyle='--', linewidth=1)

    # plt.grid(False)
    plt.xlim(left=0, right=x-0.5)
    plt.xticks([])
    plt.ylim(bottom=-27.5)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.show()
    
    
cumulative_dict = {
    '3x3': {
        'cumulative_data': {
            # 'cumulative_frac_diff_SP': {'color': '#000000'},
            # 'cumulative_frac_diff_SP_D': {'color': '#808080'},
            'cumulative_frac_diff_TCT': {'color': '#248f24'},
            'cumulative_frac_diff_TCT_D': {'color': '#88c788'},
            # 'cumulative_frac_diff_ST': {'color': '#832aa9'},
            # 'cumulative_frac_diff_ST_D': {'color': '#c38fdc'},
            # 'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            # 'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            # 'cumulative_frac_diff_K': {'color': '#cc3333'},
            # 'cumulative_frac_diff_K_D': {'color': '#cc3333'},
        }
    },
    '4x4': {
        'cumulative_data': {
            'cumulative_frac_diff_SP': {'color': '#000000'},
            'cumulative_frac_diff_SP_D': {'color': '#000000'},
            'cumulative_frac_diff_TCT': {'color': '#832aa9'},
            'cumulative_frac_diff_TCT_D': {'color': '#832aa9'},
            'cumulative_frac_diff_ST': {'color': '#248f24'},
            'cumulative_frac_diff_ST_D': {'color': '#248f24'},
            'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            'cumulative_frac_diff_KH': {'color': '#cc7f33'},
            'cumulative_frac_diff_KH_D': {'color': '#cc7f33'},
            'cumulative_frac_diff_KV': {'color': '#cc3333'},
            'cumulative_frac_diff_KV_D': {'color': '#cc3333'},
        }
    },
    '5x5': {
        'cumulative_data': {
            'cumulative_frac_diff_SP': {'color': '#000000'},
            'cumulative_frac_diff_SP_D': {'color': '#000000'},
            'cumulative_frac_diff_TCT': {'color': '#832aa9'},
            'cumulative_frac_diff_TCT_D': {'color': '#832aa9'},
            'cumulative_frac_diff_ST': {'color': '#248f24'},
            'cumulative_frac_diff_ST_D': {'color': '#248f24'},
            'cumulative_frac_diff_DG': {'color': '#8f8f24'},
            'cumulative_frac_diff_DG_D': {'color': '#8f8f24'},
            'cumulative_frac_diff_KH': {'color': '#cc7f33'},
            'cumulative_frac_diff_KH_D': {'color': '#cc7f33'},
            'cumulative_frac_diff_KV': {'color': '#cc3333'},
            'cumulative_frac_diff_KV_D': {'color': '#cc3333'},
        }
    }
}

v_3 = [2.5,7.5,11.5,19.5,24.5,27.5,29.5]
vp_3 = [0,7,18,26,27]
v_4 = [0.5,5.5,8.5,16.5,17.5,28.5,30.5,37.5,38.5,49.5,53.5]
vp_4 = [1,16,34,46,47]
v_5 = [1.5,6.5,11.5,18.5,21.5,27.5,31.5,36.5,43.5,48.5,50.5,53.5,56.5,60.5,62.5,64.5,65.5,67.5]
vp_5 = [7,31,51,64,66]

v_3=[-10]
v_4=[-10]
v_5=[-10]

# # ----------------------------------------------------------------------------
# graph_cumu_progress('3x3', v_3, [
#     'sun',
#     'python',
#     'ltspice'
#     ], 
#     # vp_3
#     )
# graph_cumu_progress('4x4', v_4, [
#     'python',
#     'ltspice'
#     ], 
#     # vp_4
#     )
# graph_cumu_progress('5x5', v_5, [
#     'python',
#     'ltspice'
#     ], 
#     # vp_5
#     )
# ----------------------------------------------------------------------------












