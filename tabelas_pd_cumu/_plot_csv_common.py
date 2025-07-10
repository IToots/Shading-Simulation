# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:49:20 2024

@author: ruiui
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tabulate import tabulate
from scipy.stats import linregress
import matplotlib as mpl

t = 'series'
directory = f"C:/Users/ruiui/Desktop/solar plug and play old stuff/common/{t}"
directory_lt = f"C:/Users/ruiui/Desktop/solar plug and play old stuff/LTspice/sunsimnew_modules_pcb/U/{t}"

# Initialize dictionaries to store data for specific files and unique prefixes
data = {}
results_by_N = {n: [] for n in range(2, 10)}
results_by_prefix = {}

# List all files in the common directory and filter for .csv files
all_files_common = os.listdir(directory)
csv_files = [f for f in all_files_common if f.endswith('.csv')]

# List all files in the LTspice directory and filter for .txt files with 't' in the name
all_files_lt = os.listdir(directory_lt)
txt_files = [f for f in all_files_lt if f.endswith('.txt') and t in f]
# Parse each txt file and load data into the dictionary
for filename in txt_files:
    file_path = os.path.join(directory_lt, filename)
    
    if os.path.exists(file_path):
        # Load the .txt file into a pandas DataFrame with specific column names
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=['vpv', 'V(Vpv)*I(D1)', 'I(D1)'])
        # Store the dataframe in the dictionary, removing the '.txt' from the filename
        data[filename.replace('.txt', '')] = df


def get_N_and_prefix_from_filename(filename):
    match = re.search(r'(.+?)_U(?:_(\d))?', filename)  # Capture the prefix before '_U' and optionally the digit after it
    if match:
        prefix = match.group(1)  # Get the string before '_U'
        N = int(match.group(2)) if match.group(2) and match.group(2).isdigit() else 9  # Default to 9 if no valid number is found
        return prefix, N
    return 'Unknown', 9  # Default if not found

# Load each CSV file and store it in the dictionary
for filename in csv_files:
    file_path = os.path.join(directory, filename)
    
    if os.path.exists(file_path):
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(file_path)
        # Store the dataframe in the dictionary, removing the '.csv' from the filename
        data[filename.replace('.csv', '')] = df


# Initialize constants
A = 12.5 * 12.5
W = A * 0.1
C = 6

def extract_parameters(voltage, current, N, label, p=None):
    try:
        # Calculate basic parameters
        power = [v * i for v, i in zip(voltage, current)]
        closest_current_index = min(range(len(current)), key=lambda i: abs(current[i]))
        Voc = voltage[closest_current_index]
        closest_voltage_index = min(range(len(voltage)), key=lambda i: abs(voltage[i]))
        Isc = current[closest_voltage_index]
        MPP = max(power)
        MPP_index = power.index(MPP)
        Vmp = voltage[MPP_index]
        Imp = current[MPP_index]
        
        FF = (MPP / (Isc * Voc)) * 100
        Ef = (MPP / (W * N)) * 100
        
        # Rsh calculation
        v_rsh_min = 0
        v_rsh_max = Vmp * 0.5
        indices_in_range_rsh = [i for i, v in enumerate(voltage) if v_rsh_min <= v <= v_rsh_max]
        voltage_in_range_rsh = [voltage[i] for i in indices_in_range_rsh]
        current_in_range_rsh = [current[i] for i in indices_in_range_rsh]
        
        if len(voltage_in_range_rsh) > 1:
            slope_rsh, intercept_rsh, _, _, _ = linregress(voltage_in_range_rsh, current_in_range_rsh)
            Rsh = -1 / slope_rsh if slope_rsh != 0 else None
        else:
            Rsh = None

        # Rs calculation
        v_rs_min = Voc*0.95
        v_rs_max = Voc
        indices_in_range_rs = [i for i, v in enumerate(voltage) if v_rs_min <= v <= v_rs_max]
        voltage_in_range_rs = [voltage[i] for i in indices_in_range_rs]
        current_in_range_rs = [current[i] for i in indices_in_range_rs]

        if len(voltage_in_range_rs) > 1:
            slope_rs, intercept_rs, _, _, _ = linregress(voltage_in_range_rs, current_in_range_rs)
            Rs = -1 / slope_rs if slope_rs != 0 else None
        else:
            Rs = None
        
        if p == 'plot':
            # Plot the V-I curve with Rs and Rsh slopes
            plot_voltage_current(voltage, current, Vmp, Voc, v_rsh_min, v_rsh_max, v_rs_min, v_rs_max, slope_rsh, intercept_rsh, slope_rs, intercept_rs, label)
        
        if t == 'parallel':
            Voc =  Voc / C
            Isc = (Isc * 1000) / ((A / C) * N)      
            MPP = (MPP * 1000) / (A * N)
            Vmp = Vmp / C
            Imp = (Imp * 1000) / ((A / C) * N)  
            Rsh = Rsh * N
            
        elif t == 'series':
            Voc =  Voc / (C * N)
            Isc = (Isc * 1000) / (A / C)
            MPP = (MPP * 1000) / (A * N)
            Vmp = Vmp / (C * N)
            Imp = (Imp * 1000) / (A / C)
            Rs = Rs / N
        
        else:
            Voc = Voc
            Isc = Isc
            MPP = MPP
            Vmp = Vmp
            Imp = Imp
            Rsh = Rsh
            Rs = Rs
            
        return Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs
    except Exception as e:
        print(f"Error extracting parameters for {label}: {e}")
        return None, None, None, None, None, None, None, None, None
    
def plot_voltage_current(voltage, current, Vmp, Voc, v_rsh_min, v_rsh_max, v_rs_min, v_rs_max, slope_rsh, intercept_rsh, slope_rs, intercept_rs, label):
    plt.figure(figsize=(10, 6))
    plt.plot(voltage, current, label="V-I Curve", color="blue", linestyle='-', linewidth=0.5)
    
    # Mark regions for Rsh and Rs
    plt.axvline(v_rsh_min, color="green", linestyle="--", label="V_rsh_min")
    plt.axvline(v_rsh_max, color="purple", linestyle="--", label="V_rsh_max")
    plt.axvline(v_rs_min, color="red", linestyle="--", label="V_rs_min")
    plt.axvline(v_rs_max, color="orange", linestyle="--", label="V_rs_max")

    # Plot Rsh slope line from min(voltage) to max(voltage)
    if slope_rsh is not None:
        rsh_x = [min(voltage), max(voltage)]
        rsh_y = [slope_rsh * x + intercept_rsh for x in rsh_x]
        plt.plot(rsh_x, rsh_y, label="Rsh Slope", color="purple", linestyle="-.")

    # Plot Rs slope line from Vmp * 0.9 to max(voltage)
    if slope_rs is not None:
        rs_x = [Vmp * 0.95, max(voltage)]
        rs_y = [slope_rs * x + intercept_rs for x in rs_x]
        plt.plot(rs_x, rs_y, label="Rs Slope", color="red", linestyle="-.")

    # Add labels and legend
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(f"V-I Curve with Rs and Rsh Ranges for {label}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Process each loaded DataFrame (for both csv and txt files)
for label, df in data.items():
    try:
        # Extract voltage and current columns based on file type
        if 'vpv' in df.columns and 'I(D1)' in df.columns:
            voltage = df['vpv'].tolist()
            current = df['I(D1)'].tolist()
        else:
            voltage = df['Voltage'].tolist()
            current = df['Current'].tolist()

        # Extract N and prefix from filename
        prefix, N = get_N_and_prefix_from_filename(label)
        
        # Call the extract_parameters function
        Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs = extract_parameters(voltage, current, N, label)
        
        # Append the result to results_by_N
        results_by_N[N].append([label, Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs])

        # Append the result to results_by_prefix if filename contains "nocell" or "switch"
        if 'nocell' in label or 'switch' in label:
            if prefix not in results_by_prefix:
                results_by_prefix[prefix] = []
            results_by_prefix[prefix].append([label, Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs])
    
    except Exception as e:
        print(f"Error processing file {label}: {e}")

# Convert results_by_N lists into separate DataFrames for each N
dataframes_by_N = {}
for N, results in results_by_N.items():
    columns = ['File', 'Voc (V)', 'Jsc (mA/cm2)', 'Vmp (V)', 'Jmp (mA/cm2)', 'MPP (mW/cm2)', 'FF (%)', 'ETA (%)', 'Rsh (ohm)', 'Rs (ohm)']
    dataframes_by_N[N] = pd.DataFrame(results, columns=columns)

# # Print each DataFrame for N
# for N, df in dataframes_by_N.items():
#     print(f"\nDataFrame for N = {N}:\n")
#     df = df.fillna("N/A")  # Replace None with "N/A" in the output
#     print(tabulate(df, headers='keys', tablefmt='psql'))

# Convert results_by_prefix lists into separate DataFrames for each prefix
dataframes_by_prefix = {}
for prefix, results in results_by_prefix.items():
    columns = ['File', 'Voc (V)', 'Jsc (mA/cm2)', 'Vmp (V)', 'Jmp (mA/cm2)', 'MPP (mW/cm2)', 'FF (%)', 'ETA (%)', 'Rsh (ohm)', 'Rs (ohm)']
    dataframes_by_prefix[prefix] = pd.DataFrame(results, columns=columns)

# # Print each DataFrame for each unique prefix
# for prefix, df in dataframes_by_prefix.items():
#     print(f"\nDataFrame for prefix '{prefix}':\n")
#     df = df.fillna("N/A")  # Replace None with "N/A" in the output
#     print(tabulate(df, headers='keys', tablefmt='psql'))

# List to store all results from directory_lt (the .txt files)
lt_data_all = []

# Process each file and collect data from directory_lt (the .txt files)
for label, df in data.items():
    try:
        # Check if the file is from `directory_lt` by looking for specific columns
        if 'vpv' in df.columns and 'I(D1)' in df.columns:
            # Extract voltage and current columns
            voltage = df['vpv'].tolist()
            current = df['I(D1)'].tolist()

            # Extract N and prefix from filename
            prefix, N = get_N_and_prefix_from_filename(label)

            # Call the extract_parameters function to get parameters
            Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs = extract_parameters(voltage, current, N, label)

            # Append the results for this file to the lt_data_all list
            lt_data_all.append([label, Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs])

    except Exception as e:
        print(f"Error processing file {label}: {e}")

# Convert the lt_data_all list into a single DataFrame
lt_data_columns = ['File', 'Voc (V)', 'Jsc (mA/cm2)', 'Vmp (V)', 'Jmp (mA/cm2)', 'MPP (mW/cm2)', 'FF (%)', 'ETA (%)', 'Rsh (ohm)', 'Rs (ohm)']
lt_data_df = pd.DataFrame(lt_data_all, columns=lt_data_columns)

# # Display the DataFrame
# print("\nCombined DataFrame for all directory_lt files:\n")
# print(tabulate(lt_data_df.fillna("N/A"), headers='keys', tablefmt='psql'))
    


# # Plot figures for each unique prefix with relevant data
# for prefix, df in dataframes_by_prefix.items():
#     plt.figure(figsize=(10, 6))
    
#     for index, row in df.iterrows():
#         # Extracting data for plotting
#         label = row['File']
#         # Assuming voltage and current data are stored in the original data
#         voltage = data[label]['Voltage'].tolist()
#         current = data[label]['Current'].tolist()
#         # Call the extract_parameters function to get the parameters
#         Voc, Isc, Vmp, Imp, MPP, FF, Ef, Rsh, Rs = extract_parameters(voltage, current, N, label)
        
#         plt.plot(voltage, current, label=label)  # Plot each file's V-I curve

#     # Add vertical line at x=0
#     plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
#     # Add horizontal line at y=0
#     plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
#     plt.xlabel("Voltage (V)")
#     plt.ylabel("Current (A)")
#     plt.title(f"V-I Curves for Prefix '{prefix}'")
#     plt.legend(loc="lower left")
#     plt.grid(True)
#     plt.show()

# # Assuming dataframes_by_prefix and dataframes structure is as per your code
# parameter_names = ['Voc (V)', 'Isc (mA/cm2)', 'Vmp (V)', 'Imp (mA/cm2)', 'MPP (mW/cm2)', 'FF (%)', 'ETA (%)', 'Rsh (ohm)', 'Rs (ohm)']

# for prefix, df in dataframes_by_prefix.items():
#     num_params = len(parameter_names)
#     nrows = (num_params // 3) + (num_params % 3 > 0)
#     fig, axs = plt.subplots(nrows, 3, figsize=(15, nrows * 4))
#     fig.suptitle(f"Parameter Values for Prefix '{prefix}'", fontsize=16)
#     axs = axs.flatten()
    
#     grouped_data = {}
#     for index, row in df.iterrows():
#         label = row['File']
#         param_values = {param: row[param] for param in parameter_names}
#         match = re.search(r'_(\d+)_', label)
#         if match:
#             N = match.group(1)
#             if N not in grouped_data:
#                 grouped_data[N] = {'nocell': {}, 'switch': {}}
#             if '_nocell' in label:
#                 grouped_data[N]['nocell'] = param_values
#             elif '_switch' in label:
#                 grouped_data[N]['switch'] = param_values

#     for i, param in enumerate(parameter_names):
#         x_nocell = []
#         y_nocell = []
#         x_switch = []
#         y_switch = []

#         for N, values in grouped_data.items():
#             if 'nocell' in values and values['nocell']:
#                 x_nocell.append(int(N))
#                 y_nocell.append(values['nocell'][param])
#             if 'switch' in values and values['switch']:
#                 x_switch.append(int(N))
#                 y_switch.append(values['switch'][param])
        
#         axs[i].scatter(x_nocell, y_nocell, color='red', marker='o', label='nocell')
#         axs[i].scatter(x_switch, y_switch, color='blue', marker='o', label='switch')

#         # Initialize slope text for each subplot
#         slope_text = ""

#         if len(x_nocell) > 1:
#             slope_nocell, intercept_nocell, _, _, _ = linregress(x_nocell, y_nocell)
#             trendline_nocell = [slope_nocell * x + intercept_nocell for x in x_nocell]
#             axs[i].plot(x_nocell, trendline_nocell, color='red', linestyle='--', label='nocell trend')
#             slope_text += f"nocell slope: {slope_nocell:.4f}  "

#         if len(x_switch) > 1:
#             slope_switch, intercept_switch, _, _, _ = linregress(x_switch, y_switch)
#             trendline_switch = [slope_switch * x + intercept_switch for x in x_switch]
#             axs[i].plot(x_switch, trendline_switch, color='blue', linestyle='--', label='switch trend')
#             slope_text += f"switch slope: {slope_switch:.4f}"

#         # Add the slope text to the top center of each subplot
#         axs[i].text(0.5, 1.1, slope_text, ha='center', va='bottom', color='black', fontsize=10, transform=axs[i].transAxes)

#         axs[i].set_ylabel(param)
#         axs[i].set_xlabel('N value')
#         axs[i].set_xticks(range(min(map(int, grouped_data.keys())), max(map(int, grouped_data.keys())) + 1))
#         axs[i].grid(True)
#         axs[i].set_title(param)
    
#     axs[-1].legend(loc='best')
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
    







# Filter lt_data_df to exclude rows with '_9' in the 'File' column
lt_data_df = lt_data_df[~lt_data_df['File'].str.contains('_9', case=False, na=False)]

# Define colors for each prefix and condition
colors = {
    'Series': {'nocell': 'red', 'switch': 'blue'},
    'Series_c+': {'nocell': 'darkred', 'switch': 'darkblue'},
    'Series_c-': {'nocell': 'lightcoral', 'switch': 'lightblue'},
    'Parallel': {'nocell': 'red', 'switch': 'blue'},
    'Parallel_c+': {'nocell': 'darkred', 'switch': 'darkblue'},
    'Parallel_c-': {'nocell': 'lightcoral', 'switch': 'lightblue'}
}

# Define the parameters to plot
parameters_to_plot = ['Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 
                      'Vmp (V)', 'Jmp (mA/cm2)', 'MPP (mW/cm2)', 
                      'ETA (%)', 'Rsh (ohm)', 'Rs (ohm)']

# num_params = len(parameters_to_plot)
# nrows = (num_params // 4) + (num_params % 4 > 0)  # 4 columns

# # Create a single figure for all prefixes
# fig, axs = plt.subplots(nrows, 4, figsize=(20, nrows * 4))
# fig.suptitle(f"Parameter Values for {t}", fontsize=16)
# axs = axs.flatten()

# # Legend items to ensure they appear even if the last subplot is empty
# legend_elements = [
#     plt.Line2D([0], [0], marker='o', color='w', label='nocell', markerfacecolor='red', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='switch', markerfacecolor='blue', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='nocell_c+', markerfacecolor='darkred', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='switch_c+', markerfacecolor='darkblue', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='nocell_c-', markerfacecolor='lightcoral', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='switch_c-', markerfacecolor='lightblue', markersize=10),
#     plt.Line2D([0], [0], marker='x', color='w', label='Ideal Value (underscore)', markerfacecolor='green', markersize=10),
#     plt.Line2D([0], [0], marker='x', color='w', label='Ideal Value (no underscore)', markerfacecolor='black', markersize=10),
# ]

# # Iterate over each prefix and its dataframe in dataframes_by_prefix
# for prefix, df in dataframes_by_prefix.items():
#     if prefix.startswith('Series') or prefix.startswith('Parallel'):
#         # Filter out rows with '_9' in the 'File' column
#         df = df[~df['File'].str.contains('_9', case=False, na=False)]
        
#         grouped_data = {}
#         for index, row in df.iterrows():
#             label = row['File']
#             param_values = {param: row[param] for param in parameters_to_plot}
#             match = re.search(r'_(\d+)_', label)
#             if match:
#                 N = match.group(1)
#                 if N not in grouped_data:
#                     grouped_data[N] = {'nocell': {}, 'switch': {}}
#                 if '_nocell' in label:
#                     grouped_data[N]['nocell'] = param_values
#                 elif '_switch' in label:
#                     grouped_data[N]['switch'] = param_values

#         # Plot each parameter for this prefix
#         for i, param in enumerate(parameters_to_plot):
#             x_nocell, y_nocell, x_switch, y_switch = [], [], [], []

#             for N, values in grouped_data.items():
#                 if 'nocell' in values and values['nocell']:
#                     x_nocell.append(int(N))
#                     y_nocell.append(values['nocell'][param])
#                 if 'switch' in values and values['switch']:
#                     x_switch.append(int(N))
#                     y_switch.append(values['switch'][param])

#             # Scatter plot for `nocell` and `switch` using defined colors
#             axs[i].scatter(x_nocell, y_nocell, color=colors[prefix]['nocell'], marker='o', label=f'{prefix} nocell')
#             axs[i].scatter(x_switch, y_switch, color=colors[prefix]['switch'], marker='o', label=f'{prefix} switch')

#             # Customize axes
#             axs[i].set_ylabel(param)
#             axs[i].set_xlabel('N value')
#             axs[i].set_title(param)
#             axs[i].grid(True)

# # Inside your parameter plotting loop for lt_data_df
# for i, param in enumerate(parameters_to_plot):
#     for index, row in lt_data_df.iterrows():
#         label = row['File']
#         match = re.search(r'_(\d+)', label)  # Extract N value from the filename
#         if match:
#             N = int(match.group(1))
#             value = row[param]

#             # Determine color based on whether the file name ends with an underscore
#             color = 'green' if label.endswith('_') else 'black'
#             marker = 'x'
#             size = 125
            
#             # Plot the point on the relevant axis
#             axs[i].scatter(N, value, color=color, marker=marker, s=size,
#                            label='Ideal Value' if color == 'black' and i == 0 and index == 0 else "")
            
# # Add legends to the last subplot
# axs[-1].legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=12)

# # Adjust layout and show
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Save the figure as SVG
# plt.savefig(f'parameter_values_{t}.svg', format='svg')

# plt.show()


# # Create a 3x3 grid for the parameters
# fig, axs = plt.subplots(3, 3, figsize=(15, 12))
# fig.suptitle(f"Parameter Values for {t}", fontsize=16)
# axs = axs.flatten()  # Flatten for easier indexing

# # Define colors and legend elements for series and parallel data
# legend_elements = [
#     mlines.Line2D([0], [0], marker='o', color='w', label='nocell', markerfacecolor='red', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='switch', markerfacecolor='blue', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='nocell_c+', markerfacecolor='darkred', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='switch_c+', markerfacecolor='darkblue', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='nocell_c-', markerfacecolor='lightcoral', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='switch_c-', markerfacecolor='lightblue', markersize=10),
# ]

# # Ideal value legend elements
# ideal_legend_elements = [
#     mlines.Line2D([0], [0], marker='x', color='green', label='Ideal Value (reverse)', markersize=10),
#     mlines.Line2D([0], [0], marker='x', color='black', label='Ideal Value', markersize=10),
# ]

# # Initialize a set to track which ideal values were added to the legend
# ideal_values_added = set()

# # Plot each parameter on its own subplot
# for i, param in enumerate(parameters_to_plot):
#     for prefix, df in dataframes_by_prefix.items():
#         # Filter the dataframe and group data by 'nocell' and 'switch'
#         df_filtered = df[~df['File'].str.contains('_9', case=False, na=False)]
        
#         grouped_data = {}
#         for _, row in df_filtered.iterrows():
#             label = row['File']
#             param_values = {p: row[p] for p in parameters_to_plot}
#             match = re.search(r'_(\d+)_', label)
#             if match:
#                 N = match.group(1)
#                 if N not in grouped_data:
#                     grouped_data[N] = {'nocell': {}, 'switch': {}}
#                 if '_nocell' in label:
#                     grouped_data[N]['nocell'] = param_values
#                 elif '_switch' in label:
#                     grouped_data[N]['switch'] = param_values

#         # Prepare data to plot for this parameter
#         x_nocell, y_nocell, x_switch, y_switch = [], [], [], []
#         for N, values in grouped_data.items():
#             if 'nocell' in values and values['nocell']:
#                 x_nocell.append(int(N))
#                 y_nocell.append(values['nocell'][param])
#             if 'switch' in values and values['switch']:
#                 x_switch.append(int(N))
#                 y_switch.append(values['switch'][param])

#         # Scatter plot for each type using predefined colors
#         axs[i].scatter(x_nocell, y_nocell, color=colors[prefix]['nocell'], marker='o', label=f'{prefix} nocell')
#         axs[i].scatter(x_switch, y_switch, color=colors[prefix]['switch'], marker='o', label=f'{prefix} switch')

#         # Customize each axis
#         axs[i].set_ylabel(param)
#         axs[i].set_xlabel('N value')
#         axs[i].set_title(param)
#         axs[i].grid(True)

#     # Plot the ideal values (lt_data_df) on each parameter axis
#     for _, row in lt_data_df.iterrows():
#         label = row['File']
#         match = re.search(r'_(\d+)', label)
#         if match:
#             N = int(match.group(1))
#             value = row[param]

#             # Determine color based on filename suffix
#             color = 'green' if label.endswith('_') else 'black'
#             marker = 'x'
#             size = 125

#             # Plot the point on the relevant axis
#             axs[i].scatter(N, value, color=color, marker=marker, s=size)

#             # Add ideal values to the legend if not already added
#             if (label.endswith('_') and 'Ideal Value (underscore)' not in ideal_values_added) or \
#                (not label.endswith('_') and 'Ideal Value (no underscore)' not in ideal_values_added):
#                 ideal_values_added.add('Ideal Value (underscore)' if label.endswith('_') else 'Ideal Value (no underscore)')

# # Hide any unused subplots
# for j in range(len(parameters_to_plot), len(axs)):
#     axs[j].axis('off')

# # Combine legend items for series/parallel data and ideal values
# all_legend_elements = legend_elements + ideal_legend_elements

# # Place the legend below the figure with 3 columns
# fig.legend(handles=all_legend_elements, loc='lower center', 
#            ncol=4, fontsize=12, frameon=False)

# # Adjust layout and save the figure
# plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for title and legend
# plt.savefig(f'parameter_values_{t}.svg', format='svg')
# plt.show()








# # Filter out rows with 'c-' in the 'File' column and store in new dataframes
# filtered_dataframes = {prefix: df[~df['File'].str.contains('c-', case=False, na=False)]
#                        for prefix, df in dataframes_by_prefix.items()}

# # Create a 3x3 grid for the parameters
# fig, axs = plt.subplots(3, 3, figsize=(15, 12))
# fig.suptitle(f"Parameter Values for {t}", fontsize=16)
# axs = axs.flatten()  # Flatten for easier indexing

# # Define colors and legend elements only for 'nocell', 'nocell_c+' and 'Ideal Value'
# legend_elements = [
#     mlines.Line2D([0], [0], marker='o', color='w', label='nocell', markerfacecolor='red', markersize=10),
#     mlines.Line2D([0], [0], marker='o', color='w', label='nocell_c+', markerfacecolor='darkred', markersize=10),
#     mlines.Line2D([0], [0], marker='x', color='black', label='Ideal Value', markersize=10),
# ]

# # Initialize a set to track which ideal values were added to the legend
# ideal_values_added = set()

# # Plot each parameter on its own subplot
# for i, param in enumerate(parameters_to_plot):
#     for prefix, df_filtered in filtered_dataframes.items():  # Use filtered dataframes
#         # Filter the dataframe to include only 'nocell' and 'nocell_c+'
#         grouped_data = {}
#         for _, row in df_filtered.iterrows():
#             label = row['File']
#             param_values = {p: row[p] for p in parameters_to_plot}
#             match = re.search(r'_(\d+)_', label)
#             if match:
#                 N = match.group(1)
#                 if N not in grouped_data:
#                     grouped_data[N] = {'nocell': {}, 'nocell_c+': {}}
#                 if 'nocell_c+' in label:
#                     grouped_data[N]['nocell_c+'] = param_values
#                 elif 'nocell' in label and 'nocell_c+' not in label:
#                     grouped_data[N]['nocell'] = param_values

#         # Prepare data to plot for this parameter
#         x_nocell, y_nocell, x_nocell_c, y_nocell_c = [], [], [], []
#         for N, values in grouped_data.items():
#             if 'nocell' in values and values['nocell']:
#                 x_nocell.append(int(N))
#                 y_nocell.append(values['nocell'][param])
#             if 'nocell_c+' in values and values['nocell_c+']:
#                 x_nocell_c.append(int(N))
#                 y_nocell_c.append(values['nocell_c+'][param])

#         # Scatter plot for each type with distinct colors
#         axs[i].scatter(x_nocell, y_nocell, color='red', marker='o', label='nocell')
#         axs[i].scatter(x_nocell_c, y_nocell_c, color='darkred', marker='o', label='nocell_c+')

#         # Customize each axis
#         axs[i].set_ylabel(param)
#         axs[i].set_xlabel('N value')
#         axs[i].set_title(param)
#         axs[i].grid(True)

    # # Plot the ideal values (lt_data_df) on each parameter axis
    # for _, row in lt_data_df.iterrows():
    #     label = row['File']
    #     match = re.search(r'_(\d+)', label)
    #     if match and not label.endswith('_'):  # Exclude 'Ideal Value (reverse)' by skipping labels with underscore
    #         N = int(match.group(1))
    #         value = row[param]

    #         # Plot the point on the relevant axis
    #         axs[i].scatter(N, value, color='black', marker='x', s=125)

    #         # Add 'Ideal Value' to the legend if not already added
    #         if 'Ideal Value' not in ideal_values_added:
    #             ideal_values_added.add('Ideal Value')

# # Hide any unused subplots
# for j in range(len(parameters_to_plot), len(axs)):
#     axs[j].axis('off')

# # Combine legend items for selected data and ideal values
# fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12, frameon=False)

# # Adjust layout and save the figure
# plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for title and legend
# plt.savefig(f'parameter_values_{t}.svg', format='svg')
# plt.show()



# Check and set the prefix variables based on `t`
if t == 'series':
    t1, t2, t3, t4 = 'Series', 'Series_c+', '_series', 'Series_c-'
elif t == 'parallel':
    t1, t2, t3, t4 = 'Parallel', 'Parallel_c+', '_parallel', 'Parallel_c-'
else:
    raise ValueError("Invalid value for 't'. Expected 'series' or 'parallel'.")

# Filter out unwanted DataFrames
filtered_dataframes = {k: v for k, v in dataframes_by_prefix.items() if k != t4}

# Define legend elements
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='#0066CC', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='Common', markerfacecolor='#CC6600', markersize=10),
    mlines.Line2D([0], [0], color='black', label='Theoretical', markersize=10),
]

# Filter and prepare DataFrames
series_filtered_df, series_cplus_filtered_df = None, None

for key, df in filtered_dataframes.items():
    # Filter rows containing the required numbers and "nocell"
    filtered_df = df[df['File'].str.contains(r'_(?:2|3|4|5|6|7|8)_nocell', regex=True)]
    # Fix for SettingWithCopyWarning
    filtered_df.loc[:, 'File'] = filtered_df['File'].str.extract(r'_(\d+)_')[0]

    if key == t1:
        series_filtered_df = filtered_df
    elif key == t2:
        filtered_df.loc[:, 'File'] = filtered_df['File'].str.replace(t2, t1, regex=False)
        series_cplus_filtered_df = filtered_df

# Merge DataFrames
if series_filtered_df is not None and series_cplus_filtered_df is not None:
    merged_df = pd.merge(series_filtered_df, series_cplus_filtered_df, on='File', suffixes=(t3, '_cplus'))
else:
    raise ValueError("Missing required DataFrames for merging.")

# Replace `desired_order` with the column names (excluding 'File') in the desired sequence
desired_order = ['Voc (V)', 'Jsc (mA/cm2)', 'FF (%)', 'Vmp (V)', 'Jmp (mA/cm2)', 'MPP (mW/cm2)', 'ETA (%)', 'Rs (ohm)', 'Rsh (ohm)']

# Ensure the columns exist in the merged DataFrame and map them to their respective suffix
ordered_columns = [
    col for col in desired_order if f"{col}{t3}" in merged_df.columns
]  # Only keep columns present in the DataFrame

# Define a global font size variable
font_size = 13  # Change this value to adjust font sizes globally

# Dynamically calculate the number of rows for subplots
num_plots = len(ordered_columns)
rows = (num_plots + 2) // 3  # Calculate the required number of rows for a 3-column grid
fig, axs = plt.subplots(rows, 3, figsize=(10, rows * 3))
axs = axs.ravel()

# Define mapping for y-axis labels with superscripts
ylabel_map = {
    'Voc (V)': r'Cell $\mathit{V_{OC}}$ (V)',
    'Jsc (mA/cm2)': r'$\mathit{J_{SC}}$ (mA/cm$^2$)',
    'FF (%)': r'$\mathit{FF}$ (%)',
    'Vmp (V)': r' Cell $\mathit{V_{MP}}$ (V)',
    'Jmp (mA/cm2)': r'$\mathit{J_{MP}}$ (mA/cm$^2$)',
    'MPP (mW/cm2)': r'$\mathit{MPP}$ (mW/cm$^2$)',
    'ETA (%)': r'$\mathit{ETA}$ (%)',
    'Rs (ohm)': r'$\mathit{R_{S}}$ (Ω)',
    'Rsh (ohm)': r'$\mathit{R_{SH}}$ (Ω)',
}

subplot_index = 0
for col_name in ordered_columns:
    column = f"{col_name}{t3}"
    if column in merged_df.columns:
        if subplot_index >= len(axs):
            print(f"Skipping column '{col_name}' as there are no more subplots available.")
            break
        
        width = 0.3
        series_data = merged_df[column]
        series_cplus_data = merged_df[column.replace(t3, '_cplus')]
        x_positions = range(len(merged_df))
        
        # Adjust y-limits to include ideal values
        # min_value = min(series_data.min(), series_cplus_data.min())
        min_value = 0
        max_value = max(series_data.max(), series_cplus_data.max())

        # Set hatch line width globally
        mpl.rcParams['hatch.linewidth'] = 0.5  # Adjust thickness (default is 1.0)
        
        # Collect and plot ideal values as bars with transparent filling and thin diagonal lines
        for _, row in lt_data_df.iterrows():
            label = row['File']
            match = re.search(r'_(\d+)', label)
            if match:
                N = int(match.group(1))
                if label.endswith('_'):
                    continue
                if N in range(2, 9):
                    value = row[col_name]
                    ideal_position = merged_df[merged_df['File'] == str(N)].index[0]
        
                    # Update min and max values
                    min_value = min(min_value, value)
                    max_value = max(max_value, value)
        
                    # Plot ideal value as a transparent bar with thin diagonal lines and solid border
                    axs[subplot_index].bar(
                        x=ideal_position,  # X-position of the bar
                        height=value,  # Height of the bar (the value you want to plot)
                        width=0.6,  # Bar width (adjust as needed)
                        color='none',  # No fill color (transparent)
                        alpha=1,  # Transparency of the filling (0 = fully transparent, 1 = opaque)
                        hatch='////',  # Diagonal hatch pattern (you can change this to other patterns like '\\', '-', etc.)
                        edgecolor='black',  # Solid border color of the bar (this won't be transparent)
                        align='center'  # Align the bars to the center
                    )
                    
        # Bar plot for 'Series' and 'Series_c+'
        axs[subplot_index].bar([pos - width / 2 for pos in x_positions], series_data, width=width, color='#0066CC', alpha=0.75)
        axs[subplot_index].bar([pos + width / 2 for pos in x_positions], series_cplus_data, width=width, color='#CC6600', alpha=0.75)

        # Set x-axis and y-axis labels
        axs[subplot_index].set_xlabel('Number of connected modules', fontsize=font_size)
        ylabel_text = ylabel_map.get(col_name, col_name)
        axs[subplot_index].set_ylabel(ylabel_text, fontsize=font_size)

        # Set x-tick labels
        axs[subplot_index].set_xticks(x_positions)
        axs[subplot_index].set_xticklabels(merged_df['File'], fontsize=font_size)

        

        # Add padding to y-limits
        padding = 0.05 * (max_value - min_value)
        axs[subplot_index].set_ylim(min_value - padding, max_value + padding)

        subplot_index += 1

# Adjust layout for better spacing
plt.tight_layout()

# Combine legend items for selected data and ideal values
fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.035),  # Move the legend slightly lower
    ncol=4,
    fontsize=14,
    frameon=False
)

# Add a title to the figure and adjust its position
fig.suptitle(
    f'Comparison of Parameter Values - {t1}', 
    fontsize=16, 
    fontweight='bold', 
    y=1.02  # Set the vertical position of the title slightly higher
)

# Save the figure as an SVG
plt.savefig(f'parameter_values_{t}.svg', format='svg', bbox_inches='tight')

# Show the plot
plt.show()