# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:00:10 2024

@author: ruiui
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

# Directory where the files are located
directory = "C:/Users/ruiui/Desktop/solar plug and play old stuff/sun simulator new/New modules testing"
# Define the new order for the File column
new_order = ['C2', 'C5', 'C7', 'C14', 'C15', 'C16', 'C17', 'C20', 'C27']


# directory = "C:/Users/ruiui/Desktop/sun simulator new/New modules testing/middle piece array"
# # Define the new order for the File column
# new_order = ['C1', 
#              # 'C3', 
#              'C21', 'C28', 'C30']



# W = Area * Irradiance
W = 12.5 * 12.5 * 0.1

# Function to calculate percentage difference between two values
def percentage_difference(value1, value2):
    try:
        return ((value2 - value1) / value1) * 100
    except ZeroDivisionError:
        return None

data = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Assuming the files are CSV
        try:
            # Extract the number from the filename (C{number} or C{number}PCB)
            if 'PCB' in filename:
                number = filename.replace('PCB.csv', '').replace('C', '')
                label = f'C{number}PCB'
            else:
                number = filename.replace('.csv', '').replace('C', '')
                label = f'C{number}'

            # Load the CSV into a pandas DataFrame
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            df_filtered = df[(df['Voltage'] > 0) & (df['Current'] > 0)]
            
            # Extract the filtered Voltage and Current as numpy arrays
            V = df_filtered['Voltage'].values
            I = df_filtered['Current'].values
            P = V * I
            
            # Interpolate the data to have the same voltage points for all curves
            interp_func = interp1d(V, I, kind='linear')
            voltage_axis = np.linspace(V.min(), V.max(), num=100)  # Choose the number of desired voltage points
            interpolated_current = interp_func(voltage_axis)

            interp_func = interp1d(V, P, kind='linear')
            interpolated_power = interp_func(voltage_axis)
            
            # Create a new DataFrame 'df' with means for 'Voltage (V)', 'Current (A)', and 'Power (W)'
            datatu = {
                'Voltage (V)': voltage_axis,
                'Current (A)': interpolated_current,
                'Power (W)': interpolated_power
            }
                
            df = pd.DataFrame(datatu)
            
            df['Power (W)'] = df['Voltage (V)'] * df['Current (A)']  # Calculate photogenerated power

            # Extract main parameters
            Voc = np.max(df['Voltage (V)']) # Open-circuit voltage
            Isc = np.max(df['Current (A)']) # Short-circuit current density
            Pmax = np.max(df['Power (W)']) # Maximum power point
            Vmp = np.array(df['Voltage (V)'].iloc[np.where(df['Power (W)'] == Pmax)])[0] # Voltage at maximum power (MP)
            Imp = np.array(df['Current (A)'].iloc[np.where(df['Power (W)'] == Pmax)])[0] # Current density at MP

            # FF: Fill Factor
            FF = (Pmax / (Isc * Voc)) * 100
            
            # Ef: Efficiency, given by (MPP / W) * 100
            Ef = (Pmax / W) * 100
            
            # Store the results for both C{number} and C{number}PCB
            if number not in data:
                data[number] = {}
            data[number][label] = [Voc, Isc, Vmp, Imp, Pmax, FF, Ef]
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Prepare a list to store the percentage difference results
percentage_differences = []

# Loop through the data to calculate percentage differences between C{number} and C{number}PCB
for number, datasets in data.items():
    if f'C{number}' in datasets and f'C{number}PCB' in datasets:
        C_values = datasets[f'C{number}']
        CPCB_values = datasets[f'C{number}PCB']
        
        # Calculate percentage differences for each parameter
        Voc_diff = percentage_difference(C_values[0], CPCB_values[0])
        Isc_diff = percentage_difference(C_values[1], CPCB_values[1])
        Vmp_diff = percentage_difference(C_values[2], CPCB_values[2])
        Imp_diff = percentage_difference(C_values[3], CPCB_values[3])
        MPP_diff = percentage_difference(C_values[4], CPCB_values[4])
        FF_diff = percentage_difference(C_values[5], CPCB_values[5])
        Ef_diff = percentage_difference(C_values[6], CPCB_values[6])
        
        # Append the percentage difference to the list
        percentage_differences.append([f'C{number}', Voc_diff, Isc_diff, Vmp_diff, Imp_diff, MPP_diff, FF_diff, Ef_diff])

# Create a DataFrame for the percentage differences
columns_diff = ['File', 'Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency']
percentage_diff_df = pd.DataFrame(percentage_differences, columns=columns_diff)

# Reindex the DataFrame according to the new order
percentage_diff_df = percentage_diff_df.set_index('File').reindex(new_order).reset_index()


# Save the percentage differences to a CSV file
percentage_diff_df.to_csv('IV_curve_percentage_differences.csv', index=False)

# ----------------------------------------------------------------------------
# Display the percentage differences as a table using tabulate
print(tabulate(percentage_diff_df, headers='keys', tablefmt='grid'))
# ----------------------------------------------------------------------------

# Reshape the dataframe to a long format
df_melted = pd.melt(percentage_diff_df, id_vars=['File'], 
                    var_name='Parameter', value_name='Percentage Difference')

# Set the plot size
plt.figure(figsize=(12, 6))

# Add the gray box from -0.2% to +0.2%
plt.axhspan(-0.2, 0.2, color='gray', alpha=0.2, zorder=-1)  # Higher transparency

# Add horizontal line at y=0
plt.axhline(0, color='black', linewidth=1.5)

# Create a grouped bar plot using seaborn
sns.barplot(x='Parameter', y='Percentage Difference', hue='File', data=df_melted)

# Add vertical lines between the groups of bar plots (between parameters)
num_params = len(percentage_diff_df.columns) - 1  # Subtract 1 for 'File' column
for i in range(1, num_params):
    plt.axvline(i - 0.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.8)

# Customize the plot
plt.title('Percentage Difference per Parameter', fontsize=16)
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('Percentage Difference (%)', fontsize=12)
plt.legend(title='Module', bbox_to_anchor=(1.005, 1), loc='upper left')
plt.tight_layout()
plt.ylim(-3,3)

# Show the plot
plt.show()




# # Load the new file into a DataFrame
# df_values = pd.read_csv('compiled_data.csv')

# # Melt the df_values to match the structure of df_melted
# df_values_melted = pd.melt(df_values, id_vars=['File'], 
#                             var_name='Parameter', value_name='Error')

# # Merge df_melted and df_values_melted on 'File' and 'Parameter' columns
# df_combined = pd.merge(df_melted, df_values_melted, on=['File', 'Parameter'], how='inner')



# N_CELLS = 9  # Number of cells for spacing in the plot
# group_offset = 0  # Initial offset for grouping bars

# # Create a color map based on unique file values
# unique_files = df_combined['File'].unique()
# colors = plt.cm.get_cmap('tab10', len(unique_files))  # Use a colormap with enough colors

# # Create a mapping from file names to colors
# file_color_map = {file_name: colors(i) for i, file_name in enumerate(unique_files)}

# # Define the custom order for parameters
# custom_order = ['Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency']

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(12, 6))

# # Add the gray box from -0.2% to +0.2%
# plt.axhspan(-0.2, 0.2, color='gray', alpha=0.2, zorder=-1)  # Higher transparency

# # Add horizontal line at y=0
# plt.axhline(0, color='black', linewidth=1.5)

# # Check if the DataFrame is not empty and has required columns
# if not df_combined.empty and all(col in df_combined.columns for col in ["Parameter", "Percentage Difference", "Error", "File"]):
#     # Ensure the Parameter column is categorical with the specified order
#     df_combined['Parameter'] = pd.Categorical(df_combined['Parameter'], categories=custom_order, ordered=True)
    
#     # Multiply Error by 100 for all parameters
#     df_combined['Error'] *= 100 

#     # Loop through each group in the custom order
#     for param_i in custom_order:
#         param_df = df_combined[df_combined['Parameter'] == param_i]
        
#         # Assign offsets using .assign() to avoid modifying the DataFrame directly
#         param_df = param_df.assign(offset=np.arange(len(param_df)) + group_offset)
        
#         # Create bar plot for each file within the parameter group
#         for file_name in param_df['File'].unique():
#             file_df = param_df[param_df['File'] == file_name]
#             ax.bar(file_df["offset"], file_df["Percentage Difference"], 
#                     yerr=file_df["Error"], 
#                     label=file_name if param_i == 'Voc' else "",  # Label only for Voc parameter
#                     color=file_color_map[file_name], 
#                     width=1)  # Adjust bar width
        
#         # Update group offset for the next group
#         group_offset += len(param_df) + 2  # Use the length of the current group to ensure spacing
    
#     # Set x-ticks at group offsets
#     ax.set_xticks(np.arange(4, group_offset, len(param_df) + 2))
    
#     # Set x-tick labels to the custom ordered parameter names
#     ax.set_xticklabels(custom_order)
    
#     # Add vertical lines between the groups of bar plots (between parameters)
#     num_params = [9.5,20.5,31.5,42.5,53.5,64.5]
#     for i in num_params:
#         plt.axvline(i, color='gray', linewidth=1.5, linestyle='--', alpha=0.8)
    
#     plt.title('Percentage Difference per Parameter', fontsize=16)
#     plt.xlabel('Parameter', fontsize=12)
#     plt.ylabel('Percentage Difference (%)', fontsize=12)
#     plt.legend(title='Module', bbox_to_anchor=(1.005, 1), loc='upper left')
    
#     plt.ylim(-3,3)
#     plt.xlim(-1.5,75.5)
    
#     # Show plot
#     plt.tight_layout()
#     plt.show()
