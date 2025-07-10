# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:53:37 2025

@author: ruiui
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tabulate import tabulate  # Import tabulate

# Directory where the files are located
directory = "C:/Users/ruiui/Desktop/solar plug and play old stuff/sun simulator new/New modules testing/middle piece array"
# Define the new order for the File column
new_order = ['C1', 'C3', 'C21', 'C28', 'C30']

# W = Area * Irradiance
W = 12.5 * 12.5 * 0.1

# Function to calculate percentage difference between two values
def percentage_difference(value1, value2):
    try:
        return ((value2 - value1) / value1) * 100
    except ZeroDivisionError:
        return None

# Initialize a list to store the data
data_list = []

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
            
            # Store the results in the list
            data_list.append({
                'Label': label,
                'Voc (V)': Voc,
                'Isc (A)': Isc,
                'Vmp (V)': Vmp,
                'Imp (A)': Imp,
                'Pmax (W)': Pmax,
                'FF (%)': FF,
                'Efficiency (%)': Ef
            })
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(data_list)

# Optionally, reorder the DataFrame based on the new_order list
result_df['Order'] = result_df['Label'].apply(lambda x: new_order.index(x.split('PCB')[0]))
result_df = result_df.sort_values(by='Order').drop(columns=['Order'])

# Print the DataFrame using tabulate
print(tabulate(result_df, headers='keys', tablefmt='grid', floatfmt=".4f"))