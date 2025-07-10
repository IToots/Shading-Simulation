# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:36:59 2024

@author: ruiui
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

from lists import x, Irradiance, order

# Rotate a matrix 90 degrees
def rotate_matrix_90(matrix):
    transpose_matrix = np.transpose(matrix)
    rotated_matrix = np.flip(transpose_matrix, axis=1)
    rotated_matrix = np.array(rotated_matrix)
    return rotated_matrix

# Apply transformations and rotations to generate matrices
def ALL(matrix, matrix_id):
    transformations = [
        (0, 1, 1, 1, 'Z')  # Transformation set Z
    ]
    
    results = []
    
    for params in transformations:
        transformed_matrix = np.where(matrix == 1, params[0], matrix)
        transformed_matrix = np.where(matrix == 2, params[1], transformed_matrix)
        transformed_matrix = np.where(matrix == 3, params[2], transformed_matrix)
        transformed_matrix = np.where(matrix == 0, params[3], transformed_matrix)
        
        rotated_matrices = [(transformed_matrix, f"{params[4]}_{matrix_id}_0")]
        current_matrix = transformed_matrix
        
        # Apply rotations
        for i in range(1, 4):
            rotated_matrix = rotate_matrix_90(current_matrix)
            rotated_matrices.append((rotated_matrix, f"{params[4]}_{matrix_id}_{i}"))
            current_matrix = rotated_matrix  # Rotate for the next iteration
        
        results.extend(rotated_matrices)
    
    # Remove duplicates
    unique_matrices = []
    for matrix, name in results:
        if not any(np.array_equal(matrix, unique_matrix) for unique_matrix, _ in unique_matrices):
            unique_matrices.append((matrix, name))
    
    return unique_matrices

# Functions to calculate power output (TCT, SP, PER circuits)
def identify_matrix_size(matrix):
    matrix=np.array(matrix)
    if matrix.size == 0:
        return "Matrix is empty"
    num_rows, num_cols = matrix.shape
    if num_rows == num_cols:
        return num_rows
    else:
        return "Matrix is not square"

def analyze_matrix(matrix):
    max_zeros_count = 0
    total_zeros = 0
    for i in range(matrix.shape[0]):
        row_zeros_count = np.sum(matrix[i] == 0)
        total_zeros += row_zeros_count
        if row_zeros_count > max_zeros_count:
            max_zeros_count = row_zeros_count
    remaining_zeros = total_zeros - max_zeros_count
    return max_zeros_count, remaining_zeros

def R_PD_cal(n):
    total_sum = 0
    for i in range(3, 3 + n):
        total_sum += np.log2(i)
    return total_sum

def C_PD_cal(n):
    total_sum = 0
    for k in range(1, n + 1):
        total_sum += (1.1) ** (-k)
    return total_sum

def calculate_total_power(shading_matrix, voltage_matrix, current_matrix):
    total_current = 0
    total_voltage = 0
    for col in range(len(shading_matrix[0])):
        column_voltage = 0
        column_current = float('inf')
        for row in range(len(shading_matrix)):
            shading = shading_matrix[row][col]
            voltage = voltage_matrix[row][col]
            current = current_matrix[row][col] * shading
            column_voltage += voltage
            column_current = min(column_current, current)
        total_voltage = column_voltage
        total_current += column_current
    total_power = total_voltage * total_current
    return total_power

def calculate_power(matrix, W, circ, voltage_matrix=None, current_matrix=None):
    matrix = np.array(matrix)  # Convert the matrix to a NumPy array if it's a list

    if circ in ['TCT', 'PER', 'C1', 'C2']:
        if circ == 'PER':
            # Transformation for 'PER'
            transformed_matrix = np.array([
                [matrix[0][0], matrix[1][1], matrix[2][2]],
                [matrix[1][0], matrix[2][1], matrix[0][2]],
                [matrix[2][0], matrix[0][1], matrix[1][2]]
            ])
        
        elif circ == 'C1':
            # Transformation for 'C1'
            transformed_matrix = np.array([
                [matrix[0][0], matrix[0][1], matrix[2][2]],
                [matrix[1][0], matrix[2][1], matrix[0][2]],
                [matrix[2][0], matrix[1][1], matrix[1][2]]
            ])
        
        elif circ == 'C2':
            # Transformation for 'C2'
            transformed_matrix = np.array([
                [matrix[0][0], matrix[0][1], matrix[2][2]],
                [matrix[1][0], matrix[1][1], matrix[0][2]],
                [matrix[2][0], matrix[2][1], matrix[1][2]]
            ])
        
        else:
            # Default case for 'TCT'
            transformed_matrix = matrix

        n = identify_matrix_size(transformed_matrix)
        
        if n == "Matrix is not square":
            return n
        elif n == "Matrix is empty":
            return None
        
        r, c = analyze_matrix(transformed_matrix)
        R_MF = 71.07 / (n ** 1.3)
        C_MF = 13.87 / (n ** 1.19)
        R_PD = R_MF * R_PD_cal(r)
        C_PD = C_MF * C_PD_cal(c)
        PD = R_PD + C_PD
        P = W * (100 - PD) / 100
        
        if P < 0:
            P = 0
        
        return P
    
    elif circ == 'SP':
        if voltage_matrix is None or current_matrix is None:
            return "Voltage and current matrices are required for SP calculation"
        
        total_output_power = calculate_total_power(matrix, voltage_matrix, current_matrix)
        return total_output_power

# Matrices to process
matrices = {
    "AA": np.array([[1, 2, 0], [2, 3, 0], [0, 0, 0]]),
    "AB": np.array([[1, 1, 2], [2, 2, 3], [0, 0, 0]]),
    "AC": np.array([[1, 1, 1], [2, 2, 2], [0, 0, 0]]),
    
    "AD": np.array([[1, 1, 2], [1, 2, 3], [2, 3, 0]]),
    "AE": np.array([[1, 1, 1], [1, 2, 2], [1, 2, 0]]),
    "AF": np.array([[1, 1, 1], [1, 1, 2], [1, 2, 3]]),
    
    "AG": np.array([[1, 2, 3], [2, 1, 2], [3, 2, 3]]),
    "AH": np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]]),
    
    "AI": np.array([[2, 3, 0], [1, 2, 0], [2, 3, 0]]),
    "AJ": np.array([[2, 2, 3], [1, 1, 2], [2, 2, 3]]),
    "AK": np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]]),
    
    "AL": np.array([[1, 2, 3], [1, 1, 2], [1, 2, 3]]),
    "AM": np.array([[2, 1, 2], [1, 1, 1], [2, 1, 2]]),
    "AN": np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]]),
    "AO": np.array([[1, 1, 2], [1, 1, 1], [1, 1, 2]]),
    
    "AP": np.array([[3, 2, 3], [2, 1, 2], [3, 2, 3]]),
    "AQ": np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]]),
    "AR": np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]),
    
    "AS": np.array([[1, 2, 0], [2, 3, 0], [1, 2, 0]]),
    "AT": np.array([[1, 1, 2], [1, 2, 3], [1, 1, 2]]),
    "AU": np.array([[1, 1, 1], [1, 2, 2], [1, 1, 1]]),
    
    "AV": np.array([[1, 1, 2], [2, 2, 3], [1, 1, 2]]),
    "AW": np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]]),
    
    "AX": np.array([[1, 2, 0], [2, 3, 2], [0, 2, 1]]),
    "AY": np.array([[1, 1, 2], [2, 2, 2], [2, 1, 1]]),
    "AZ": np.array([[1, 1, 2], [2, 1, 2], [2, 1, 1]]),
    
    "BA": np.array([[1, 1, 2], [1, 1, 2], [2, 2, 3]]),
    "BB": np.array([[1, 1, 2], [1, 1, 2], [1, 2, 3]]),
    "BC": np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]]),
    
    "BD": np.array([[1, 2, 0], [1, 2, 3], [1, 1, 2]]),
    "BE": np.array([[2, 1, 2], [2, 1, 2], [2, 1, 1]])
}


# Example voltage and current matrices
voltage_matrix_example = [
    [2.729, 2.732, 2.822], 
    [2.832, 2.732, 2.733],
    [2.788, 2.826, 2.731]
]

current_matrix_example = [
    [0.9084, 0.9244, 0.8855],
    [0.8985, 0.9282, 0.8976],
    [0.9073, 0.9075, 0.9282]
]

# voltage_matrix_example = [
#     [2.733, 2.733, 2.733], 
#     [2.733, 2.733, 2.733],
#     [2.733, 2.733, 2.733]
# ]

# current_matrix_example = [
#     [0.9075, 0.9075, 0.9075],
#     [0.9075, 0.9075, 0.9075],
#     [0.9075, 0.9075, 0.9075]
# ]

# Reference value for power calculation
reference_value = 22.5089475498911

ref = 22.5089475498911


# Create a DataFrame to store the results
results = []

# Iterate over each matrix and its transformations
for matrix_id, matrix in matrices.items():
    transformed_matrices = ALL(matrix, matrix_id)
    
    for transformed_matrix, transformed_id in transformed_matrices:
        # Calculate power for SP circuit
        sp_power = calculate_power(transformed_matrix, reference_value, 'SP', voltage_matrix_example, current_matrix_example)
        
        # Calculate power for TCT circuit
        tct_power = calculate_power(transformed_matrix, reference_value, 'TCT')
        
        # Calculate power for PER circuit
        per_power = calculate_power(transformed_matrix, reference_value, 'PER')
        
        # Calculate power for PER circuit
        c1_power = calculate_power(transformed_matrix, reference_value, 'C1')
        
        # Calculate power for PER circuit
        c2_power = calculate_power(transformed_matrix, reference_value, 'C2')
        
        # Append the results to the list
        results.append({
            "it": transformed_id,
            "sp": sp_power,
            "tct": tct_power,
            "per": per_power,
            "c1": c1_power,
            "c2": c2_power
        })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Extract base code by stripping the numbers
df['base_code'] = df['it'].str.extract(r'Z_([A-Z]+)_')

# Get the highest and lowest values for each base code
max_df = df.groupby('base_code').max().reset_index()
min_df = df.groupby('base_code').min().reset_index()


# Drop the base_code column from the result dataframes
max_df = max_df.drop(columns='it')
min_df = min_df.drop(columns='it')

# Filter out only numeric columns
numeric_df = df.select_dtypes(include='number')

# Calculate the median for each base code
median_df = df.groupby('base_code')[numeric_df.columns].median().reset_index()

# Function to calculate percentage difference
def percentage_difference(df, ref_value):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[f'{col}_percent_diff'] = ((df[col] - ref_value) / ref_value)
    return df

# Calculate percentage differences
max_df = percentage_difference(max_df, ref)
min_df = percentage_difference(min_df, ref)
median_df = percentage_difference(median_df, ref)

# Create a custom order column
max_df['order'] = max_df['base_code'].apply(lambda x: order.index(x) if x in order else float('inf'))
min_df['order'] = min_df['base_code'].apply(lambda x: order.index(x) if x in order else float('inf'))
median_df['order'] = median_df['base_code'].apply(lambda x: order.index(x) if x in order else float('inf'))

# Sort by the custom order column
max_df = max_df.sort_values('order').drop(columns='order')
min_df = min_df.sort_values('order').drop(columns='order')
median_df = median_df.sort_values('order').drop(columns='order')

# Function to calculate cumulative percentage difference
def cumulative_percentage_difference(df):
    numeric_cols = [col for col in df.columns if '_percent_diff' in col]
    for col in numeric_cols:
        df[f'{col}_cumulative'] = df[col].cumsum()
    return df

# Calculate cumulative percentage differences
max_df = cumulative_percentage_difference(max_df)
min_df = cumulative_percentage_difference(min_df)
median_df = cumulative_percentage_difference(median_df)

# Plotting function for selected columns with shaded areas
def plot_cumulative_diff_combined(selected_columns):
    plt.figure(figsize=(14, 7))
    
    # Define colors for each column
    color_map = {
        'sp': 'blue',
        'tct': 'green',
        'per': 'red',
        'c1': 'orange',
        'c2': 'purple'
    }
    
    # Plot for each selected column
    for col in selected_columns:
        # Plot shaded area between min and max
        plt.fill_between(max_df['base_code'], 
                         min_df[f'{col}_percent_diff_cumulative'], 
                         max_df[f'{col}_percent_diff_cumulative'], 
                         color=color_map.get(col, 'black'), 
                         alpha=0.1)
        
        # Plot max_df, min_df, and median_df lines
        plt.plot(max_df['base_code'], max_df[f'{col}_percent_diff_cumulative'], label=f'{col}', color=color_map.get(col, 'black'), linestyle='-')
        plt.plot(min_df['base_code'], min_df[f'{col}_percent_diff_cumulative'], color=color_map.get(col, 'black'), linestyle='-')
        plt.plot(median_df['base_code'], median_df[f'{col}_percent_diff_cumulative'], color=color_map.get(col, 'black'), linestyle='-')
    
    plt.xlabel('Base Code')
    plt.ylabel('Cumulative Percentage Difference')
    plt.title('Cumulative Percentage Difference Comparison')
    plt.legend(loc='upper right')
    
    plt.xlim(left=-0.5, right=30.5)
    
    plt.tight_layout()
    plt.show()

# Example of selecting columns
selected_columns = [
    'sp', 
    'tct', 
    'per',
    'c1',
    'c2'
    ]
plot_cumulative_diff_combined(selected_columns)