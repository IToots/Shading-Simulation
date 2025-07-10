# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:07:42 2024

@author: ruiui
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

# Rotate a matrix 90 degrees
def rotate_matrix_90(matrix):
    transpose_matrix = np.transpose(matrix)
    rotated_matrix = np.flip(transpose_matrix, axis=1)
    rotated_matrix = np.array(rotated_matrix)
    return rotated_matrix

# Apply transformations and rotations to generate matrices
def ALL(matrix, matrix_id):
    transformations = [
        # (0.1, 1, 1, 1, 'U'),    # Transformation set U
        # (0.1, 0.6, 0.6, 1, 'US'), # Transformation set US
        # (0.1, 0.6, 1, 1, 'USD'),   # Transformation set USD
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
    # Find the maximum value in the matrix
    max_value = np.max(matrix)
    
    max_below_max_count = 0
    values_below_max_in_row = []  # To store the values below max in the row with most of them
    
    # Collect all values below the maximum from the matrix
    all_below_max = []
    
    # Iterate over each row in the matrix
    for i in range(matrix.shape[0]):
        # Find values in the row that are less than the maximum value
        row_below_max_values = matrix[i][matrix[i] < max_value]
        
        # Add row values to the list of all values below max
        all_below_max.extend(row_below_max_values)
        
        # Count how many values are below the maximum value in this row
        row_below_max_count = len(row_below_max_values)
        
        # Update if this row has the most values below the max so far
        if row_below_max_count > max_below_max_count:
            max_below_max_count = row_below_max_count
            # Store the values as they appear in the row
            values_below_max_in_row = [value for value in matrix[i] if value < max_value]
    
    # Find values that are below the maximum but not in the row with the most values below the maximum
    values_below_max_in_row_count = {}
    for value in values_below_max_in_row:
        if value in values_below_max_in_row_count:
            values_below_max_in_row_count[value] += 1
        else:
            values_below_max_in_row_count[value] = 1
    
    remaining_values = []
    for value in all_below_max:
        if value in values_below_max_in_row_count and values_below_max_in_row_count[value] > 0:
            values_below_max_in_row_count[value] -= 1
        else:
            remaining_values.append(value)
    
    # Sort the remaining values
    remaining_values = sorted(remaining_values)
    
    return values_below_max_in_row, remaining_values

# def R_PD_cal(row, max_value):
#     # Handle case where max_value is zero
#     if max_value == 0:
#         return 100
    
#     total_sum = 0
#     for i, value in enumerate(row):
#         # Compute the updated expression
#         expression = 3 * (1 - value / max_value) + i
        
#         # Ensure the expression is positive before computing log2
#         if expression <= 0:
#             raise ValueError("Expression must be positive for log2 computation")
        
#         # Compute the log2 of the expression
#         total_sum += np.log2(expression)
    
#     return total_sum

# def C_PD_cal(values):
#     total_sum = 0
#     for k, value in enumerate(values, start=2):
#         numerator = 1 - value
#         denominator = 1.1 ** (k - 1)  # Power of (k - 1) as per the formula
#         total_sum += numerator / denominator
#     return total_sum

def R_PD_cal(row):
    total_sum = 0
    for i in range(len(row)):
        # Compute the log2 of (3 + i)
        expression = 3 + i
        
        # Ensure the expression is positive before computing log2
        if expression <= 0:
            raise ValueError("Expression must be positive for log2 computation")
        
        total_sum += np.log2(expression)
    
    return total_sum

def C_PD_cal(values):
    total_sum = 0
    for j in range(2, len(values) + 2):  # Adjusting to iterate from 2 to S_c + 1
        # Compute the term (1.1)^(-(j-1))
        term = (1.1) ** (-(j - 1))
        total_sum += term
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
    
    if circ in ['TCT', 'KV', 'KH', 'DG', 'ST']:
        if circ == 'KV':
            # Transformation for 'PER'            
            transformed_matrix = np.array([
                [matrix[0][0], matrix[2][1], matrix[0][2], matrix[2][3]],
                [matrix[1][0], matrix[3][1], matrix[1][2], matrix[3][3]],
                [matrix[2][0], matrix[0][1], matrix[2][2], matrix[0][3]],
                [matrix[3][0], matrix[1][1], matrix[3][2], matrix[1][3]]
            ])
            
        elif circ == 'KH':
            # Transformation for 'PER'            
            transformed_matrix = np.array([
                [matrix[0][0], matrix[2][1], matrix[3][2], matrix[1][3]],
                [matrix[1][0], matrix[3][1], matrix[0][2], matrix[2][3]],
                [matrix[2][0], matrix[0][1], matrix[1][2], matrix[3][3]],
                [matrix[3][0], matrix[1][1], matrix[2][2], matrix[0][3]]
            ])    
        
        elif circ == 'DG':
            # Transformation for 'C1'
            transformed_matrix = np.array([
                [matrix[0][0], matrix[0][1], matrix[0][2], matrix[3][3]],
                [matrix[1][0], matrix[1][1], matrix[3][2], matrix[0][3]],
                [matrix[2][0], matrix[3][1], matrix[1][2], matrix[1][3]],
                [matrix[3][0], matrix[2][1], matrix[2][2], matrix[2][3]]
            ])
        
        elif circ == 'ST':
            # Transformation for 'C2'
            transformed_matrix = np.array([
                [matrix[0][0], matrix[0][1], matrix[3][2], matrix[3][3]],
                [matrix[1][0], matrix[1][1], matrix[0][2], matrix[0][3]],
                [matrix[2][0], matrix[2][1], matrix[1][2], matrix[1][3]],
                [matrix[3][0], matrix[3][1], matrix[2][2], matrix[2][3]]
            ])
        
        else:
            # Default case for 'TCT'
            transformed_matrix = matrix
    
        n = identify_matrix_size(transformed_matrix)
        
        if n == "Matrix is not square":
            return n
        elif n == "Matrix is empty":
            return None
        
        max_value = np.max(matrix)
        
        r, c = analyze_matrix(transformed_matrix)
        
        if n == 3:
            R_MF = 16.87
            C_MF = 3.76
        elif n == 4:
            R_MF = 11.72
            C_MF = 2.69
        elif n == 5:
            R_MF = 8.79
            C_MF = 2.06
        elif n == 6:
            R_MF = 6.98
            C_MF = 1.64
        # R_MF = 71.07 / (n ** 1.3)
        # C_MF = 13.87 / (n ** 1.19)
        R_PD = R_MF * R_PD_cal(r)
        C_PD = C_MF * C_PD_cal(c)
        PD = R_PD + C_PD
        P = W * (100 - PD) / 100
        
        if P < 0:
            P = 0
        
        return P
    
    if circ == 'SP':
        if voltage_matrix is None or current_matrix is None:
            return "Voltage and current matrices are required for SP calculation"
        
        total_output_power = calculate_total_power(matrix, voltage_matrix, current_matrix)
        return total_output_power

# Matrices
matrices = {
    "AA": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AB": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AC": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "AD": np.array([[1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]]),
    
    "AE": np.array([[1, 1, 2, 0], [1, 2, 3, 0], [2, 3, 0, 0], [0, 0, 0, 0]]),
    "AF": np.array([[1, 1, 1, 2], [1, 2, 2, 3], [1, 2, 0, 0], [2, 3, 0, 0]]),
    "AG": np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 0, 0], [1, 2, 0, 0]]),
    "AH": np.array([[1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 0]]),
    "AI": np.array([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 3, 0], [1, 2, 0, 0]]),
    "AJ": np.array([[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 0]]),
    
    "AK": np.array([[1, 2, 3, 0], [2, 1, 2, 0], [3, 2, 3, 0], [0, 0, 0, 0]]),
    "AL": np.array([[1, 2, 3, 0], [2, 1, 2, 3], [3, 2, 1, 2], [0, 3, 2, 3]]),
    "AM": np.array([[1, 2, 3, 0], [2, 1, 2, 3], [3, 2, 1, 2], [0, 3, 2, 1]]),
    
    "AN": np.array([[2, 3, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0]]),
    "AO": np.array([[2, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0]]),
    "AP": np.array([[2, 2, 2, 3], [1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 3]]),
    "AQ": np.array([[2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]),
    
    "AR": np.array([[1, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0], [1, 2, 3, 0]]),
    "AS": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 2, 3]]),
    "AT": np.array([[2, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 2]]),
    "AU": np.array([[2, 1, 1, 2], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 2]]),
    "AV": np.array([[1, 1, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 2]]),
    
    "AW": np.array([[3, 2, 2, 3], [2, 1, 1, 2], [2, 1, 1, 2], [3, 2, 2, 3]]),
    "AX": np.array([[1, 2, 2, 1], [2, 3, 3, 2], [2, 3, 3, 2], [1, 2, 2, 1]]),
    "AY": np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]),
    
    "AZ": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [2, 3, 0, 0], [1, 2, 0, 0]]),
    "BA": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [2, 2, 3, 0], [1, 1, 2, 0]]),
    "BB": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [2, 2, 2, 3], [1, 1, 1, 2]]),
    "BC": np.array([[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [1, 1, 1, 1]]),
    
    "BD": np.array([[1, 1, 2, 0], [1, 2, 3, 0], [1, 2, 3, 0], [1, 1, 2, 0]]),
    "BE": np.array([[1, 1, 1, 2], [1, 1, 2, 3], [1, 1, 2, 3], [1, 1, 1, 2]]),
    "BF": np.array([[2, 1, 1, 1], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1]]),
    "BG": np.array([[1, 1, 1, 2], [1, 2, 2, 3], [1, 2, 2, 3], [1, 1, 1, 2]]),
    "BH": np.array([[2, 1, 1, 1], [2, 1, 2, 2], [2, 1, 2, 2], [2, 1, 2, 2]]),
    "BI": np.array([[1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 1, 1]]),
    "BJ": np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 2, 2], [1, 1, 1, 1]]),
    
    "BK": np.array([[1, 2, 0, 0], [2, 3, 0, 0], [0, 0, 3, 2], [0, 0, 2, 1]]),
    "BL": np.array([[1, 1, 2, 0], [2, 2, 3, 0], [0, 3, 2, 2], [0, 2, 1, 1]]),
    "BM": np.array([[1, 1, 1, 2], [2, 2, 2, 3], [3, 2, 2, 2], [2, 1, 1, 1]]),
    "BN": np.array([[1, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1]]),
    
    "BO": np.array([[1, 1, 2, 0], [1, 2, 3, 2], [2, 3, 2, 1], [0, 2, 1, 1]]),
    "BP": np.array([[1, 1, 1, 2], [1, 2, 2, 1], [1, 2, 2, 1], [2, 1, 1, 1]]),
    "BQ": np.array([[1, 1, 2, 0], [1, 1, 2, 2], [2, 2, 1, 1], [0, 2, 1, 1]]),
    
    "BR": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0], [0, 0, 0, 0]]),
    "BS": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0], [2, 2, 3, 0]]),
    "BT": np.array([[1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0], [1, 1, 2, 0]]),
    "BU": np.array([[1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 3]]),
    
    "BV": np.array([[1, 1, 1, 2], [1, 1, 1, 2], [1, 1, 2, 3], [2, 2, 3, 0]]),
    "BW": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [2, 2, 3, 0]]),
    "BX": np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 2, 2], [1, 1, 2, 0]]),
    
    "BY": np.array([[1, 2, 0, 0], [1, 2, 3, 0], [1, 1, 2, 0], [1, 1, 2, 0]]),
    "BZ": np.array([[2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 1], [2, 1, 1, 1]]),
    
    "CA": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 2, 3], [3, 2, 3, 0]]),
    "CB": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 1, 2], [3, 2, 2, 3]]),
    "CC": np.array([[1, 1, 2, 3], [1, 1, 1, 2], [2, 1, 1, 1], [3, 2, 1, 1]]),
    }

voltage_matrix_example = [
    [2.52, 2.52, 2.52, 2.52], 
    [2.52, 2.52, 2.52, 2.52],
    [2.52, 2.52, 2.52, 2.52],
    [2.52, 2.52, 2.52, 2.52]
]

current_matrix_example = [
    [0.919, 0.919, 0.919, 0.919],
    [0.919, 0.919, 0.919, 0.919],
    [0.919, 0.919, 0.919, 0.919],
    [0.919, 0.919, 0.919, 0.919]
]

ref = 37.06

# Create a DataFrame to store the results
results = []

# Iterate over each matrix and its transformations
for matrix_id, matrix in matrices.items():
    transformed_matrices = ALL(matrix, matrix_id)
    
    for transformed_matrix, transformed_id in transformed_matrices:
        # Calculate power for SP circuit
        sp_power = calculate_power(transformed_matrix, ref, 'SP', voltage_matrix_example, current_matrix_example)
        
        # Calculate power for TCT circuit
        tct_power = calculate_power(transformed_matrix, ref, 'TCT')
        
        # Calculate power for PER circuit
        per_power = calculate_power(transformed_matrix, ref, 'KV')
        
        # Calculate power for PER circuit
        l_power = calculate_power(transformed_matrix, ref, 'KH')
        
        # Calculate power for PER circuit
        c1_power = calculate_power(transformed_matrix, ref, 'DG')
        
        # Calculate power for PER circuit
        c2_power = calculate_power(transformed_matrix, ref, 'ST')
        
        # Append the results to the list
        results.append({
            "it": transformed_id,
            "SP": sp_power,
            "TCT": tct_power,
            "KV": per_power,
            "KH": l_power,
            "DG": c1_power,
            "ST": c2_power
        })

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Extract base code by stripping the numbers
df['base_code'] = df['it'].str.extract(r'Z_([A-Z]+)_')

a_df_python = df[['it','base_code','SP','TCT','ST','DG','KV','KH']]


# Save the final merged DataFrame to a CSV file
output_file_path = "C:/Users/ruiui/Desktop/iteration data/tabelas_pd_cumu/python_4x4_results.csv"
a_df_python.to_csv(output_file_path, index=False)

# Optionally, print a message confirming the save
print(f"DataFrame saved to {output_file_path}")















