# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:43:26 2024

@author: ruiui
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel file (replace 'your_file.xlsx' with your actual file path)
file_path = "C:/Users/ruiui/Desktop/plsplspls.xlsx"

# Load the specific sheet '3x3'
df = pd.read_excel(file_path, sheet_name='4x4')

# Rename columns if needed
df.columns = ['Circuit', 'Point', 'Python', 'LTspice']

# Fill NaN values in the 'Circuit' column with the last valid value (forward fill)
df['Circuit'].fillna(method='ffill', inplace=True)

# Pivot the DataFrame to separate max, median, and min
df_pivot = df.pivot(index='Circuit', columns='Point')

# Define the new order of circuits
new_order = ['SP', 'SP_D', 'TCT', 'TCT_D', 'R2R3', 'R2R3_D', 'R1', 'R1_D', 'PER', 'PER_D', 'L', 'L_D']

# Reorder the DataFrame based on this new order
df_pivot = df_pivot.reindex(new_order)

# Calculate errors for each source
df_pivot['Python', 'yerr'] = df_pivot['Python', 'max'] - df_pivot['Python', 'min']
df_pivot['LTspice', 'yerr'] = df_pivot['LTspice', 'max'] - df_pivot['LTspice', 'min']

# Set up the plot
plt.figure(figsize=(14, 7))

# Get the positions for the circuits on the x-axis
x = np.arange(len(df_pivot.index))

# Define an offset value to separate the groups of points
offset = 0.2

# Plot for Python with no offset (centered)
plt.errorbar(x, df_pivot['Python', 'median'], yerr=df_pivot['Python', 'yerr'], fmt='s', label='Python', capsize=5)

# Plot for LTspice with an offset to the right (+offset)
plt.errorbar(x + offset, df_pivot['LTspice', 'median'], yerr=df_pivot['LTspice', 'yerr'], fmt='^', label='LTspice', capsize=5)

# Add labels and title
plt.title('Comparison of IRL, Python, and LTspice', fontsize=16)
plt.xlabel('Circuit', fontsize=12)
plt.ylabel('Losses (%)', fontsize=12)

# Set x-ticks to be the circuit names
plt.xticks(x, df_pivot.index, rotation=45)

plt.grid(False)

# Show legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()