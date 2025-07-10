# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:55:57 2024

@author: ruiui
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    'File': ['C2', 'C5', 'C7', 'C14', 'C15', 'C16', 'C17', 'C20', 'C27'],
    'Voc': [0.4338203141396356, 0.2914112527425648, 0.2821620223905703, 0.7508516189483273, 1.07520792469261, 0.8569886480764639, 0.8967583194337622, 0.748992431665264, 0.7910754101632502],
    'Isc': [0.2287920756349549, 0.4607339491810631, 0.5692578039109137, 0.8880663488328969, 0.8835180212886654, 0.34083937376443424, 0.2116663450241379, 0.34045257495628545, 0.22204422789057462],
    'Vmp': [1.715585666543971, 0.27698972276861483, 0.2862768750914769, 1.9886379035352935, 2.3230363339225213, 0.8992548160048193, 2.1197978115772202, 2.036958555151449, 2.0781711789848925],
    'Imp': [-1.4734350866525423, 0.2806014779941208, 0.36072728756283856, 0.06440127362918123, -0.8785143191136188, 0.5901209381170789, -0.6755213957315594, -0.5598992477609088, -1.3722893840779964],
    'MPP': [0.21687253873900078, 0.5583684380187243, 0.6480368414607611, 2.054319885302229, 1.4241138079771853, 1.4946824450781815, 1.429956728082212, 1.4656543917630551, 0.6773632724347199],
    'FF': [-0.4437874024277637, -0.19366018852113295, -0.20325539271352847, 0.4021166810426061, -0.5336091616521011, 0.2904459418219726, 0.31612393197285227, 0.36962315486028563, -0.3341219785487603],
    'Efficiency': [0.21687253873898443, 0.558368438018722, 0.6480368414607612, 2.054319885302231, 1.4241138079771687, 1.4946824450782024, 1.4299567280822052, 1.4656543917630551, 0.6773632724347223]
}

percentage_diff_df = pd.DataFrame(data)

# Calculate the median values of each column except 'File'
median_values = percentage_diff_df.drop(columns=['File']).median()

# Convert the medians into a DataFrame for plotting
median_df = pd.DataFrame({
    'Parameter': median_values.index,
    'Median Percentage Difference': median_values.values
})

# Add the additional values as a new DataFrame
c21_data = {
    'Parameter': ['Voc', 'Isc', 'Vmp', 'Imp', 'MPP', 'FF', 'Efficiency'],
    'Percentage Difference': [1.453441942181761, -2.0637941709279994, 2.7367242191400796, -2.5741544487304084, 
                               0.09212226217319922, 0.7371911317236598, 0.09212226217321576],
    'Type': ['C21'] * 7
}

c21_df = pd.DataFrame(c21_data)

# Update the median DataFrame to include a "Type" column
median_df['Type'] = 'Median'
median_df = median_df.rename(columns={'Median Percentage Difference': 'Percentage Difference'})

# Combine the two DataFrames
combined_df = pd.concat([median_df, c21_df])

# Set the plot size
plt.figure(figsize=(12, 6))

# Add the gray box from -0.2% to +0.2%
plt.axhspan(-0.2, 0.2, color='gray', alpha=0.2, zorder=-1)

# Add a horizontal line at y=0
plt.axhline(0, color='black', linewidth=1.5)

# Create a grouped bar plot using seaborn
sns.barplot(x='Parameter', y='Percentage Difference', hue='Type', data=combined_df, palette='Set2')

# Add vertical lines between the groups of bar plots (between parameters)
num_params = len(median_df['Parameter'])
for i in range(1, num_params):
    plt.axvline(i - 0.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.8)

# Customize the plot
plt.title('Percentage Difference per Parameter (Median and C21)', fontsize=16)
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('Percentage Difference (%)', fontsize=12)
plt.legend(title='Type', bbox_to_anchor=(1.005, 1), loc='upper left')
plt.tight_layout()
plt.ylim(-3, 3)

# Show the plot
plt.show()