# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:32:16 2024

@author: ruiui
"""

import matplotlib.pyplot as plt
import numpy as np

# Main component names and weights
main_components = ['Solar panel', 'PCB', 'Case', 'Components']
main_weights = np.array([62.5, 95.0, 77.2, 8.1 + 5.8 + 0.3])  # Grouped "Components" weight
total_main_weight = np.sum(main_weights)

# Sub-component names and weights for "Components"
sub_components = ['Connectors', 'Switches', 'Diode']
sub_weights = np.array([8.1, 5.8, 0.3])  # Individual weights of sub-components
total_sub_weight = np.sum(sub_weights)

# Main components plot
fig1, ax1 = plt.subplots(figsize=(6, 8))

# Plot the main vertical bar
bottom_main = 0  # Start at the bottom of the main bar
main_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
for weight, color, component in zip(main_weights, main_colors, main_components):
    ax1.bar(0, weight, color=color, edgecolor='black', width=0.4, bottom=bottom_main)
    
    # Add percentage inside the bar
    percentage = (weight / total_main_weight) * 100
    ax1.text(
        0, 
        bottom_main + weight / 2, 
        f"{percentage:.1f}%", 
        ha='center', 
        va='center', 
        fontsize=15, 
        color='black'
    )
    
    # Add component name on the side
    ax1.text(
        0.25, 
        bottom_main + weight / 2, 
        component, 
        ha='left', 
        va='center', 
        fontsize=15
    )
    bottom_main += weight

# Adding cumulative lines for the main bar
main_cumulative_weights = np.cumsum(main_weights)
for weight in main_cumulative_weights:
    ax1.plot([-0.2, 0.2], [weight, weight], color='black', lw=2)

# Title and labels for the main components figure
ax1.set_xlim(-0.35, 0.6)
ax1.set_ylim(0, 270)
ax1.set_title('Weight Distribution', fontsize=18)
ax1.set_ylabel('Weight (g)', fontsize=18)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=13)

# Show the first figure
plt.savefig('cost.svg', format='svg', bbox_inches='tight')
plt.show()

# Sub-components plot
fig2, ax2 = plt.subplots(figsize=(6, 8))

# Plot the sub-components vertical bar
bottom_sub = 0  # Start at the bottom of the sub-components bar
sub_colors = ['lightsteelblue', 'lightpink', 'lightgray']
for weight, color, sub_component in zip(sub_weights, sub_colors, sub_components):
    ax2.bar(0, weight, color=color, edgecolor='black', width=0.4, bottom=bottom_sub)
    
    # Add sub-component name on the side
    ax2.text(
        0.25, 
        bottom_sub + weight / 2, 
        sub_component, 
        ha='left', 
        va='center', 
        fontsize=15
    )
    
    # Add percentage inside the bar
    percentage = (weight / total_sub_weight) * 100
    if sub_component == 'Diode':
        # Adjust the vertical position slightly higher for "Diode"
        ax2.text(
            0, 
            bottom_sub + weight / 2 + 0.4,  # Adjusted higher by 0.3
            f"{percentage:.1f}%", 
            ha='center', 
            va='center', 
            fontsize=15, 
            color='black'
        )
    else:
        ax2.text(
            0, 
            bottom_sub + weight / 2, 
            f"{percentage:.1f}%", 
            ha='center', 
            va='center', 
            fontsize=15, 
            color='black'
        )
    
    bottom_sub += weight  # Update the bottom position for the next section

# Adding cumulative lines for the sub-components bar
sub_cumulative_weights = np.cumsum(sub_weights)
for weight in sub_cumulative_weights:
    ax2.plot([-0.2, 0.2], [weight, weight], color='black', lw=2)

# Title and labels for the sub-components figure
ax2.set_xlim(-0.35, 0.6)
ax2.set_ylim(0, 15)
ax2.set_title('Components Breakdown', fontsize=18)
ax2.set_ylabel('Weight (g)', fontsize=18)
ax2.set_xticks([])
ax2.tick_params(axis='y', labelsize=13)

# Show the second figure
plt.savefig('cost_sub.svg', format='svg', bbox_inches='tight')
plt.show()



connectors = 1.9965
switches = 2.48
diode = 0.165
panel = 4.94
pcb = 10.95

# Main component names and weights
main_components = ['Solar panel', 'PCB', 'Case', 'Connectors', 'Switches', 'Diode']
main_weights = np.array([4.94, 10.95, 0.611115, 1.9965, 2.48, 0.165])  # Prices (€)

# Total price for percentage calculation
total_price = np.sum(main_weights)

# Main components plot
fig3, ax3 = plt.subplots(figsize=(6, 8))

# Plot the main vertical bar
bottom_main = 0  # Start at the bottom of the main bar
main_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']

for weight, color, component in zip(main_weights, main_colors, main_components):
    ax3.bar(0, weight, color=color, edgecolor='black', width=0.4, bottom=bottom_main)
    
    # Calculate percentage contribution
    percentage = (weight / total_price) * 100
    
    # Add the label with the percentage difference
    label_text = f"{component} ({percentage:.1f}%)"
    if component == 'Connectors':
        ax3.text(0.25, bottom_main + weight/2, label_text, ha='left', va='center', fontsize=14)
    else:
        ax3.text(0.25, bottom_main + weight / 2, label_text, ha='left', va='center', fontsize=14)
    
    bottom_main += weight

# Adding cumulative lines for the main bar
main_cumulative_weights = np.cumsum(main_weights)
for weight in main_cumulative_weights:
    ax3.plot([-0.2, 0.2], [weight, weight], color='black', lw=2)

# Title and labels for the main components figure
ax3.set_xlim(-0.35, 0.7)
# ax3.set_ylim(0, 30)
ax3.set_title('Price distribution', fontsize=18)
ax3.set_ylabel('Price (€)', fontsize=18)
ax3.set_xticks([])
ax3.tick_params(axis='y', labelsize=13)

# Show the figure
plt.savefig('price.svg', format='svg', bbox_inches='tight')
plt.show()
