import numpy as np
import matplotlib.pyplot as plt

titles = ["TIMO", "Llama3.1-8B", "Llama3.1-70B"]
y_titles = ['Exact Match','','']

# Data for the bar chart
categories = ["DP", "CoT", "RAG", "Self-MRAG"]  # Labels for bars within each group
group_labels = ['TimeQA', 'SituatedQA']  # Labels for each group

# Values for each group
group1_values = [[16.2, 15.8, 48.2, 44.6],[16.0, 16.8, 49.2, 54.2],[31.0 , 33.2 , 58.0, 61.2]]
group2_values = [[50.6, 49.4, 61.4, 62.0],[42.8, 49.6, 65.8, 66.4],[59.0 , 69.0 , 69.2, 72.2]]

# Colors for each bar
group1_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

# Bar settings
bar_width = 0.15  # Width of each bar
spacing_within_group = 0  # Smaller spacing between bars within each group
group_spacing = len(categories) * (bar_width + spacing_within_group) + 0.15  # Large spacing between the groups

# Positions for each bar
x = np.arange(len(categories)) * (bar_width + spacing_within_group)  # Position bars closely within each group
x_shifted = x + group_spacing  # Shift the positions for Group 2 to maintain distance between the groups

# Create the subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 6))

# Loop through each subplot (each column)
for i, ax in enumerate(axs):
    # Plotting Group 1 bars
    for j in range(len(categories)):
        ax.bar(x[j], group1_values[i][j], width=bar_width, color=group1_colors[j], alpha=0.8, label=categories[j] if i == 0 else "")
    
    # Plotting Group 2 bars
    for j in range(len(categories)):
        ax.bar(x_shifted[j], group2_values[i][j], width=bar_width, color=group1_colors[j], alpha=0.8)
    
    # Customizing each subplot
    ax.set_ylabel(y_titles[i], fontsize=18, labelpad=10)
    ax.set_xticks([x.mean(), x_shifted.mean()])
    ax.set_xticklabels(group_labels, fontsize=18)
    ax.set_yticks(range(0, 81, 20))
    ax.set_yticklabels([f'{val}%' for val in range(0, 81, 20)], fontsize=16)
    ax.grid(True, linestyle='-', color='lightgrey', axis='y')
    
    # Add title for each subplot
    # ax.set_title(titles[i], fontsize=16)

# Adjust legend to move it further down
fig.legend(loc='upper center', fontsize=16, ncol=4, bbox_to_anchor=(0.5, 0.99))

# Adjust layout to make space for the legend and subplots
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.subplots_adjust(wspace=0.3)

plt.savefig('combined_bars.pdf')  # Save as "bar_plot.pdf"
plt.show()
