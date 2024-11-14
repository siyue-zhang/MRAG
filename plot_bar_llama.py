import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Data for the bar chart
categories = ["DP", "CoT", "RAG", "DiP"] # Labels for bars within each group
group_labels = ['TimeQA', 'SituatedQA']  # Labels for each group

# Values for each group
group1_values = [13.0, 16.4, 46.5, 0]
group2_values = [39.0, 48.4, 63.2, 0]

# Colors for each bar
group1_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
# group2_colors = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

# Bar settings
bar_width = 0.15  # Width of each bar
spacing_within_group = 0  # Smaller spacing between bars within each group
group_spacing = len(categories) * (bar_width + spacing_within_group) + 0.15 # Large spacing between the groups

# Positions for each bar
x = np.arange(len(categories)) * (bar_width + spacing_within_group)  # Position bars closely within each group
x_shifted = x + group_spacing  # Shift the positions for Group 2 to maintain distance between the groups

# Create the figure and plot the bars
plt.figure(figsize=(6, 8))


# Plotting Group 1 bars
for i in range(len(categories)):
    plt.bar(x[i], group1_values[i], width=bar_width, color=group1_colors[i], alpha=0.8, label=categories[i])

# Plotting Group 2 bars
for i in range(len(categories)):
    plt.bar(x_shifted[i], group2_values[i], width=bar_width, color=group1_colors[i], alpha=0.8)

# Customizing the plot
# plt.xlabel('Groups', fontsize=20,labelpad=10)
plt.ylabel('EM', fontsize=20, labelpad=10)
plt.yticks(range(0, 81, 20), [f'{val}%' for val in range(0, 81, 20)], fontsize=16)
plt.xticks([x.mean(), x_shifted.mean()], group_labels, fontsize=14)  # Position group labels below the bars

# # Format y-axis as percentage
# def to_percent(y, pos):
#     return f'{y}%'

# formatter = FuncFormatter(to_percent)
# plt.gca().yaxis.set_major_formatter(formatter)

plt.legend(loc='upper left', ncol=2, fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey', axis='y')

# Display the plot
plt.tight_layout()
plt.savefig('bar_plot_llama.pdf')  # Save as "bar_plot.pdf"
plt.show()
