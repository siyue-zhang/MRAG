import numpy as np
import matplotlib.pyplot as plt

# # Performance metrics for two datasets (normal vs perturbed)
# # Let's assume some example values for the sake of illustration
# # Replace these values with your actual data

# # Performance on SituatedQA (left subplot)
# recall_metrics_situatedqa_normal = [0.86, 1, 1, 1]    # Recall@1, Recall@5, Recall@10, Recall@20 for normal case
# recall_metrics_situatedqa_perturbed = [0.36, 0.77, 0.84, 0.86]  # Recall@1, Recall@5, Recall@10, Recall@20 for perturbed case

# # Performance on TimeQA (right subplot)
# recall_metrics_timeqa_normal = [0.86, 1, 1, 1]        # Recall@1, Recall@5, Recall@10, Recall@20 for normal case
# recall_metrics_timeqa_perturbed = [0.36, 0.77, 0.84, 0.86]     # Recall@1, Recall@5, Recall@10, Recall@20 for perturbed case

# # Recall metric names
# recall_labels = ['R@1', 'R@5', 'R@10', 'R@20']

# # Set up the figure and subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Define bar width and x-axis positions
# bar_width = 0.35
# index = np.arange(len(recall_labels))

# # Increase font sizes for both subplots
# font_size = 22  # Adjust the value to increase/decrease text size
# title_size = 24  # Title font size

# # Plot for SituatedQA
# ax1.bar(index, recall_metrics_situatedqa_normal, bar_width, label='Original', color='#1f77b4')
# ax1.bar(index + bar_width, recall_metrics_situatedqa_perturbed, bar_width, label='Perturbed', color='#1f77b4', alpha=0.4)

# # Plot for TimeQA
# ax2.bar(index, recall_metrics_timeqa_normal, bar_width, label='Original', color='#ff7f0e')
# ax2.bar(index + bar_width, recall_metrics_timeqa_perturbed, bar_width, label='Perturbed', color='#ff7f0e', alpha=0.4)

# # Labels and Titles for SituatedQA plot (left)
# # ax1.set_xlabel('Recall Metrics')
# # ax1.set_ylabel('Performance')
# ax1.set_title('SituatedQA',  fontsize=title_size)
# ax1.set_xticks(index + bar_width / 2)
# ax1.set_xticklabels(recall_labels)
# ax1.legend(fontsize=14,loc='upper left')
# ax1.tick_params(axis='both', which='major', labelsize=font_size)

# # Labels and Titles for TimeQA plot (right)
# # ax2.set_xlabel('Recall Metrics')
# # ax2.set_ylabel('Performance')
# ax2.set_title('TimeQA', fontsize=title_size)
# ax2.set_xticks(index + bar_width / 2)
# ax2.set_xticklabels(recall_labels)
# ax2.legend(fontsize=14,loc='upper left')
# ax2.tick_params(axis='both', which='major', labelsize=font_size)

# # Display the plot
# plt.tight_layout()
# # plt.show()

# plt.savefig('retriever_perf.png', dpi=300)



import numpy as np
import matplotlib.pyplot as plt

# Example Performance metrics for two datasets (normal vs perturbed)
# Replace these values with your actual data

# Performance on SituatedQA (left subplot)
answer_recall_situatedqa_normal = [85.8, 96.7, 96.7, 99.2]    # Recall@1, Recall@5, Recall@10, Recall@20 for normal case
answer_recall_situatedqa_perturbed = [54.7, 82.6, 89.5, 94.0]  # Perturbed case

# Performance on TimeQA (right subplot)
answer_recall_timeqa_normal = [76.4, 91.9, 92.7, 92.7]        # Normal case
answer_recall_timeqa_perturbed = [46.7, 82.5, 86.5, 87.8]     # Perturbed case

# Gold Evidence Recall (for bottom row)
# SituatedQA
gold_recall_situatedqa_normal = [45.0, 72.5, 80.0, 83.3]  # Gold Recall@1, @5, @10, @20 for normal case
gold_recall_situatedqa_perturbed = [20.3, 45.3, 51.8, 55.5]  # Perturbed case

# TimeQA
gold_recall_timeqa_normal = [58.5, 81.3, 82.1, 82.1]      # Normal case
gold_recall_timeqa_perturbed = [26.0, 66.6, 71.6, 73.2]   # Perturbed case

# Recall metric names
recall_labels = ['R@1', 'R@5', 'R@10', 'R@20']

# Set up the figure and subplots (2 rows x 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(12, 7))  # Increased the figure size for better readability

# Define bar width and x-axis positions
bar_width = 0.35
index = np.arange(len(recall_labels))

# Font sizes
font_size = 18  # Adjust the value to increase/decrease text size
title_size = 22  # Title font size

# === Plotting ===

# Answer Recall for SituatedQA
axs[0, 1].bar(index, answer_recall_situatedqa_normal, bar_width, label='Original', color='#ff7f0e')
axs[0, 1].bar(index + bar_width, answer_recall_situatedqa_perturbed, bar_width, label='Perturbed', color='#ff7f0e', alpha=0.4)
axs[0, 1].set_title('SituatedQA', fontsize=title_size+3)
axs[0, 1].set_xticks(index + bar_width / 2)
axs[0, 1].set_xticklabels(recall_labels)
axs[0, 1].set_ylim(0,100)
axs[0, 1].set_yticklabels([])
axs[0, 1].legend(fontsize=16, loc='lower right')
axs[0, 1].tick_params(axis='both', which='major', labelsize=font_size)

# Answer Recall for TimeQA
axs[0, 0].bar(index, answer_recall_timeqa_normal, bar_width, label='Original', color='#1f77b4')
axs[0, 0].bar(index + bar_width, answer_recall_timeqa_perturbed, bar_width, label='Perturbed', color='#1f77b4', alpha=0.4)
axs[0, 0].set_title('TimeQA', fontsize=title_size+3)
axs[0, 0].set_xticks(index + bar_width / 2)
axs[0, 0].set_xticklabels(recall_labels)
axs[0, 0].set_ylim(0,100)
axs[0, 0].set_ylabel('Answer Recall (%)', fontsize=title_size)
axs[0, 0].legend(fontsize=16, loc='lower right')
axs[0, 0].tick_params(axis='both', which='major', labelsize=font_size)

# Gold Evidence Recall for SituatedQA
axs[1, 1].bar(index, gold_recall_situatedqa_normal, bar_width, label='Original', color='tan')
axs[1, 1].bar(index + bar_width, gold_recall_situatedqa_perturbed, bar_width, label='Perturbed', color='tan', alpha=0.4)
# axs[1, 1].set_title('SituatedQA', fontsize=title_size)
axs[1, 1].set_xticks(index + bar_width / 2)
axs[1, 1].set_xticklabels(recall_labels)
axs[1, 1].set_ylim(0,100)
axs[1, 1].set_yticklabels([])
axs[1, 1].legend(fontsize=16, loc='lower right')
axs[1, 1].tick_params(axis='both', which='major', labelsize=font_size)

# Gold Evidence Recall for TimeQA
axs[1, 0].bar(index, gold_recall_timeqa_normal, bar_width, label='Original', color='cadetblue')
axs[1, 0].bar(index + bar_width, gold_recall_timeqa_perturbed, bar_width, label='Perturbed', color='cadetblue', alpha=0.4)
# axs[1, 0].set_title('TimeQA', fontsize=title_size)
axs[1, 0].set_xticks(index + bar_width / 2)
axs[1, 0].set_xticklabels(recall_labels)
axs[1, 0].set_ylim(0,100)
axs[1, 0].set_ylabel('Evidence Recall (%)', fontsize=title_size)
axs[1, 0].legend(fontsize=16, loc='lower right')
axs[1, 0].tick_params(axis='both', which='major', labelsize=font_size)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('retriever_perf.pdf')
