import matplotlib.pyplot as plt

# Data
x = [0, 2, 6, 10, 20, 30, 40]

y2 = [42.8, 53.2, 60.0, 60.0, 61.0, 61.4, 60.8]
y22 = [45.0, 55.9, 62.7, 62.5, 63.0, 63.6, 63.2]

y3 = [42.8, 57.6, 65.8, 64.0, 62.8, 61.0, 59.6]
y33 = [45.0, 59.1, 68.0, 66.2, 64.6, 63.5, 62.2]


# Plotting
plt.figure(figsize=(8, 5))

l=2
# Plotting the line
# plt.plot(x, y, color='yellowgreen', label='TableQA', linestyle='dashed', linewidth=l)
plt.plot(x, y2, color='tab:green', label='Standard (EM)', linewidth=l)
plt.plot(x, y22, color='lightgreen', label='Standard (F1)', linestyle='dashed', linewidth=l)
plt.plot(x, y3, color='tab:purple', label='MRAG (EM)', linewidth=l)
plt.plot(x, y33, color='thistle', label='MRAG (F1)', linestyle='dashed', linewidth=l)
# plt.plot(x, y4, color='blue', label='Oracle', marker='o', linewidth=l)


# Customizing the plot
# plt.xlabel('Number of context chunks', fontsize=20,labelpad=10)
# plt.ylabel('Score', fontsize=20, labelpad=10)
plt.xticks(x, ['0', '1', '5', '10', '20', '30', '40'], fontsize=16)
plt.yticks(range(30, 81, 10), [f'{val}%' for val in range(30, 81, 10)], fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey')

#
plt.legend(loc='lower right', fontsize=16, ncol=2) 

# plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
#           fancybox=True, shadow=False, ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig('line_plot_situatedqa.pdf')




# # Data
# x = [0, 1, 5, 10, 20, 30, 40, 50, 60]

# y2 = [35.4, 42.4, 52.4, 55.6, 59.2, 57.6, 58.0, 32.0, 23.8]
# y22 = [37.1, 45.1, 54.8, 58.2, 61.8, 59.8, 60.2, 38.3, 31.3]

# y3 = [35.4, 50.4, 60.2, 59.6, 63.2, 61.2, 60.2, 27.2, 22.2]
# y33 = [37.1, 52.8, 63.1, 63.1, 67.0, 64.0, 62.9, 33.9, 29.8]

# # Plotting
# plt.figure(figsize=(8, 5))

# l=2
# # Plotting the line
# # plt.plot(x, y, color='yellowgreen', label='TableQA', linestyle='dashed', linewidth=l)
# plt.plot(x, y2, color='tab:blue', label='Standard (EM)', linewidth=l)
# plt.plot(x, y22, color='lightblue', label='Standard (F1)', linestyle='dashed', linewidth=l)
# plt.plot(x, y3, color='tab:green', label='Modular (EM)', linewidth=l)
# plt.plot(x, y33, color='lightgreen', label='Modular (F1)', linestyle='dashed', linewidth=l)
# # plt.plot(x, y4, color='blue', label='Oracle', marker='o', linewidth=l)


# # Customizing the plot
# # plt.xlabel('Number of context chunks', fontsize=20,labelpad=10)
# # plt.ylabel('Score', fontsize=20, labelpad=10)
# plt.xticks(x, ['0', '1', '5', '10', '20', '30', '40', '50', '60'], fontsize=16)
# plt.yticks(range(20, 71, 10), [f'{val}%' for val in range(20, 71, 10)], fontsize=16)
# plt.grid(True, linestyle='-', color='lightgrey')

# #
# plt.legend(loc='lower left', fontsize=16, ncol=2) 

# # plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
# #           fancybox=True, shadow=False, ncol=2, fontsize=16)

# plt.tight_layout()
# plt.savefig('line_plot_situatedqa.pdf')




