import matplotlib.pyplot as plt

# Data
x = [1, 5, 10, 20, 30, 50]
# y = [62.6, 62.6, 62.6, 62.6, 62.6]
y2 = [51.5, 56.9, 61.7, 64.7, 62.6, 62.6]
y22 = [55, 58, 65, 66, 66, 66]

y3 = [67.6, 68.5, 70.3, 71.6, 72, 72.6]
y33 = [70, 70, 70.3, 76, 76, 76]

# Plotting
plt.figure(figsize=(8, 5))

l=3
# Plotting the line
# plt.plot(x, y, color='yellowgreen', label='TableQA', linestyle='dashed', linewidth=l)
plt.plot(x, y2, color='tab:blue', label='2-Stage (S)', marker='o', linewidth=l)
plt.plot(x, y22, color='lightblue', label='2-Stage (T)', marker='o', linewidth=l)
plt.plot(x, y3, color='tab:green', label='MR (S)', marker='o', linewidth=l)
plt.plot(x, y33, color='lightgreen', label='MR (T)', marker='o', linewidth=l)
# plt.plot(x, y4, color='blue', label='Oracle', marker='o', linewidth=l)


# Customizing the plot
plt.xlabel('Number of context chunks', fontsize=20,labelpad=10)
plt.ylabel('EM', fontsize=20, labelpad=10)
plt.xticks(x, ['1', '5', '10', '20', '30', '50'], fontsize=16)
plt.yticks(range(40, 81, 10), [f'{val}%' for val in range(40, 81, 10)], fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey')

#
plt.legend(loc='lower right', fontsize=16, ncol=2) 

# plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
#           fancybox=True, shadow=False, ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig('line_plot.pdf')




