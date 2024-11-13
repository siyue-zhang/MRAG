import matplotlib.pyplot as plt

# Data
x = [0, 1, 5, 10, 20, 30, 40, 50]
# y = [62.6, 62.6, 62.6, 62.6, 62.6]
y2 = [35.4, 42.4, 52.4, 55.6, 59.2, 57.6, 58.0, 23.2]
y22 = [37.1, 45.1, 54.8, 58.2, 61.8, 59.8, 60.2, 24.5]

y3 = [35.4, 50.4, 60.2, 59.6, 63.2, 61.2, 60.2, 10.4]
y33 = [37.1, 52.8, 63.1, 63.1, 67.0, 64.0, 62.9, 11.1]

# Plotting
plt.figure(figsize=(8, 5))

l=2
# Plotting the line
# plt.plot(x, y, color='yellowgreen', label='TableQA', linestyle='dashed', linewidth=l)
plt.plot(x, y2, color='tab:blue', label='2-Stage (EM)', marker='o', linewidth=l)
plt.plot(x, y22, color='lightblue', label='2-Stage (F1)', marker='o', linewidth=l)
plt.plot(x, y3, color='tab:green', label='MR (EM)', marker='o', linewidth=l)
plt.plot(x, y33, color='lightgreen', label='MR (F1)', marker='o', linewidth=l)
# plt.plot(x, y4, color='blue', label='Oracle', marker='o', linewidth=l)


# Customizing the plot
plt.xlabel('Number of context chunks (SituatedQA)', fontsize=20,labelpad=10)
plt.ylabel('Score', fontsize=20, labelpad=10)
plt.xticks(x, ['0', '1', '5', '10', '20', '30', '40', '50'], fontsize=16)
plt.yticks(range(20, 81, 10), [f'{val}%' for val in range(20, 81, 10)], fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey')

#
plt.legend(loc='lower right', fontsize=16, ncol=2) 

# plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
#           fancybox=True, shadow=False, ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig('line_plot_situatedqa.pdf')




