import matplotlib.pyplot as plt

# Data
x = [0, 2, 6, 10, 20, 30, 40]

y2 = [16.0, 35.0, 44.0, 40.8, 37.2, 35.8, 34.6]
y22 = [23.9, 46.5, 52.8, 49.2, 46.1, 45.9, 44.2]


y3 = [16.0, 43.2, 49.2, 46.8, 40.0, 40.6, 37.8]
y33 = [23.9, 54.2, 59.2, 56.4, 49.8, 50.4, 48.3]



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
plt.xticks(x, ['0', '1', '5', '10', '20', '30','40'], fontsize=16)
plt.yticks(range(10, 61, 10), [f'{val}%' for val in range(10, 61, 10)], fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey')

#
plt.legend(loc='lower right', fontsize=16, ncol=2) 

# plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
#           fancybox=True, shadow=False, ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig('line_plot_timeqa.pdf')




# import matplotlib.pyplot as plt

# # Data
# x = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70]
# # y = [62.6, 62.6, 62.6, 62.6, 62.6]
# y2 = [16.4, 31.5, 42.7, 40.3, 38.9, 36.3, 32.1, 34.9, 1.2, 0.4]
# y22 = [25.5, 42.3, 52.0, 48.1, 48.2, 45.6, 41.8, 45.2, 11.8, 11.3]

# y3 = [16.4, 39.3, 46.5, 43.7, 42.3, 41.5, 39.9, 38.1, 0.4, 0.2]
# y33 = [25.2, 50.0, 56.5, 52.6, 51.8, 51.4, 49.0, 48.6, 10.9, 10.0]

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
# plt.xticks(x, ['0', '1', '5', '10', '20', '30', '40', '50', '60', '70'], fontsize=16)
# plt.yticks(range(0, 61, 10), [f'{val}%' for val in range(0, 61, 10)], fontsize=16)
# plt.grid(True, linestyle='-', color='lightgrey')

# #
# plt.legend(loc='lower left', fontsize=16, ncol=2) 

# # plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
# #           fancybox=True, shadow=False, ncol=2, fontsize=16)

# plt.tight_layout()
# plt.savefig('line_plot_timeqa.pdf')



