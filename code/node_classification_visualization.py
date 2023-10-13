# This file is used to generate Figure 4
import matplotlib.pyplot as plt

x_labels = ['DB', 'AI', 'DM', 'IR']
#y_values1 = [DB_all_number, AI_all_number, DM_all_number, IR_all_number]
#y_values2 = [DB_current_number, AI_current_number, DM_current_number, IR_current_number]
y_values1 = [242.0, 239.0, 120.0, 204.0]
y_values2 = [233.0, 228.0, 112.0, 195.0]

differences = [y2 / y1 for y1, y2 in zip(y_values1, y_values2)]


bar_width = 0.3
index = range(len(x_labels))
bar1 = plt.bar(index, y_values1, bar_width, label='Total quantity')
bar2 = plt.bar([i + bar_width for i in index], y_values2, bar_width, label='Correct quantity')
plt.title("Training set proportion: 80%", fontsize=17)
plt.xlabel("Four Areas", fontsize=17)
plt.ylabel("Number of authors", fontsize=17)

x_positions = [rect.get_x() for rect in bar1]
print(x_positions)


for i in range(len(x_labels)):
    x_pos = x_positions[i] + bar_width
    y_pos = max(y_values1[i], y_values2[i]) + 0.1  
    plt.text(x_pos, y_pos, f'{differences[i]:.4f}', ha='center', va='bottom')

plt.xticks([i + bar_width/2 for i in index], x_labels)
plt.legend()
plt.tight_layout()
plt.show()


# Statistical results of four groups of experiments (taking the average of ten experiments per group)
#20%
# y_values1 = [981.0, 973.0, 484.0, 799.0]
# y_values2 = [943.0, 918.0, 434.0, 745.0]

#40%
# y_values1 = [739.0, 737.0, 356.0, 601.0]
# y_values2 = [707.0, 694.0, 324.0, 569.0]

#60%
# y_values1 = [489.0, 488.0, 243.0, 398.0]
# y_values2 = [468.0, 464.0, 221.0, 382.0]

#80%
# y_values1 = [242.0, 239.0, 120.0, 204.0]
# y_values2 = [233.0, 228.0, 112.0, 195.0]