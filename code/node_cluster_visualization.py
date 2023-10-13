# This file is used to generate Figure 5 in the paper
import matplotlib.pyplot as plt

x_labels = ['DB', 'AI', 'DM', 'IR']
#y_values1 = [DB_all_number, AI_all_number, DM_all_number, IR_all_number]
#y_values2 = [DB_current_number, AI_current_number, DM_current_number, IR_current_number]
y_values1 = [1232, 1217, 601, 1007]
y_values2 = [1196, 1132, 579, 930]
#differences = [y2 / y1 for y1, y2 in zip(y_values1, y_values2)]


bar_width = 0.3
index = range(len(x_labels))
bar1 = plt.bar(index, y_values1, bar_width, label='Total nodes')
bar2 = plt.bar([i + bar_width for i in index], y_values2, bar_width, label='Clustered nodes')
plt.title("Author node clustering result statistics-ARMHGNN")
plt.xlabel("Four Areas")
plt.ylabel("Number of author node")

x_positions = [rect.get_x() for rect in bar1]
print(x_positions)

for i in range(len(x_labels)):
    x_pos = x_positions[i] + bar_width / 2
    y_pos = y_values1[i] + 0.1  
    plt.text(x_pos, y_pos, f'{y_values1[i]}', ha='center', va='bottom')
for i in range(len(x_labels)):
    x_pos = x_positions[i] + bar_width + bar_width / 2
    y_pos = y_values2[i] + 0.1  
    plt.text(x_pos, y_pos, f'{y_values2[i]}', ha='center', va='bottom')

plt.xticks([i + bar_width/2 for i in index], x_labels)
plt.legend()
plt.show()