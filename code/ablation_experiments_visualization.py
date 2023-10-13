# Generate code for Figures 7 and 8
# Draw a comparison chart between the ablation experiment and the original experiment, using a line chart
# Draw eight graphs with four different scales, macro-F1 and micro-F1 for each scale
import matplotlib.pyplot as plt
import seaborn as sns
x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
y_ARMHGNN = [0.9526, 0.9568, 0.9579, 0.9593, 0.9493, 0.9468, 0.9496, 0.9521, 0.9489, 0.9551]
y_no_node_type = [0.9341, 0.9384, 0.9376, 0.9398, 0.9364, 0.9394, 0.9371, 0.9376, 0.9391, 0.9418]
y_no_metapath_type = [0.939, 0.9416, 0.9449, 0.94, 0.9456, 0.9469, 0.9402, 0.9448, 0.9433, 0.9321]

fig, ax = plt.subplots()

sns.set(style = "whitegrid")

#plt.rcParams.update({'font.size': 20})

plt.xticks(fontsize=17)
plt.yticks(fontsize=17)

ax.plot(x, y_ARMHGNN, linestyle = '-.',marker = '>', label = 'ARMHGNN', linewidth = 3)
ax.plot(x, y_no_node_type, linestyle = '--', marker = '*', label = 'no_node_type', linewidth = 3)
ax.plot(x, y_no_metapath_type, linestyle = ':', marker = 'o', label = 'no_metapath_type', linewidth = 3)

ax.set_xlabel('experimental iteration', fontsize=17)
ax.set_ylabel('micro-F1', fontsize=17)
ax.set_title('Training set proportion: 80%', fontsize=17)

ax.legend(loc = 'lower right', bbox_to_anchor = (0.5, 0.5, 0.5, 0.5), fontsize=10)

plt.tight_layout()

plt.show()


# Experimental results for each group of experiments
# There are a total of 8 sets of experimental results, 
# corresponding to the 8 subgraphs in Figure 7 and Figure 8

# 20%, macro-F1
# y_ARMHGNN = [0.9347, 0.9342, 0.9325, 0.9351, 0.9325, 0.9329, 0.9335, 0.9334, 0.9366, 0.934]
# y_no_node_type = [0.9129, 0.9121, 0.9126, 0.9114, 0.9160, 0.9158, 0.9161, 0.9148, 0.9076, 0.9087]
# y_no_metapath_type = [0.9288, 0.9207, 0.917, 0.9232, 0.9212, 0.919, 0.9183, 0.9171, 0.92, 0.9184]

# 20%, micro-F1
# y_ARMHGNN = [0.9391, 0.9398, 0.9379, 0.9405, 0.9383, 0.9376, 0.9388, 0.939, 0.9412, 0.9393]
# y_no_node_type = [0.9209, 0.9214, 0.9211, 0.9189, 0.9244, 0.9236, 0.9254, 0.9227, 0.9175, 0.918]
# y_no_metapath_type = [0.9354, 0.9279, 0.9244, 0.93, 0.9273, 0.9264, 0.9256, 0.925, 0.9266, 0.9256]

# 40%, macro-F1
# y_ARMHGNN = [0.9346, 0.9368, 0.9341, 0.9422, 0.9334, 0.9372, 0.934, 0.9348, 0.9343, 0.9356]
# y_no_node_type = [0.923, 0.9169, 0.9191, 0.9187, 0.9173, 0.9187, 0.918, 0.9153, 0.9109, 0.9131]
# y_no_metapath_type = [0.9247, 0.924, 0.9263, 0.9256, 0.9234, 0.9257, 0.9236, 0.9263, 0.9261, 0.9247]

# 40%, micro-F1
# y_ARMHGNN = [0.9395, 0.9423, 0.9398, 0.9471, 0.9387, 0.942, 0.9388, 0.9382, 0.9397, 0.9424]
# y_no_node_type = [0.9322, 0.9252, 0.9277, 0.9276, 0.9258, 0.9262, 0.9255, 0.9243, 0.9208, 0.9203]
# y_no_metapath_type = [0.9312, 0.929, 0.9317, 0.9325, 0.9303, 0.9311, 0.9308, 0.9323, 0.9336, 0.9295]

# 60%, macro-F1
# y_ARMHGNN = [0.9361, 0.9359, 0.9372, 0.9394, 0.937, 0.938, 0.9361, 0.9364, 0.9397, 0.9423]
# y_no_node_type = [0.9196, 0.9233, 0.9225, 0.9218, 0.9201, 0.9239, 0.9198, 0.9185, 0.9237, 0.9243]
# y_no_metapath_type = [0.9278, 0.9272, 0.927, 0.9276, 0.9295, 0.9277, 0.9264, 0.9262, 0.9278, 0.9276]

# 60%, micro-F1
# y_ARMHGNN = [0.9397, 0.9415, 0.9424, 0.9431, 0.941, 0.9393, 0.9435, 0.9419, 0.9432, 0.947]
# y_no_node_type = [0.9254, 0.9324, 0.9306, 0.9294, 0.9272, 0.9295, 0.9275, 0.9284, 0.9303, 0.9335]
# y_no_metapath_type = [0.9348, 0.9328, 0.9338, 0.933, 0.9361, 0.936, 0.9323, 0.9325, 0.9343, 0.9347]

# 80%, macro-F1
# y_ARMHGNN = [0.9485, 0.955, 0.9534, 0.9556, 0.9476, 0.9461, 0.9446, 0.9477, 0.9458, 0.9523]
# y_no_node_type = [0.9285, 0.9316, 0.928, 0.9332, 0.9312, 0.9333, 0.9308, 0.93, 0.9331, 0.9353]
# y_no_metapath_type = [0.9361, 0.9385, 0.9398, 0.9361, 0.9401, 0.94, 0.9347, 0.9404, 0.9369, 0.9247]

# 80%, micro-F1
# y_ARMHGNN = [0.9526, 0.9568, 0.9579, 0.9593, 0.9493, 0.9468, 0.9496, 0.9521, 0.9489, 0.9551]
# y_no_random_walk = [0.9341, 0.9384, 0.9376, 0.9398, 0.9364, 0.9394, 0.9371, 0.9376, 0.9391, 0.9418]
# y_no_metapath = [0.939, 0.9416, 0.9449, 0.94, 0.9456, 0.9469, 0.9402, 0.9448, 0.9433, 0.9321]

