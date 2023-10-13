# In the classification task, count the total number of authors in each category 
# and the number of correctly classified authors. Conduct ten experiments 
# and calculate the average value

from Parameter_settings import set_param
import matplotlib.pyplot as plt
import numpy as np

DB_current_number = 0
DB_all_number = 0
DM_current_number = 0
DM_all_number = 0
AI_current_number = 0
AI_all_number = 0
IR_current_number = 0
IR_all_number = 0
with open(set_param.data_path + "a_class_test.txt", 'r') as target_f, open(set_param.data_path + "NC_prediction.txt", 'r') as predict_f:
    lines1 = target_f.readlines()
    lines2 = predict_f.readlines()
        
for line_num, (line1, line2) in enumerate(zip(lines1, lines2)):
    line1.strip()
    line1 = line1.split(",")
    target_label = int(line1[1])
    line2.strip()
    line2 = line2.split(",")
    predict_f = line2[1]
    if target_label == 0:
        DB_all_number += 1
    elif target_label == 1:
        DM_all_number += 1
    elif target_label == 2:
        AI_all_number += 1
    elif target_label == 3:
        IR_all_number += 1    
    if target_label == int(predict_f[0]):
        if target_label == 0:
            DB_current_number += 1
        elif target_label == 1:
            DM_current_number += 1
        elif target_label == 2:
            AI_current_number += 1
        elif target_label == 3:
            IR_current_number += 1
        
print("DB_all_number:{0}, DB_current_number:{1}".format(DB_all_number, DB_current_number))
print("DM_all_number:{0}, DM_current_number:{1}".format(DM_all_number, DM_current_number))
print("AI_all_number:{0}, AI_current_number:{1}".format(AI_all_number, AI_current_number))
print("IR_all_number:{0}, IR_current_number:{1}".format(IR_all_number, IR_current_number))


