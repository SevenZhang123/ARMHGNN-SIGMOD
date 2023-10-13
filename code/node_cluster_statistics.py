# In the node clustering task, count the total number of nodes in each category 
# and the number of nodes successfully clustered together
from Parameter_settings import set_param
kmeans_labels_f = open(set_param.data_path + 'kmeans_labels.txt', 'r')
areas_cluster_f = open(set_param.data_path + 'areas_clusterid.txt', 'r')
kmeans_areas_labels_f = open(set_param.data_path + 'kmeans_areas_labels.txt', 'w')

areas_clusterid = {'DB':[], 'DM':[], 'AI':[], 'IR':[]}

for line in areas_cluster_f:
    line.strip()
    line = line.split(':')
    area = line[0]
    cluster_id = line[1]
    cluster_id_list = cluster_id.split(',')
    areas_clusterid[area] = cluster_id_list


for line in kmeans_labels_f:
    line.strip()
    line = line.split(',')
    if line[0] in areas_clusterid['DB']:
        kmeans_areas_labels_f.write(f'DB,{line[1]}\n')
    elif line[0] in areas_clusterid['DM']:
        kmeans_areas_labels_f.write(f'DM,{line[1]}\n')
    elif line[0] in areas_clusterid['AI']:
        kmeans_areas_labels_f.write(f'AI,{line[1]}\n')
    elif line[0] in areas_clusterid['IR']:
        kmeans_areas_labels_f.write(f'IR,{line[1]}\n')

print(len(areas_clusterid['DB']), len(areas_clusterid['DM']), len(areas_clusterid['AI']), len(areas_clusterid['IR']))
areas_cluster_f.close()
kmeans_areas_labels_f.close()
kmeans_labels_f.close()


kmeans_areas_labels_f = open(set_param.data_path + 'kmeans_areas_labels.txt', 'r')
DB_all = 0
DB_correct = 0
DM_all = 0
DM_correct = 0
AI_all = 0
AI_correct = 0
IR_all = 0
IR_correct = 0

for line in kmeans_areas_labels_f:
    line.strip()
    line = line.split(',')
    area = line[0]
    if area == 'DB':
        DB_all += 1
        #print(len(line[1]))
        if line[1] == '0\n':
            DB_correct += 1
    elif area == 'DM':
        DM_all += 1
        if line[1] == '2\n':
            DM_correct += 1
    elif area == 'AI':
        AI_all += 1
        if line[1] == '1\n':
            AI_correct += 1
    elif area == 'IR':
        IR_all += 1
        if line[1] == '3\n':
            IR_correct += 1
print("DB_all_number:{0}, DB_current_number:{1}".format(DB_all, DB_correct))
print("DM_all_number:{0}, DM_current_number:{1}".format(DM_all, DM_correct))
print("AI_all_number:{0}, AI_current_number:{1}".format(AI_all, AI_correct))
print("IR_all_number:{0}, IR_current_number:{1}".format(IR_all, IR_correct))
