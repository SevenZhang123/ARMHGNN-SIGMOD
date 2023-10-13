import string
import re
import numpy as np
import os
import sys
import random
from itertools import *
import argparse
from Parameter_settings import set_param
import node_classification_model as Node_classification
import node_clustering_model as Node_cluster

def a_class_feature_setting():
	a_embed = np.around(np.random.normal(0, 0.01, [set_param.A_n, set_param.embed_d]), 4)
	embed_f = open(set_param.data_path + "node_embedding_200.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
	embed_f.close()

	a_p_list_train = [[] for k in range(set_param.A_n)]
	a_p_list_train_f = open(set_param.data_path + "a_p_list.txt", "r")
	for line in a_p_list_train_f:
		line = line.strip()
		node_id = int(re.split(':', line)[0])
		neigh_list = re.split(':', line)[1]
		neigh_list_id = re.split(',', neigh_list)
		for j in range(len(neigh_list_id)):
			a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
	a_p_list_train_f.close()

	p_v = [0] * set_param.P_n
	p_v_f = open(set_param.data_path + 'p_v_list.txt', "r")
	for line in p_v_f:
		line = line.strip()
		p_id = int(re.split(':',line)[0])
		v_id = int(re.split(':',line)[1])
		p_v[p_id] = v_id
	p_v_f.close()

	a_v_list_train = [[] for k in range(set_param.A_n)]
	a_v_f = open(set_param.data_path + 'a_v_list.txt', "r")
	for line in a_v_f:
		line = line.strip()
		a_id = int(re.split(':', line)[0])
		v_id = re.split(':', line)[1]
		v_id_list = re.split(',', v_id)[0:]
		for i in range(len(v_id_list)):
			v_id_list[i] = int(v_id_list[i])
			a_v_list_train[a_id].append(v_id_list[i])

	# for i in range(len(a_p_list_train)):#tranductive node classification
	# 	for j in range(len(a_p_list_train[i])):
	# 		p_id = int(a_p_list_train[i][j][1:])
	# 		a_v_list_train[i].append(p_v[p_id])

	a_v_num = [[0 for k in range(set_param.V_n)] for k in range(set_param.A_n)]
	for i in range(set_param.A_n):
		for j in range(len(a_v_list_train[i])):
			v_index = int(a_v_list_train[i][j])
			a_v_num[i][v_index] += 1

	a_max_v = [0] * set_param.A_n
	for i in range(set_param.A_n):
		a_max_v[i] = a_v_num[i].index(max(a_v_num[i]))

	a_class_list = [[] for k in range(set_param.C_n)]
	num_hidden = set_param.embed_d
	for i in range(set_param.A_n):
		if len(a_p_list_train[i]):
			if a_max_v[i] == 5 or a_max_v[i] == 6 or a_max_v[i] == 13 or a_max_v[i] == 16 or a_max_v[i] == 17:#DB:EDBT, ICDE, PODS, SIGMOD, VLDB
				a_class_list[0].append(i)
			elif a_max_v[i] == 7 or a_max_v[i] == 10 or a_max_v[i] == 11 or a_max_v[i] == 12 or a_max_v[i] == 14: #DM:ICDM, KDD, PAKDD, PKDD, SDM
				a_class_list[1].append(i)
			elif a_max_v[i] == 0 or a_max_v[i] == 2 or a_max_v[i] == 4 or a_max_v[i] == 8 or a_max_v[i] == 9: #AI:AAAI, CVPR, ECML, ICML, IJCAI
				a_class_list[2].append(i)
			elif a_max_v[i] == 1 or a_max_v[i] == 3 or a_max_v[i] == 15 or a_max_v[i] == 18 or a_max_v[i] == 19: #IR:CIKM, ECIR, SIGIR, WWW, WSDM
				a_class_list[3].append(i)
	#print(len(a_class_list[0]), len(a_class_list[1]), len(a_class_list[2]), len(a_class_list[3]))

	a_class_train_f = open(set_param.data_path + "a_class_train.txt", "w")
	a_class_test_f = open(set_param.data_path + "a_class_test.txt", "w")
	train_class_feature_f = open(set_param.data_path + "train_class_feature.txt", "w")
	test_class_feature_f = open(set_param.data_path + "test_class_feature.txt", "w")
	train_num = 0
	test_num = 0
	for i in range(set_param.C_n):
		for j in range(len(a_class_list[i])):
			randvalue = random.random()
			if randvalue < 0.2:
				a_class_train_f.write("%d,%d\n"%(a_class_list[i][j],i))
				train_class_feature_f.write("%d,%d," %(a_class_list[i][j],i))
				for d in range(num_hidden - 1):
					train_class_feature_f.write("%lf," %a_embed[a_class_list[i][j]][d])
				train_class_feature_f.write("%lf" %a_embed[a_class_list[i][j]][num_hidden-1])
				train_class_feature_f.write("\n")
				train_num += 1
			else:
				a_class_test_f.write("%d,%d\n"%(a_class_list[i][j],i))
				test_class_feature_f.write("%d,%d," %(a_class_list[i][j],i))
				for d in range(num_hidden - 1):
					test_class_feature_f.write("%lf," %a_embed[a_class_list[i][j]][d])
				test_class_feature_f.write("%lf" %a_embed[a_class_list[i][j]][num_hidden-1])
				test_class_feature_f.write("\n")
				test_num += 1
	a_class_train_f.close()
	a_class_test_f.close()
	# print("train_num: " + str(train_num))
	# print("test_num: " + str(test_num))
	# print("train_ratio: " + str(float(train_num) / (train_num + test_num)))


	return train_num, test_num

def a_cluster_feature_setting():
	a_embed = np.around(np.random.normal(0, 0.01, [set_param.A_n, set_param.embed_d]), 4)
	embed_f = open(set_param.data_path + "node_embedding_200.txt", "r")
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
			type_label = node_id[0]
			index = int(node_id[1:])
			embed = np.asarray(re.split(' ',line)[1:], dtype='float32')
			if type_label == 'a':
				a_embed[index] = embed
	embed_f.close()

	a_p_list_train = [[] for k in range(set_param.A_n)]
	a_p_list_train_f = open(set_param.data_path + "a_p_list.txt", "r")
	for line in a_p_list_train_f:
		line = line.strip()
		node_id = int(re.split(':', line)[0])
		neigh_list = re.split(':', line)[1]
		neigh_list_id = re.split(',', neigh_list)
		for j in range(len(neigh_list_id)):
			a_p_list_train[node_id].append('p'+str(neigh_list_id[j]))
	a_p_list_train_f.close()

	p_v = [0] * set_param.P_n
	p_v_f = open(set_param.data_path + 'p_v_list.txt', "r")
	for line in p_v_f:
		line = line.strip()
		p_id = int(re.split(':',line)[0])
		v_id = int(re.split(':',line)[1])
		p_v[p_id] = v_id
	p_v_f.close()

	a_v_list_train = [[] for k in range(set_param.A_n)]
	for i in range(len(a_p_list_train)):#tranductive node classification
		for j in range(len(a_p_list_train[i])):
			p_id = int(a_p_list_train[i][j][1:])
			a_v_list_train[i].append(p_v[p_id])

	a_v_num = [[0 for k in range(set_param.V_n)] for k in range(set_param.A_n)]
	for i in range(set_param.A_n):
		for j in range(len(a_v_list_train[i])):
			v_index = int(a_v_list_train[i][j])
			a_v_num[i][v_index] += 1

	a_max_v = [0] * set_param.A_n
	for i in range(set_param.A_n):
		a_max_v[i] =  a_v_num[i].index(max(a_v_num[i]))

	cluster_f = open(set_param.data_path + "cluster.txt", "w")
	cluster_embed_f = open(set_param.data_path + "cluster_embed.txt", "w")
	areas_clusterid_f = open(set_param.data_path + 'areas_clusterid.txt', 'w')
	cluster_id = 0
	areas_clusterid = {'DB':[], 'DM':[], 'AI':[], 'IR':[]}
	num_hidden = set_param.embed_d
	for i in range(set_param.A_n):
		if len(a_p_list_train[i]):
			if a_max_v[i] == 6 or a_max_v[i] == 13 or a_max_v[i] == 16 or a_max_v[i] == 17:#DB
				cluster_f.write("%d,%d\n"%(cluster_id,0))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				areas_clusterid['DB'].append(cluster_id)
				cluster_id += 1
			elif a_max_v[i] == 10 or a_max_v[i] == 11 or a_max_v[i] == 12 or a_max_v[i] == 14: #DM
				cluster_f.write("%d,%d\n"%(cluster_id,1))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				areas_clusterid['DM'].append(cluster_id)
				cluster_id += 1
			elif a_max_v[i] == 2 or a_max_v[i] == 4 or a_max_v[i] == 8 or a_max_v[i] == 9: #AI
				cluster_f.write("%d,%d\n"%(cluster_id,2))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				areas_clusterid['AI'].append(cluster_id)
				cluster_id += 1
			elif a_max_v[i] == 3 or a_max_v[i] == 15 or a_max_v[i] == 18 or a_max_v[i] == 19: #IR
				cluster_f.write("%d,%d\n"%(cluster_id,3))
				cluster_embed_f.write("%d "%(cluster_id))
				for k in range(num_hidden):
					cluster_embed_f.write("%lf "%(a_embed[i][k]))
				cluster_embed_f.write("\n")
				areas_clusterid['IR'].append(cluster_id)
				cluster_id += 1
	
	areas_clusterid_f.write('DB:')
	for i in areas_clusterid['DB']:
		areas_clusterid_f.write(f'{i},')
	areas_clusterid_f.write('\n')
	areas_clusterid_f.write('DM:')
	for i in areas_clusterid['DM']:
		areas_clusterid_f.write(f'{i},')
	areas_clusterid_f.write('\n')
	areas_clusterid_f.write('AI:')
	for i in areas_clusterid['AI']:
		areas_clusterid_f.write(f'{i},')
	areas_clusterid_f.write('\n')
	areas_clusterid_f.write('IR:')
	for i in areas_clusterid['IR']:
		areas_clusterid_f.write(f'{i},')
	areas_clusterid_f.write('\n')

	areas_clusterid_f.close()
	cluster_f.close()
	cluster_embed_f.close()
	#print id_
	# print("train_num: " + str(train_num))
	# print("test_num: " + str(test_num))
	# print("train_ratio: " + str(float(train_num) / (train_num + test_num)))

	return cluster_id


#The application of author classification
"""
print("------author classification------")
train_num, test_num = a_class_feature_setting() #setup of author classification task
Node_classification.model(train_num, test_num)
print("------author classification end------")
"""

#The application of author clustering
print("------author clustering------")
cluster_id = a_cluster_feature_setting() #setup of author clustering task
Node_cluster.model(cluster_id)
print("------author clustering end------")
