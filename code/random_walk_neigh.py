import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from Parameter_settings import set_param
import networkx as nx


class random_walk_neigh(object):
	def __init__(self):

		a_p_list = [[] for k in range(set_param.A_n)]
		p_a_list = [[] for k in range(set_param.P_n)]
		a_v_list = [[] for k in range(set_param.A_n)]
		a_a_cite_list = [[] for k in range(set_param.A_n)]
		v_p_list = [[] for k in range(set_param.V_n)]

		relation_f = ["a_p_list.txt", "p_a_list.txt",\
		 "a_a_cooperate.txt", "v_p_list.txt", "a_v_list.txt"]

		#store academic relational data 
		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(set_param.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				#print("f_name:{0},node_id:{1},neigh_id:{2}".format(f_name,node_id,neigh_list_id))
				if len(neigh_list_id) and neigh_list_id[-1][-1] == "\n":
					neigh_list_id[-1] = neigh_list_id[-1][:-1]
				if f_name == 'a_p_list.txt':
					for j in range(len(neigh_list_id)):
						a_p_list[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'p_a_list.txt':
					for j in range(len(neigh_list_id)):
						p_a_list[node_id].append('a'+str(neigh_list_id[j]))
				elif f_name == 'a_a_cooperate.txt':
					for j in range(len(neigh_list_id)):
						a_a_cite_list[node_id].append('a'+str(neigh_list_id[j]))
				elif f_name == 'v_p_list.txt':
					for j in range(len(neigh_list_id)):
						v_p_list[node_id].append('p'+str(neigh_list_id[j]))
				elif f_name == 'a_v_list.txt':
					for j in range(len(neigh_list_id)):
						a_v_list[node_id].append('v'+str(neigh_list_id[j]))
			neigh_f.close()
		a_v_list_top = [[] for k in range(set_param.A_n)]
		for i in range(set_param.A_n):
			a_v_list_temp = Counter(a_v_list[i])
			top_list = a_v_list_temp.most_common(3)
			for k in range(len(top_list)):
				a_v_list_top[i].append(top_list[k][0])
			a_v_list[i] = a_v_list_top[i]

		p_v = [0] * set_param.P_n
		p_v_f = open(set_param.data_path + 'p_v_list.txt', "r")
		for line in p_v_f:
			line = line.strip()
			p_id = int(re.split(':',line)[0])
			v_id = int(re.split(':',line)[1])
			p_v[p_id] = v_id
		p_v_f.close()

		p_neigh_list = [[] for k in range(set_param.P_n)]
		for i in range(set_param.P_n):
			p_neigh_list[i] += p_a_list[i]
			p_neigh_list[i].append('v' + str(p_v[i]))

		a_neigh_list = [[] for k in range(set_param.A_n)]
		for i in range(set_param.A_n):
			a_neigh_list[i] += a_p_list[i]
			a_neigh_list[i] += a_a_cite_list[i]
			a_neigh_list[i] += a_v_list[i]

		self.a_p_list =  a_p_list
		self.p_a_list = p_a_list
		self.a_a_cite_list = a_a_cite_list
		self.p_neigh_list = p_neigh_list
		self.a_neigh_list = a_neigh_list
		self.v_p_list = v_p_list
		self.a_v_list = a_v_list

	def het_walk_restart(self):
		a_random_neigh_list = [[] for k in range(set_param.A_n)]
		p_random_neigh_list = [[] for k in range(set_param.P_n)]
		v_random_neigh_list = [[] for k in range(set_param.V_n)]

		node_n = [set_param.A_n, set_param.P_n, set_param.V_n]
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_temp = self.a_neigh_list[j]	
					neigh_train = a_random_neigh_list[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_temp = self.p_neigh_list[j]
					neigh_train = p_random_neigh_list[j]
					curNode = "p" + str(j)
				else:
					neigh_temp = self.v_p_list[j]
					neigh_train = v_random_neigh_list[j]
					curNode = "v" + str(j)
				if len(neigh_temp):
					neigh_L = 0
					a_L = 0
					p_L = 0
					v_L = 0
					while neigh_L < 100: 
						rand_p = random.random() #return p
						if rand_p > 0.4:
							if curNode[0] == "a":
								curNode = random.choice(self.a_neigh_list[int(curNode[1:])])
								if  curNode[0] == 'a' and a_L < 26: 
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'p' and p_L < 66:
									neigh_train.append(curNode)
									neigh_L += 1
									p_L += 1
							elif curNode[0] == "p":
								curNode = random.choice(self.p_neigh_list[int(curNode[1:])])
								if  curNode[0] == 'a' and a_L < 26:
									neigh_train.append(curNode)
									neigh_L += 1
									a_L += 1
								elif curNode[0] == 'v' and v_L < 11:
									neigh_train.append(curNode)
									neigh_L += 1
									v_L += 1
							elif curNode[0] == "v":
								curNode = random.choice(self.v_p_list[int(curNode[1:])])
								if p_L < 66:
									neigh_train.append(curNode)
									neigh_L +=1
									p_L += 1
						else:
							if i == 0:
								curNode = ('a' + str(j))
							elif i == 1:
								curNode = ('p' + str(j))
							else:
								curNode = ('v' + str(j))

				if i == 0:
					a_random_neigh_list[j] = neigh_train
				elif i == 1:
					p_random_neigh_list[j] = neigh_train
				elif i == 2:
					v_random_neigh_list[j] = neigh_train


		neigh_f = open(set_param.data_path + "random_neigh_node.txt", "w")
		for i in range(3):
			for j in range(node_n[i]):
				if i == 0:
					neigh_train = a_random_neigh_list[j]
					curNode = "a" + str(j)
				elif i == 1:
					neigh_train = p_random_neigh_list[j]
					curNode = "p" + str(j)
				else:
					neigh_train = v_random_neigh_list[j]
					curNode = "v" + str(j)
				if len(neigh_train):
					neigh_f.write(curNode + ":")
					for k in range(len(neigh_train) - 1):
						neigh_f.write(neigh_train[k] + ",")
					neigh_f.write(neigh_train[-1] + "\n")
		neigh_f.close()

	def gen_het_rand_walk(self):
		het_walk_f = open(set_param.data_path + "random_walk_result.txt", "w")
		#print len(self.p_neigh_list_train)
		for i in range(set_param.walk_n):
			for j in range(set_param.A_n):
				if len(self.a_neigh_list[j]):
					curNode = "a" + str(j)
					het_walk_f.write(curNode + " ")
					walk_length = 1
					while walk_length <= set_param.walk_L:
						if walk_length % 11 != 0:
							if curNode[0] == "a":
								curNode = int(curNode[1:])
								curNode = random.choice(self.a_neigh_list[curNode])
								het_walk_f.write(curNode + " ")
								walk_length += 1
							elif curNode[0] == "p":
								curNode = int(curNode[1:])
								curNode = random.choice(self.p_neigh_list[curNode])
								het_walk_f.write(curNode + " ")
								walk_length += 1
							elif curNode[0] == "v": 
								curNode = int(curNode[1:])
								curNode = random.choice(self.v_p_list[curNode])
								het_walk_f.write(curNode + " ")
								walk_length += 1
						else:
							curNode = "a" + str(j)
							het_walk_f.write(curNode + " ")
							walk_length += 1
					het_walk_f.write("\n")
		het_walk_f.close()


generate_random_walk_neigh_file = random_walk_neigh()
generate_random_walk_neigh_file.gen_het_rand_walk()
