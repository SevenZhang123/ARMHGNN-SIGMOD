import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from Parameter_settings import set_param
import networkx as nx


class input_data(object):
	def __init__(self):

		a_p_list = [[] for k in range(set_param.A_n)]
		a_v_list = [[] for k in range(set_param.A_n)]
		p_a_list = [[] for k in range(set_param.P_n)]
		a_a_cite_list = [[] for k in range(set_param.A_n)]
		v_p_list = [[] for k in range(set_param.V_n)]

		relation_f = ["a_p_list.txt", "p_a_list.txt",\
		 "a_a_cooperate.txt", "v_p_list.txt", "a_v_list.txt"]

		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(set_param.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
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

		#store paper venue 
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

		self.triple_sample_p = self.computer_sample_p()

		p_title_embed = np.zeros((set_param.P_n, set_param.in_f_d))
		p_t_e_f = open(set_param.data_path + "p_title_embed.txt", "r")
		for line in islice(p_t_e_f, 1, None):
			index = int(line.split(":")[0])
			values = line.split(":")[1]
			values = values.split()
			embeds = np.asarray(values, dtype='float32')
			p_title_embed[index] = embeds
		p_t_e_f.close()

		self.p_title_embed = p_title_embed

		a_net_embed = np.zeros((set_param.A_n, set_param.in_f_d))
		p_net_embed = np.zeros((set_param.P_n, set_param.in_f_d))
		v_net_embed = np.zeros((set_param.V_n, set_param.in_f_d)) 
		net_e_f = open(set_param.data_path + "All_node_net_embedding.txt", "r")
		for line in islice(net_e_f, 1, None):
			line = line.strip()
			index = re.split(' ', line)[0]
			if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p'):
				embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
				if index[0] == 'a':
					a_net_embed[int(index[1:])] = embeds
				elif index[0] == 'v':
					v_net_embed[int(index[1:])] = embeds
				else:
					p_net_embed[int(index[1:])] = embeds
		net_e_f.close()

		p_v_net_embed = np.zeros((set_param.P_n, set_param.in_f_d))
		p_v = [0] * set_param.P_n
		p_v_f = open(set_param.data_path + "p_v_list.txt", "r")
		for line in p_v_f:
			line = line.strip()
			p_id = int(re.split(':', line)[0])
			v_id = int(re.split(':', line)[1])
			p_v[p_id] = v_id
			p_v_net_embed[p_id] = v_net_embed[v_id]
		p_v_f.close()

		p_a_net_embed = np.zeros((set_param.P_n, set_param.in_f_d))
		for i in range(set_param.P_n):
			if len(p_a_list[i]):
				for j in range(len(p_a_list[i])):
					a_id = int(p_a_list[i][j][1:])
					p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
				p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list[i])

		a_coop_net_embed = np.zeros((set_param.A_n, set_param.in_f_d))
		for i in range(set_param.A_n):
			if len(a_a_cite_list[i]):
				for j in range(len(a_a_cite_list[i])):
					a_id = int(a_a_cite_list[i][j][1:])
					a_coop_net_embed[i] = np.add(a_coop_net_embed[i], a_net_embed[a_id])
				a_coop_net_embed[i] = a_coop_net_embed[i] / len(a_a_cite_list[i])
			else:
				a_coop_net_embed[i] = p_net_embed[i]

		a_name_embed = np.zeros((set_param.A_n, set_param.in_f_d))
		a_name_f = open(set_param.data_path + "a_name_embed.txt", "r")
		for line in a_name_f:
			author_number = int(line.split(":")[0])
			author_name = line.split(":")[1]
			author_name = author_name.split()
			embeds = np.asarray(author_name, dtype='float32')
			a_name_embed[author_number] = embeds
		a_name_f.close()

		a_text_embed = np.zeros((set_param.A_n, set_param.in_f_d * 3))
		for i in range(set_param.A_n):
			feature_temp = []
			feature_temp.append(a_name_embed[i])
			if len(a_p_list[i]):
				if len(a_p_list[i]) >= 2:
					for j in range(2):
						feature_temp.append(p_title_embed[int(a_p_list[i][j][1:])])
				else:
					for j in range(len(a_p_list[i])):
						feature_temp.append(p_title_embed[int(a_p_list[i][j][1:])])
					for k in range(len(a_p_list[i]), 2):
						feature_temp.append(p_title_embed[int(a_p_list[i][-1][1:])])

				feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
				a_text_embed[i] = feature_temp


		v_text_embed = np.zeros((set_param.V_n, set_param.in_f_d * 5))
		for i in range(set_param.V_n):
			if len(v_p_list[i]):
				feature_temp = []
				if len(v_p_list[i]) >= 5:
					for j in range(5):
						feature_temp.append(p_title_embed[int(v_p_list[i][j][1:])])
				else:
					for j in range(len(v_p_list[i])):
						feature_temp.append(p_title_embed[int(v_p_list[i][j][1:])])
					for k in range(len(v_p_list[i]), 5):
						feature_temp.append(p_title_embed[int(v_p_list[i][-1][1:])])

				feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
				v_text_embed[i] = feature_temp

		self.p_v = p_v
		self.p_v_net_embed = p_v_net_embed
		self.p_a_net_embed = p_a_net_embed
		self.a_coop_net_embed = a_coop_net_embed
		self.p_net_embed = p_net_embed
		self.a_net_embed = a_net_embed
		self.a_text_embed = a_text_embed
		self.v_net_embed = v_net_embed
		self.v_text_embed = v_text_embed


		a_neigh_list_train = [[[] for i in range(set_param.A_n)] for j in range(3)]
		p_neigh_list_train = [[[] for i in range(set_param.P_n)] for j in range(3)]
		v_neigh_list_train = [[[] for i in range(set_param.V_n)] for j in range(3)]

		het_neigh_train_f = open(set_param.data_path + "random_neigh_node.txt", "r")
		for line in het_neigh_train_f:
			line = line.strip()
			node_id = re.split(':', line)[0]
			neigh = re.split(':', line)[1]
			neigh_list = re.split(',', neigh)
			if node_id[0] == 'a' and len(node_id) > 1:
				for j in range(len(neigh_list)):
					if neigh_list[j][0] == 'a':
						a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
					elif neigh_list[j][0] == 'p':
						a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
					elif neigh_list[j][0] == 'v':
						a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
			elif node_id[0] == 'p' and len(node_id) > 1:
				for j in range(len(neigh_list)):
					if neigh_list[j][0] == 'a':
						p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
					if neigh_list[j][0] == 'p':
						p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
					if neigh_list[j][0] == 'v':
						p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
			elif node_id[0] == 'v' and len(node_id) > 1:
				for j in range(len(neigh_list)):
					if neigh_list[j][0] == 'a':
						v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
					if neigh_list[j][0] == 'p':
						v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
					if neigh_list[j][0] == 'v':
						v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))	
		het_neigh_train_f.close()

		a_neigh_list_train_top = [[[] for i in range(set_param.A_n)] for j in range(3)]
		p_neigh_list_train_top = [[[] for i in range(set_param.P_n)] for j in range(3)]
		v_neigh_list_train_top = [[[] for i in range(set_param.V_n)] for j in range(3)]
		top_k = [10, 20, 3] #fix each neighor type size 
		for i in range(set_param.A_n):
			for j in range(3):
				a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
				top_list = a_neigh_list_train_temp.most_common(top_k[j])
				neigh_size = 0
				if j == 0:
					neigh_size = 10
				elif j == 1:
					neigh_size = 20
				else:
					neigh_size = 3
				for k in range(len(top_list)):
					a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
				if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
					a_neigh_list_train_top_cnt = len(a_neigh_list_train_top[j][i])
					for l in range(a_neigh_list_train_top_cnt, neigh_size):
						if l - a_neigh_list_train_top_cnt < len(top_list):
							a_neigh_list_train_top[j][i].append(int(top_list[l-a_neigh_list_train_top_cnt][0]))
						else:
							a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

		for i in range(set_param.P_n):
			for j in range(3):
				p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
				top_list = p_neigh_list_train_temp.most_common(top_k[j])
				neigh_size = 0
				if j == 0:
					neigh_size = 10
				elif j == 1:
					neigh_size = 20
				else:
					neigh_size = 3
				for k in range(len(top_list)):
					p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
				if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
					p_neigh_list_train_top_cnt = len(p_neigh_list_train_top[j][i])
					for l in range(p_neigh_list_train_top_cnt, neigh_size):
						if l - p_neigh_list_train_top_cnt < len(top_list):
							p_neigh_list_train_top[j][i].append(int(top_list[l-p_neigh_list_train_top_cnt][0]))
						else:
							p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

		for i in range(set_param.V_n):
			for j in range(3):
				v_neigh_list_train_temp = Counter(v_neigh_list_train[j][i])
				top_list = v_neigh_list_train_temp.most_common(top_k[j])
				neigh_size = 0
				if j == 0:
					neigh_size = 10
				elif j == 1:
					neigh_size = 20
				else:
					neigh_size = 3
				for k in range(len(top_list)):
					v_neigh_list_train_top[j][i].append(int(top_list[k][0]))
				if len(v_neigh_list_train_top[j][i]) and len(v_neigh_list_train_top[j][i]) < neigh_size:
					v_neigh_list_train_top_cnt = len(v_neigh_list_train_top[j][i])
					for l in range(v_neigh_list_train_top_cnt, neigh_size):
						if l - v_neigh_list_train_top_cnt < len(top_list):
							v_neigh_list_train_top[j][i].append(int(top_list[l-v_neigh_list_train_top_cnt][0]))
						else:
							v_neigh_list_train_top[j][i].append(random.choice(v_neigh_list_train_top[j][i]))
						

		a_neigh_list_train[:] = []
		p_neigh_list_train[:] = []
		v_neigh_list_train[:] = []

		self.a_neigh_list_train = a_neigh_list_train_top
		self.p_neigh_list_train = p_neigh_list_train_top
		self.v_neigh_list_train = v_neigh_list_train_top

		train_id_list = [[] for i in range(3)]
		for i in range(3):
			if i == 0:
				for l in range(set_param.A_n):
					if len(a_neigh_list_train_top[i][l]):
						train_id_list[i].append(l)
				self.a_train_id_list = np.array(train_id_list[i])
			elif i == 1:
				for l in range(set_param.P_n):
					if len(p_neigh_list_train_top[i][l]):
						train_id_list[i].append(l)
				self.p_train_id_list = np.array(train_id_list[i])
			else:
				for l in range(set_param.V_n):
					if len(v_neigh_list_train_top[i][l]):
						train_id_list[i].append(l)
				self.v_train_id_list = np.array(train_id_list[i])
	

	def computer_sample_p(self):
		window = set_param.window
		walk_L = set_param.walk_L
		A_n = set_param.A_n
		P_n = set_param.P_n
		V_n = set_param.V_n

		total_triple_n = [0.0] * 9
		het_walk_f = open(set_param.data_path + "random_walk_result.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1	#aa
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1	#ap
								elif neighNode[0] == 'v':
									total_triple_n[2] += 1	#av
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[3] += 1	#pa
								elif neighNode[0] == 'p':
									total_triple_n[4] += 1	#pp
								elif neighNode[0] == 'v':
									total_triple_n[5] += 1	#pv
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[6] += 1	#va
								elif neighNode[0] == 'p':
									total_triple_n[7] += 1	#vp
								elif neighNode[0] == 'v':
									total_triple_n[8] += 1	#vv
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = set_param.batch_s / (total_triple_n[i] * 10)
		print("sampling ratio computing finish.")

		return total_triple_n

	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(9)]
		window = set_param.window
		walk_L = set_param.walk_L
		A_n = set_param.A_n
		P_n = set_param.P_n
		V_n = set_param.V_n
		triple_sample_p = self.triple_sample_p

		het_walk_f = open(set_param.data_path + "random_walk_result.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
									a_negNode = 'a' + str(negNode)
									while len(self.a_p_list[negNode]) == 0 or (a_negNode in self.a_a_cite_list[int(centerNode[1:])]) or (a_negNode in path):
										negNode = random.randint(0, A_n - 1)
										a_negNode = 'a' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									p_negNode = 'p' + str(negNode)
									while len(self.p_a_list[negNode]) == 0 or (p_negNode in self.a_p_list[int(centerNode[1:])]) or (p_negNode in path):
										negNode = random.randint(0, P_n - 1)
										p_negNode = 'p' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									v_negNode = 'v' + str(negNode)
									while len(self.v_p_list[negNode]) == 0 or (v_negNode in self.a_v_list[int(centerNode[1:])]) or (v_negNode in path):
										negNode = random.randint(0, V_n - 1)
										v_negNode = 'v' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, A_n - 1)
									a_negNode = 'a' + str(negNode)
									while len(self.a_p_list[negNode]) == 0 or (a_negNode in self.p_a_list[int(centerNode[1:])]) or (a_negNode in path):
										negNode = random.randint(0, A_n - 1)
										a_negNode = 'a' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, P_n - 1)
									p_negNode = 'p' + str(negNode)
									while len(self.p_a_list[negNode]) == 0 or (p_negNode in path):
										negNode = random.randint(0, P_n - 1)
										p_negNode = 'p' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, V_n - 1)
									v_negNode = 'v' + str(negNode)
									while len(self.v_p_list[negNode]) == 0 or (negNode == self.p_v[int(centerNode[1:])]) or (v_negNode in path):
										negNode = random.randint(0, V_n - 1)
										v_negNode = 'v' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, A_n - 1)
									a_negNode = 'a' + str(negNode)
									while len(self.a_p_list[negNode]) == 0 or (a_negNode in path):
										negNode = random.randint(0, A_n - 1)
										a_negNode = 'a' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, P_n - 1)
									p_negNode = 'p' + str(negNode)
									while len(self.p_a_list[negNode]) == 0 or (p_negNode in self.v_p_list[int(centerNode[1:])]) or (p_negNode in path):
										negNode = random.randint(0, P_n - 1)
										p_negNode = 'p' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, V_n - 1)
									v_negNode = 'v' + str(negNode)
									while len(self.v_p_list[negNode]) == 0 or (v_negNode in path):
										negNode = random.randint(0, V_n - 1)
										v_negNode = 'v' + str(negNode)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
		het_walk_f.close()

		return triple_list