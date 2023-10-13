import torch
import torch.optim as optim
import data_process
import model
from Parameter_settings import set_param
import computer_metapath
from torch.autograd import Variable
import numpy as np
import re
import random
torch.set_num_threads(2)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self):
		super(model_class, self).__init__()
		self.gpu = set_param.cuda

		input_data = data_process.input_data()
		self.input_data = input_data

		feature_list = [input_data.p_title_embed,input_data.p_v_net_embed,\
		input_data.p_a_net_embed, input_data.p_net_embed,\
		input_data.a_coop_net_embed,input_data.a_net_embed, input_data.a_text_embed,\
		input_data.v_net_embed, input_data.v_text_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()

		a_neigh_list_train = input_data.a_neigh_list_train
		p_neigh_list_train = input_data.p_neigh_list_train
		v_neigh_list_train = input_data.v_neigh_list_train

		a_train_id_list = input_data.a_train_id_list
		p_train_id_list = input_data.p_train_id_list
		v_train_id_list = input_data.v_train_id_list
		a_metapath_neigh_list, p_metapath_neigh_list, v_metapath_neigh_list = self.get_metapath_neigh()

		self.model = model.ARMHGNN(feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list, a_metapath_neigh_list, p_metapath_neigh_list, v_metapath_neigh_list)

		if self.gpu:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=set_param.lr, weight_decay = 0)
		self.model.init_weights()

	def get_metapath_neigh(self):
		a_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.A_n)]
		p_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.P_n)]
		v_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.V_n)]
		a_metapath_neigh_f = open(set_param.data_path + "a_metapath_neigh_list.txt", "r")
		for line in a_metapath_neigh_f:
			line = line.strip()
			node_id = re.split(':', line)[0]
			metapath_type = re.split(':', line)[1]
			metapath = re.split(':', line)[2]
			metapath_list = eval(metapath)
			if len(node_id[1:]) > 0:
				a_metapath_neigh_list[int(node_id[1:])][int(metapath_type)] = [[int(x) for x in sublist] for sublist in metapath_list]
		a_metapath_neigh_f.close()

		p_metapath_neigh_f = open(set_param.data_path + "p_metapath_neigh_list.txt", "r")
		for line in p_metapath_neigh_f:
			line = line.strip()
			node_id = re.split(':', line)[0]
			metapath_type = re.split(':', line)[1]
			metapath = re.split(':', line)[2]
			metapath_list = eval(metapath)
			if len(node_id[1:]) > 0:
				p_metapath_neigh_list[int(node_id[1:])][int(metapath_type)] = [[int(x) for x in sublist] for sublist in metapath_list]
		p_metapath_neigh_f.close()

		v_metapath_neigh_f = open(set_param.data_path + "v_metapath_neigh_list.txt", "r")
		for line in v_metapath_neigh_f:
			line = line.strip()
			node_id = re.split(':', line)[0]
			metapath_type = re.split(':', line)[1]
			metapath = re.split(':', line)[2]
			metapath_list = eval(metapath)
			if len(node_id[1:]) > 0:
				v_metapath_neigh_list[int(node_id[1:])][int(metapath_type)] = [[int(x) for x in sublist] for sublist in metapath_list]
		v_metapath_neigh_f.close()

		return a_metapath_neigh_list, p_metapath_neigh_list, v_metapath_neigh_list

	def model_train(self):
		print ('model training ...')
		
		self.model.train()
		mini_batch_s = set_param.mini_batch_s
		embed_d = set_param.embed_d
	
		for iter_i in range(1, set_param.train_iter_n + 1):
			print ('iteration ' + str(iter_i) + ' ...')
			triple_list = self.input_data.sample_het_walk_triple()
			min_len = 1e10
			for ii in range(len(triple_list)):
				if len(triple_list[ii]) < min_len:
					min_len = len(triple_list[ii])
			batch_n = int(min_len / mini_batch_s)
			print (batch_n)
			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])

				for triple_index in range(len(triple_list)):
					triple_list_temp = triple_list[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s]
					c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

					c_out[triple_index] = c_out_temp
					p_out[triple_index] = p_out_temp
					n_out[triple_index] = n_out_temp

				loss = model.cross_entropy_loss(c_out, p_out, n_out, embed_d)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step() 

				if k % 100 == 0:
					print ("loss: " + str(loss))

			if iter_i % set_param.save_model_freq == 0:
				torch.save(self.model.state_dict(), set_param.model_path + "ARMHGNN_" + str(iter_i) + ".pt")
				triple_index = 9 
				a_out, p_out, v_out = self.model([], triple_index)
			print ('iteration ' + str(iter_i) + ' finish.')


if __name__ == '__main__':
	random.seed(10)
	np.random.seed(10)
	torch.manual_seed(10)
	torch.cuda.manual_seed_all(10)
	
    #model 
	model_object = model_class()
	model_object.model_train()
	