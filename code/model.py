import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import string
import re
import math
from Parameter_settings import set_param

class ARMHGNN(nn.Module):
    def __init__(self, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train,\
		 a_train_id_list, p_train_id_list, v_train_id_list, a_metapath_neigh_list,\
         p_metapath_neigh_list, v_metapath_neigh_list):
        super(ARMHGNN, self).__init__()
        embed_d = set_param.embed_d
        in_f_d = set_param.in_f_d
        self.P_n = set_param.P_n
        self.A_n = set_param.A_n
        self.V_n = set_param.V_n
        self.feature_list = feature_list
        self.a_neigh_list_train = a_neigh_list_train
        self.p_neigh_list_train = p_neigh_list_train
        self.v_neigh_list_train = v_neigh_list_train
        self.a_train_id_list = a_train_id_list
        self.p_train_id_list = p_train_id_list
        self.v_train_id_list = v_train_id_list
        self.a_metapath_neigh_list = a_metapath_neigh_list
        self.p_metapath_neigh_list = p_metapath_neigh_list
        self.v_metapath_neigh_list = v_metapath_neigh_list

        #Attribute aggregation of nodes
        self.a_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_content_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

        #Aggregation of neighboring nodes of the same type
        self.a_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_neigh_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

        #Aggregation of nodes within the metapath
        self.a_metapath_inner_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.a_metapath_inner_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_metapath_inner_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_metapath_inner_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_metapath_inner_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_metapath_inner_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

        #Aggregation of Similar Metapaths
        self.a_metapath_neigh_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.a_metapath_neigh_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_metapath_neigh_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.p_metapath_neigh_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_metapath_neigh_type1_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)
        self.v_metapath_neigh_type2_rnn = nn.LSTM(embed_d, int(embed_d/2), 1, bidirectional = True)

        self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)
        self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad = True)

        self.softmax = nn.Softmax(dim = 1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p = 0.5)
        self.bn = nn.BatchNorm1d(embed_d)
        self.embed_d = embed_d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def a_content_agg(self, id_batch): 
        embed_d = self.embed_d
        a_coop_embed_batch = self.feature_list[4][id_batch]
        a_net_embed_batch = self.feature_list[5][id_batch]
        a_text_embed_batch_1 = self.feature_list[6][id_batch, :embed_d][0]
        a_text_embed_batch_2 = self.feature_list[6][id_batch, embed_d : embed_d * 2][0]
        a_text_embed_batch_3 = self.feature_list[6][id_batch, embed_d * 2 : embed_d * 3][0]
        
        concate_embed = torch.cat((a_coop_embed_batch, a_net_embed_batch, a_text_embed_batch_1,\
        a_text_embed_batch_2, a_text_embed_batch_3), 1).view(len(id_batch[0]), 5, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.a_content_rnn(concate_embed)

        return torch.mean(all_state, 0)
    
    def p_content_agg(self, id_batch):
        embed_d = self.embed_d
        p_t_embed_batch = self.feature_list[0][id_batch]
        p_v_net_embed_batch = self.feature_list[1][id_batch]
        p_a_net_embed_batch = self.feature_list[2][id_batch]
        p_net_embed_batch = self.feature_list[3][id_batch]

        concate_embed = torch.cat((p_t_embed_batch, p_v_net_embed_batch,\
		 p_a_net_embed_batch, p_net_embed_batch), 1).view(len(id_batch[0]), 4, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.p_content_rnn(concate_embed)

        return torch.mean(all_state, 0)
    
    def v_content_agg(self, id_batch):
        embed_d = self.embed_d
        v_net_embed_batch = self.feature_list[7][id_batch]
        v_text_embed_batch_1 = self.feature_list[8][id_batch, :embed_d][0]
        v_text_embed_batch_2 = self.feature_list[8][id_batch, embed_d: 2 * embed_d][0]
        v_text_embed_batch_3 = self.feature_list[8][id_batch, 2 * embed_d: 3 * embed_d][0]
        v_text_embed_batch_4 = self.feature_list[8][id_batch, 3 * embed_d: 4 * embed_d][0]
        v_text_embed_batch_5 = self.feature_list[8][id_batch, 4 * embed_d:][0]

        concate_embed = torch.cat((v_net_embed_batch, v_text_embed_batch_1, v_text_embed_batch_2, v_text_embed_batch_3,\
			v_text_embed_batch_4, v_text_embed_batch_5), 1).view(len(id_batch[0]), 6, embed_d)

        concate_embed = torch.transpose(concate_embed, 0, 1)
        all_state, last_state = self.v_content_rnn(concate_embed)
		
        return torch.mean(all_state, 0)
    
    def node_random_neigh_agg(self, id_batch, node_type):  
        embed_d = self.embed_d

        if node_type == 1:
            batch_s = int(len(id_batch[0]) / 10)
        elif node_type == 2:
            batch_s = int(len(id_batch[0]) / 20) 
        else:
            batch_s = int(len(id_batch[0]) / 3)

        if node_type == 1:
            neigh_agg = self.a_content_agg(id_batch).view(batch_s, 10, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state  = self.a_neigh_rnn(neigh_agg)
        elif node_type == 2:
            neigh_agg = self.p_content_agg(id_batch).view(batch_s, 20, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state  = self.p_neigh_rnn(neigh_agg)
        else:
            neigh_agg = self.v_content_agg(id_batch).view(batch_s, 3, embed_d)
            neigh_agg = torch.transpose(neigh_agg, 0, 1)
            all_state, last_state  = self.v_neigh_rnn(neigh_agg)
        neigh_agg = torch.mean(all_state, 0).view(batch_s, embed_d)
		
        return neigh_agg
    
    def node_metapath_neigh_agg(self, id_batch, node_type):
        if node_type == 1:  
            a_metapath_neigh_agg = [[[] for j in range(2)] for i in range(len(id_batch))]
            for i in range(len(id_batch)):  
                metapath_type1_number = len(self.a_metapath_neigh_list[id_batch[i]][0])
                if metapath_type1_number > 0:
                    metapath_type1_inner = [[[] for j in range(3)] for _ in range(metapath_type1_number)]            
                    for j in range(metapath_type1_number):  
                        for k in range(3):  
                            if k == 0 or k == 2:    
                                a_node_id = [self.a_metapath_neigh_list[id_batch[i]][0][j][k]]
                                a_node_id = np.reshape(a_node_id, (1, -1))
                                inner_node_agg = self.a_content_agg(a_node_id)
                            elif k == 1:    
                                p_node_id = [self.a_metapath_neigh_list[id_batch[i]][0][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            metapath_type1_inner[j][k] = inner_node_agg
                    
                    metapath_type1_inner = torch.stack([torch.stack(row) for row in metapath_type1_inner])  
                    metapath_type1_inner = torch.squeeze(metapath_type1_inner, dim = 2)

                    metapath_type1_inner_agg = torch.transpose(metapath_type1_inner, 0, 1)
                    all_state_type1, last_state = self.a_metapath_inner_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_inner_agg = torch.mean(all_state_type1, 0).view(metapath_type1_number, self.embed_d) 
                    metapath_type1_inner_agg = torch.unsqueeze(metapath_type1_inner_agg, 1) 

                    all_state_type1_neigh_agg, last_state = self.a_metapath_neigh_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_neigh_agg = torch.mean(all_state_type1_neigh_agg, 0)
                    a_metapath_neigh_agg[i][0] = metapath_type1_neigh_agg
                elif metapath_type1_number == 0:  
                    a_metapath_neigh_agg[i][0] = torch.tensor(self.feature_list[5][id_batch[i]]).unsqueeze(0)

                metapath_type2_number = len(self.a_metapath_neigh_list[id_batch[i]][1])
                if metapath_type2_number > 0:
                    metapath_type2_inner = [[[] for j in range(5)] for _ in range(metapath_type2_number)]
                    for j in range(metapath_type2_number):  
                        for k in range(5):
                            if k == 0 or k == 4:
                                a_node_id = [self.a_metapath_neigh_list[id_batch[i]][1][j][k]]
                                a_node_id = np.reshape(a_node_id, (1, -1))
                                inner_node_agg = self.a_content_agg(a_node_id)
                            elif k == 1 or k == 3:
                                p_node_id = [self.a_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            elif k == 2:
                                v_node_id = [self.a_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n-set_param.P_n]
                                v_node_id = np.reshape(v_node_id, (1, -1))
                                inner_node_agg = self.v_content_agg(v_node_id)
                            metapath_type2_inner[j][k] = inner_node_agg
                
                    metapath_type2_inner = torch.stack([torch.stack(row) for row in metapath_type2_inner])
                    metapath_type2_inner = torch.squeeze(metapath_type2_inner, dim = 2)
                    metapath_type2_inner_agg = torch.transpose(metapath_type2_inner, 0, 1)
                    all_state_type2, last_state = self.a_metapath_inner_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_inner_agg = torch.mean(all_state_type2, 0).view(metapath_type2_number, self.embed_d)
                    metapath_type2_inner_agg = torch.unsqueeze(metapath_type2_inner_agg, 1)
                    all_state_type2_neigh_agg, last_state = self.a_metapath_neigh_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_neigh_agg = torch.mean(all_state_type2_neigh_agg, 0)
                    a_metapath_neigh_agg[i][1] = metapath_type2_neigh_agg
                elif metapath_type2_number == 0:
                    a_metapath_neigh_agg[i][1] = torch.tensor(self.feature_list[5][id_batch[i]]).unsqueeze(0)
            
            return a_metapath_neigh_agg

        elif node_type == 2:  
            p_metapath_neigh_agg = [[[] for j in range(2)] for i in range(len(id_batch))]
            for i in range(len(id_batch)):  
                metapath_type1_number = len(self.p_metapath_neigh_list[id_batch[i]][0])
                if metapath_type1_number > 0:
                    metapath_type1_inner = [[[] for j in range(3)] for _ in range(metapath_type1_number)]   
                    for j in range(metapath_type1_number):  
                        for k in range(3):  
                            if k == 0 or k == 2:    
                                p_node_id = [self.p_metapath_neigh_list[id_batch[i]][0][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            elif k == 1:    
                                a_node_id = [self.p_metapath_neigh_list[id_batch[i]][0][j][k]]
                                a_node_id = np.reshape(a_node_id, (1, -1))
                                inner_node_agg = self.a_content_agg(a_node_id)
                            metapath_type1_inner[j][k] = inner_node_agg
                    
                    metapath_type1_inner = torch.stack([torch.stack(row) for row in metapath_type1_inner])
                    metapath_type1_inner = torch.squeeze(metapath_type1_inner, dim = 2)
                    #metapath_type1_inner = torch.tensor(metapath_type1_inner)
                    metapath_type1_inner_agg = torch.transpose(metapath_type1_inner, 0, 1)
                    all_state_type1, last_state = self.a_metapath_inner_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_inner_agg = torch.mean(all_state_type1, 0).view(metapath_type1_number, self.embed_d)              
                    metapath_type1_inner_agg = torch.unsqueeze(metapath_type1_inner_agg, 1)
                    all_state_type1_neigh_agg, last_state = self.a_metapath_neigh_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_neigh_agg = torch.mean(all_state_type1_neigh_agg, 0)
                    p_metapath_neigh_agg[i][0] = metapath_type1_neigh_agg
    
                elif metapath_type1_number == 0:  
                    p_metapath_neigh_agg[i][0] = torch.tensor(self.feature_list[3][id_batch[i]]).unsqueeze(0)

                metapath_type2_number = len(self.p_metapath_neigh_list[id_batch[i]][1])
                if metapath_type2_number > 0:
                    metapath_type2_inner = [[[] for j in range(5)] for _ in range(metapath_type2_number)]
                    for j in range(metapath_type2_number):  
                        for k in range(5):
                            if k == 0 or k == 4:
                                p_node_id = [self.p_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            elif k == 1 or k == 3:
                                a_node_id = [self.p_metapath_neigh_list[id_batch[i]][1][j][k]]
                                a_node_id = np.reshape(a_node_id, (1, -1))
                                inner_node_agg = self.a_content_agg(a_node_id)
                            elif k == 2:
                                v_node_id = [self.p_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n-set_param.P_n]
                                v_node_id = np.reshape(v_node_id, (1, -1))
                                inner_node_agg = self.v_content_agg(v_node_id)
                            metapath_type2_inner[j][k] = inner_node_agg
                
                    metapath_type2_inner = torch.stack([torch.stack(row) for row in metapath_type2_inner])
                    metapath_type2_inner = torch.squeeze(metapath_type2_inner, dim = 2)
                    #metapath_type2_inner = torch.tensor(metapath_type2_inner)
                    metapath_type2_inner_agg = torch.transpose(metapath_type2_inner, 0, 1)
                    all_state_type2, last_state = self.a_metapath_inner_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_inner_agg = torch.mean(all_state_type2, 0).view(metapath_type2_number, self.embed_d)
                    metapath_type2_inner_agg = torch.unsqueeze(metapath_type2_inner_agg, 1)
                    all_state_type2_neigh_agg, last_state = self.a_metapath_neigh_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_neigh_agg = torch.mean(all_state_type2_neigh_agg, 0)
                    p_metapath_neigh_agg[i][1] = metapath_type2_neigh_agg
                elif metapath_type2_number == 0:
                    p_metapath_neigh_agg[i][1] = torch.tensor(self.feature_list[3][id_batch[i]]).unsqueeze(0)       

            return p_metapath_neigh_agg

        elif node_type == 3:  
            v_metapath_neigh_agg = [[[] for j in range(2)] for i in range(len(id_batch))]
            for i in range(len(id_batch)):  
                metapath_type1_number = len(self.v_metapath_neigh_list[id_batch[i]][0])
                if metapath_type1_number > 0:
                    metapath_type1_inner = [[[] for j in range(3)] for _ in range(metapath_type1_number)]              
                    for j in range(metapath_type1_number):  
                        for k in range(3):  
                            if k == 0 or k == 2:    
                                v_node_id = [self.v_metapath_neigh_list[id_batch[i]][0][j][k]-set_param.A_n-set_param.P_n]
                                v_node_id = np.reshape(v_node_id, (1, -1))
                                inner_node_agg = self.v_content_agg(v_node_id)
                            elif k == 1:    
                                p_node_id = [self.v_metapath_neigh_list[id_batch[i]][0][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            metapath_type1_inner[j][k] = inner_node_agg
                    
                    metapath_type1_inner = torch.stack([torch.stack(row) for row in metapath_type1_inner])
                    metapath_type1_inner = torch.squeeze(metapath_type1_inner, dim = 2)
                    #metapath_type1_inner = torch.tensor(metapath_type1_inner)
                    metapath_type1_inner_agg = torch.transpose(metapath_type1_inner, 0, 1)
                    all_state_type1, last_state = self.v_metapath_inner_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_inner_agg = torch.mean(all_state_type1, 0).view(metapath_type1_number, self.embed_d) 
                    metapath_type1_inner_agg = torch.unsqueeze(metapath_type1_inner_agg, 1)
                    all_state_type1_neigh_agg, last_state = self.v_metapath_neigh_type1_rnn(metapath_type1_inner_agg)
                    metapath_type1_neigh_agg = torch.mean(all_state_type1_neigh_agg, 0)
                    v_metapath_neigh_agg[i][0] = metapath_type1_neigh_agg
                elif metapath_type1_number == 0:
                    v_metapath_neigh_agg[i][0] = torch.tensor(self.feature_list[7][id_batch[i]]).unsqueeze(0)

                metapath_type2_number = len(self.v_metapath_neigh_list[id_batch[i]][1])
                if metapath_type2_number > 0:
                    metapath_type2_inner = [[[] for j in range(5)] for _ in range(metapath_type2_number)]
                    for j in range(metapath_type2_number):  
                        for k in range(5):
                            if k == 0 or k == 4:
                                v_node_id = [self.v_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n-set_param.P_n]
                                v_node_id = np.reshape(v_node_id, (1, -1))
                                inner_node_agg = self.v_content_agg(v_node_id)
                            elif k == 1 or k == 3:
                                p_node_id = [self.v_metapath_neigh_list[id_batch[i]][1][j][k]-set_param.A_n]
                                p_node_id = np.reshape(p_node_id, (1, -1))
                                inner_node_agg = self.p_content_agg(p_node_id)
                            elif k == 2:
                                a_node_id = [self.v_metapath_neigh_list[id_batch[i]][1][j][k]]
                                a_node_id = np.reshape(a_node_id, (1, -1))
                                inner_node_agg = self.a_content_agg(a_node_id)
                            metapath_type2_inner[j][k] = inner_node_agg
                
                    metapath_type2_inner = torch.stack([torch.stack(row) for row in metapath_type2_inner])
                    metapath_type2_inner = torch.squeeze(metapath_type2_inner, dim = 2)
                    #metapath_type2_inner = torch.tensor(metapath_type2_inner)
                    metapath_type2_inner_agg = torch.transpose(metapath_type2_inner, 0, 1)
                    all_state_type2, last_state = self.v_metapath_inner_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_inner_agg = torch.mean(all_state_type2, 0).view(metapath_type2_number, self.embed_d)
                    metapath_type2_inner_agg = torch.unsqueeze(metapath_type2_inner_agg, 1)
                    all_state_type2_neigh_agg, last_state = self.v_metapath_neigh_type2_rnn(metapath_type2_inner_agg)
                    metapath_type2_neigh_agg = torch.mean(all_state_type2_neigh_agg, 0)
                    v_metapath_neigh_agg[i][1] = metapath_type2_neigh_agg
                elif metapath_type2_number == 0:
                    v_metapath_neigh_agg[i][1] = torch.tensor(self.feature_list[7][id_batch[i]]).unsqueeze(0)

            return v_metapath_neigh_agg
        
    def node_het_agg(self, id_batch, node_type):
        a_neigh_batch = [[0] * 10] * len(id_batch)
        p_neigh_batch = [[0] * 20] * len(id_batch)
        v_neigh_batch = [[0] * 3] * len(id_batch)
        for i in range(len(id_batch)):
            if node_type == 1:	#author
                a_neigh_batch[i] = self.a_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.a_neigh_list_train[1][id_batch[i]]
                v_neigh_batch[i] = self.a_neigh_list_train[2][id_batch[i]]
            elif node_type == 2:
                a_neigh_batch[i] = self.p_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.p_neigh_list_train[1][id_batch[i]]
                v_neigh_batch[i] = self.p_neigh_list_train[2][id_batch[i]]
            else:
                a_neigh_batch[i] = self.v_neigh_list_train[0][id_batch[i]]
                p_neigh_batch[i] = self.v_neigh_list_train[1][id_batch[i]]
                v_neigh_batch[i] = self.v_neigh_list_train[2][id_batch[i]]
        
        a_neigh_batch = np.reshape(a_neigh_batch, (1, -1))  
        a_agg_batch = self.node_random_neigh_agg(a_neigh_batch, 1)	
        p_neigh_batch = np.reshape(p_neigh_batch, (1, -1))
        p_agg_batch = self.node_random_neigh_agg(p_neigh_batch, 2)	
        v_neigh_batch = np.reshape(v_neigh_batch, (1, -1))
        v_agg_batch = self.node_random_neigh_agg(v_neigh_batch, 3)	

        metapath_neigh_batch = self.node_metapath_neigh_agg(id_batch, node_type)
        #metapath_neigh_batch = torch.tensor([item.cpu().detach().numpy() for item in metapath_neigh_batch]).cuda()
        metapath_neigh_batch = torch.stack([torch.stack(row) for row in metapath_neigh_batch])
        metapath_neigh_batch_type1 = metapath_neigh_batch[:,0:1]    
        metapath_neigh_batch_type2 = metapath_neigh_batch[:,1:2]
        metapath_neigh_batch_type1 = torch.transpose(metapath_neigh_batch_type1, 0, 1)  
        metapath_neigh_batch_type2 = torch.transpose(metapath_neigh_batch_type2, 0, 1)
        metapath_neigh_batch_type1 = torch.squeeze(metapath_neigh_batch_type1)  
        metapath_neigh_batch_type2 = torch.squeeze(metapath_neigh_batch_type2)

        #attention module
        id_batch = np.reshape(id_batch, (1, -1))
        if node_type == 1:
            center_agg_batch = self.a_content_agg(id_batch)    
        elif node_type == 2:
            center_agg_batch = self.p_content_agg(id_batch)
        elif node_type == 3:
            center_agg_batch = self.v_content_agg(id_batch)
        
        center_agg_batch_2 = torch.cat((center_agg_batch, center_agg_batch), 1).view(len(center_agg_batch), self.embed_d*2)
        a_agg_batch_2 = torch.cat((center_agg_batch, a_agg_batch), 1).view(len(center_agg_batch), self.embed_d*2)
        p_agg_batch_2 = torch.cat((center_agg_batch, p_agg_batch), 1).view(len(center_agg_batch), self.embed_d*2)
        v_agg_batch_2 = torch.cat((center_agg_batch, v_agg_batch), 1).view(len(center_agg_batch), self.embed_d*2)
        #print(center_agg_batch.size(), metapath_neigh_batch_type1.size(), metapath_neigh_batch.size())
        metapath_type1_agg_batch_2 = torch.cat((center_agg_batch, metapath_neigh_batch_type1), 1).view(len(center_agg_batch), self.embed_d*2)
        metapath_type2_agg_batch_2 = torch.cat((center_agg_batch, metapath_neigh_batch_type2), 1).view(len(center_agg_batch), self.embed_d*2)

        concate_embed = torch.cat((center_agg_batch_2, a_agg_batch_2, p_agg_batch_2,\
                                   v_agg_batch_2, metapath_type1_agg_batch_2,\
                                    metapath_type2_agg_batch_2), 1).view(len(center_agg_batch), 6, self.embed_d*2)
        if node_type == 1:
            atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(center_agg_batch),\
                                                                                        *self.a_neigh_att.size())))
        elif node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(center_agg_batch),\
                                                                                        *self.p_neigh_att.size())))
        elif node_type == 3:
            atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(center_agg_batch),\
                                                                                        *self.v_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(center_agg_batch), 1, 6)

        concate_embed = torch.cat((center_agg_batch, a_agg_batch, p_agg_batch,\
                                   v_agg_batch, metapath_neigh_batch_type1,\
                                    metapath_neigh_batch_type2), 1).view(len(center_agg_batch), 6, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(center_agg_batch), self.embed_d)  

        return weight_agg_batch
    
    def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        embed_d = self.embed_d
        if triple_index == 0:	
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 1:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 2)
            n_agg = self.node_het_agg(neg_id_batch, 2)
        elif triple_index == 2:
            c_agg = self.node_het_agg(c_id_batch, 1)
            p_agg = self.node_het_agg(pos_id_batch, 3)
            n_agg = self.node_het_agg(neg_id_batch, 3)
        elif triple_index == 3:
            c_agg = self.node_het_agg(c_id_batch, 2)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)
        elif triple_index == 4:
            c_agg = self.node_het_agg(c_id_batch, 2)
            p_agg = self.node_het_agg(pos_id_batch, 2)
            n_agg = self.node_het_agg(neg_id_batch, 2)	
        elif triple_index == 5:
            c_agg = self.node_het_agg(c_id_batch, 2)
            p_agg = self.node_het_agg(pos_id_batch, 3)
            n_agg = self.node_het_agg(neg_id_batch, 3)	
        elif triple_index == 6:
            c_agg = self.node_het_agg(c_id_batch, 3)
            p_agg = self.node_het_agg(pos_id_batch, 1)
            n_agg = self.node_het_agg(neg_id_batch, 1)		
        elif triple_index == 7:
            c_agg = self.node_het_agg(c_id_batch, 3)
            p_agg = self.node_het_agg(pos_id_batch, 2)
            n_agg = self.node_het_agg(neg_id_batch, 2)	
        elif triple_index == 8:
            c_agg = self.node_het_agg(c_id_batch, 3)
            p_agg = self.node_het_agg(pos_id_batch, 3)
            n_agg = self.node_het_agg(neg_id_batch, 3)
        elif triple_index == 9: 
            embed_file = open(set_param.data_path + "node_embedding.txt", "w")
            save_batch_s = set_param.mini_batch_s
            for i in range(3):
                if i == 0:
                    batch_number = int(len(self.a_train_id_list) / save_batch_s)
                elif i == 1:
                    batch_number = int(len(self.p_train_id_list) / save_batch_s)
                else:
                    batch_number = int(len(self.v_train_id_list) / save_batch_s)
                for j in range(batch_number):
                    if i == 0:
                        id_batch = self.a_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 1) 
                    elif i == 1:
                        id_batch = self.p_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 2)
                    else:
                        id_batch = self.v_train_id_list[j * save_batch_s : (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, 3)
                    out_temp = out_temp.data.cpu().numpy()
                    for k in range(len(id_batch)):
                        index = id_batch[k]
                        if i == 0:
                            embed_file.write('a' + str(index) + " ")
                        elif i == 1:
                            embed_file.write('p' + str(index) + " ")
                        else:
                            embed_file.write('v' + str(index) + " ")
                        for l in range(embed_d - 1):
                            embed_file.write(str(out_temp[k][l]) + " ")
                        embed_file.write(str(out_temp[k][-1]) + "\n")

                if i == 0:
                    id_batch = self.a_train_id_list[batch_number * save_batch_s : -1]
                    out_temp = self.node_het_agg(id_batch, 1) 
                elif i == 1:
                    id_batch = self.p_train_id_list[batch_number * save_batch_s : -1]
                    out_temp = self.node_het_agg(id_batch, 2) 
                else:
                    id_batch = self.v_train_id_list[batch_number * save_batch_s : -1]
                    out_temp = self.node_het_agg(id_batch, 3) 
                out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    index = id_batch[k]
                    if i == 0:
                        embed_file.write('a' + str(index) + " ")
                    elif i == 1:
                        embed_file.write('p' + str(index) + " ")
                    else:
                        embed_file.write('v' + str(index) + " ")
                    for l in range(embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")
            embed_file.close()
            return [], [], []

        return c_agg, p_agg, n_agg
    
    def aggregate_all(self, triple_list_batch, triple_index):
        c_id_batch = [x[0] for x in triple_list_batch]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]

        #debug
        #print("triple_index:{0}".format(triple_index))
        #print("c_id_batch:{0}".format(c_id_batch))
        #print("pos_id_batch:{0}".format(pos_id_batch))
        #print("neg_id_batch:{0}".format(neg_id_batch))

        c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

        return c_agg, pos_agg, neg_agg
    
    def forward(self, triple_list_batch, triple_index):
        c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
        return c_out, p_out, n_out
    
def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, embed_d):
	batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
	
	c_embed = c_embed_batch.view(batch_size, 1, embed_d)
	pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
	neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

	out_p = torch.bmm(c_embed, pos_embed)
	out_n = - torch.bmm(c_embed, neg_embed)

	sum_p = F.logsigmoid(out_p)
	sum_n = F.logsigmoid(out_n)
	loss_sum = - (sum_p + sum_n)


	return loss_sum.mean()


