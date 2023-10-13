from Parameter_settings import set_param
import numpy as np
import re
import networkx as nx

class metapath_seek(object):
    def build_adjacency_matrix(self):
        node_number = set_param.A_n + set_param.P_n + set_param.V_n
        type_mask = np.zeros((node_number), dtype=np.int16)
        type_mask[set_param.A_n:set_param.A_n+set_param.P_n] = 1
        type_mask[set_param.A_n+set_param.P_n:] = 2
        adjM = np.zeros((node_number, node_number), dtype=np.int16)

        connect_relation = open(set_param.data_path + 'a_p_list.txt', 'r')
        for line in connect_relation:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if neigh_list[-1][-1] == "\n":
                neigh_list[-1] = neigh_list[-1][:-1]
            for neigh_list_id in neigh_list:
                adjM[int(node_id)][int(neigh_list_id)+set_param.A_n] = 1
                adjM[int(neigh_list_id)+set_param.A_n][int(node_id)] = 1
        connect_relation.close()
        connect_relation = open(set_param.data_path + 'a_a_cooperate.txt', 'r')
        for line in connect_relation:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if neigh_list[-1][-1] == "\n":
                neigh_list[-1] = neigh[-1][:-1]
            for neigh_list_id in neigh_list:
                adjM[int(node_id)][int(neigh_list_id)] = 1
                adjM[int(neigh_list_id)][int(node_id)] = 1
        connect_relation.close()
        connect_relation = open(set_param.data_path + 'a_v_list.txt', 'r')
        for line in connect_relation:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if neigh_list[-1][-1] == "\n":
                neigh_list[-1] = neigh_list[-1][:-1]
            for neigh_list_id in neigh_list:
                if neigh_list_id:
                    adjM[int(node_id)][int(neigh_list_id)+set_param.A_n+set_param.P_n] = 1
                    adjM[int(neigh_list_id)+set_param.A_n+set_param.P_n][int(node_id)] = 1
        connect_relation.close()
        connect_relation = open(set_param.data_path + 'v_p_list.txt', 'r')
        for line in connect_relation:
            line = line.strip()
            node_id = re.split(':', line)[0]
            neigh = re.split(':', line)[1]
            neigh_list = re.split(',', neigh)
            if neigh_list[-1][-1] == "\n":
                neigh_list[-1] = neigh_list[-1][:-1]
            for neigh_list_id in neigh_list:
                adjM[int(node_id)+set_param.A_n+set_param.P_n][int(neigh_list_id)+set_param.A_n] = 1
                adjM[int(neigh_list_id)+set_param.A_n][int(node_id)+set_param.A_n+set_param.P_n] = 1
        connect_relation.close()
        connect_relation = open(set_param.data_path + 'p_a_list.txt', 'r')
        for line in connect_relation:
            line = line.strip()
            node_id = re.split(":", line)[0]
            neigh = re.split(":", line)[1]
            neigh_list = re.split(",", neigh)
            if neigh_list[-1][-1] == "\n":
                neigh_list[-1] = neigh[-1][:-1]
            for neigh_list_id in neigh_list:
                adjM[int(node_id)+set_param.A_n][int(neigh_list_id)] = 1
                adjM[int(neigh_list_id)][int(node_id)+set_param.A_n] = 1
        connect_relation.close()

        return adjM, type_mask
        #self.adjM = adjM
        #self.type_mask = type_mask

    def get_metapath_neighbor_pairs(self, M, type_mask, expected_metapaths, center_node_type):
        """
        param M: the raw adjacency matrix
        param type_mask: an array of types of all node
        param expected_metapaths: a list of expected metapaths
        return: a list of python lists, consisting of metapath-based neighbor pairs and intermediate paths
        for example:for node type author,has two metapath:APA,APVPA
        the author a123's metapath-based lists is :a_metapath_neigh_list[123]=[[a123,p12,a456],[a123,p43,v47,p43,a568]]
        """
        a_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.A_n)]
        p_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.P_n)]
        v_metapath_neigh_list = [[[] for j in range(2)] for i in range(set_param.V_n)]

        cnt = 0
        for metapath in expected_metapaths:
            # consider only the edges relevant to the expected metapath
            mask = np.zeros(M.shape, dtype=bool)
            for i in range((len(metapath) - 1) // 2):
                temp = np.zeros(M.shape, dtype=bool)
                temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
                temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
                mask = np.logical_or(mask, temp)
            partial_g_nx = nx.from_numpy_matrix((M * mask).astype(np.int16))

            # only need to consider the former half of the metapath
            # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
            metapath_to_target = {}
            for source in (type_mask == metapath[0]).nonzero()[0]:
                for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                    # check if there is a possible valid path from source to target node
                    has_path = False
                    single_source_paths = nx.single_source_shortest_path(
                        partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
                    if target in single_source_paths:
                        has_path = True

                    if has_path:
                        shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                    len(p) == (len(metapath) + 1) // 2]
                        if len(shortests) > 0:
                            metapath_to_target[target] = metapath_to_target.get(target, []) + shortests

            metapath_neighbor_paris = {}
            for key, value in metapath_to_target.items():
                for p1 in value:
                    for p2 in value:
                        #metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        #    p1 + p2[-2::-1]]
                        if center_node_type == 0:
                            if len(a_metapath_neigh_list[p1[0]][cnt]) < 10 and p1[0] != p2[0]:
                                a_metapath_neigh_list[p1[0]][cnt].append(p1 + p2[-2::-1])
                            #a_metapath_neigh_list[p2[0]][cnt].append(metapath_neighbor_paris[((p1[0], p2[0]))])
                        elif center_node_type == 1:
                            if len(p_metapath_neigh_list[p1[0]-set_param.A_n][cnt]) < 10 and p1[0] != p2[0]:
                                p_metapath_neigh_list[p1[0]-set_param.A_n][cnt].append(p1 + p2[-2::-1])
                            #p_metapath_neigh_list[p2[0]][cnt].append(metapath_neighbor_paris[((p1[0], p2[0]))])
                        elif center_node_type == 2:
                            if len(v_metapath_neigh_list[p1[0]-set_param.A_n-set_param.P_n][cnt]) < 10 and p1[0] != p2[0]:
                                v_metapath_neigh_list[p1[0]-set_param.A_n-set_param.P_n][cnt].append(p1 + p2[-2::-1])
                            #v_metapath_neigh_list[p2[0]][cnt].append(metapath_neighbor_paris[((p1[0], p2[0]))])

            for key, value in metapath_to_target.items():
                for p1 in value:
                    for p2 in value:
                        if center_node_type == 0:
                            if len(a_metapath_neigh_list[p1[0]][cnt]) < 10 and p1[0] == p2[0]:
                                a_metapath_neigh_list[p1[0]][cnt].append(p1 + p2[-2::-1])
                        elif center_node_type == 1:
                            if len(p_metapath_neigh_list[p1[0]-set_param.A_n][cnt]) < 10 and p1[0] == p2[0]:
                                p_metapath_neigh_list[p1[0]-set_param.A_n][cnt].append(p1 + p2[-2::-1])
                        elif center_node_type == 2:
                            if len(v_metapath_neigh_list[p1[0]-set_param.A_n-set_param.P_n][cnt]) < 10 and p1[0] == p2[0]:
                                v_metapath_neigh_list[p1[0]-set_param.A_n-set_param.P_n][cnt].append(p1 + p2[-2::-1])

            cnt += 1
        if center_node_type == 0:
            self.a_metapath_neigh_list = a_metapath_neigh_list
            f_a = open(set_param.data_path + "a_metapath_neigh_list.txt", 'w')
            for i in range(set_param.A_n):
                for j in range(2):
                    f_a.write("a" + str(i) + ":" + str(j) + ":" + str(a_metapath_neigh_list[i][j]) + "\n")
            f_a.close()

        if center_node_type == 1:
            self.p_metapath_neigh_list = p_metapath_neigh_list
            f_p = open(set_param.data_path + "p_metapath_neigh_list.txt", 'w')
            for i in range(set_param.P_n):
                for j in range(2):
                    f_p.write("p" + str(i) + ":" + str(j) + ":" + str(p_metapath_neigh_list[i][j]) + "\n")
            f_p.close()

        if center_node_type == 2:
            self.v_metapath_neigh_list = v_metapath_neigh_list
            f_v = open(set_param.data_path + "v_metapath_neigh_list.txt", 'w')
            for i in range(set_param.V_n):
                for j in range(2):
                    f_v.write("v" + str(i) + ":" + str(j) + ":" + str(v_metapath_neigh_list[i][j]) + "\n")
            f_v.close()
            #outs.append(metapath_neighbor_paris)
        #return outs