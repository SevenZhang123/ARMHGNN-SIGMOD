import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
from itertools import *
from Parameter_settings import set_param

dimen = 128
window = 4


def read_random_walk_corpus():
	walks=[]
	inputfile = open(set_param.data_path + "random_walk_result.txt", "r")
	for line in inputfile:
		path = []
		node_list=re.split(' ',line)
		for i in range(len(node_list)):
			path.append(node_list[i])			
		walks.append(path)
	inputfile.close()
	return walks


walk_corpus = read_random_walk_corpus()
model = Word2Vec(walk_corpus, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)


print("Output...")
model.wv.save_word2vec_format(set_param.data_path + "All_node_net_embedding.txt")

