import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv
from Parameter_settings import set_param

def model(cluster_id_num):
	cluter_embed = numpy.around(numpy.random.normal(0, 0.01, [cluster_id_num, set_param.embed_d]), 4)
	cluster_embed_f = open(set_param.data_path + "cluster_embed.txt", "r")
	for line in cluster_embed_f:
		line=line.strip()
		author_index=int(re.split(' ',line)[0])
		embed_list=re.split(' ',line)[1:]
		for i in range(len(embed_list)):
			cluter_embed[author_index][i] = embed_list[i]

	kmeans = KMeans(n_clusters = 4, random_state = 0, max_iter = 100).fit(cluter_embed) 

	

	pca = PCA(n_components = 2)
	cluster_embed_pca = pca.fit_transform(cluter_embed)
	plt.scatter(cluster_embed_pca[:, 0], cluster_embed_pca[:, 1], c=kmeans.labels_, cmap='rainbow')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x')
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.title('KMeans Clustering with PCA')
	plt.show()

	cluster_id_list = [0] * cluster_id_num
	cluster_id_f = open(set_param.data_path + "cluster.txt", "r")
	for line in cluster_id_f:
		line = line.strip()
		author_index = int(re.split(',',line)[0])
		cluster_id = int(re.split(',',line)[1])
		cluster_id_list[author_index] = cluster_id

	kmeans_labels = open(set_param.data_path + 'kmeans_labels.txt', 'w')
	node_id = 0
	for i in range(len(kmeans.labels_)):
		kmeans_labels.write(f'{node_id},{kmeans.labels_[node_id]},{cluster_id_list[node_id]}\n')
		node_id += 1
	kmeans_labels.close()

	#NMI
	print ("NMI: " + str(normalized_mutual_info_score(cluster_id_list, kmeans.labels_)))
	print ("ARI: " + str(adjusted_rand_score(cluster_id_list, kmeans.labels_)))
