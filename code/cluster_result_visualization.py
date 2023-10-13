# This file is used to generate Figure 6
import numpy
import re
from Parameter_settings import set_param
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

kmeans_label_f = open('../DBLP/kmeans_labels.txt', 'r')
label_pre = []
label_true = []
for line in kmeans_label_f:
    line.strip()
    line = line.split(',')
    label_pre.append(int(line[1]))
    label_true.append(int(line[2][:-1]))

cluter_embed = numpy.around(numpy.random.normal(0, 0.01, [4057, set_param.embed_d]), 4)
cluster_embed_f = open(set_param.data_path + "cluster_embed.txt", "r")
for line in cluster_embed_f:
	line=line.strip()
	author_index=int(re.split(' ',line)[0])
	embed_list=re.split(' ',line)[1:]
	for i in range(len(embed_list)):
		cluter_embed[author_index][i] = embed_list[i]

pca = PCA(n_components = 2)
cluster_embed_pca = pca.fit_transform(cluter_embed)
plt.scatter(cluster_embed_pca[:, 0], cluster_embed_pca[:, 1], c=label_pre, cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Node clustering visualization')
plt.show()
