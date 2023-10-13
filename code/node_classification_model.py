import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
from Parameter_settings import set_param

def load_data(data_file_name, n_features, n_samples):
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        for i, d in enumerate(data_file):
            data[i] = numpy.asarray(d[:], dtype=numpy.float)
        f.close

        return data


def model(train_num, test_num):
    train_data_f = set_param.data_path + "train_class_feature.txt"
    train_data = load_data(train_data_f, set_param.embed_d + 2, train_num)
    train_features = train_data.astype(numpy.float32)[:,2:-1]
    train_target = train_data.astype(numpy.float32)[:,1]

    #print(train_target[1])
    learner = linear_model.LogisticRegression(max_iter = 500)
    learner.fit(train_features, train_target)
    train_features = None
    train_target = None

    print("training finish!")

    test_data_f = set_param.data_path + "test_class_feature.txt"
    test_data = load_data(test_data_f, set_param.embed_d + 2, test_num)
    test_id = test_data.astype(numpy.int32)[:,0]
    test_features = test_data.astype(numpy.float32)[:,2:-1]
    test_target = test_data.astype(numpy.float32)[:,1]
    test_predict = learner.predict(test_features)
    test_features = None

    print("test prediction finish!")

    output_f = open(set_param.data_path + "NC_prediction.txt", "w")
    for i in range(len(test_predict)):
        output_f.write('%d,%lf\n'%(test_id[i],test_predict[i]))
    output_f.close()

    print ("MacroF1: ")
    print (sklearn.metrics.f1_score(test_target,test_predict,average='macro'))

    print ("MicroF1: ")
    print (sklearn.metrics.f1_score(test_target,test_predict,average='micro'))