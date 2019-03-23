import sys
import os
import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='exercise4 kmeans')
parser.add_argument('--train')
parser.add_argument('--test')
parser.add_argument('--Kmin')
parser.add_argument('--Kmax')
parser.add_argument('--output')
args = parser.parse_args()

train = pd.read_csv(args.train, sep = '\t', index_col = 0)
train = train.transpose()
test = pd.read_csv(args.test, sep = '\t', index_col = 0)
test = test.transpose()

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def myKmeans(data, k, centers, maxit):
    n = data.shape[0] #no of samples
    d = data.shape[1] #no of features
    clusters = np.empty(shape = n)
    itr = 0
    while(itr < maxit):
        #assign clusters
        for i in range(n):
            distances = np.empty(shape = k)
            for j in range(k):
                distances[j] = distance(centers[j,:], data[i,:])
                clusters[i] = distances.argmin()
        #recompute centers
        for j in range(k):
            whr_j = np.where(clusters == j)[0]
            centers[j,:] = np.mean(data[whr_j,:], axis=0)
        itr += 1
    return(clusters,centers)

def predict(model, test_data):
    k = len(model[1])
    n = test_data.shape[0]
    centers = model[1]
    clusters = np.empty(shape = n)
    for i in range(n):
        distances = np.empty(shape = k)
        for j in range(k):
            distances[j] = distance(centers[j,:], test_data[i,:])
        clusters[i] = distances.argmin()
    return clusters

def avg_distance_from_centroid(model,data,k):
    distances = np.empty(shape = k)
    clusters = model[0]
    for j in range(k):
        whr_j = np.where(clusters == j)[0]
        s = 0
        temp_mat = data[whr_j,]
        for i in range(len(whr_j)):
            s += distance(model[1][j],temp_mat[i])
        distances[j] = s
    return np.mean(distances)

kmin = int(args.Kmin)
kmax = int(args.Kmax)
tries = kmax - kmin + 1
rng = range(kmin, kmax + 1)
mean_distances = np.empty(shape = tries)
try_no = 0
for i in rng:
    data = train.values
    model = myKmeans(data,i,data[0:i],10)
    mean_distances[try_no] = avg_distance_from_centroid(model,data,i) 
    try_no += 1

#Using elbow method and detecting elbow at less than 25% change
chk = 0
i_for_k = 0
for i in range(len(mean_distances) - 1):
    if chk != 1:
        if ((mean_distances[i] - mean_distances[i+1])/mean_distances[i]) < 0.25:
            i_for_k = i
            chk = 1

if chk == 0:
    i_for_k = len(mean_distances) - 1
optimal_k = rng[i_for_k] 

test = test.values
test = test.transpose()
test_v = test.values
model_optimal = myKmeans(train.values, optimal_k, train.values[0:optimal_k,],10)
y_pred = predict(model_optimal,test_v)

#print output
f = open(args.output,'w')
for i in range(len(y_pred)):
    line = str(test.index[i]) + '\t' + str(y_pred[i]) + '\n'
    f.write(line)
f.close()