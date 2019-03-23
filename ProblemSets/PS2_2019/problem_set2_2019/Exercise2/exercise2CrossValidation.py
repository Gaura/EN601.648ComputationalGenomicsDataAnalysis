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

parser = argparse.ArgumentParser(description='svm buillding')
parser.add_argument('--features')
parser.add_argument('--labels')
parser.add_argument('--output')
parser.add_argument('--kernel')
args = parser.parse_args()

features = pd.read_csv(args.features, header = None, sep = '\t')
features = features.values
features = np.array_split(features,5)
labels = pd.read_csv(args.labels, header = None, sep = ' ')
labels = labels.iloc[:][2].array
labels = np.array_split(labels,5)
kernel = args.kernel
degree = 3
if kernel == 'polynomial_3':
	kernel = 'poly'
elif kernel == 'polynomial_6':
	kernel = 'poly'
	degree = 6
elif kernel == 'gaussian':
	kernel = 'rbf'
output = args.output

def modelfit(kernel, x, y, testx, degree = 3):
    model = SVC(kernel = kernel, degree = degree, gamma ='auto')
    model.fit(x,y)
    y_pred = model.predict(testx)
    return y_pred

def accuracy2(true_class,predicted_class):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	accuracy = 0.0
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 'S'):
			if(true_class[i] == 'S'):
				tp += 1
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 'S'):
			if(true_class[i] == 'N'):
				fp += 1
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 'N'):
			if(true_class[i] == 'S'):
				fn += 1
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 'N'):
			if(true_class[i] == 'N'):
				tn += 1
	accuracy = (tp + tn)/(tp + tn + fp + fn)
	# print(accuracy)
	return accuracy

accuracies = []
for i in range(5):
	test_data = features[i]
	test_y = labels[i]
	chk = 0
	for j in range(5):
		if j != i:
			if chk == 0:
				train_data = np.concatenate([features[j]])
				train_y = np.concatenate([labels[j]])
				chk = 1
			else:
				train_data = np.concatenate([train_data,features[j]])
				train_y = np.concatenate([train_y,labels[j]])
	y_pred = modelfit(kernel = kernel, x = train_data, y = train_y, testx = test_data, degree = degree)
	accuracy = accuracy2(test_y, y_pred)
	accuracies.append(accuracy)

f = open(output,"w")
mean_a = np.mean(accuracies)
print(mean_a)
f.write(str(mean_a))
f.close()


