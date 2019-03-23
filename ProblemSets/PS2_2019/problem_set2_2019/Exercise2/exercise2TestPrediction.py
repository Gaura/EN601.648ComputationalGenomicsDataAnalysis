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
parser.add_argument('--features_train')
parser.add_argument('--labels_train')
parser.add_argument('--features_test')
parser.add_argument('--labels_test')
parser.add_argument('--kernel')
parser.add_argument('--output')
args = parser.parse_args()

features = pd.read_csv(args.features_train, header = None, sep = '\t')
features = features.values
labels = pd.read_csv(args.labels_train, header = None, sep = ' ')
labels = labels.iloc[:][2].array
features_test = pd.read_csv(args.features_test, header = None, sep = '\t')
features_test = features_test.values
labels_test = pd.read_csv(args.labels_test, header = None, sep = ' ')
labels_test = labels_test.iloc[:][2].array
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


y_pred = modelfit(kernel = kernel, x = features, y = labels, testx = features_test, degree = degree)
accuracy = accuracy2(labels_test, y_pred)
print(accuracy)
f = open(output,"w")
f.write(str(accuracy))
f.close()


