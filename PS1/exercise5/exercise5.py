import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description='Logistic Regression Model on data')
parser.add_argument('--X', help="get the training exp data")
parser.add_argument('--Y', help="get training phenotype data")
parser.add_argument('--testX', help="get the test exp data")
parser.add_argument('--testY', help="get test phenotype data")

def precision_recall(true_class,predicted_class):
	tp = 0
	fp = 0
	fn = 0
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 1):
			if(true_class[i] == 1):
				tp += 1
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 1):
			if(true_class[i] == 0):
				fp += 1
	for i in range(len(predicted_class)):
		if(predicted_class[i] == 0):
			if(true_class[i] == 1):
				fn += 1
	precision = tp/(tp + fp)
	recall = tp/(tp + fn)
	return(precision,recall)

def predict_class(train_exp,test_exp,train_phen):
	model = LogisticRegression(solver = 'liblinear')
	train_phen = train_phen['Basal_or_Luminal']
	model.fit(train_exp.transpose().values,train_phen.values)
	predicted_class = model.predict(test_exp.transpose().values)
	return(predicted_class)

def main():
	args = parser.parse_args()
	train_exp = pd.read_csv(args.X)
	train_phen = pd.read_csv(args.Y)
	test_exp = pd.read_csv(args.testX)
	test_phen = pd.read_csv(args.testY)
	train_exp.set_index('Gene',inplace = True)
	test_exp.set_index('Gene',inplace = True)
	true_class = test_phen['Basal_or_Luminal'].values
	precision1, recall1 = precision_recall(true_class,predict_class(train_exp,test_exp,train_phen))
	print("Part (a): Precision = " + str(precision1) + " Recall = " + str(recall1))
	train_data = train_exp[:10]
	test_data = test_exp[:10]
	precision2, recall2 = precision_recall(true_class,predict_class(train_data,test_data,train_phen))
	print("Part (b): Precision = " + str(precision2) + " Recall = " + str(recall2))

if __name__ == "__main__": main()