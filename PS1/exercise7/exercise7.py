import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge


parser = argparse.ArgumentParser(description='Logistic Regression Model on data')
parser.add_argument('--X', help="get the training exp data")
parser.add_argument('--Y', help="get training phenotype data")
parser.add_argument('--testX', help="get the test exp data")
parser.add_argument('--testY', help="get test phenotype data")
parser.add_argument('--output', help="output text file")

args = parser.parse_args()
train_exp = pd.read_csv(args.X,index_col = 0)
train_phen = pd.read_csv(args.Y,index_col = 0)
test_exp = pd.read_csv(args.testX,index_col = 0)
test_phen = pd.read_csv(args.testY,index_col = 0)
train_exp = np.array(train_exp[train_phen.index])
train_phen  = np.array(train_phen['V1'])
test_exp = np.array(test_exp[test_phen.index])
test_phen  = np.array(test_phen['V1'])

def findalpha(range):
	rss = []
	for alpha in range:
		model = Ridge(alpha=alpha)
		model.fit(train_exp.transpose(), train_phen)
		yhat = model.predict(test_exp.transpose())
		rss.append(np.sum([x**2 for x in test_phen - yhat]))
	range_len = len(range)
	min_idx = np.argmin(np.array(rss))
	print("In the given range, the optimal alpha is " + str(range[min_idx]))
	return(model)

model = findalpha([0.01,0.1,1,10,100])
output_file = args.output
f = open(output_file,'w')
for i in range(train_exp.shape[0]):
	f.write('beta' + str(i+1) + ' ' + str(model.coef_[i]) + '\n')
f.close()