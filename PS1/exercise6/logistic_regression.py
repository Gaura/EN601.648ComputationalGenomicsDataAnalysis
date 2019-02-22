import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser(description='Exercise 6')
parser.add_argument('--X', help="get the genotype file")
parser.add_argument('--output', help="output text file")
parser.add_argument('--Y', help="get the phenotype file")
args = parser.parse_args()

genotype = pd.read_csv(args.X)
phenotype = pd.read_csv(args.Y)
output_file = args.output

snp05 = [] #store snps >= 0.05 MAF
for i in range(genotype.shape[1]):
	freq = np.sum(genotype.iloc[:,i])/len(genotype.iloc[:,i])/2
	maf = np.min([freq,1- freq])
	if maf >= 0.05:
		snp05.append(genotype.columns[i])
g_subset = genotype[snp05] #subset of relevant snps
y = phenotype.iloc[:,1].values
intercept = np.repeat(int(1),len(y))
print(snp05[9])
pvalues = []
for i in range(len(snp05)):
	z = zip(intercept,g_subset.iloc[:,i].values)
	design_matrix = np.array(list(z))
	model = Logit(y,design_matrix)
	result = model.fit()
	pvalues.append(result.pvalues[1])

pvalues = np.asarray(pvalues)
check = np.logical_not(np.isnan(pvalues)) #non nan values
pvalues = pvalues[check]
snp05 = np.asarray(snp05)[check]
qvals = multipletests(pvalues,alpha=0.05,method = 'fdr_bh')
qvals = qvals[1]
count_qval = 0
f = open(output_file,'w')
for i in range(len(qvals)):
	if qvals[i] <= 0.05:
		f.write(str(snp05[i]) + '\t' + str(qvals[i]) + '\n')
		count_qval += 1
f.close()