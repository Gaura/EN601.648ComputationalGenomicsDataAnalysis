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
args = parser.parse_args()

genotype = pd.read_csv(args.X, sep=',', index_col = 0)
output_file = args.output

snp03 = 0
snp05 = 0
snp1 = 0
f = open(output_file,'w')
for i in range(genotype.shape[1]):
	freq = np.sum(genotype.iloc[:,i])/len(genotype.iloc[:,i])/2
	maf = np.min([freq,1- freq])
	if maf > 0.03:
		snp03 += 1
	if maf > 0.05:
		snp05 += 1
	if maf > 0.1:
		snp1 +=1
	f.write(genotype.columns[i] + "\t" + str(maf) + '\n')
print(snp03,snp05,snp1)
f.close()