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

parser = argparse.ArgumentParser(description='kmer extraction')
parser.add_argument('--input', help="sequence to find kmer")
parser.add_argument('--output', help="output kmer frequency")
parser.add_argument('--order', help="order of output")
args = parser.parse_args()

order = pd.read_csv(args.order, header = None, names = ['kmers'])
f = open(args.input,'r')
seqs = f.read().split('\n')
seqs = seqs[:len(seqs) - 1]
nseqs = len(seqs)

df = pd.DataFrame(data = np.zeros((nseqs,336)),columns=order.loc[:,'kmers'])

def generate_kmers(s,l):
	mers = [] #generate kmers
	for i in range(len(s) - l - 1):
		mers.append(s[i:i+l])
	return mers

for i in range(nseqs):
	m2 = generate_kmers(seqs[i],2)
	m3 = generate_kmers(seqs[i],3)
	m4 = generate_kmers(seqs[i],4)
	m = m2 + m3 + m4
	mer_counts = pd.value_counts(m)
	indices = mer_counts.index
	for j in range(336):
		if df.columns[j] in indices:
			df.iloc[i][j] =  mer_counts[df.columns[j]]

# print(df.head())

cols = df.columns
means = []
sdevs = []
for col in cols:
	means.append(df.loc[:,col].mean())
	sdevs.append(np.std(df.loc[:,col],ddof = 1))

for j in range(336):
	for i in range(nseqs):
		df.iloc[i][j] = (df.iloc[i][j] - means[j])/sdevs[j]

# print(df.head())

df.to_csv(args.output,sep='\t',header = False, index = False)