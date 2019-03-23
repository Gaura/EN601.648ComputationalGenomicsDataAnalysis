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
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()

counts = pd.read_csv(args.input,sep = '\t', index_col = 0)
nrow = counts.shape[0]
ncol = counts.shape[1]
row_means = counts.mean(axis = 1)
row_std = []
for i in range(nrow):
    row_std.append(np.std(counts.iloc[i],ddof = 1))

for i in range(nrow):
    for j in range(ncol):
        val = counts.iloc[i][j]
        counts.iloc[i][j] = (np.log2(val + 1) - row_means[i])/row_std[i]
        counts.iloc[i][j] = round(counts.iloc[i][j],3)

counts.to_csv(args.output, sep = '\t')
