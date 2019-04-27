import argparse
import numpy as np
import pandas as pd
from numpy import linalg

parser = argparse.ArgumentParser(description='GMM clustering')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--k')
parser.add_argument('--max_iter', default = 100)
parser.add_argument('--conv_tol', default = 1e-6)
args = parser.parse_args()

def pdf_multivar_normal(x,mu,sigma):
    sigma_inv =linalg.inv(sigma)
    x = x
    mu = mu
    t33 = np.matmul(np.inner(x - mu,sigma_inv),x - mu)
    d = x.shape[0] #Number of dimensions
    pi = np.pi
    t1 = 1/np.sqrt(np.power(2*pi,d))
    #print(t1)
    t2 = 1/np.sqrt(linalg.det(sigma))
    #print(t2)
    t3 = np.exp(-0.5*t33)
    #print(t3)
    #print(t33)
    return(t1*t2*t3)

def marginal_x(pi,xn,mu,cov_mats,k):
    marginal = 0
    for j in range(k):
        pdf = pdf_multivar_normal(xn,mu[j],cov_mats[j])
        #print(pdf)
        prob = pi[j]*pdf
        #print(prob)
        marginal += prob
        #print(marginal)
    return(marginal)

gmm = pd.read_csv(args.input)
gmm = gmm.values

max_iter = int(args.max_iter)
conv_tol = float(args.conv_tol)
k = int(args.k)

class return_myGMM:
	def __init__(self,pi,mean,z,assign):
		self.pi = pi
		self.mean = mean
		self.z = z
		self.assign = assign

def myGMM(data,k,max_iter,conv_tol):
	np.random.seed(20)
	pi = np.random.uniform(size = k) #array of probabilities of a cluster
	tot = sum(pi)
	#gmm = (gmm - np.mean(gmm, axis= 0))/(np.std(gmm, axis= 0))
	for j in range(k):
	    pi[j] = pi[j]/tot
	rand_idx = np.random.randint(data.shape[0], size=k)
	mu = data[rand_idx,:]
	soft_assignments = np.zeros((300,k), dtype = "float64")
	cov_mats = []
	for i in range(k):
	    cov_mats.append(np.identity(data.shape[1], dtype = "float64"))
	loglikelihood = []
	#log likelihood
	ll = 0
	for i in range(data.shape[0]):
	    ll_k = 0
	    for j in range(k):
	        ll_k += pi[j]*pdf_multivar_normal(data[i,:],mu[j],cov_mats[j])
	    ll += np.log(ll_k)
	loglikelihood.append(ll)

	#     mat = np.random.rand(data.shape[1],data.shape[1])
	#     mat = np.dot(mat,mat.transpose())
	#     cov_mats.append(mat)
	iteration = 0
	done = 0
	while iteration < max_iter and done == 0:
	    for i in range(data.shape[0]):
	        #print(i)
	        marginal_xi = marginal_x(pi,data[i,:],mu,cov_mats,k)
	        for j in range(k):
	            soft_assignments[i,j] = pi[j]*pdf_multivar_normal(data[i,:],mu[j],cov_mats[j])/marginal_xi
	    #M Step
	    for j in range(k):
	        #mu 
	        Nj = np.sum(soft_assignments[:,j])
	        numerator_mu = 0
	        numerator_sigma = 0
	        for m in range(data.shape[0]):
	            numerator_mu += soft_assignments[m,j]*data[m,:]
	        mu[j] = numerator_mu/Nj
	        
	        #sigma
	        for m in range(data.shape[0]):
	            xm = data[m,:]
	            numerator_sigma += soft_assignments[m,j]*np.outer(xm - mu[j],xm - mu[j])
	        cov_mats[j] = numerator_sigma/Nj
	        
	        #pi
	        pi[j] = Nj/data.shape[0] 
	    
	    #log likelihood
	    ll = 0
	    for i in range(data.shape[0]):
	        ll_k = 0
	        for j in range(k):
	            ll_k += pi[j]*pdf_multivar_normal(data[i,:],mu[j],cov_mats[j])
	        ll += np.log(ll_k)
	    loglikelihood.append(ll)
	    if abs(loglikelihood[iteration + 1] - loglikelihood[iteration]) < conv_tol:
	        done = 1
	    iteration += 1
	assign = []
	for i in range(data.shape[0]):
		assign.append(np.argmax(soft_assignments[i,:]))
	gmm_return = return_myGMM(pi,mu,soft_assignments,assign)
	return(gmm_return)

gmm_return = myGMM(gmm,k,max_iter,conv_tol)
#print(z)

f = open(args.output,'w')
for i in range(gmm.shape[0]):
    line = 'data point ' + str(i+1) + '\t' + str(gmm_return.assign[i]) + '\n'
    f.write(line)
f.close()