#synthetic data
import numpy as np
from scipy.stats import t,multivariate_t, multivariate_normal,chi2
from riemannian_geometry import exp_riemann, mean_riemann
from classifiers import MDM
import scipy
import random
from pyriemann.utils.base import logm, sqrtm,invsqrtm,expm
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
    
def unvectorize(v):
    m = len(v)
    #n(n+1)/2 = m then n^2+n-2m=0
    n = int((np.sqrt(1+8*m)-1)/2)
    A = np.zeros((n,n))
    k= 0
    for j in range(n):
        for i in range(j+1):
            if i==j:
                A[i,j] = v[k]
            else: 
                A[i,j] = v[k]/np.sqrt(2)
                A[j,i] = A[i,j]
            k += k
    return A



eps = 1e-3
p = 3
X = np.random.randn(p,p)
Sigma = X@X.T+np.eye(p)

"""
Y = np.random.randn(p*(p+1)//2,p*(p+1)//2)
S = Y@Y.T+np.eye(p*(p+1)//2)
S = S/np.linalg.det(S)
"""
def huber(t,p,q=0.9):
    r = chi2.ppf(q,df=p)
    a = chi2.cdf(r,df=p+2)
    k = a+(r/p)*(1-q)
    if t<r:
        return 1/k
    else:
        return r/(t*k)
boolean=True 
    
samples = [10+10*i for i in range(80)]
s= 5
dff=3
res = {"x":[],"y":[], "Robustification Type":[]}
for n in tqdm(samples):
    error_n = []
    iters=20
    z =0
    j=0
    while j<iters:
        SPDs = np.zeros((n,p,p))
        for k in range(n): 
            gamma =multivariate_t.rvs(shape=(s**2)*np.eye(p*(p+1)//2),df=dff)
            #gamma =multivariate_t.rvs(shape=eps*S,df=5)
            #gamma = multivariate_normal.rvs(cov=s*np.eye(p*(p+1)//2))
            m = exp_riemann(Sigma, unvectorize(gamma))
            #print(min(np.linalg.eigvals(m)))
            SPDs[k,:,:]= m  
            #eigs.append(np.linalg.eigvals(m))
        try:
            geo_mean1 = mean_riemann(SPDs)
        except:
            boolean =False
        try:
            geo_mean3 = mean_riemann(SPDs,u_prime=lambda t: (p+dff)/(dff+t) )
        except:
            boolean=False
        try:
            geo_mean2 = mean_riemann(SPDs,u_prime=lambda t: huber(t,p) )
        except:
            iters+=1
            boolean=False
        #geo_mean4 = mean_riemann(SPDs,u_prime=lambda t : p/t)
        #geo_mean = mean_riemann(SPDs )
        #print(np.linalg.norm(geo_mean,'fro'))
        if boolean:
            
            error1 = np.linalg.norm(geo_mean1-Sigma,'fro')
            error2 = np.linalg.norm(geo_mean2-Sigma,'fro')
            error3 = np.linalg.norm(geo_mean3-Sigma,'fro')
            print("done")
            #error4 = np.linalg.norm(geo_mean4-Sigma,'fro')
            #res["y"].append(error4)
            #res["Robustification Type"].append("Tyler Robustification")
            if (error1 <1) and (error2<1) and (error3<1):
                j +=1
                print(j)
                res["y"].append(error1)
                res["Robustification Type"].append("No Robustification")
            
                res["y"].append(error2)
                res["Robustification Type"].append("Huber Robustification")
            
                res["y"].append(error3)
                res["Robustification Type"].append("Student Robustification")
                
                res["x"].extend([n,n,n])
            
        
df = pd.DataFrame(res)
fig = plt.figure()
g = sns.lineplot(x="x",y="y",hue="Robustification Type",data=df)
plt.xlabel("n")
plt.ylabel(" ")
plt.title("$\hat{E}(||\overline{\Gamma}-\Gamma||_F)$ as function of n \n p="
          +str(p)+" / $\sigma$="+str(s)+ " /Elliptical distribution = $t$(df="
          +str(dff)+")")
plt.show()
