import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
import scipy
import scipy.integrate as integrate
from tqdm import tqdm

def scm(X):
    #unbiased
    n,m = X.shape
    Y = np.dot(X,X.T)/(m-1)
    return Y
    

def mle(X):
    #baised
    n,m = X.shape
    return np.dot(X,X.T)/m

def hill(X,kn):
    p,n = X.shape
    norms = [X[:,i].T@X[:,i] for i in range(n)]
    norms.sort(reverse=True)
    s=0
    for i in range(kn):
        s+=np.log(norms[i]/norms[kn])
    return (2*kn)/s

def AutomaticAdaptDOF(X,kappa,max_iter=3):
    n,m = X.shape
    nu = 2/max(1e-8,kappa)+4
    for t in range(max_iter):
        u = lambda x : (n + nu)/(nu+x)
        estim = m_estimator(X, u)
        ratio = 0
        S = scm(X)
        ratio = np.trace(S)/np.trace(estim)
        nu_new = 2*ratio/(ratio-1)
        if np.abs(nu_new-nu)/nu < 0.01:
            break
        nu = nu_new
    return nu_new


def ddl_pop(X, maxiter= 100,threshold=1e-2):
    n,m = X.shape
    kappabis=0
    for p in range(n):
        moment_2 = 0
        moment_4 = 0
        for i in range(m):
            moment_2 += X[p,i]**2
            moment_4 += X[p,i]**4
        k_p = m*moment_4/(moment_2**2)
        k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
        kappabis += k_p 
    kappa = max(-2/(n+2),kappabis/(3*n))
    nu= 2/max(kappa, 0)+4
    iteration = 0
    cov = scm(X)
    error = np.inf
    
    while (iteration < maxiter) and (error > threshold):
        iteration +=1
        s = 0
        for i in range(m):
            u = lambda t: (n+nu)/(t+nu)
            inv_cov = np.linalg.pinv(cov)
            diag = []
            for j in range(m):
                if j!=i:
                    diag.append(u(X[:,j].T@inv_cov@X[:,j]))
            U = np.diag(diag)
            indx = list(range(i))+list(range(i+1,m))
            X_i = X[:, np.asarray(indx)]
            cov_i = (X_i@U@X_i.T)/m
            s += X[:,i].T@ np.linalg.pinv(cov_i)@X[:,i]
        s = (1/n -1/m)*(1/m)*s
        new_nu = 2*s/(s-1)
        error = np.abs(new_nu - nu)/nu
        nu = new_nu
    return nu
    
    

def rho_prime_fusion(r,c):
    if np.abs(r) <c:
        return r*(1-(r/c)**2)**2
    else:
        return 0

def fusing_eigs(X,eta,c=1,max_iter=100,threshold=1e-8):
    rho_prime = lambda r : rho_prime_fusion(r,c)
    S= scm(X)
    n,m = X.shape
    l,v = np.linalg.eigh(S)
    ll = list(l)
    sorted(ll,reverse=True)
    ll.reverse()
    l=np.asarray(ll)
    #print(l)
    loglamda = [np.log(l[j]) for j in range(len(l))]
    s =np.std(np.asarray([np.log(l[j])-np.log(l[j+1]) for j in range(n-1)]))
    t=0
    d = np.zeros(n)
    w = np.zeros(n)
    error=np.inf
    while (t<max_iter)and (error>threshold):
        old_loglamda = loglamda.copy()
        t+=1
        #print(d)
        for j in range(n-1):
            d[j] = loglamda[j]-loglamda[j+1]
            x = d[j]/s
            w[j] = rho_prime(x)*x
            
            if j==0:
                if w[j]>5e-1:
                    loglamda[j] = (s**2/(w[j]*eta))*(l[0]/np.exp(loglamda[0])-1)+loglamda[1]
                
            else:
                if w[j]+w[j-1] >5e-1:
                    loglamda[j] = 1/(w[j]+w[j-1])*((s**2/eta)*(l[j]/np.exp(loglamda[j])-1)+w[j]*loglamda[j+1]+w[j-1]*loglamda[j-1])
            #print(j," ",t," ",loglamda[j])       
        if w[n-2]>5e-1:
            loglamda[n-1] = (s**2/(w[n-2]*eta))*(l[n-1]/np.exp(loglamda[n-1])-1)+loglamda[n-2]
        ####
        for j in range(n):
            if np.exp(loglamda[j])==0:
                print(t,"_",j,"_",w[j],"/",w[j-1],"/",l[j]/np.exp(loglamda[j]),
                      "//")
        ####
        error = np.linalg.norm(np.asarray(loglamda)-np.asarray(old_loglamda))
    lamda = [np.exp(min(x,10)) for x in loglamda]    
    return np.asarray(lamda)
    
    
def m_estimator(X,u,threshold=1e-3,max_iter=50,rho=0,constraint_trace=False,initial=None):
#def m_estimator(X,u,threshold=1e-7,max_iter=100,rho=0,constraint_trace=False):

    #for mle : u(x) = 1
    #for tyler: u(x) = n/x ## add power information
    #for huber of parameter r : u(x) = k if (|x|<r) ;else kr/|x|
    n,m = X.shape
    #Z = np.random.rand(n,m)
    #cov_0 = Z@Z.T+np.random.rand()*np.eye(n)
    if initial is None:
        cov_0 = scm(X)#np.eye(n) #possible init with the scm
    else:
        cov_0 = initial
    cov = cov_0 #of shape (n,n)
    error = np.inf
    i=0
    while (error > threshold) and (i < max_iter):
        i +=1
        inv_cov = np.linalg.pinv(cov)
        W = np.diag([u(X[:,j].T@inv_cov@X[:,j]) for j in range(m)])
        new_cov = (1-rho)*(X@W@X.T)/m+rho*np.eye(n)
        if constraint_trace:
            new_cov = n*new_cov/np.matrix.trace(new_cov)
        error = np.linalg.norm(np.eye(n)-new_cov@inv_cov,ord=np.inf)

        cov = new_cov 
        #print(" error = ",error)
    #print("convergence check : iterations = ",i)
    return cov


def huber(x,k,r):
    if x < r:
        return 1/k
    else:
        return r/(k*x)
    
def huber_loss(x,k,r):
    if x < r:
        return x/k
    else:
        return (r/k)*(np.log(x/r)+1)
    
        

def tyler_adaptive(X_trials,method = 1):
    #power factor regularization
    N,n,m = X_trials.shape #m is the number of points "(per session:(32,24,768) => m =768
    u   = lambda x : n/x
    tyler_covs = []
    
    for k in range(N):
        X = X_trials[k,:,:]
        cov  =  m_estimator(X,u)
        cov = cov/np.exp(np.log(np.linalg.det(cov))/n)
        taux = []
        for i in range(m):
            Xi = X[:,i]
            taux.append(Xi.T@np.linalg.pinv(cov)@Xi)
        if method==2:
            tau_mean =0
            for s in taux:
                tau_mean +=np.log(s)
            tau_mean = tau_mean/m
            new_cov = np.exp(tau_mean)*cov
        if method==1:#fisher consistent
        
            median = np.median(np.asarray(taux))
            new_cov = (median/scipy.stats.chi2.median(df=n))*cov
        tyler_covs.append(new_cov)
        
    return tyler_covs


def huber_adaptive_param(X_trials, clean_prop = 0.9):
    N,n,m = X_trials.shape
    scms = [scm(X_trials[k,:,:]) for k in range(N)]
    params = []
    
    for k in range(N): 
        # for the k^th trial
        X = X_trials[k,:,:]
        all_arg = [X[:,i].T@np.linalg.pinv(scms[k])@X[:,i] for i in range(m)]
        indx = sorted(range(len(all_arg)), key=lambda k: all_arg[k])
        nb_clean_data = int(m*clean_prop)
        params.append(all_arg[indx[nb_clean_data]])

    return params
    
    
def huber_non_adaptive_param(X_trials,q):
    N,n,m = X_trials.shape
    param1 = scipy.stats.chi2.ppf(q,df=n)
    a = scipy.stats.chi2.cdf(param1,df=n+2)
    param2 = a+(param1/n)*(1-q)
    return param1,param2
    
                

        
def covariances(X_trials,estimator, clean_prop= 0.9, ddl = 5):
    N,n,m = X_trials.shape
    
    
    if estimator=="SCM":
        res = [scm(X_trials[k,:,:]) for k in range(N)]
        
        
            
    if estimator=="Scaled Tyler (1)":
        res= tyler_adaptive(X_trials,method=1)
        
        
    if estimator=="Scaled Tyler (2)":
        res = tyler_adaptive(X_trials,method=2)
        
    
    if estimator=="Tyler + tr = dim":
        u = lambda x: n/x
        res = [m_estimator(X_trials[k,:,:],u)for k in range(N)]
        for k in range(N):
            res[k] = n*res[k]/np.matrix.trace(res[k])
                
    
    if estimator=="Tyler + det = 1":
        u = lambda x: n/x
        res= [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        for k in range(N):
            res[k]=res[k]/(np.exp(np.log(np.linalg.det(res[k]))/n))
        
    
    if estimator=="Huber (2)":
        params1 = huber_adaptive_param(X_trials,clean_prop)
        params2 = [scipy.stats.chi2.cdf(r,df=n+2)+(r/n)*(1-scipy.stats.chi2.cdf(r,df=n)) for r in params1]
        res = [m_estimator(X_trials[k,:,:],lambda x : huber(x,params2[k],params1[k])) for k in range(N)]
        
         
    if estimator=="Huber (1)":
        param1,param2 = huber_non_adaptive_param(X_trials,clean_prop)
        res = [m_estimator(X_trials[k,:,:],lambda x : huber(x,param2,param1)) for k in range(N)]
        
        
    if estimator=="Student":
        u = lambda x : (n + ddl)/(ddl+x)
        res = [m_estimator(X_trials[k,:,:],u) for k in range(N)]
        
    if estimator=="Student POP":
        res= []
        for k in range(N):
            X= X_trials[k,:,:]
            ddl = ddl_pop(X)
            u = lambda x : (n + ddl)/(ddl+x)
            res.append(m_estimator(X,u) for k in range(N))
    
    if estimator=="Student (Fisher consistent)":
        b = ((n+ddl)/n)*integrate.quad(lambda x : (x/(x+ddl))*scipy.stats.chi2.pdf(x,df=n),0,np.inf)[0]
        u = lambda x : (n + ddl)/(b*(ddl+x))
        res= [m_estimator(X_trials[k,:,:],u) for k in range(N)]
    
    if estimator=="RSCM-LW":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            betabis = 0
            for i in range(m):
                betabis += np.linalg.norm(X[:,i]@(X[:,i].T)-res_k)**2
            betabis = betabis/((np.linalg.norm(res_k-mu*np.eye(n))**2)*n*n)
            beta = min(1,betabis)
            res_k = (1-beta)*res_k+beta*mu*np.eye(n)
            res.append(res_k)
    
    if estimator=="RSCM-Ell1":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            beta = (gamma-1)/((gamma-1)+kappa*(2*gamma+n)/m+(gamma+n)/(m-1))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)
            
    if estimator=="RSCM-Ell2":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            gammabis = np.trace(res_k@res_k)/(mu**2*n)-(1+kappa)*(n/m)
            gamma = min(p,max(1,gammabis))
            
            beta = (gamma-1)/((gamma-1)+kappa*(2*gamma+n)/m+(gamma+n)/(m-1))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)
        
        
    if estimator=="RSCM-Ell3":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma1 = (n*m/(m-1))*(np.trace(M@M)-1/m)
 
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            gammabis = np.trace(res_k@res_k)/(mu**2*n)-(1+kappa)*(n/m)
            gamma2 = min(p,max(1,gammabis))
            
            gamma = min(gamma1,gamma2)
            
            beta = (gamma-1)/((gamma-1)+kappa*(2*gamma+n)/m+(gamma+n)/(m-1))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)
        
        
    if estimator=="Reg2-Gauss":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
 
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            psi1= 1+kappa
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            #print(beta)
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)
        
    if estimator=="Reg2-Huber":
        res = []
        params1 = huber_adaptive_param(X_trials,clean_prop)
        params2 = [scipy.stats.chi2.cdf(r,df=n+2)+(r/n)*(1-scipy.stats.chi2.cdf(r,df=n)) for r in params1]
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            u = lambda x : huber(x,params2[k],params1[k])
            res_huber_k = m_estimator(X,u)
            inv_cov = np.linalg.pinv(res_huber_k)
            t = [X[:,i].T@inv_cov@X[:,i] for i in range(m)]
            U = np.diag([u(t[i]) for i in range(m)])
            res_k = (X@U@X.T)/m

            mu = np.trace(res_k)/n
            
            psi1= 0
            for i in range(m):
                psi1 += (u(t[i])*t[i])**2
            psi1 = psi1/(m*n*(n+2))
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
 
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)        

    if estimator=="Reg2-Student":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            ddl = hill(X,int(m**0.25))
            u = lambda x : (n + ddl)/(ddl+x)
            res_student_k = m_estimator(X,u)
            inv_cov = np.linalg.pinv(res_student_k)
            t = [X[:,i].T@inv_cov@X[:,i] for i in range(m)]
            U = np.diag([u(t[i]) for i in range(m)])
            res_k = (X@U@X.T)/m

            mu = np.trace(res_k)/n
            
            psi1= 0
            for i in range(m):
                psi1 += (u(t[i])*t[i])**2
            psi1 = psi1/(m*n*(n+2))
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
             
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k) 
        
    


    if estimator=="Reg2bis-Huber":
        res = []
        params1 = huber_adaptive_param(X_trials,clean_prop)
        params2 = [scipy.stats.chi2.cdf(r,df=n+2)+(r/n)*(1-scipy.stats.chi2.cdf(r,df=n)) for r in params1]
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            u = lambda x : huber(x,params2[k],params1[k])
            res_huber_k = m_estimator(X,u)
            inv_cov = np.linalg.pinv(res_huber_k)
            t = [X[:,i].T@inv_cov@X[:,i] for i in range(m)]
            U = np.diag([u(t[i]) for i in range(m)])
            res_k = (X@U@X.T)/m

            mu = np.trace(res_k)/n
            
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
             
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = (X[:,i]/np.sqrt(params2[k]))*min(1,np.sqrt(params1[k])/np.linalg.norm(scipy.linalg.sqrtm(inv_cov)@X[:,i]))
            kappawbis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += W[p,i]**2
                    moment_4 += W[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappawbis += k_p 
            kappaw = max(-2/(n+2),kappawbis/(3*n))
            psi1 = 1+kappaw
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)      
        
    if estimator=="Reg2bis-Tyler":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            res_k = scm(X)
            mu = np.trace(res_k)/n
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            psi1= p/(p+2)
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k)
        
    if estimator=="Reg2bis-Student":
        res = []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            ddl = AutomaticAdaptDOF(X, kappa)
            u = lambda x : (n + ddl)/(ddl+x)
            res_student_k = m_estimator(X,u)
            inv_cov = np.linalg.pinv(res_student_k)
            t = [X[:,i].T@inv_cov@X[:,i] for i in range(m)]
            U = np.diag([u(t[i]) for i in range(m)])
            res_k = (X@U@X.T)/m

            mu = np.trace(res_k)/n
            
            psi1= 1-2/(n+ddl+2)
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = (n*m/(m-1))*(np.trace(M@M)-1/m)
 
            
            
            beta = (gamma-1)/((gamma-1)*(1-1/m)+psi1*(1-1/n)*((2*gamma+n)/m))
            res_k = beta*res_k+(1-beta)*mu*np.eye(n)
            res.append(res_k) 
        
    
    if estimator=="Reg1-Gauss":
        res= []
        for k in range(N):
            X = X_trials[k,:,:]
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            s = (n**2)*(np.trace(M@M)-1/m)
            beta = (s+n**2)/(s*(m+1)+n**2-n*m)
            res_k = (1-beta)*scm(X)+beta*np.eye(n)
            res.append(res_k)
        
    if estimator=="Reg1-Student":
        res= []
        b = 0.25
        for k in range(N):
            X = X_trials[k,:,:]
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            s = (n**2)*(np.trace(M@M)-1/m)
            ddl = hill(X,int(m**b))
            #print(ddl)
            u = lambda x: (ddl+n)/(ddl+x)
            beta = (s*(1+(ddl-2)/n)+n*(n+ddl))/(s*(m+1)*(1+ddl/n)+(n+ddl)*(n-m)-2*m)
            res_k = m_estimator(X,u,rho=beta,constraint_trace=True,initial=np.eye(n))
            res.append(res_k)
            
    if estimator=="Reg1bis-Student":
        res= []
        b = 0.5
        for k in range(N):
            X = X_trials[k,:,:]
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            s = (n**2)*(np.trace(M@M)-1/m)
            
            kappabis=0
            for p in range(n):
                moment_2 = 0
                moment_4 = 0
                for i in range(m):
                    moment_2 += X[p,i]**2
                    moment_4 += X[p,i]**4
                k_p = m*moment_4/(moment_2**2)
                k_p = ((m+1)*k_p+6)*(m-1)/((m-2)*(m-3))
                kappabis += k_p 
            kappa = max(-2/(n+2),kappabis/(3*n))
            
            ddl = AutomaticAdaptDOF(X, kappa)
            #print(ddl)
            u = lambda x: (ddl+n)/(ddl+x)
            beta = (s*(1+(ddl-2)/n)+n*(n+ddl))/(s*(m+1)*(1+ddl/n)+(n+ddl)*(n-m)-2*m)
            res_k = m_estimator(X,u,rho=beta,constraint_trace=True,initial=np.eye(n))
            res.append(res_k)
        
    if estimator=="Reg1-Tyler":
        res= []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            s = (n**2)*(np.trace(M@M)-1/m)
            u = lambda x: n/x
            beta = (s*(1-2/n)+n**2)/(s*(m+1+(2*m-2)/n)+n**2-n*m-2*m)
            res_k = m_estimator(X,u,rho=beta,constraint_trace=True,initial=np.eye(n))
            res.append(res_k)
        
        
    if estimator=="Reg1-Tyler-sign":
        res= []
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
           
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            s = (n**2)*(np.trace(M@M)-1/m)
            u = lambda x: n/x
            beta = (s*(1-2/n)+n**2)/(s*(m+1+(2*m-2)/n)+n**2-n*m-2*m)
            res_k = m_estimator(W,u,rho=beta,constraint_trace=True,initial=np.eye(n))
            res.append(res_k)
        
    if estimator=="Reg3-Tyler":
        res=[]
        threshold = 1e-10
        max_iter = 50
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            
            W = np.zeros(X.shape)
            for i in range(m):
                W[:,i] = X[:,i]/np.linalg.norm(X[:,i])
            M = scm(W)
            gamma = n*(np.trace(M@M)-1/m)
 
            beta = (gamma+n)/(m*(gamma-1)+n)
            cov =  np.eye(n)
            error = np.inf
            iteration =0
            while (error > threshold) and (iteration < max_iter):
                iteration +=1
                inv_cov = np.linalg.pinv(cov)
                new_cov = np.zeros((n,n))
                for i in range(m):
                    Z_i = (1-beta)*X[:,i]@X[:,i].T+beta*(np.linalg.norm(X[:,i])**2/n)*np.eye(n)
                    Z_i = Z_i/np.trace(Z_i@inv_cov)
                    new_cov += Z_i
                new_cov = (n**2/(m*np.trace(new_cov)))*new_cov
                #error = np.linalg.norm(np.eye(n)-new_cov@inv_cov,ord=)
                error = np.linalg.norm(cov-new_cov,ord=np.inf)
                #print(error)
                cov = new_cov
            #print(iteration)
            res.append(new_cov)

    if estimator=="Spec-RSCM":
        res=[]
        eta =1.5##TOMODIFY : cross valiation
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            l,v = np.linalg.eigh(scm(X))
            inds = list(np.argsort(l))
            inds.reverse()
            w = v.copy()
            for i in range(len(inds)):
                w[:,i]=v[:,inds[i]]
            target_eigs = fusing_eigs(X, eta)
            print(np.mean(np.abs(l-target_eigs)/l),np.std(np.abs(l-target_eigs)/l))
            res_k = w@(np.diag(target_eigs)@w.T)
            
            res.append(res_k)
            
    if estimator=="Spec-Tyler":
        res=[]
        maxiter=50
        threshold= 1e-8
        eta =1.5##TOMODIFY : cross valiation
        beta = 0.9 ##TOMODIFY : cross validation
        for k in tqdm(range(N)):
            X = X_trials[k,:,:]
            target_eigs = fusing_eigs(X, eta)
            cov = np.eye(n)
            t=0
            error = np.inf
            while (t< maxiter) and (error>threshold):
                inv_cov = np.linalg.pinv(cov)
                dist=[X[:,i].T@inv_cov@X[:,i] for i in range(m)]
                U = np.diag([1/d for d in dist])
                cov_bis = n*(X@U@X.T)/m
            
                l,v = np.linalg.eigh(cov_bis)
                inds = list(np.argsort(l))
                inds.reverse()
                w = v.copy()
                for i in range(len(inds)):
                    w[:,i]=v[:,inds[i]]
                diagonal = (1-beta)*l+beta*target_eigs
                new_cov = w@(np.diag(diagonal)@w.T)
                error = np.linalg.norm(cov-new_cov)
                cov = new_cov
                t+=1
            res.append(cov)
    res = np.asarray(res)       
    return res
        

class Covariances(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='SCM',clean_prop = 0.9, ddl = 5):
        self.estimator = estimator
        self.clean_prop = clean_prop
        self.ddl = ddl
        

    def fit(self, X, y):
        return self

    def transform(self, X):
        covmats = covariances(X, self.estimator,self.clean_prop,self.ddl)

        return covmats

