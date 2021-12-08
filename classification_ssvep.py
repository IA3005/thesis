#from SSVEP1 import TrialsBuilding ,SsvepLoading
from estimation import Covariances
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyriemann.estimation import Covariances as COV
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from classifiers import TangentSpace,MDM
from sklearn.pipeline import make_pipeline
import pandas as pd
from pyriemann.stats import pairwise_distance
from sklearn.manifold import TSNE
from preprocess_ssvep import contaminate_within_subject,contaminate_cross_subject,augment_trials,whiten_covs

colors=[ 'darkgoldenrod',
       'lime', 'black', 'mediumpurple', 'darkolivegreen','navy', 'red', 'grey',
       'fuchsia','yellow','darkkhaki',]
markers=['1','x','+','.','s']







def build_train_test_sets(database,classify_method,idx_per_class,n_sessions,shuffling,
                          train_prop,nb_classes):
    
    if database=="MAMEM3":
        if n_sessions==6 :
            sessions = [2,5,5,5,5,5]
        else:
            sessions= [5 for i in range(n_sessions)]
    
    if database=="ExoSkeleton":
        sessions = [8 for i in range(n_sessions)]
        
        
    total_trials = np.sum(np.asarray(sessions))*nb_classes
    
    if classify_method == "train and test on same session":
        kfolds = 100
        all_train_idx,all_test_idx = [],[]
        n =0
        m = 0
        for s in range(n_sessions):
            p = sessions[s]
            if shuffling : 
                for iter_ in range(kfolds):
                    train_idx = []
                    for k in range(1,1+nb_classes):
                        train_idx.extend(np.random.choice(idx_per_class[k][n:n+p],int(p*train_prop),replace=False))
                    test_idx = []
                    for i in range(m,m+nb_classes*p):
                        if not(i in train_idx):
                            test_idx.append(i)
                    all_train_idx.append(train_idx)
                    all_test_idx.append(test_idx)
            else:
                train_idx,test_idx = [],[]
                r = int(train_prop*p)
                for k in range(1,1+nb_classes):
                    l = idx_per_class[k][n:n+p]
                    train_idx.extend(l[:r])
                    test_idx.extend(l[r:])
                all_train_idx.append(train_idx)
                all_test_idx.append(test_idx)
            n = n+ p
            m =m +nb_classes*p
                        
            
    if classify_method == "train on all sessions except one":
        all_train_idx,all_test_idx = [],[]
        n = 0
        for s in range(n_sessions):
            p =sessions[s]
            test_idx = list(range(n,nb_classes*p+n))
            train_idx = []
            for i in range(total_trials):
                if not(i in test_idx):
                    train_idx.append(i)
            all_train_idx.append(train_idx)
            all_test_idx.append(test_idx)
            n = n + nb_classes*p
        
        
    if classify_method == "train on a session and test on another":
        all_train_idx,all_test_idx = [],[]
        n=0
        for s in range(n_sessions):
            p = sessions[s]
            test_idx = list(range(n,n+nb_classes*p))
            n = n + nb_classes*p
            m =0
            for r in range(n_sessions):
                q= sessions[r]
                if r!=s:
                    train_idx = list(range(m,m+nb_classes*q))
                    m = m+nb_classes*q
                    all_train_idx.append(train_idx)
                    all_test_idx.append(test_idx)

        
    if classify_method == "mix sessions":
        all_train_idx,all_test_idx = [],[]
        kfolds= 100
        if shuffling:
            for iter_ in range(kfolds):
                train_idx,test_idx = [],[]
                for k in range(1,1+nb_classes):
                    m= int(len(idx_per_class[k])*train_prop)
                    train_idx.extend(np.random.choice(idx_per_class[k],m,replace=False))
                    for i in range(total_trials):
                        if not(i in train_idx):
                            test_idx.append(i)
                all_train_idx.append(train_idx)
                all_test_idx.append(test_idx)
                
        else:
            train_idx,test_idx = [],[]
            for k in range(1,1+nb_classes):
                m = int(len(idx_per_class[k])*train_prop)
                train_idx.extend(idx_per_class[k][:m])
                test_idx.extend(idx_per_class[k][m:])
            all_train_idx = [train_idx]
            all_test_idx = [test_idx]
            
    return all_train_idx,all_test_idx



def fill_dict(results,subject,tmin,delta_t,freq_band,train_acc,
              test_acc,estimator,train_idx,test_idx,confusion,
              method,x,y):
    results['subject'].append(subject)
    results['train score'].append(train_acc)
    results['test score'].append(test_acc)
    results['estimator'].append(estimator)
    results['train idx'].append(train_idx)
    results['test idx'].append(test_idx)
    results['confusion'].append(confusion)
    results['classifier'].append(method)
    results['tmin'].append(tmin)
    results['delta_t'].append(delta_t)
    results['freq_band'].append(freq_band)
    results['x'].append(x)
    results["y"].append(y)
    return results  





def choose_classifier(method,u_prime):
    if method =="MDM":
        classifier = MDM(u_prime=u_prime)
    if method =="TS+logReg":
        classifier = make_pipeline(TangentSpace(u_prime=u_prime),LogisticRegression())
    if method =="TS+SVM":
        classifier = make_pipeline(TangentSpace(u_prime=u_prime),SVC(gamma="auto"))
    if method =="TS+LDA":
        classifier = make_pipeline(TangentSpace(u_prime=u_prime),LDA())
    return classifier




def classify(database,data,estimators,classify_method,freq_band,
             x,y,tmin,delta_t,method,nb_classes,
             freqs,u_prime =lambda t :1, adaptation=False,
             train_prop=0.625,shuffling=False, same_class=False,
             resampling=256, ddl=5, clean_prop=0.9):

    results =  {'subject':[],'estimator':[],'tmin':[],'delta_t':[],'freq_band':[],'train score':[],
                'test score':[],"confusion":[] ,'train idx':[],'test idx':[],
                'classifier':[],'x':[],'y':[]}
    
    classifier = choose_classifier(method, u_prime)
    nb_subjects = len(list(data.keys()))
    
    for subject in tqdm(range(nb_subjects)):
            trials,labels = data[subject]
            #contaminate all data
            if x !=1:
                trials = contaminate_within_subject(trials, labels, delta_t, x,resampling,nb_classes,same_class)
                
            if y !=1:
                trials = contaminate_cross_subject(subject,trials,labels ,data, delta_t, nb_classes, y,resampling,same_class)
            
                    
            #define idx for each class
            idx_per_class = {k:[] for k in range(1,1+nb_classes)}
            for j in range(len(labels)):
                idx_per_class[labels[j]].append(j)
            
            #choose indices for train 
            if database=="ExoSkeleton":
                n_sessions = trials.shape[0]//32
                
            if database=="MAMEM3":
                n_sessions = 5 + adaptation
                
            all_train_idx,all_test_idx = build_train_test_sets(database,classify_method,idx_per_class,n_sessions,shuffling,train_prop,nb_classes)
            
             
            for train_idx,test_idx in zip(all_train_idx,all_test_idx):
                
                
                train_trials = trials[train_idx,:,:]
                train_labels = labels[train_idx]
                test_trials = trials[test_idx,:,:]
                
                """
                #contaminate only training data
                if x !=1:
                    train_trials = contaminate_within_subject(train_trials, train_labels, delta_t, x,resampling,nb_classes,same_class)
                
                if y !=1:
                    train_trials = contaminate_cross_subject(subject, train_trials,train_labels ,data, delta_t, nb_classes, y,resampling,same_class)
                """
                
                if freq_band !=0:#extended trials
                    train_trials = augment_trials(train_trials, freq_band, tmin, delta_t,resampling,freqs)
                    test_trials = augment_trials(test_trials, freq_band, tmin, delta_t,resampling,freqs)
                
                #recenter    
                train_trials = train_trials - np.tile(train_trials.mean(axis=2).reshape(train_trials.shape[0], 
                                        train_trials.shape[1], 1), (1, 1, train_trials.shape[2]))
                test_trials = test_trials - np.tile(test_trials.mean(axis=2).reshape(test_trials.shape[0], 
                                        test_trials.shape[1], 1), (1, 1, test_trials.shape[2]))
        
                
                for estimator in tqdm(estimators):
                    if estimator=="Ledoit Wolf":
                        COVS = COV(estimator="lwf")
                    else:      
                        if estimator =="Student Optimized":
                            ddl  = optim_Student(subject,train_trials,train_labels,
                                                 classifier)
                            print(subject,"=",ddl)
                            COVS = Covariances(estimator="Student",ddl=ddl)
                        else:
                            COVS = Covariances(estimator=estimator,clean_prop=clean_prop,ddl=ddl)
                        
                    COVS.fit(trials,labels)
                    Xtrain = COVS.transform(train_trials)
                    Xtest  = COVS.transform(test_trials)
                    
                    test_labels = labels[test_idx]
            
                    classifier.fit(Xtrain,train_labels)
                    preds_train= classifier.predict(Xtrain)
                    preds_test= classifier.predict(Xtest)
                    train_acc = len(train_labels[train_labels==preds_train])/len(train_labels)
                    test_acc = len(test_labels[test_labels==preds_test])/len(test_labels)
                    confusion = np.zeros((nb_classes,nb_classes))
                    for j in range(len(test_labels)):
                        confusion[test_labels[j]-1,preds_test[j]-1] +=1
                    l_confusion = []
                    for k in range(nb_classes):
                        for l in range(nb_classes):
                            l_confusion.append(confusion[k,l])
                    results = fill_dict(results,subject+1,tmin,delta_t,freq_band,train_acc,test_acc,
                                        estimator,train_idx,test_idx,l_confusion,method,x,y)
               
    
    df= pd.DataFrame(results)
    for estimator in estimators:
        dfe = df[df["estimator"]==estimator]
        for i in range(nb_subjects):
            dfi = dfe[dfe["subject"]==i+1]
            scorei= dfi["test score"]
            score = np.mean(np.asarray(scorei))
            results = fill_dict(results,"mean",tmin,delta_t,freq_band,0,score,estimator,
                  train_idx,test_idx,[],method,x,y)
    df1=pd.DataFrame(results)
    plt.figure()
    sns.barplot(data=df1,x="subject",y="test score",hue="estimator")
    if x !=1:
        plt.title(database+" : within-subject contamination (x = "+str(x)+
              ") \n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+
              "s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method)
    else:
        if y!=1:
            plt.title(database+" : cross-subject contamination (y = "+str(y)+
              ") \n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+
              "s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method)
        else:
            plt.title(database+" : No contamination"+
              "\n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+
              "s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method)
            
    plt.legend(fontsize=8,loc="best")
    plt.show()
    
    return df1



def classify_single_subject(subject,database,data,estimators,classify_method,
                             freq_band,x,y,tmin,delta_t,nb_classes,method,
                             u_prime, freqs, results,
                             adaptation=False,tsne_plot=False,whiten=False,
                             train_prop=0.625,shuffling=False, same_class=False,
                             resampling=256, ddl=5, clean_prop=0.9,
                             p_train=10,lr_train=500,p_test=5,lr_test=100):

    classifier = choose_classifier(method, u_prime)
    trials,labels = data[subject]
    
    #contaminate all data
    if x !=1:
        trials = contaminate_within_subject(trials, labels, delta_t, x,resampling,nb_classes,same_class)
        
    if y !=1:
        trials = contaminate_cross_subject(subject,trials,labels ,data, delta_t, nb_classes, y,resampling,same_class)
        
    
    #define idx for each class
    idx_per_class = {k:[] for k in range(1,1+nb_classes)}
    for j in range(len(labels)):
        idx_per_class[labels[j]].append(j)
    
    #choose indices for train 
    if database=="ExoSkeleton":
        n_sessions = trials.shape[0]//32
        nb_trials_per_session = 32
        
    if database=="MAMEM3":
        n_sessions = 5 + adaptation
        nb_trials_per_session = 5 #if no adaptation
        
    all_train_idx,all_test_idx = build_train_test_sets(database,classify_method,idx_per_class,n_sessions,shuffling,train_prop,nb_classes)
    
     
    for train_idx,test_idx in zip(all_train_idx,all_test_idx):
        
        
        train_trials = trials[train_idx,:,:]
        train_labels = labels[train_idx]
        test_trials = trials[test_idx,:,:]
        
        """
        #contaminate only training data
        if x !=1:
            train_trials = contaminate_within_subject(train_trials, train_labels, delta_t, x,resampling,nb_classes,same_class)
        
        if y !=1:
            train_trials = contaminate_cross_subject(subject, train_trials,train_labels ,data, delta_t, nb_classes, y,resampling,same_class)
        """
        
        if freq_band !=0:#extended trials
            train_trials = augment_trials(train_trials, freq_band, tmin, delta_t,resampling,freqs)
            test_trials = augment_trials(test_trials, freq_band, tmin, delta_t,resampling,freqs)
        
        #recenter    
        train_trials = train_trials - np.tile(train_trials.mean(axis=2).reshape(train_trials.shape[0], 
                                train_trials.shape[1], 1), (1, 1, train_trials.shape[2]))
        test_trials = test_trials - np.tile(test_trials.mean(axis=2).reshape(test_trials.shape[0], 
                                test_trials.shape[1], 1), (1, 1, test_trials.shape[2]))

        
        for estimator in tqdm(estimators):
            if estimator=="Ledoit Wolf":
                COVS = COV(estimator="lwf")
            else:      
                if estimator =="Student Optimized":
                    ddl  = optim_Student(subject,train_trials,train_labels,
                                         classifier)
                    print(subject,"=",ddl)
                    COVS = Covariances(estimator="Student",ddl=ddl)
                else:
                    COVS = Covariances(estimator=estimator,clean_prop=clean_prop,ddl=ddl)
                
            COVS.fit(trials,labels)
            Xtrain = COVS.transform(train_trials)
            if whiten:
                Xtrain = whiten_covs(Xtrain, n_sessions, nb_trials_per_session)
            
            Xtest  = COVS.transform(test_trials)
            test_labels = labels[test_idx]
            
            classifier.fit(Xtrain,train_labels)
            covmeans = classifier.covmeans_
            
            if tsne_plot:
                if x!=1:
                    title_bis = estimator+" \\ within-subject contamination (x = "+str(x)+") \n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+"s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method
                else:
                    if y!=1:
                        title_bis = estimator+" \\ cross-subject contamination (y = "+str(y)+") \n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+"s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method
                    else:
                        title_bis = estimator+" \\ No contamination"+"\n half-window="+str(freq_band)+"Hz \\tmin="+str(tmin)+"s \\ duration="+str(delta_t)+"s \n paradigm="+classify_method
                        
                
                title = database+" : Training Covariance Matrices of Subject = "+str(subject+1)+"\n"+title_bis
                _ = tsne(covmeans,Xtrain,train_labels,n_sessions,nb_trials_per_session,title,p=p_train,lr=lr_train)
                
                title = database+"Testing Covariance Matrices of Subject = "+str(subject+1)+"\n"+title_bis
                _ = tsne(covmeans,Xtest,test_labels,n_sessions,nb_trials_per_session,title,p=p_test,lr=lr_test)
   
            preds_train= classifier.predict(Xtrain)
            preds_test= classifier.predict(Xtest)
            train_acc = len(train_labels[train_labels==preds_train])/len(train_labels)
            test_acc = len(test_labels[test_labels==preds_test])/len(test_labels)
            confusion = np.zeros((nb_classes,nb_classes))
            for j in range(len(test_labels)):
                confusion[test_labels[j]-1,preds_test[j]-1] +=1
            l_confusion = []
            for k in range(nb_classes):
                for l in range(nb_classes):
                    l_confusion.append(confusion[k,l])
            results = fill_dict(results,subject+1,tmin,delta_t,freq_band,train_acc,test_acc,
                                estimator,train_idx,test_idx,l_confusion,method,x,y)
       
    return results



def optim_Student(subject,trials,labels,classifier):
     
    #optimize ddl
    max_score = 0
    best_ddl = 0
    list_ddl = [0.1*i for i in range(1,101)]
    for ddl in tqdm(list_ddl):
        COVS = Covariances(estimator="Student",ddl = ddl)
        COVS.fit(trials,labels)
        covs = COVS.transform(trials)
        classifier.fit(covs,labels)
        preds_train= classifier.predict(covs)
        train_acc = len(labels[labels==preds_train])/len(labels)
        
        if train_acc > max_score:
            max_score = train_acc
            best_ddl = ddl

    return best_ddl


def tsne(centers,covs,labels,nb_sessions,nb_trials_per_session,title,p=10,lr=500):
    fig=plt.figure()
    ax= fig.add_subplot(111)
    #all_covs = np.zeros((len(centers)+covs.shape[0],covs.shape[1],covs.shape[2]))
    #for i in range(len(centers)):
    #    all_covs[i,:,:]=centers[i] 
    #all_covs[len(centers):,:,:]=covs
    
    #pair_dist = pairwise_distance(all_covs, metric='riemann')
    pair_dist = pairwise_distance(covs, metric='riemann')
    tsne = TSNE(metric='precomputed', perplexity=p, learning_rate=lr,
                        early_exaggeration=4,random_state=665)
           
    out = tsne.fit_transform(pair_dist)
    legs = ["session"+str(s+1) for s in range(nb_sessions)]
    
    #for i in range(len(centers)):
        #line=ax.scatter([out[i, 0]], [out[i, 1]],marker="s",c=colors[i])
        #line.set_label(str(i+1))
        
    for i in range(len(covs)):
        s = i//nb_trials_per_session
        #line=ax.scatter([out[i+len(centers), 0]], [out[i+len(centers), 1]],marker=markers[s],c=colors[labels[i]-1])
        line=ax.scatter([out[i, 0]], [out[i, 1]],marker=markers[s],c=colors[labels[i]-1])
        if i%nb_trials_per_session==0:
            line.set_label(legs[s])
            
    plt.title(title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return None
    