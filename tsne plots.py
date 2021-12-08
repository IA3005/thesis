from classification_ssvep import classify_single_subject,tsne
from preprocess_ssvep import ExoSkeleton
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

database="ExoSkeleton"
classify_method =  "train on all sessions except one"
estimators=["SCM"]
#estimators=["SCM","Student","Huber (2)","Huber (1)","Scaled Tyler (1)"]
method="MDM"
freq_band = 0
tmin=2
delta_t=2
x = 1
nb_classes= 4
u_prime=lambda t:1
resampling =128
y = 1
assert (x ==1) or (y==1),"Either within-subject or cross-subject contamination"
results =  {'subject':[],'estimator':[],'tmin':[],'delta_t':[],'freq_band':[],'train score':[],
                'test score':[],"confusion":[] ,'train idx':[],'test idx':[],
                'classifier':[],'x':[],'y':[]}
    


dataset = ExoSkeleton(freq_band,tmin,delta_t,resampling)
freqs = dataset.freqs
subject = 11

subj_list,records = dataset.SsvepLoading()
data = dataset.load_all_data(records,subj_list)

res = classify_single_subject(subject,database,data,estimators,classify_method,
                             freq_band,x,y,tmin,delta_t,nb_classes,method,
                             u_prime, freqs, results,
                             tsne_plot=True,
                             whiten=False,
                             same_class=False,
                             resampling=resampling, ddl=5,
                             clean_prop=min(x,y,0.9),
                             p_train=20,lr_train=200,p_test=5,lr_test=300)
