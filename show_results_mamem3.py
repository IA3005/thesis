from classification_ssvep import classify
from preprocess_ssvep import MAMEM3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

database="MAMEM3"
classify_method =  "train on all sessions except one"
#estimators=["SCM"]
estimators=["SCM","Student","Huber (2)","Huber (1)"]
method="MDM"
freq_band = 0
tmin=0.5
delta_t=4.5
x =1
nb_classes=5
u_prime=lambda t:1
resampling =128
adaptation=False
same_class=False
y = 1
assert (x ==1) or (y==1),"Either within-subject or cross-subject contamination"

dataset = MAMEM3(freq_band,tmin,delta_t,resampling,adaptation)
freqs = dataset.freqs


dataset.prepare_data()
data = dataset.load_all_data()

dfs = []

for x in [0.6,0.7,0.8,0.9,1]:
    df = classify(database,data,estimators,classify_method,freq_band,
                x,y,tmin,delta_t,method,nb_classes,
                freqs,u_prime=u_prime , 
                adaptation=adaptation,same_class=same_class,
                resampling=resampling,
                ddl=5,clean_prop=min(x,y,0.9))
    dfs.append(df)
    
    
df_all = pd.concat(dfs)
  
for i  in df_all["subject"].unique():
    dfi = df_all[df_all["subject"]==i]
    plt.figure()
    plt.grid()
    sns.lineplot(x="x",y="test score",hue="estimator",data=dfi,markers=True, dashes=True)
    plt.title(database+": Subject = "+str(i)+
              "\n half-window="+str(freq_band)+"Hz \\tmin="+
              str(tmin)+"s \\duration="+str(delta_t)+"\n resampling ="+
              str(resampling)+"Hz \\paradigm="+
              classify_method)
    plt.legend(fontsize=8,loc="best")
    plt.xticks([0.6,0.7,0.8,0.9,1])
    plt.show()
