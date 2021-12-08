import mne
import numpy as np
import os
from scipy.signal import filtfilt, butter,lfilter
import random
import scipy
#import scipy.stats.multivariate_t as t
from pyriemann.utils.mean import mean_riemann


def filter_bandpass(signal, lowcut, highcut, fs, order=4, filttype='forward-backward'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if filttype == 'forward':
        filtered = lfilter(b, a, signal, axis=-1)
    elif filttype == 'forward-backward':
        filtered = filtfilt(b, a, signal, axis=-1)
    else:
        raise ValueError("Unknown filttype:", filttype)    
    return filtered    




class ExoSkeleton():

    def __init__(self,freq_band,tmin,delta_t,resampling=256):
        
        self.data_path = "C:/Users/Imen Ayadi/EEG/ssvep_moabb/"
        self.freqs = [13,17,21]
        self.nb_trials_per_session = 32
        self.sampling = 256
        self.event_code_fif = [1,2,3,4]
        self.lowcut = 1
        self.highcut = 40
        self.channels = ['Oz','O1','O2','PO3','POz','PO7','PO8','PO4']
        self.names=['resting','stim13','stim21','stim17']
        
        self.freq_band = freq_band
        self.tmin = tmin
        self.delta_t = delta_t
        self.resampling = resampling
        
        
    def SsvepLoading(self):
        """
        Outputs:
            subj_list : a list of subjects ["subject1",...]
            records : a dictionnary of subjects and their associated sessions
        """
        subj_list = os.listdir(self.data_path)
        records = {s: [] for s in range(len(subj_list))}
        for s in range(len(subj_list)):
            subj = subj_list[s]
            record_all = os.listdir(self.data_path+subj+'/')
            for file in record_all:
                if file[len(file)-8:]=="_raw.fif":
                    records[s].append(file[:len(file)-8])
        return subj_list,records
    
            
   
    def extract_trials(self,records,subj_list, subject,tmin=0, tmax=5):     
        chosen_subject = subj_list[subject]
        all_labels = []
        all_trials = []
        
        for session in  range(len(records[subject])): 
            fname = chosen_subject+'/'+records[subject][session]
            raw = mne.io.read_raw_fif(self.data_path + fname + '_raw.fif',preload=True)
            events = mne.find_events(raw)
            if self.resampling != self.sampling:
                raw_signal,events = raw.copy().resample(self.resampling, npad='auto', events=events)
                raw_signal = raw_signal.get_data()
            else:
                raw_signal = raw.get_data()
            raw_signal = raw_signal[:-1,:]
            raw_signal = filter_bandpass(raw_signal, lowcut=self.lowcut, highcut=self.highcut, fs=self.resampling)
                
            event_pos = events.T[0]
            event_type = events.T[2]
            
            
            labels = []
            for e in event_type:
                if (e  in self.event_code_fif):
                    labels.append(e)
            all_labels.extend(labels)
            
            trials = list()
            for i in range(len(event_type)):
                if event_type[i] in self.event_code_fif: 
                    t = event_pos[i]
                    start = int(t + tmin*self.resampling)
                    stop  = int(t + tmax*self.resampling)
                    epoch = raw_signal[:, start:stop]
                    trials.append(epoch)
            trials = np.array(trials)
            #trials = trials - np.tile(trials.mean(axis=2).reshape(trials.shape[0], 
                                            #trials.shape[1], 1), (1, 1, trials.shape[2]))
            all_trials.append(trials)
        
        nb_total_trials = 0
        all_labels = np.asarray(all_labels)
        
        for k in range(len(all_trials)):
            nb_total_trials += all_trials[k].shape[0]
        all_trials_ = np.zeros((nb_total_trials,all_trials[0].shape[1],all_trials[0].shape[2]))
        m=0
        for k in range(len(all_trials)):
            n = all_trials[k].shape[0]
            all_trials_[m:m+n,:,:]=all_trials[k]
            m +=n 
            
        return all_trials_,all_labels
                     
    

    def load_all_data(self,records,subj_list):
        data = { subject : None for subject in range(len(subj_list))}
        for subject in range(len(subj_list)):
            trials,labels = self.extract_trials(records,subj_list, subject)
            data[subject] = [trials,labels]
        return data


class MAMEM3():

    def __init__(self,freq_band,tmin,delta_t,resampling=128,adaptation=False):
        
        self.data_path = "C:/Users/Imen Ayadi/EEG/MAME3_mat"
        self.saving_path = "C:/Users/Imen Ayadi/EEG/MAMEM3"
        self.freqs = [6.66,7.5,8.57,10,12]
        self.n_subjects = 11
        self.sampling = 128
        self.event_code = [33025,33026,33027,33028,33029]
        self.lowcut = 1
        self.highcut = 40
        
        self.freq_band = freq_band
        self.tmin = tmin
        self.delta_t = delta_t
        self.resampling = resampling
        self.adaptation = adaptation
        
        
    def prepare_data(self):
            
        for extension in ["x","ai","aii","bi","bii","ci","cii","di","dii","ei","eii"]:
            if extension=="x":
                session="/adaptation"
            else:
                session="/session"+str(ord(extension[0])-ord("a")+1)+"_run"+str(len(extension)-1)
            
            for i in range(1,self.n_subjects+1):
                if i<10:
                    subj = "/U00"+str(i)+extension+".mat"
                else:
                    subj = "/U0"+str(i)+extension+".mat"
                path_subj = self.saving_path +"/Subject"+str(i)+"/"+session
                if not os.path.exists(path_subj):
                    os.makedirs(path_subj)
                mat = scipy.io.loadmat(self.data_path+subj)
                x = mat['eeg'][:-1] #(14,130*128)
                y= mat['events'].T
                labels =[]
                starts =[]
                ends= []
                for j in range(len(y[1])):
                    if y[1][j] in self.event_code:
                        labels.append(33030-y[1][j])
                    if y[1][j] ==32779:
                        starts.append(y[2][j])
                        ends.append(y[2][j+1])
               
                #save starts and ends
                starts = np.asarray(starts)
                with open(path_subj+'/starts.npy', 'wb') as f:
                    np.save(f, starts)
                ends= np.asarray(ends)
                with open(path_subj+'/ends.npy', 'wb') as f:
                    np.save(f, ends)    
                    
                #save signal
                with open(path_subj+'/raw_signal.npy', 'wb') as f:
                    np.save(f, x)
                    
                    
                #save labels
                with open(path_subj+'/labels.npy', 'wb') as f:
                    np.save(f, np.asarray(labels))
                    
        return None
                        
    def extract_trials(self, subject):
         
        path_subj = self.saving_path +"/Subject"+str(subject+1)
        all_labels = []
        all_trials = [] 
        sessions = os.listdir(path_subj)
    
        if not(self.adaptation):
            sessions.remove("adaptation")
            
        for session in sessions:
            
            path_session = path_subj+"/"+session
            with open(path_session+'/raw_signal.npy', 'rb') as f:
                x = np.load(f)
                
            with open(path_session+'/labels.npy', 'rb') as f:
                labels = np.load(f)
                
            with open(path_session+'/starts.npy', 'rb') as f:
                starts = np.load(f)
                
            with open(path_session+'/ends.npy', 'rb') as f:
                ends = np.load(f)
                
                
            trials= []
            for j in range(len(starts)):
                start = int(starts[j])
                stop  = int(ends[j])
                epoch = x[:, start:stop]
                trials.append(epoch)
            trials = np.array(trials)
            
            all_labels.extend(list(labels))
            
            all_trials.append(trials)
    
        all_labels = np.asarray(all_labels)
        nb_total_trials = 0
        for k in range(len(all_trials)):
            nb_total_trials += all_trials[k].shape[0]
        all_trials_ = np.zeros((nb_total_trials,all_trials[0].shape[1],all_trials[0].shape[2]))
        m=0
        for k in range(len(all_trials)):
            n = all_trials[k].shape[0]
            all_trials_[m:m+n,:,:]=all_trials[k]
            m +=n    
            
        #print(len(all_labels), all_trials_.shape)
        return all_trials_,all_labels
        
     

    def load_all_data(self):
        data = { subject : None for subject in range(self.n_subjects)}
        for subject in range(self.n_subjects):
            trials,labels = self.extract_trials(subject)
            data[subject] = [trials,labels]
        return data



"""
class SyntheticSSVEP():

    def __init__(self,nb_subjects,nb_sessions_per_subject,sampling,nb_channels,
                 nb_trials_per_session,nb_classes,freq_band,tmin,delta_t):
        
        self.sampling = sampling
        self.nb_subjects = nb_subjects
        self.nb_sessions_per_subject = nb_sessions_per_subject
        self.nb_trials_per_session = nb_trials_per_session
        self.nb_channels = nb_channels
        self.nb_classes = nb_classes
        
        self.freq_band = freq_band
        self.tmin = tmin
        self.delta_t = delta_t
        self.duration = 5 

        
    def generate_trial(self,ddl,scatter):
        nb_samples = self.duration*self.sampling
        mean = [0 for i in range(self.nb_channels)]
        raw_signal = t.rvs(mean,scatter,df=ddl,size=nb_samples)
        return raw_signal
    
    def centers_of_classes(self):
        centers= []
        for k in range(self.nb_classes):
            #generate random SPD
            Z = np.random.rand(self.nb_channels,self.nb_channels)
            cov = Z@Z.T+np.random.rand()*np.eye(self.nb_channels) 
            centers.append(cov)
        return centers
        
    def genrate_
    
    def build_data(self):
        data = { subject : None for subject in range(self.nb_subjects)}
        centers = self.centers_of_classes()
        
        #occurence of a class in a session
        occ = self.nb_trials_per_session//self.nb_classes
        label_per_session = [k for k in range(1,1+self.nb_classes) for i in range(occ)]
        labels = []
        trials = np.zeros((self.nb_trials_per_session*self.nb_sessions_per_subject,self.nb_channels,self.sampling*self.duration))
        
        for subject in range(self.nb_subjects):
            #fix a ddl for each subject between 1 and 30
            ddl = 1+np.random.random_sample()*29
            
            for s in range(self.nb_sessions_per_subject):
                labels.extend(label_per_sessions)
                self.generate_trial(ddl, centers[k])
                
                
            
            
            data[subject] = [trials,labels]
        return data
"""

#########################################################################
def augment_trials(trials, freq_band, tmin, delta_t,resampling,freqs):
        
    start = int(tmin*resampling)
    stop  = int((tmin+delta_t)*resampling)
    ext_trials = np.zeros((trials.shape[0],len(freqs)*trials.shape[1],stop-start))
    for i in range(trials.shape[0]):
        ext_signal = np.empty_like(trials[0,0,:])    #(1,n)
        for f in freqs:
            ext_signal = np.vstack((ext_signal, filter_bandpass(trials[i,:,:], f-freq_band,f+freq_band, fs=resampling)))
        
        ext_trials[i,:,:] = ext_signal[1:,start:stop]
        
    return ext_trials       

def whiten_covs(covs, nb_sessions,nb_trials_per_session):
    for session in range(nb_sessions):
        n= session*nb_trials_per_session
        center = np.mean(covs[n:n+nb_trials_per_session,:,:],axis=0)
        inv_center = np.linalg.pinv(center)
        for i in range(nb_trials_per_session):
            covs[i+n,:,:]= inv_center@covs[i+n,:,:]
    return covs  
    
   
def contaminate_within_subject(train_trials,train_labels,delta_t,clean_prop,
                               resampling,nb_classes,same_class):
    
    #classify training trials per class 
    indx_trials_per_class = {k: [] for k in range(1,1+nb_classes)}
    for i in range(len(train_trials)):
        indx_trials_per_class[train_labels[i]].append(i)
    
    #build contaminated training trials
    new_trials = np.zeros(train_trials.shape)
    limit = int(delta_t*clean_prop*resampling)
    
    for i in range(len(train_trials)):
        #current class
        k = train_labels[i]
        
        #choose random class for contamination
        if same_class:
            l = k
        else:
            l = random.choice(list(range(1,k))+list(range(k+1,1+nb_classes)))
        
        #pick clean part of the trial
        new_trials[i,:,:limit] = train_trials[i,:,:limit]
        
        #pick the contaminated part
        j = random.choice(indx_trials_per_class[l])
        new_trials[i,:,limit:] = train_trials[j,:,limit:]
        
    return new_trials
        
        
def contaminate_cross_subject(subject, train_trials,train_labels ,data,
                              delta_t,nb_classes,clean_prop,resampling,
                              same_class):
    
    nb_subjects = len(list(data.keys()))
    limit = int(delta_t*clean_prop*resampling)
    
    new_trials = np.zeros(train_trials.shape)
    
    
    
    for i in range(len(train_labels)):
        #current class
        k = train_labels[i]
        
        #choose random subject for contamination
        contaminating_subject = random.choice(list(range(subject))+list(range(subject+1,nb_subjects)))
        contaminating_trials,contaminating_labels = data[contaminating_subject]

        indx_trials_per_class = {k: [] for k in range(1,1+nb_classes)}
        for j in range(len(contaminating_labels)):
            indx_trials_per_class[contaminating_labels[j]].append(j)
        
        #choose random class 
        if same_class:
            l=k
        else:
            l = random.choice(list(range(1,k))+list(range(k+1,1+nb_classes)))
        
        #pick clean part of the trial
        new_trials[i,:,:limit] = train_trials[i,:,:limit]
        
        #pick the contaminated part
        j = random.choice(indx_trials_per_class[l])
        new_trials[i,:,limit:] = contaminating_trials[j,:,limit:]
    
    return new_trials

def contaminate_cross_subject_bis(subject, train_trials,train_labels ,data,
                              delta_t,nb_classes,clean_prop,resampling,
                              same_class):
    
    nb_subjects = len(list(data.keys()))
    limit = int(delta_t*clean_prop*resampling)
    
    new_trials = np.zeros(train_trials.shape)
    
    
    
    for i in range(len(train_labels)):
        #current class
        k = train_labels[i]
        
        #pick clean part of the trial
        new_trials[i,:,:limit] = train_trials[i,:,:limit]
        
        #mean contamination
        for subj in range(nb_subjects):
            if subj !=subject:
                contaminating_trials,contaminating_labels = data[subj]

                indx_trials_per_class = {k: [] for k in range(1,1+nb_classes)}
                for j in range(len(contaminating_labels)):
                    indx_trials_per_class[contaminating_labels[j]].append(j)
                
                #choose random class 
                if same_class:
                    l=k
                else:
                    l = random.choice(list(range(1,k))+list(range(k+1,1+nb_classes)))
            
                #pick the contaminated part
                j = random.choice(indx_trials_per_class[l])
        
                new_trials[i,:,limit:] += contaminating_trials[j,:,limit:]
          
        new_trials[i,:,limit:] = new_trials[i,:,limit:] /(nb_subjects-1)
    
    return new_trials
