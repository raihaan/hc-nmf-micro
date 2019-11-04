import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import scipy.io
import pickle

import sys
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import random
from sklearn.model_selection import StratifiedShuffleSplit

n_subjects = 329
n_splits = 10
max_gran = 10

out_dir = "n" + str(n_splits) + "/"

cols = ["Granularity","Iteration","Euclidean_mean","Euclidean_median","Euclidean_std","Cosine_mean","Cosine_median","Cosine_std","Corr_mean","Corr_median","Corr_std","Recon_errorA","Recon_errorB"]
df = pd.DataFrame(columns = cols)

stab_dir = "/data/chamal/projects/raihaan/projects/inprogress/hc-nmf-micro/analysis/329subject_singleshellNMF/stability/n10/pnmf_out/"
stab_dir = "" #MODIFY to point to dir containing nmf outputs of each split of data 

#RIGHT HC STABILITY
for i in range(0,n_splits):
    
    for g in range(2,max_gran+1):
        #load split input, run nmf for each split
        fname = stab_dir + "k" + str(g) + "/rightA_" + str(i) + "_res.mat" #MODIFY as needed
        print fname
        resA = scipy.io.loadmat(fname)
        Wa = resA['W']
        ea = resA['recon'][0,0]
        
        fname = stab_dir + "k" + str(g) + "/rightB_" + str(i) + "_res.mat" #MODIFY as needed
        print fname
        resB = scipy.io.loadmat(fname)
        Wb = resB['W']
        eb = resB['recon'][0,0]
         
        #assess correlation of identified parcel component scores - which parcels vary together?
        c_Wa = cosine_similarity(Wa)
        c_Wb = cosine_similarity(Wb)
        
        cosine_dist = np.zeros((1,np.shape(c_Wa)[0]))
        euclid_dist = np.zeros((1,np.shape(c_Wa)[0]))
        corr = np.zeros((1,np.shape(c_Wa)[0]))

        for parcel in range(0,np.shape(c_Wa)[0]):
            cosine_dist[0,parcel] = scipy.spatial.distance.cosine(c_Wa[parcel,:], c_Wb[parcel,:])
            euclid_dist[0,parcel] = scipy.spatial.distance.euclidean(c_Wa[parcel,:], c_Wb[parcel,:])
            corr[0,parcel] = np.corrcoef(c_Wa[parcel,:],c_Wb[parcel,:])[0,1]

        df_curr = pd.DataFrame(data = [[g, i+1, np.mean(euclid_dist), np.median(euclid_dist),np.std(euclid_dist),
                                        np.mean(cosine_dist), np.median(cosine_dist),np.std(cosine_dist),np.mean(corr),np.median(corr),np.std(corr),ea,eb]], columns = cols)

        df = df.append(df_curr)
        df.to_csv(out_dir + 'temppnmf_cosine-sim_righthc_corr.csv')
        del Wa,Wb,ea,eb,resA,resB
    
df.to_csv(out_dir + 'pnmf_cosine-sim_righthc_corr.csv')
del df, df_curr
