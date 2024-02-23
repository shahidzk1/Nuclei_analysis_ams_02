import numpy as np
import pandas as pd
import uproot
class TestClass(object): 
    def check(self): 
        print ("object is alive!") 
    def __del__(self): 
        print ("object deleted")    
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(10)
import optuna
import warnings
import os
import gc
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold,cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import xgboost as xgb
from xgboost import XGBClassifier
from datetime import date
today = date.today()
print("xgb is ",xgb.__version__)
print("uproot is ",uproot.__version__)

def add_data_signal_to_mc(df_mc, df_data):
    #this func was created because the previous mc model was trained on data which had two classes for each fragment there are 3 now
    dataframes = []
    label_mapping = {i: j for i, j in zip(range(32, 90, 3), range(21, 60, 2))}
    for i in range(0,90,1):
        df = df_all[df_all['label']==i]
        if i not in label_mapping:
            #because signal for charge greater than have been selected with tight manual selection and xgboost
            dataframes.extend([df])
        else:
            df = df_all[df_all['label']==i]
            df.reset_index(drop=True, inplace=True)
            df_signal = df_data[df_data['label']==label_mapping[i]]
            df_signal['label']=i
            df_signal.reset_index(drop=True, inplace=True)
            if df_signal.shape[0] >5000:
                df_signal = df_signal.sample(n=5000)
                df_red = df.drop(df.index[0:5000])
            else:
                df_red = df.drop(df.index[0:df_signal.shape[0]])
            dataframes.extend([df_signal])
            del df
            dataframes.extend([df_red])
            del df_signal, df_red
            gc.collect()
    df_tot = pd.concat(dataframes)
    return df_tot
