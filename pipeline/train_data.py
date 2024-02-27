#convert normal root files for easier upload for next time
import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(6)
import gc
import logging

def feature_engineering(df):
    df.columns = [col.replace('[', '_').replace(']', '') for col in df.columns]
    df['tof_oq_0_0/tof_ql_0']       = df['tof_oq_0_0'] / ((df['tof_ql_1']+df['tof_ql_0'])/2)
    df['tof_oq_0_1/tof_ql_0']       = df['tof_oq_0_1'] / ((df['tof_ql_1']+df['tof_ql_0'])/2)
    df['tof_oq_1_0/tof_ql_1']       = df['tof_oq_1_0'] / ((df['tof_ql_1']+df['tof_ql_0'])/2)
    df['tof_oq_1_1/tof_ql_1']       = df['tof_oq_1_1'] / ((df['tof_ql_1']+df['tof_ql_0'])/2)
    df['tof_oq_2_0/tof_ql_2']       = df['tof_oq_2_0'] / ((df['tof_ql_2']+df['tof_ql_3'])/2)
    df['tof_oq_2_1/tof_ql_2']       = df['tof_oq_2_1'] / ((df['tof_ql_2']+df['tof_ql_3'])/2)
    df['tof_oq_3_0/tof_ql_3']       = df['tof_oq_3_0'] / ((df['tof_ql_2']+df['tof_ql_3'])/2)
    df['tof_oq_3_1/tof_ql_3']       = df['tof_oq_3_1'] / ((df['tof_ql_2']+df['tof_ql_3'])/2)
    df['betah2r/tk_rigidity_0_2_2'] = df['betah2r']    /   df['tk_rigidity_0_2_2']
    df = df[df['tk_qin0_2']>0]
    df['betah2q/tk_qin0_2']         = df['betah2q']    /   df['tk_qin0_2']
    df['tk_oq_0/tk_qin0_2']         = df['tk_oq_0']    /   df['tk_qin0_2']
    df['tk_oq_1/tk_qin0_2']         = df['tk_oq_1']    /   df['tk_qin0_2']
    return df


#import awkward as ak
rigidity_low = 2.15
rigidity_up  = 1200

path0 = '/eos/user/k/khansh/data/train_test/without_chi2_sel'
tree_name = "t1"
n_jobs = 4
n1 = 70000

json_path = sys.argv[1]
with open(json_path, 'r') as json_file:
    files_info = json.load(json_file)

dataframes = []
for file_info in files_info:
    file_path = f"{path0}{file_info[0]}"
    df = uproot.open(f"{file_path}:t1").arrays(library='pd', decompression_executor=executor, interpretation_executor=executor)
    df = feature_engineering(df)
    df = df[(df['tk_rigidity_0_2_2']>rigidity_low) & (df['tk_rigidity_0_2_2']<rigidity_up)]
    print(file_info[1])
    df1 = df[df['label']==file_info[1]].sample(n=n1)
    df2 = df[df['label']==file_info[2]].sample(n=n1)
    df3 = df[df['label']==file_info[3]].sample(n=n1)
    del df
    print(f"file {file_path} processed")
    dataframes.extend([df1])
    dataframes.extend([df2])
    dataframes.extend([df3])
    del df1, df2, df3
    gc.collect()
    
    
df_all = pd.concat(dataframes)
del dataframes   # Delete the list to free up memory
gc.collect()


file = uproot.recreate(f"/eos/user/k/khansh/data/train_test_Li_to_Ni_without_chi2_89_classes_rigi_{rigidity_low}_{rigidity_up}.root", compression=uproot.ZLIB(4))
file["t1"] = df_all
del df_all
gc.collect()
print("train data creation complete")
