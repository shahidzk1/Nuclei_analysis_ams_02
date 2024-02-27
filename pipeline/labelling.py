#This code labels the data and converts normal root files for easier upload for next time
import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
from tree_converter_and_selector import TreeHandler
import json
import sys

rigidity_low = 2.15
rigidity_up  = 1200
rigidity_var = 'tk_rigidity_0[2][2]'
tree_name = "t1"
n_jobs = 6
layer_info = 13


def process_tree(file_path, label_frag, label_frag_L0, label_non_frag, charge):
    with TreeHandler(file_path, tree_name, n_jobs) as tree_handler:
        df_frag, df_non_frag = tree_handler.labeled(label_frag, label_non_frag, layer_info, rigidity_var, rigidity_low, rigidity_up, charge_sel=True, nuclei_charge=charge)
        df_frag_L0, df_non_frag1 = tree_handler.labeled(label_frag_L0, label_non_frag, 0, rigidity_var, rigidity_low, rigidity_up, charge_sel=True, nuclei_charge=charge)
        del df_non_frag1
    gc.collect()
    return df_frag, df_non_frag, df_frag_L0
#the json file should contain the path/s of the file/s to be processed
#for example [['/eos/user/k/khansh/data/train_test/without_chi2_sel/Li6/amsd68nMCLi6l1_B1236_NGenQ3/Li6.root', 0, 1, 2, 3],]
json_path = sys.argv[1]
with open(json_path, 'r') as json_file:
    files_info = json.load(json_file)

output = []
for file_info in files_info:
    file_path = f"{file_info[0]}"
    df_frag, df_non_frag, df_frag_L0 = process_tree(file_path, file_info[1], file_info[2], file_info[3], file_info[4])
    #df_non_frag = df_non_frag[df_non_frag['tk_qin0[2]']>0]
    print(f"file{file_info[0]} processed")
    dataframes = []
    dataframes.extend([df_frag, df_non_frag, df_frag_L0])
    df_all = pd.concat(dataframes)
    del dataframes,df_frag, df_non_frag, df_frag_L0
    gc.collect()
    df_all.columns = [col.replace('[', '_').replace(']', '') for col in df_all.columns]
    df_all = df_all[(df_all['tk_rigidity_0_2_2']>rigidity_low) & (df_all['tk_rigidity_0_2_2']<rigidity_up)]
    file_path = file_path.replace('.root', '')
    file = uproot.recreate(f"{file_path}_rig_{rigidity_low}_{rigidity_up}.root", compression=uproot.ZLIB(4))
    output.append({
        "file_path": f"{file_path}_rig_{rigidity_low}_{rigidity_up}.root",
        "label_frag_belo_L1": file_info[1],
        "label_frag_abv_L1": file_info[2],
        "label_non_frag": file_info[3],
        "charge": file_info[4]
    })
    file["t1"] = df_all
    print("file created: ",f"{file_path}_rig_{rigidity_low}_{rigidity_up}.root")
    del df_all, file
    gc.collect()

#to store the paths of the output files
output_file_path = sys.argv[2]
with open(output_file_path, 'w') as json_file_out:
    json.dump(output, json_file_out)
print("labelling complete")
