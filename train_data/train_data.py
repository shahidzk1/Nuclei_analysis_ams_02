#convert normal root files for easier upload for next time
import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(4)
import gc
import logging
#import awkward as ak
rigidity_low = 2.15
rigidity_up  = 1200

path0 = '/eos/user/k/khansh/data/train_test/without_chi2_sel'
tree_name = "t1"
n_jobs = 4
n1 = 70000

# Process and concatenate the DataFrames
dataframes = []
files_info = [
    ("/Li6/amsd68nMCLi6l1_B1236_NGenQ3/Li6_rig_2.15_1200.root", 0, 1, 2),
    ("/Li7/amsd68nMCLi7l1_B1236_NGenQ37/Li7_rig_2.15_1200.root", 3, 4, 5),
    ("/Be7/amsd68nMCBe7l1_B1236_NGenQ4/Be7_0_3_rig_2.15_1200.root", 6, 7, 8),
    ("/Be9/amsd68nMCBe9l1_B1236_NGenQ49/Be9_0_3_rig_2.15_1200.root", 9, 10, 11),
    ("/Be10/amsd68nMCBe10l1_B1236_NGenQ410/Be10_0_3_rig_2.15_1200.root", 12, 13, 14),
    ("/B10/amsd68nMCB10l1_B1236_NGenQ5/B10_0_3_rig_2.15_1200.root", 15, 16, 17),
    ("/B11/amsd68nMCB11l1_B1236_NGenQ511/B11_0_3_rig_2.15_1200.root", 18, 19, 20),
    ("/C/amsd68nMCC12l1_B1236_NGenQ6/C_0_3_rig_2.15_1200.root", 21, 22, 23),
    ("/N14/amsd68nMCN14l1_B1236_NGenQ7/N14_0_3_rig_2.15_1200.root", 24, 25, 26),
    ("/N15/amsd68nMCN15l1_B1236_NGenQ715/N15_0_3_rig_2.15_1200.root", 27, 28, 29),
    ("/O/amsd68nMCO16l1_B1236_NGenQ8/O_0_3_rig_2.15_1200.root", 30, 31, 32),
    ("/F/amsd68nMCF19l1_B1236_NGenQ9/F_0_3_rig_2.15_1200.root", 33, 34, 35),
    ("/Ne/amsd68nMCNe20l1_B1236_NGenQ10/Ne_0_3_rig_2.15_1200.root", 36, 37, 38),
    ("/Na/amsd68nMCNa23l1_B1236_NGenQ11/Na_0_3_rig_2.15_1200.root", 39, 40, 41),
    ("/Mg/amsd68nMCMg24l1_B1236_NGenQ12/Mg_0_3_rig_2.15_1200.root", 42, 43, 44),
    ("/Al/amsd68nMCAl27l1_B1236_NGenQ13/Al_0_3_rig_2.15_1200.root", 45, 46, 47),
    ("/Si/amsd68nMCSi28l1_B1236_NGenQ14/Si_0_3_rig_2.15_1200.root", 48, 49, 50),
    ("/P/amsd68nMCP31l1_B1236_NGenQ15/P_0_3_rig_2.15_1200.root", 51, 52, 53),
    ("/S/amsd68nMCS32l1_B1236_NGenQ16/S_0_3_rig_2.15_1200.root", 54, 55, 56),
    ("/Cl/amsd68nMCCl35l1_B1236_NGenQ17/Cl_0_3_rig_2.15_1200.root", 57, 58, 59),
    ("/Ar/amsd68nMCAr36l1_B1236_NGenQ18/Ar_0_3_rig_2.15_1200.root", 60, 61, 62),
    ("/K/amsd68nMCK39l1_B1236_NGenQ19/K_0_3_rig_2.15_1200.root", 63, 64, 65),
    ("/Ca/amsd68nMCCa40l1_B1236_NGenQ20/Ca_0_3_rig_2.15_1200.root", 66, 67, 68),
    ("/Sc/amsd68nMCSc45l1_B1236_NGenQ21/Sc_0_3_rig_2.15_1200.root", 69, 70, 71),
    ("/Ti/amsd68nMCTi48l1_B1236_NGenQ22/Ti_0_3_rig_2.15_1200.root", 72, 73, 74),
    ("/V/amsd68nMCV51l1_B1236_NGenQ23/V_0_3_rig_2.15_1200.root", 75, 76, 77),
    ("/Cr/amsd68nMCCr52l1_B1236_NGenQ24/Cr_0_3_rig_2.15_1200.root", 78, 79, 80),
    ("/Mn/amsd68nMCMn55l1_B1236_NGenQ25/Mn_0_3_rig_2.15_1200.root", 81, 82, 83),
    ("/Fe/amsd68nMCFe56l1_B1236_NGenQ26/Fe_0_3_rig_2.15_1200.root", 84, 85, 86),
    ("/Ni/amsd68nMCNi58l1_B1236_NGenQ28/Ni_0_3_rig_2.15_1200.root", 87, 88, 89),
]

for file_info in files_info:
    file_path = f"{path0}{file_info[0]}"
    df = uproot.open(f"{file_path}:t1").arrays(library='pd', decompression_executor=executor, interpretation_executor=executor)
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
