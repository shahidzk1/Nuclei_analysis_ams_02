#convert normal root files for easier upload for next time
import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
from tree_converter_and_selector import TreeHandler




rigidity_low = 2.15
rigidity_up  = 1200

path0 = '/eos/user/k/khansh/data/train_test/without_chi2_sel'
tree_name = "t1"
n_jobs = 2
layer_info = 13
variable_name = 'tk_rigidity_0[2][2]'

def process_tree(file_path, label_frag, label_frag_L0, label_non_frag):
    with TreeHandler(file_path, tree_name, n_jobs) as tree_handler:
        df_frag, df_non_frag = tree_handler.labeled(label_frag, label_non_frag, layer_info, variable_name, rigidity_low, rigidity_up)
        df_frag_L0, df_non_frag1 = tree_handler.labeled(label_frag_L0, label_non_frag, 0, variable_name, rigidity_low, rigidity_up)
        del df_non_frag1
    gc.collect()
    
    return df_frag, df_non_frag, df_frag_L0

# Process and concatenate the DataFrames
files_info = [
    ("/Li6/amsd68nMCLi6l1_B1236_NGenQ3/Li6.root", 0, 1, 2, 3),
    ("/Li7/amsd68nMCLi7l1_B1236_NGenQ37/Li7.root", 3, 4, 5, 3),
    ("/Be7/amsd68nMCBe7l1_B1236_NGenQ4/Be7_0_3.root", 6, 7, 8, 4),
    ("/Be9/amsd68nMCBe9l1_B1236_NGenQ49/Be9_0_3.root", 9, 10, 11, 4),
    ("/Be10/amsd68nMCBe10l1_B1236_NGenQ410/Be10_0_3.root", 12, 13, 14, 4),
    ("/B10/amsd68nMCB10l1_B1236_NGenQ5/B10_0_3.root", 15, 16, 17, 5),
    ("/B11/amsd68nMCB11l1_B1236_NGenQ511/B11_0_3.root", 18, 19, 20, 5),
    ("/C/amsd68nMCC12l1_B1236_NGenQ6/C_0_3.root", 21, 22, 23, 6),
    ("/N14/amsd68nMCN14l1_B1236_NGenQ7/N14_0_3.root", 24, 25, 26, 7),
    ("/N15/amsd68nMCN15l1_B1236_NGenQ715/N15_0_3.root", 27, 28, 29, 7),
    ("/O/amsd68nMCO16l1_B1236_NGenQ8/O_0_3.root", 30, 31, 32, 8),
    ("/F/amsd68nMCF19l1_B1236_NGenQ9/F_0_3.root", 33, 34, 35, 9),
    ("/Ne/amsd68nMCNe20l1_B1236_NGenQ10/Ne_0_3.root", 36, 37, 38, 10),
    ("/Na/amsd68nMCNa23l1_B1236_NGenQ11/Na_0_3.root", 39, 40, 41, 11),
    ("/Mg/amsd68nMCMg24l1_B1236_NGenQ12/Mg_0_3.root", 42, 43, 44, 12),
    ("/Al/amsd68nMCAl27l1_B1236_NGenQ13/Al_0_3.root", 45, 46, 47, 13),
    ("/Si/amsd68nMCSi28l1_B1236_NGenQ14/Si_0_3.root", 48, 49, 50, 14),
    ("/P/amsd68nMCP31l1_B1236_NGenQ15/P_0_3.root", 51, 52, 53, 15),
    ("/S/amsd68nMCS32l1_B1236_NGenQ16/S_0_3.root", 54, 55, 56, 16),
    ("/Cl/amsd68nMCCl35l1_B1236_NGenQ17/Cl_0_3.root", 57, 58, 59, 17),
    ("/Ar/amsd68nMCAr36l1_B1236_NGenQ18/Ar_0_3.root", 60, 61, 62, 18),
    ("/K/amsd68nMCK39l1_B1236_NGenQ19/K_0_3.root", 63, 64, 65, 19),
    ("/Ca/amsd68nMCCa40l1_B1236_NGenQ20/Ca_0_3.root", 66, 67, 68, 20),
    ("/Sc/amsd68nMCSc45l1_B1236_NGenQ21/Sc_0_3.root", 69, 70, 71, 21),
    ("/Ti/amsd68nMCTi48l1_B1236_NGenQ22/Ti_0_3.root", 72, 73, 74, 22),
    ("/V/amsd68nMCV51l1_B1236_NGenQ23/V_0_3.root", 75, 76, 77, 23),
    ("/Cr/amsd68nMCCr52l1_B1236_NGenQ24/Cr_0_3.root", 78, 79, 80, 24),
    ("/Mn/amsd68nMCMn55l1_B1236_NGenQ25/Mn_0_3.root", 81, 82, 83, 25),
    ("/Fe/amsd68nMCFe56l1_B1236_NGenQ26/Fe_0_3.root", 84, 85, 86, 26),
    ("/Ni/amsd68nMCNi58l1_B1236_NGenQ28/Ni_0_3.root", 87, 88, 89, 28),
]

for file_info in files_info:
    file_path = f"{path0}{file_info[0]}"
    df_frag, df_non_frag, df_frag_L0 = process_tree(file_path, file_info[1], file_info[2], file_info[3])
    df_non_frag = df_non_frag[df_non_frag['tk_qin0[2]']>0]
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
    file["t1"] = df_all
    print("file created: ",f"{file_path}_rig_{rigidity_low}_{rigidity_up}.root")
    del df_all, file
    gc.collect()
