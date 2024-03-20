import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor
import gc
import logging

class TreeHandler:
    def __init__(self, file_path, tree_name, n_jobs=None):
        """"
This class imports a root tree file and can convert to a dataframe. It can also apply selections on the data and store it as a dataframe
use case:
path = f"/path/of/the/root/file"
df_O_frag_man, df_O_non_frag_man = TreeHandler(f"{path}/O.root",
                                               "t1",2).labeled(20, 21, 13, 'tk_rigidity_0[2][2]', rigidity_low, rigidity_up)
Args:
    file_path_name (string)   : The address of the root file and its name
    tree_name      (string)   : The name of the TTree object inside the file
    n_jobs         (int)      : The number of parallel jobs to run    
"""
        self.file_path = file_path
        self.tree_name = tree_name
        self.n_jobs = n_jobs or 1
        self.executor = None
        self._file = None         
        
    def __enter__(self):
        self.executor = ThreadPoolExecutor(self.n_jobs)
        self._open_file()
        return self
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
            
    def _open_file(self):
        """Imports a root file into python"""
        try:
            self._file = uproot.open(f"{self.file_path}:{self.tree_name}",
                                     decompression_executor=self.executor,
                                     interpretation_executor=self.executor)
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}") from e
        
    def _get_dataframe(self):
        """Converts the tree in the file to a dataframe"""
        if self._file is None:
            self._open_file()
        try:
            df = self._file.arrays(library="pd", decompression_executor=self.executor, interpretation_executor=self.executor)
            gc.collect()
            return df
        except Exception as e:
            raise ValueError(f"Error converting tree to dataframe: {str(e)}") from e
        
    def selection_bounds_based(self, variable_name, lower_bound, upper_bound, df=None):
        """Returns data from dataframe by applying selection on a variable"""
        df = self._get_dataframe() if df is None else df
        try:
            selected_df = df[(df[variable_name] >= lower_bound) & (df[variable_name] < upper_bound)]
            del df
            gc.collect()
            return selected_df
        except KeyError as e:
            raise KeyError(f"Variable not found in dataframe: {variable_name}") from e                   
        
    def fragmentation_selection(self, layer_info , frag_option, variable_name, lower_bound, upper_bound, charge_sel=None,
                                var_inner_trcker_charge=None,
                                nuclei_charge=None,
                                nuclei_charge_low=None,
                                nuclei_charge_up=None,
                                df_option=None):
        '''
        The function applies the MC selection on detector parts to check whether a nuclei has fragmented or not. 

        Args:
            layer_info (int)        : The number of layers to check for
            frag_option (string)    : fragmented or non-fragmented options

        Returns:
            df (Pandas.Dataframe)   : A subset of df after the application of selection
        '''
        frag_separation_var = [f"mevmom1[{i}]" for i in range(21)]
        
        df = self.selection_bounds_based(variable_name, lower_bound, upper_bound)
        
        if frag_option == 'non-fragmented':
            # the initial implementation
            selected_df = df[df[frag_separation_var[layer_info]] > 0]
            if charge_sel == True:
                # updated because at the end the var_inner_trcker_charge will be used
                # so select also candidates to be fragmented or not on this basis. 
                # For example, if GEANT info gives that it has fragmented but its charge lies in the z charge 3.5 sigma
                # then it is a non-fragmented nucleus
                selected_df = selected_df[ (selected_df[var_inner_trcker_charge]>= (nuclei_charge-3.5*(selected_df[var_inner_trcker_charge].std())))
                                         & (selected_df[var_inner_trcker_charge]<= (nuclei_charge+3.5*(selected_df[var_inner_trcker_charge].std())))]
                selected_df1 = df[ (df[frag_separation_var[layer_info]] < 0) &
                                 (selected_df[var_inner_trcker_charge]>= (nuclei_charge-2*(selected_df[var_inner_trcker_charge].std()))) &
                                         (selected_df[var_inner_trcker_charge]<= (nuclei_charge+2*(selected_df[var_inner_trcker_charge].std())))]
                selected_df = selected_df.reset_index(drop=True)
                selected_df1 = selected_df1.reset_index(drop=True)
                selected_df = pd.concat([selected_df, selected_df1])
                
        elif frag_option == 'fragmented':
            selected_df = df[df[frag_separation_var[layer_info]] < 0]
            if charge_sel == True:
                if layer_info==0:
                    selected_df = selected_df[ (selected_df[var_inner_trcker_charge]<= (nuclei_charge-0.1*(selected_df[var_inner_trcker_charge].std()))) ]
                if layer_info >0:
                    selected_df1 = selected_df[selected_df[frag_separation_var[0]] < 0]
                    selected_df1 = selected_df1[ (selected_df1[var_inner_trcker_charge]<= (nuclei_charge-0.1*(selected_df1[var_inner_trcker_charge].std()))) ]
                    selected_df = selected_df[ (selected_df[var_inner_trcker_charge]<= (nuclei_charge-0.1*(selected_df[var_inner_trcker_charge].std()))) ]
                    mask = selected_df.isin(selected_df1.to_dict('list')).all(axis=1)
                    mask = ~mask
                    selected_df = selected_df[mask]
        else:
            raise ValueError("Invalid frag_option value. Use 'fragmented' or 'non-fragmented'.")
        del df
        gc.collect()
        return selected_df
    
            
    def labeled(self, label_frag,
                label_non_frag,
                layer_info, variable_name,
                lower_bound,
                upper_bound,charge_sel=None,
                var_inner_trcker_charge=None,
                nuclei_charge=None,
                df_option=None):
        '''
        The function labels the data by using fragmentation_selection function.  

        Args:
            label_frag (int)        : The label for the fragmented data
            label_non_frag (int)    : The label for the non-fragmented data

        Returns:
            df_frag (Pandas.Dataframe)    : A subset of df after the application of fragmentation_selection with label "label_frag"
            df_non_frag (Pandas.Dataframe): A subset of df after the application of fragmentation_selection with label "label_non_frag"
        '''
        # variables definition
        charge_sel = charge_sel or False
        var_inner_trcker_charge = var_inner_trcker_charge or 'tk_qin0[2]'
        nuclei_charge= nuclei_charge or 9
        #application of fragmentation_selection
        df_frag = self.fragmentation_selection(layer_info, 'fragmented', variable_name, lower_bound, upper_bound, charge_sel=charge_sel,
                                               var_inner_trcker_charge=var_inner_trcker_charge,
                                               nuclei_charge = nuclei_charge)
        df_non_frag = self.fragmentation_selection(layer_info, 'non-fragmented', variable_name, lower_bound, upper_bound, charge_sel=charge_sel,
                                               var_inner_trcker_charge=var_inner_trcker_charge,
                                               nuclei_charge = nuclei_charge)
        #labelling
        df_frag['label'] = label_frag
        df_non_frag['label'] = label_non_frag
        gc.collect()
        return df_frag, df_non_frag
    
    def tight_nuclei_sig_selec(self,z, var_L1_charge=None,
                           L1_charge_low=None,L1_charge_up=None,
                           var_inner_trcker_charge=None,
                           inner_trcker_charge_low=None,
                           inner_trcker_charge_up = None,
                           var_upper_tof_1=None,
                           var_upper_tof_2=None,
                           upper_tof_low=None,
                           upper_tof_up=None,
                           var_low_tof_1 = None,
                           var_low_tof_2 = None,
                           low_tof_low=None,
                           low_tof_up=None,
                           var_scnd_trcker_trck_bit_y=None,
                           var_scnd_trcker_trck_bit_xy=None,
                           var_2nd_trckr_trk_rig = None,
                              ):
        """"
            This function selects a nuclei species by
            applying tight manual selections based on the 
            charge of a nuclei species for the AMS-02 Pass8 data.
            Args:
                df                  (pd.core.frame.DataFrame)    : Pass-8 data converted to flat trees
                z                   (int)                        : charge of nuclei
        """
        #variables
        var_L1_charge = var_L1_charge or 'tk_qln0_0_2'
        var_inner_trcker_charge = var_inner_trcker_charge or 'tk_qin0_2'
        var_upper_tof_1 = var_upper_tof_1 or 'tof_ql_0'
        var_upper_tof_2 = var_upper_tof_2 or 'tof_ql_1'
        var_upper_tof_avg = 'tof_up_avg'
        var_low_tof_1 = var_low_tof_1 or 'tof_ql_2'
        var_low_tof_2 = var_low_tof_2 or 'tof_ql_3'
        var_low_tof_avg = 'tof_low_avg'
        var_scnd_trcker_trck_bit_y = var_scnd_trcker_trck_bit_y or 'betah2hby_sum'
        var_scnd_trcker_trck_bit_xy = var_scnd_trcker_trck_bit_xy or 'betah2hbxy_sum'
        var_2nd_trckr_trk_rig = var_2nd_trckr_trk_rig or 'betah2r'
        #selections definition
        L1_charge_low = L1_charge_low or (z - 0.3)
        L1_charge_up  = L1_charge_up or (z + 0.3)
        upper_tof_low = upper_tof_low or (z-(0.625-0.0225*(z-9)))
        upper_tof_up = upper_tof_up or z + 1
        low_tof_low = low_tof_low or (z-0.625-0.0225*(z-9))
        low_tof_up = low_tof_up or z-0
        inner_trcker_charge_low = inner_trcker_charge_low or (z - 0.3)
        inner_trcker_charge_up  = inner_trcker_charge_up or (z+0.3)
        #selections applications
        df = self.selection_bounds_based(var_L1_charge,
                                         L1_charge_low,
                                         L1_charge_up)
        df = self.selection_bounds_based(var_inner_trcker_charge,
                                         inner_trcker_charge_low,
                                         inner_trcker_charge_up, df)
        #new variable definition
        df[var_upper_tof_avg] = (df[var_upper_tof_1]+df[var_upper_tof_2])/2
        df[var_low_tof_avg]   = (df[var_low_tof_1]+df[var_low_tof_2])/2
        if z>=9:
            df = df[ (df[var_upper_tof_avg]>= upper_tof_low) & (df[var_upper_tof_avg]<=upper_tof_up)]
            df = df[ df[var_low_tof_avg]> low_tof_low]
        else:
            df = df[(df[var_upper_tof_avg]>= z-0.6) & (df[var_upper_tof_avg]<=upper_tof_up) ]
            df = df[ df[var_low_tof_avg] > z-0.6]
        df = df[(df[var_scnd_trcker_trck_bit_y]<3) & (df[var_scnd_trcker_trck_bit_xy]<5) & (df[var_2nd_trckr_trk_rig]<0.5)]
        return df
    
    def nuclei_fragments_selec(self, z,
                               var_L1_charge=None,
                               L1_charge_low=None,
                               L1_charge_up=None,
                               var_inner_trcker_charge=None,
                               inner_trcker_charge_low=None):
        """"
            This function selects fragments of a nuclei species by
            applying manual selections based on the charge
            of a nuclei species for the AMS-02 Pass8 data.
            Args:
                df                  (pd.core.frame.DataFrame)    : Pass-8 data converted to flat trees
                z                   (int)                        : charge of nuclei
        """
        var_L1_charge = var_L1_charge or 'tk_qln0_0_2'
        var_inner_trcker_charge = var_inner_trcker_charge or 'tk_qin0_2'
        L1_charge_low = L1_charge_low or (z - 0.5)
        L1_charge_up  = L1_charge_up or (z + 0.5)
        inner_trcker_charge_low = inner_trcker_charge_low or (z - 0.5)
        df = self.selection_bounds_based(var_L1_charge,
                                         L1_charge_low,
                                         L1_charge_up)
        df = df[df[var_inner_trcker_charge]<=inner_trcker_charge_low]
        return df
