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
        
    def selection_bounds_based(self, variable_name, lower_bound, upper_bound):
        """Returns data from dataframe by applying selection on a variable"""
        df = self._get_dataframe()
        try:
            selected_df = df[(df[variable_name] >= lower_bound) & (df[variable_name] < upper_bound)]
            del df
            gc.collect()
            return selected_df
        except KeyError as e:
            raise KeyError(f"Variable not found in dataframe: {variable_name}") from e
        
    def fragmentation_selection(self, layer_info, frag_option, variable_name, lower_bound, upper_bound, df_option=None):
        '''
        This method applies MC selection inside detector parts to check
        whether a nuclei has fragmented or not inside this part

        Args:
            layer_info (int)        : The number of layers to check for
            frag_option (string)    : fragmented or non-fragmented options

        Returns:
            df (Pandas.Dataframe)   : A subset of df after the application of selection
        '''
        frag_separation_var = [f"mevmom1[{i}]" for i in range(21)]
        
        df = self.selection_bounds_based(variable_name, lower_bound, upper_bound)
        
        if frag_option == 'non-fragmented':
            selected_df = df[df[frag_separation_var[layer_info]] > 0]
        elif frag_option == 'fragmented':
            selected_df = df[df[frag_separation_var[layer_info]] < 0]
        else:
            raise ValueError("Invalid frag_option value. Use 'fragmented' or 'non-fragmented'.")
        del df
        gc.collect()
        return selected_df
    
            
    def labeled(self, label_frag, label_non_frag, layer_info, variable_name, lower_bound, upper_bound, df_option=None):
        '''
        The function applies the fragmentation_selection function and labels the data

        Args:
            label_frag (int)        : The label for the fragmented data
            label_non_frag (int)    : The label for the non-fragmented data

        Returns:
            df_frag (Pandas.Dataframe)    : A subset of df after the application of fragmentation_selection with label "label_frag"
            df_non_frag (Pandas.Dataframe): A subset of df after the application of fragmentation_selection with label "label_non_frag"
        '''
        df_frag = self.fragmentation_selection(layer_info, 'fragmented', variable_name, lower_bound, upper_bound, df_option)
        df_non_frag = self.fragmentation_selection(layer_info, 'non-fragmented', variable_name, lower_bound, upper_bound, df_option)
        df_frag['label'] = label_frag
        df_non_frag['label'] = label_non_frag
        gc.collect()
        return df_frag, df_non_frag
