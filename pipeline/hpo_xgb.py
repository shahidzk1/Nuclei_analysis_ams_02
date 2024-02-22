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

rigidity_low = 2.15
rigidity_up  = 1200
df_all = uproot.open(f"/eos/user/k/khansh/data/train_test_Li_to_Ni_without_chi2_89_classes_rigi_{rigidity_low}_{rigidity_up}.root:t1").arrays(library='pd', 
                                                            decompression_executor=executor, interpretation_executor=executor)
print("MC file uploaded")
n1 = 10000
df_all = df_all.groupby('label', group_keys=False).apply(lambda x: x.sample(n=n1))
df_all.reset_index(drop=True, inplace=True)
#df_all = df_all[~df_all.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
#df_all.reset_index(drop=True, inplace=True)

#new lines added from here to include data signal

signal_all = uproot.open(f"/eos/user/k/khansh/data_iss_pass8/signal.root:t1").arrays(library='pd', 
                                                            decompression_executor=executor, interpretation_executor=executor)

df = add_data_signal_to_mc(df_all, signal_all)
del df_all
df['trd_nhitk/trd_onhitk']= df['trd_nhitk']/df['trd_onhitk']
#new lines end above

new_cuts = ['anti_nhit', 'betah2hbxy_sum', 'betah2hby_sum', 'betah2p','betah2q/tk_qin0_2', 'betah2r/tk_rigidity_0_2_2', 'nbetah',
       'ntrack', 'tk_exdis_0', 'tk_exqln_0_0_0', 'tk_exqln_0_0_1', 'tk_exqln_0_0_2', 
            'tk_exqls_0_0', 'tk_exqls_0_1', 'tk_exqls_0_2','tk_exqls_0_3', 'tk_exqls_0_4', 'tk_exqls_0_5', 'tk_exqls_0_6','tk_exqls_0_7',
            'tk_iso_0_0', 'tk_iso_0_1', 'tk_iso_1_0', 'tk_iso_1_1', 'tk_iso_2_0', 'tk_iso_2_1', 'tk_iso_3_0','tk_iso_3_1', 'tk_iso_4_0', 'tk_iso_4_1', 'tk_iso_5_0',
       'tk_iso_5_1', 'tk_iso_6_0', 'tk_iso_6_1', 'tk_iso_7_0','tk_iso_7_1',
            'tk_oel_0', 'tk_oel_1', 'tk_oel_2', 'tk_oel_3','tk_oel_4', 'tk_oel_5', 'tk_oel_6', 'tk_oel_7',
            'tk_ohitl_0','tk_ohitl_1', 'tk_ohitl_2', 'tk_ohitl_3', 'tk_ohitl_4','tk_ohitl_5', 'tk_ohitl_6', 'tk_ohitl_7',
            'tk_oq_0/tk_qin0_2', 'tk_oq_1/tk_qin0_2', 'tk_qin0_0', 'tk_qin0_1', 'tk_qin0_2',
            'tk_qln0_0_0', 'tk_qln0_0_1', 'tk_qln0_0_2', 'tk_qln0_1_0','tk_qln0_1_1', 'tk_qln0_1_2', 'tk_qln0_2_0', 'tk_qln0_2_1',
       'tk_qln0_2_2', 'tk_qln0_3_0', 'tk_qln0_3_1', 'tk_qln0_3_2', 'tk_qln0_4_0', 'tk_qln0_4_1', 'tk_qln0_4_2', 'tk_qln0_5_0',
       'tk_qln0_5_1', 'tk_qln0_5_2', 'tk_qln0_6_0', 'tk_qln0_6_1', 'tk_qln0_6_2', 'tk_qln0_7_0', 'tk_qln0_7_1', 'tk_qln0_7_2',
       'tof_oncl',  'tof_oq_0_0/tof_ql_0', 'tof_oq_0_1/tof_ql_0','tof_oq_1_0/tof_ql_1', 'tof_oq_1_1/tof_ql_1','tof_oq_2_0/tof_ql_2', 'tof_oq_2_1/tof_ql_2',
       'tof_oq_3_0/tof_ql_3', 'tof_oq_3_1/tof_ql_3', 'tof_ql_0',     'tof_ql_1', 'tof_ql_2', 'tof_ql_3', 'tk_qrmn','trd_oampk','trd_nhitk/trd_onhitk']

x = df[new_cuts].copy()
y =pd.DataFrame(df_all['label'], dtype='int8')
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=324, stratify=y)
#del x, y, df_all
del df
gc.collect()

class XGB_Hyp_par_Optimizer:
    def __init__(self, x_train, y_train, n_trials, xgb_hyp_par, with_feature_selection,
                 early_stopping_rounds=None, eval_metric='logloss'):
        self.x_train = x_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.xgb_hyp_par = xgb_hyp_par
        self.with_feature_selection = with_feature_selection
        self.selected_features = None
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric or 'logloss'

    def xgb_objective(self, trial):
        if self.with_feature_selection == "with_fs":
            k_best = trial.suggest_int("k_best", 1, self.x_train.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_best)
            x_train_fs = selector.fit_transform(self.x_train, self.y_train)
            self.selected_features = selector.get_support(indices=True)
        else:
            x_train_fs = self.x_train

        param = {
            'booster': 'gbtree',
            "n_estimators": trial.suggest_int("n_estimators", self.xgb_hyp_par['n_estimators'][0], self.xgb_hyp_par['n_estimators'][1]),
            "alpha": trial.suggest_int("alpha", self.xgb_hyp_par['alpha'][0], self.xgb_hyp_par['alpha'][1]),
            "gamma": trial.suggest_float("gamma", self.xgb_hyp_par['gamma'][0],self.xgb_hyp_par['gamma'][1]),
            "learning_rate": trial.suggest_float("learning_rate", self.xgb_hyp_par['learning_rate'][0], self.xgb_hyp_par['learning_rate'][1], log=True),
            "max_depth": trial.suggest_int("max_depth", self.xgb_hyp_par['max_depth'][0], self.xgb_hyp_par['max_depth'][1]),
            'objective': 'multi:softprob',  'num_class': len(np.unique(self.y_train)),
            'tree_method': 'hist',  'eval_metric': self.eval_metric}
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_fs, self.y_train, test_size=0.2, random_state=42)
        clf = XGBClassifier(**param)
        clf.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=self.early_stopping_rounds, verbose=False)
        cv = StratifiedKFold(n_splits=3)
        scores = cross_val_score(clf, x_train_fs, self.y_train, scoring='f1_macro', cv=cv)
        print(scores)
        return scores.mean()

    def optimize_xgb(self):
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                   sampler=optuna.samplers.TPESampler())
        study.optimize(self.xgb_objective, n_trials=self.n_trials, gc_after_trial=True, n_jobs=1)
        return study

if __name__ == "__main__":
    n_trials = 12
    xgb_hyp_par = { 'n_estimators': (100, 1000), 'alpha': (2, 50),'gamma': (0, 1),'learning_rate': (0.01, 1), 'max_depth': (1, 15)  }
    print("hyperparameters defined")
    xgb_optimizer = XGB_Hyp_par_Optimizer(x, y, n_trials, xgb_hyp_par, "without_fs", early_stopping_rounds=10, eval_metric='auc')
    print("optimization starts")
    study_xgb = xgb_optimizer.optimize_xgb()
    print("optimization ends")
    best_params = study_xgb.best_params
    print("XGBoost Best param:", best_params)

    with open("/afs/cern.ch/user/k/khansh/private/python/hpo_xgb/best_hyperparameters.json", "w") as json_file:
        json.dump(best_params, json_file)
        
    #export EOS_MGM_URL=root://eosuser.cern.ch
    #eos root://eosuser.cern.ch mkdir /eos/user/k/khansh/SWAN_projects/nuclei_analysis/xgb_models/today
    #xrdcp -v /afs/cern.ch/user/k/khansh/private/python/hpo_xgb/best_hyperparameters.json root://eosuser.cern.ch//eos/user/k/khansh/SWAN_projects/nuclei_analysis/xgb_models/today
    #xrdcp -v /afs/cern.ch/user/k/khansh/private/python/hpo_xgb/hpo_xgb.py root://eosuser.cern.ch//eos/user/k/khansh/SWAN_projects/nuclei_analysis/xgb_models/today
    #xrdcp -v /afs/cern.ch/user/k/khansh/private/python/hpo_xgb/train_xgb.sh root://eosuser.cern.ch//eos/user/k/khansh/SWAN_projects/nuclei_analysis/xgb_models/today
