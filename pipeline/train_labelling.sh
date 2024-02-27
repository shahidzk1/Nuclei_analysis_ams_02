#!/bin/bash
#source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh

python3 -m virtualenv myvenv
source myvenv/bin/activate
pip install optuna==3.2.0
pip install numpy
pip install pandas
pip install uproot==4.3.7
pip install xgboost==2.0.2
pip install scikit-learn==1.2.2

log_dir=/afs/cern.ch/user/k/khansh/private/python/labelling/log/
output_dir=/afs/cern.ch/user/k/khansh/private/python/labelling/output/
error_dir=/afs/cern.ch/user/k/khansh/private/python/labelling/errors


mkdir -p ${log_dir} ${output_dir} ${error_dir}
cd /afs/cern.ch/user/k/khansh/private/python/labelling/
python3 --version

input_json_path='/afs/cern.ch/user/k/khansh/private/python/labelling/files_info_labelling_updated.json'
output_json_path='/afs/cern.ch/user/k/khansh/private/python/labelling/output_labelling.json'
python3 labelling.py $input_json_path $output_json_path
python3 train_data.py $output_json_path
#python3 hpo_xgb.py
