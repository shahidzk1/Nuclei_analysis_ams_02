#!/bin/bash
#source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh

python3 -m virtualenv myvenv
source myvenv/bin/activate
pip install numpy
pip install pandas
pip install uproot==4.3.7
pip install xgboost==2.0.2
pip install scikit-learn==1.2.2

log_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/log/
output_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/output/
error_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/errors

mkdir -p ${log_dir} ${output_dir} ${error_dir}
cd /afs/cern.ch/user/k/khansh/private/python/create_train_data/
python3 --version
python3 train_data.py