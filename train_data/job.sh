universe                = vanilla
executable              = train.sh
requirements = 
#MY.WantOS = "el9"

log_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/log/
output_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/output/
error_dir=/afs/cern.ch/user/k/khansh/private/python/create_train_data/errors

log                     = $(log_dir)/test.log
output                  = $(output_dir)/outfile.$(Cluster).$(Process).out
error                   = $(error_dir)/errors.$(Cluster).$(Process).err
should_transfer_files   = NO
#+testJob = True
+JobFlavour = "tomorrow"
#espresso     = 20 minutes
#microcentury = 1 hour
#longlunch    = 2 hours
#workday      = 8 hours
#tomorrow     = 1 day
#testmatch    = 3 days
#nextweek     = 1 week
# Remaining job parameters
RequestCpus = 20
queue 1
