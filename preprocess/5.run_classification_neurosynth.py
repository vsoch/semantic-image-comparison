#!/usr/bin/python
from glob import glob
import sys
import pickle
import numpy
import pandas
import os

base = sys.argv[1]
decode_folder = "%s/decode" %base
output_folder = "%s/results" %decode_folder

# Images by Concepts data frame
X_pickle = "%s/neurosynth_X.pkl" %decode_folder
Y_pickle = "%s/neurosynth_Y.pkl" %decode_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 28K voxels for 11K pmids
X = pickle.load(open(X_pickle,"rb"))

# Generate a list of [start,stop] indices for holdout groups, each ~1% of total 
image_groups = []
group_size =  int(numpy.floor(0.01*X.shape[0]))
pmids = X.index.tolist()
start = 0
end = start + group_size
while end < len(pmids)-1:
    if start + group_size <= len(pmids)-1:
        end = start + group_size
    else:
        end = len(pmids)-1
    image_groups.append([start,end])
    start = end 

#len(image_pairs)
# 101

for holdouts in image_groups:
    holdout_start = holdouts[0]
    holdout_end = holdouts[1]
    print "Parsing holdout from %s to %s" %(holdout_start,holdout_end)
    output_file = "%s/%s_%s_predict.pkl" %(output_folder,holdout_start,holdout_end)
    if not os.path.exists(output_file):
        job_id = "%s_%s" %(holdout_start,holdout_end)
        filey = ".job/class_%s.job" %(job_id)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=%s\n" %(job_id))
        filey.writelines("#SBATCH --output=.out/%s.out\n" %(job_id))
        filey.writelines("#SBATCH --error=.out/%s.err\n" %(job_id))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=64000\n")
        filey.writelines("python 5.classification_neurosynth.py %s %s %s %s %s" %(holdout_start, holdout_end, X_pickle, Y_pickle, output_file))
        filey.close()
        os.system("sbatch -p russpold --qos=russpold " + ".job/class_%s.job" %(job_id))
