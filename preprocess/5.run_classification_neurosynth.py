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

# 28K images by 11K concepts
X = pickle.load(open(X_pickle,"rb"))

# Generate a list of all possible pairs to randomly sample from
image_pairs = []
for pmid1_holdout in X.index.tolist():
    for pmid2_holdout in X.index.tolist():
        if (pmid1_holdout != pmid2_holdout) and (pmid1_holdout < pmid2_holdout):
            image_pairs.append([pmid1_holdout,pmid2_holdout])

#len(image_pairs)
#

for holdouts in image_pairs:
    pmid1_holdout = image_pairs[0]
    pmid2_holdout = image_pairs[1]
    print "Parsing %s and %s" %(pmid1_holdout,pmid2_holdout)
    output_file = "%s/%s_%s_predict.pkl" %(output_folder,pmid1_holdout,pmid2_holdout)
    if not os.path.exists(output_file):
        job_id = "%s_%s" %(pmid1_holdout,pmid2_holdout)
        filey = ".job/class_%s.job" %(job_id)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=%s\n" %(job_id))
        filey.writelines("#SBATCH --output=.out/%s.out\n" %(job_id))
        filey.writelines("#SBATCH --error=.out/%s.err\n" %(job_id))
        filey.writelines("#SBATCH --time=2-00:00\n")
        filey.writelines("#SBATCH --mem=32000\n")
        filey.writelines("python 5.classification_neurosynth.py %s %s %s %s %s" %(pmid1_holdout, pmid2_holdout, X_pickle, Y_pickle, output_file))
        filey.close()
        os.system("sbatch -p russpold " + ".job/class_%s.job" %(job_id))
