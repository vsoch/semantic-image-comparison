#!/usr/bin/python
from glob import glob
import pickle
import sys
import os

base = sys.argv[1]
queue_limit = 1000

data = "%s/data" %base        # mostly images
likelihood_pickles = glob("%s/likelihood/*.pkl" %(data))
scores_folder = "%s/individual_scores" %(data)     # output folder for individual scores
tables_folder = "%s/likelihood/tables" %(data)

if not os.path.exists(scores_folder):
    os.mkdir(scores_folder)

# Make a ridiculosly long list of likelihood pickle/image id pairs
pairs = []
for i in range(0,len(likelihood_pickles)):
    node = likelihood_pickles[i]
    group = pickle.load(open(node,"rb"))
    all_images = group["in"] + group["out"]
    for image in all_images:
        pairs.append([i,image])

pairs_to_run = pairs[:]

while len(pairs_to_run) > 0:
    queue_count = int(os.popen("squeue -u vsochat | wc -l").read().strip("\n"))
    if queue_count < queue_limit:
        i,image = pairs_to_run.pop(0)
        node = likelihood_pickles[i]
        group = pickle.load(open(node,"rb"))
        image_id = os.path.split(image)[1].replace(".nii.gz","")
        output_pkl = "%s/%s_%s.pkl" %(scores_folder,group["nid"],image_id)
        if not os.path.exists(output_pkl):
            print ".jobs/oc_%s_%s.job" %(i,image_id)
            filey = ".jobs/oc_%s_%s.job" %(i,image_id)
            filey = open(filey,"w")
            filey.writelines("#!/bin/bash\n")
            filey.writelines("#SBATCH --job-name=%s\n" %(image_id))
            filey.writelines("#SBATCH --output=.out/%s_%s.out\n" %(i,image_id))
            filey.writelines("#SBATCH --error=.out/%s_%s.err\n" %(i,image_id))
            filey.writelines("#SBATCH --time=2-00:00\n")
            filey.writelines("#SBATCH --mem=64000\n")
            filey.writelines("python 3.calculate_reverse_inference.py %s %s %s %s" %(image, node, output_pkl, tables_folder))
            filey.close()
            os.system("sbatch -p russpold " + ".job/oc_%s_%s.job" %(i,image_id))
