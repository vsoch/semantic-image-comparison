#!/usr/bin/python
from glob import glob
import pickle
import os

# For each concept, (N=140):
#    Select a number G from 1...[total "in" group] as the size of the set to investigate
#        For each image in (entire) "in"set:
#            For some number of iterations:
#                Randomly select G other images for "in" set, calculate P(concept|image)
#                Take mean score of iterations as P(concept|image)


base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison"
data = "%s/data" %base        # mostly images
likelihood_pickles = glob("%s/likelihood/*.pkl" %(data))
scores_folder = "%s/group_size_vary_scores" %(data)     # output folder for group size scores

if not os.path.exists(scores_folder):
    os.mkdir(scores_folder)

# Make a ridiculosly long list of likelihood pickle/image id pairs
pairs = []
for i in range(0,len(likelihood_pickles)):
    node = likelihood_pickles[i]
    group = pickle.load(open(node,"rb"))
    all_images = group["in"] + group["out"]
    for j in range(1,len(group["in"])-1):
        all_images = group["in"] + group["out"]
        for image in all_images:
            pairs.append([i,image,j])

pairs_to_run = pairs[:]

while len(pairs_to_run) > 0:
    queue_count = int(os.popen("squeue -u vsochat | wc -l").read().strip("\n"))
    if queue_count < 1000:
        i,image,j = pairs_to_run.pop(0)
        node = likelihood_pickles[i]
        group = pickle.load(open(node,"rb"))
        # vary the size of the "in" set from 1 to the number of "in" images (j)
        all_images = group["in"] + group["out"]
        image_id = os.path.split(image)[1].replace(".nii.gz","")
        # Output convention is [node]_size_[size]_[image_id].pkl
        run_id = "%s_size_%s_%s" %(group["nid"],j,image_id)
        output_pkl = "%s/%s.pkl" %(scores_folder,run_id)
        if not os.path.exists(output_pkl):        
            filey = ".jobs/ri_%s.job" %(run_id)
            filey = open(filey,"w")
            filey.writelines("#!/bin/bash\n")
            filey.writelines("#SBATCH --job-name=%s\n" %(run_id))
            filey.writelines("#SBATCH --output=.out/%s.out\n" %(run_id))
            filey.writelines("#SBATCH --error=.out/%s.err\n" %(run_id))
            filey.writelines("#SBATCH --time=2-00:00\n")
            filey.writelines("#SBATCH --mem=64000\n")
            filey.writelines("python /home/vsochat/SCRIPT/python/brainmeta/ontological_comparison/cluster/classification-framework/5.explore_group_size.py %s %s %s %s" %(image, node, output_pkl,j))
            filey.close()
            os.system("sbatch -p russpold " + ".jobs/ri_%s.job" %(run_id))
