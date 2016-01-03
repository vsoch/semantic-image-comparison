#!/usr/bin/python
from glob import glob
import pickle
import os

base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison"

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
    if queue_count < 1000:
        i,image = pairs_to_run.pop(0)
        node = likelihood_pickles[i]
        group = pickle.load(open(node,"rb"))
        image_id = os.path.split(image)[1].replace(".nii.gz","")
        output_pkl = "%s/%s_%s.pkl" %(scores_folder,group["nid"],image_id)
        if not os.path.exists(output_pkl):
            print ".job/oc_%s_%s.job" %(i,image_id)
            filey = ".job/oc_%s_%s.job" %(i,image_id)
            filey = open(filey,"w")
            filey.writelines("#!/bin/bash\n")
            filey.writelines("#SBATCH --job-name=%s\n" %(image_id))
            filey.writelines("#SBATCH --output=.outs/%s_%s.out\n" %(i,image_id))
            filey.writelines("#SBATCH --error=.outs/%s_%s.err\n" %(i,image_id))
            filey.writelines("#SBATCH --time=2-00:00\n")
            filey.writelines("#SBATCH --mem=64000\n")
            filey.writelines("python /home/vsochat/SCRIPT/python/brainmeta/ontological_comparison/cluster/classification-framework/3.calculate_reverse_inference.py %s %s %s %s" %(image, node, output_pkl, tables_folder))
            filey.close()
            os.system("sbatch -p russpold " + ".job/oc_%s_%s.job" %(i,image_id))


# This will produce data for a LOO cross validation procedure:
# For each image node (defined with the likelihood group pickles above)
#     For each image: select him to leave out
#     With the remaining images, calculate likelihood tables, prior, and, RI for the query image
#     Condition A: With actual labels, do reverse inference procedure for correct/real tags. For each query image:
# We want to save a pickle/data object with:
# result["id"]: "trm*"
# result["prior"]: {"in": 0.98, "out": 0.02}
# result["query_RI"]: panda data frame with:
#               reverse_inference_score  bayes_factor
#  queryimage1
#  queryimage2
#  queryimage3

# Then for analysis:
# For each image node (defined with the likelihood group pickles above)
#     For each image: select him to leave out
#     Condition A: load scores of image for "correct/real" tags
#     Condition B: randomly select incorrect tags, load scores, compare
#     For each of the above, calculate a CE score, and compare distributions.
#     We would want "correct" tags to have higher scores
# This means that we need, for each node: to save a reverse inference score for ALL query images in the databases (against the node) and to save the priors values (not including the image) and then a reverse inference score.
