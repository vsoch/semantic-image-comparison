#!/usr/bin/python
from random import shuffle
from glob import glob
import sys
import numpy
import pandas
import os

base = sys.argv[1]
data = "%s/data" %base
node_folder = "%s/groups" %data
output_folder = "%s/classification_null" %base
results = "%s/results" %base  # any kind of tsv/result file

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
images = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Generate a list of all possible pairs to randomly sample from
image_pairs = []
for image1_holdout in images.index.tolist():
    for image2_holdout in images.index.tolist():
        if (image1_holdout != image2_holdout) and (image1_holdout < image2_holdout):
            image_pairs.append("%s|%s" %(image1_holdout,image2_holdout))

#len(image_pairs)
#4186

# We want approximately 1/4 for each group
group_size = int(numpy.floor(len(image_pairs)/4.0))

# SHERLOCK CLUSTER (slurm environment) #################################

# Here is a function for writing and running the job script
def run_job(group_list,iter_number,group_number):
    job_id = "%s_%s" %(iter_number,group_number)
    output_file = "%s/%s_predict.pkl" %(output_folder,job_id)
    group_list = ",".join(group_list)
    if not os.path.exists(output_file):
        filey = ".job/null_%s.job" %(job_id)
        filey = open(filey,"w")
        filey.writelines("#!/bin/bash\n")
        filey.writelines("#SBATCH --job-name=%s\n" %(job_id))
        filey.writelines("#SBATCH --output=.out/%s.out\n" %(job_id))
        filey.writelines("#SBATCH --error=.out/%s.err\n" %(job_id))
        filey.writelines("#SBATCH --time=1-00:00\n")
        filey.writelines("#SBATCH --mem=12000\n")
        filey.writelines('python 3.calculate_null.py "%s" %s %s %s' %(group_list, node_folder, output_file, labels_tsv))
        filey.close()
        os.system("sbatch -p russpold " + ".job/null_%s.job" %(job_id))

# We want to start off with 250 iterations, meaning splitting ~4000 images into four groups, so we get 1000 jobs in queue at once, each job should take ~19 hours.
for i in range(250):
    image_choices = image_pairs[:]
    shuffle(image_choices)
    # Split into four groups of images
    group1 = image_choices[0:group_size]
    group2 = image_choices[group_size:group_size*2]
    group3 = image_choices[group_size*2:group_size*3]
    group4 = image_choices[group_size*3:]
    run_job(group1,i,1)
    run_job(group2,i,2)
    run_job(group3,i,3)
    run_job(group4,i,4)

# TACC CLUSTER (launch with slurm environment) ############################
# The analysis was not done here, this is provided as an option only

# Here is a function for outputting all commands to a single file (for launch environment)
def write_job(group_list,iter_number,run,group_number,filey):
    job_id = "%s_%s_%s" %(run,iter_number,group_number)
    output_file = "%s/%s_predict.pkl" %(output_folder,job_id)
    group_list = ",".join(group_list)
    if not os.path.exists(output_file):
        filey.writelines('python 3.calculate_null.py "%s" %s %s %s\n' %(group_list, node_folder, output_file, labels_tsv))


for run in range(10):
    job_file = ".job/image_comparison_%s.job" %run
    filey = open(job_file,"w")
    for i in range(100):
        image_choices = image_pairs[:]
        shuffle(image_choices)
        # Split into four groups of images
        group1 = image_choices[0:group_size]
        group2 = image_choices[group_size:group_size*2]
        group3 = image_choices[group_size*2:group_size*3]
        group4 = image_choices[group_size*3:]
        write_job(group1,run,i,1,filey)
        write_job(group2,run,i,2,filey)
        write_job(group3,run,i,3,filey)
        write_job(group4,run,i,4,filey)
    filey.close()
    os.system("launch -s %s --runtime=2-00:00 --jobname='semantic-comparison-%s' -N 1" %(job_file,run))
