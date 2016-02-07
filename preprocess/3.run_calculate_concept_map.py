#!/usr/bin/python
from glob import glob
import sys
import pandas
import os

# Classification framework
# for image1 in all images:
#    for image2 in allimages:
#        if image1 != image2:
#            hold out image 1 and image 2, generate regression parameter matrix using other images
#            generate predicted image for image 1 [PR1]
#            generate predicted image for image 2 [PR2]
#            classify image 1 as fitting best to PR1 or PR2
#            classify image 2 as fitting best to PR1 or PR2

base = sys.argv[1]
data = "%s/data" %base
node_folder = "%s/groups" %data
output_folder = "%s/classification" %base  # any kind of tsv/result file
results = "%s/results" %base  # any kind of tsv/result file

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
images = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for image1_holdout in images.index.tolist():
    print "Parsing %s" %(image1_holdout)
    for image2_holdout in images.index.tolist():
        if (image1_holdout != image2_holdout) and (image1_holdout < image2_holdout):
            output_file = "%s/%s_%s_predict.pkl" %(output_folder,image1_holdout,image2_holdout)
            if not os.path.exists(output_file):
                job_id = "%s_%s" %(image1_holdout,image2_holdout)
                filey = ".job/class_%s.job" %(job_id)
                filey = open(filey,"w")
                filey.writelines("#!/bin/bash\n")
                filey.writelines("#SBATCH --job-name=%s\n" %(job_id))
                filey.writelines("#SBATCH --output=.out/%s.out\n" %(job_id))
                filey.writelines("#SBATCH --error=.out/%s.err\n" %(job_id))
                filey.writelines("#SBATCH --time=2-00:00\n")
                filey.writelines("#SBATCH --mem=64000\n")
                filey.writelines("python 3.calculate_concept_map.py %s %s %s %s %s" %(image1_holdout, image2_holdout, node_folder, output_file, labels_tsv))
                filey.close()
                os.system("sbatch -p russpold " + ".job/class_%s.job" %(job_id))
