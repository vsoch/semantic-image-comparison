#!/usr/bin/python
from pyneurovault import api
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
update = "%s/update" %base
output_folder = "%s/pls" %update  # any kind of tsv/result file
results = "%s/results" %update  # any kind of tsv/result file
old_results = "%s/results" %base  # any kind of tsv/result file

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %old_results
images = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
image_lookup = "%s/image_nii_lookup.pkl" %update

# Image metadata with number of subjects included
contrast_file = "%s/filtered_contrast_images.tsv" %results
#image_df = api.get_images(pks=images.index.tolist())
#image_df.to_csv(contrast_file,sep="\t")

method_type = "encoding"
output_folder = "%s/%s" %(output_folder,method_type)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

number_components = [2,3,4,5,6,7,8,9,10]

for number_c in number_components:
    output_folder_c = "%s/%s" %(output_folder,number_c)
    if not os.path.exists(output_folder_c):
        os.mkdir(output_folder_c)
    for image1_holdout in images.index.tolist():
        print "Parsing %s" %(image1_holdout)
        for image2_holdout in images.index.tolist():
            if (image1_holdout != image2_holdout) and (image1_holdout < image2_holdout):
                output_file = "%s/%s_%s_pls.pkl" %(output_folder_c,image1_holdout,image2_holdout)
                if not os.path.exists(output_file):
                    job_id = "%s_%s_%s" %(image1_holdout,image2_holdout,output_folder_c)
                    filey = ".job/pls_%s.job" %(job_id)
                    filey = open(filey,"w")
                    filey.writelines("#!/bin/bash\n")
                    filey.writelines("#SBATCH --job-name=%s\n" %(job_id))
                    filey.writelines("#SBATCH --output=.out/%s.out\n" %(job_id))
                    filey.writelines("#SBATCH --error=.out/%s.err\n" %(job_id))
                    filey.writelines("#SBATCH --time=2-00:00\n")
                    filey.writelines("#SBATCH --mem=32000\n")
                filey.writelines("python %s_pls.py %s %s %s %s %s %s %s" %(method_type,image1_holdout, image2_holdout, output_file, labels_tsv, image_lookup, contrast_file, number_components))
                filey.close()
                os.system("sbatch -p russpold --qos russpold " + ".job/pls_%s.job" %(job_id))
