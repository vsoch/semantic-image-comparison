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


# Here is to run with all images to produce a regression parameter data frame to make images from
X = pickle.load(open(X_pickle,"rb"))
Y = pickle.load(open(Y_pickle,"rb"))

# Only include labels that have at least 10 occurences
counts = Y.sum()
grten = counts[counts>=10].index.tolist()
Y = Y.loc[:,grten]

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

###################################################################################
# TRAINING
###################################################################################

regression_params = pandas.DataFrame(0,index=X.columns,columns=Y.columns)

# Keep count of how many voxels we couldn't build a model for (labels of all one type)
missed = 0

print "Training voxels..."
for voxel in X.columns:
    if len(X.loc[:,voxel].unique())>0:
        # We can only build a model for voxels with 0/1 data
        if len(X.loc[:,voxel].unique()) > 1:
            yy = X.loc[:,voxel].tolist() 
            Xtrain = Y.loc[:,:]
            clf = linear_model.LogisticRegression()
            clf.fit(Xtrain,yy)
            regression_params.loc[voxel,:] = clf.coef_[0]
        else:
            missed +=1

regression_params.to_csv("%s/regression_params.tsv" %decode_folder,sep="\t")

brainmap_folder = "%s/regparam_maps" %decode_folder
if not os.path.exists(brainmap_folder):
    os.mkdir(brainmap_folder)

# Write a brain map for each
import nibabel
for concept in regression_params.columns.tolist():
    brainmap = regression_params[concept]
    nii_data = numpy.zeros(standard_mask.shape)
    nii_data[standard_mask.get_data()!=0] = brainmap.tolist()
    nii = nibabel.Nifti1Image(nii_data,affine=standard_mask.get_affine())
    nibabel.save(nii,"%s/%s_regparams.nii.gz" %(brainmap_folder,concept))


###################################################################################
# CONFUSION
###################################################################################

# Load a result file
