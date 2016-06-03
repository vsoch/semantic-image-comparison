#!/usr/bin/python

# Classification framework
# for image1 in all images:
#    for image2 in allimages:
#        if image1 != image2:
#            hold out image 1 and image 2, generate regression parameter matrix using other images
#            generate predicted image for image 1 [PR1]
#            generate predicted image for image 2 [PR2]
#            classify image 1 as fitting best to PR1 or PR2
#            classify image 2 as fitting best to PR1 or PR2

from pybraincompare.compare.maths import calculate_correlation
from pybraincompare.compare.mrutils import get_images_df
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.mr.transformation import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from glob import glob
import pickle
import pandas
import nibabel
import sys
import os

base = sys.argv[1]
update = "%s/update" %base
output_folder = "%s/classification" %update  # any kind of tsv/result file
results = "%s/results" %update  # any kind of tsv/result file
output_file = "%s/elasticNet_regressionParams.tsv" %update

# Images by Concepts data frame (NOT including all levels of ontology)
labels_tsv = "%s/concepts_binary_df.tsv" %update
image_lookup = "%s/image_nii_lookup.pkl" %update

# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

# Dictionary to look up image files (4mm)
lookup = pickle.load(open(image_lookup,"rb"))

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# We will save data to dictionary
result = dict()

concepts = X.columns.tolist()

# We will go through each voxel (column) in a data frame of image data
image_paths = lookup.values()
mr = get_images_df(file_paths=image_paths,mask=standard_mask)
image_ids = [int(os.path.basename(x).split(".")[0]) for x in image_paths]
mr.index = image_ids
   
# what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

regression_params = pandas.DataFrame(0,index=mr.columns,columns=concepts)

print "Training voxels..."
for voxel in mr.columns:
    train = mr.index
    Y = mr.loc[train,voxel].tolist()
    Xtrain = X.loc[train,:] 
    # Use regularized regression
    clf = linear_model.ElasticNet(alpha=0.1)
    clf.fit(Xtrain,Y)
    regression_params.loc[voxel,:] = clf.coef_.tolist()

regression_params.to_csv(output_file,sep="\t")

# GENERATE BRAIN IMAGES FOR REGRESSION PARAMS
image_folder = "%s/regression_param_images" %(update)
if not os.path.exists(image_folder):
    os.mkdir(image_folder)

for concept in regression_params.columns.tolist():
    data = regression_params[concept].tolist()
    empty_nii = numpy.zeros(standard_mask.shape)
    empty_nii[standard_mask.get_data()!=0] = data
    nii = nibabel.Nifti1Image(empty_nii,affine=standard_mask.get_affine())
    try:
        name = get_concept(id=concept).json[0]["name"]
        nii_file = "%s/%s_regparam.nii.gz" %(image_folder,name.replace(" ","_"))
    except:
        # There are a few ids with bugs (not returning from cognitive atlas)
        nii_file = "%s/%s_regparam.nii.gz" %(image_folder,concept)
    nibabel.save(nii,nii_file)
