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
from sklearn import linear_model
from random import shuffle
from glob import glob
import pickle
import pandas
import nibabel
import sys
import os

image_pairs = sys.argv[1]
output_file = sys.argv[2]
labels_tsv = sys.argv[3]
image_lookup = sys.argv[4]

# image_pairs should be string of image pairs 1|2,3|4 for about 2000 (half) the image pairs
# We will calculate an accuracy across the image set, and build a null distribution by doing
# this procedure many times
image_pairs = image_pairs.split(",")
total = 0
correct = 0

# We will save a vector of 
# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# Dictionary to look up image files (4mm)
lookup = pickle.load(open(image_lookup,"rb"))

concepts = X.columns.tolist()

# We will go through each voxel (column) in a data frame of image data
image_paths = lookup.values()
mr = get_images_df(file_paths=image_paths,mask=standard_mask)
image_ids = [int(os.path.basename(x).split(".")[0]) for x in image_paths]
mr.index = image_ids


# We will go through each voxel (column) in a data frame of image data
mr = get_images_df(file_paths=group["in"] + group["out"],mask=standard_mask)
image_paths = group["in"] + group["out"]
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
image_ids = image_ids_in + image_ids_out
mr.index = image_ids

# We will save a data frame of pearson scores (to calculate accuracies later)
comparison_dfs = pandas.DataFrame()

for image_pair in image_pairs:
    
    image1_holdout,image2_holdout = [int(x) for x in image_pair.split("|")]

    # Get the labels for holdout images
    holdout1Y = X.loc[image1_holdout,:]
    holdout2Y = X.loc[image2_holdout,:]

    # what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

    regression_params = pandas.DataFrame(0,index=mr.columns,columns=concepts)

    # We will use the above df to get the test data, and the below (shuffled) to train
    Xshuffled = X.copy()
    train = [x for x in Xshuffled.index if x not in [image1_holdout,image2_holdout] and x in mr.index]
    Xtrain = Xshuffled.loc[train,:]
    Xshuffled_index = Xtrain.index.tolist()
    shuffle(Xshuffled_index)
    Xtrain.index = Xshuffled_index

    for voxel in mr.columns:
        Y = mr.loc[Xshuffled_index,voxel].tolist()
        # Use regularized regression
        clf = linear_model.ElasticNet(alpha=0.1)
        clf.fit(Xtrain,Y)
        regression_params.loc[voxel,:] = clf.coef_.tolist()

    print "Making predictions for %s vs %s..." %(image1_holdout,image2_holdout)
    # Use regression parameters to generate predicted images
    concept_vector1 =  pandas.DataFrame(holdout1Y)
    concept_vector2 =  pandas.DataFrame(holdout2Y)
    predicted_nii1 =  regression_params.dot(concept_vector1)
    predicted_nii2 =  regression_params.dot(concept_vector2)

    # Turn into nifti images
    nii1 = numpy.zeros(standard_mask.shape)
    nii2 = numpy.zeros(standard_mask.shape)
    nii1[standard_mask.get_data()!=0] = predicted_nii1[image1_holdout].tolist()
    nii2[standard_mask.get_data()!=0] = predicted_nii2[image2_holdout].tolist()
    nii1 = nibabel.Nifti1Image(nii1,affine=standard_mask.get_affine())
    nii2 = nibabel.Nifti1Image(nii2,affine=standard_mask.get_affine())
    # Turn the holdout image data back into nifti
    actual1 = mr.loc[image1_holdout,:]
    actual2 = mr.loc[image2_holdout,:]
    actual_nii1 = numpy.zeros(standard_mask.shape)
    actual_nii2 = numpy.zeros(standard_mask.shape)
    actual_nii1[standard_mask.get_data()!=0] = actual1.tolist()
    actual_nii2[standard_mask.get_data()!=0] = actual2.tolist()
    actual_nii1 = nibabel.Nifti1Image(actual_nii1,affine=standard_mask.get_affine())
    actual_nii2 = nibabel.Nifti1Image(actual_nii2,affine=standard_mask.get_affine())
    # Make a dictionary to lookup images based on nifti
    lookup = dict()
    lookup[actual_nii1] = image1_holdout
    lookup[actual_nii2] = image2_holdout
    lookup[nii1] = image1_holdout
    lookup[nii2] = image2_holdout
    comparison_df = pandas.DataFrame(columns=["actual","predicted","cca_score"])
    comparisons = [[actual_nii1,nii1],[actual_nii1,nii2],[actual_nii2,nii1],[actual_nii2,nii2]]
    count=0
    for comp in comparisons:
        name1 = lookup[comp[0]]
        name2 = lookup[comp[1]]
        corr = calculate_correlation(comp,mask=standard_mask)
        comparison_df.loc[count] = [name1,name2,corr] 
        count+=1

    #   actual  predicted  cca_score
    #0    3186       3186   0.908997
    #1    3186        420   0.485644
    #2     420       3186   0.044668
    #3     420        420   0.657109
    comparison_dfs = comparison_dfs.append(comparison_df)

    # Calculate accuracy
    acc1 = comparison_df[comparison_df.actual==image1_holdout]
    acc2 = comparison_df[comparison_df.actual==image2_holdout]

    # Did we predict image1 to be image1?
    if acc1.loc[acc1.predicted==image1_holdout,"cca_score"].tolist()[0] > acc1.loc[acc1.predicted==image2_holdout,"cca_score"].tolist()[0]:
        correct+=1

    # Did we predict image2 to be image2?
    if acc2.loc[acc2.predicted==image2_holdout,"cca_score"].tolist()[0] > acc2.loc[acc2.predicted==image1_holdout,"cca_score"].tolist()[0]:
        correct+=1

    total = total + 2

# Once we finish, save the single value
result = dict()
result["total"] = total
result["correct"] = correct
result["comparison_df"] = comparison_dfs
result["accuracy"] = correct/float(total)
pickle.dump(result,open(output_file,"wb"))
