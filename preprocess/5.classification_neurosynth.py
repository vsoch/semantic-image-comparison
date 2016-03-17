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
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.mr.transformation import *
from sklearn import linear_model
import pickle
import pandas
import nibabel
import sys
import os

pmid1_holdout = int(sys.argv[1])
pmid2_holdout = int(sys.argv[2])
X_pickle = sys.argv[3]
Y_pickle = sys.argv[4]
output_file = sys.argv[5]

X = pickle.load(open(X_pickle,"rb"))
Y = pickle.load(open(Y_pickle,"rb"))

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# We will save data to dictionary
result = dict()
   
# Get the labels for holdout images
holdout1Y = Y.loc[pmid1_holdout,:]
holdout2Y = Y.loc[pmid2_holdout,:]

# what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

regression_params = pandas.DataFrame(0,index=X.columns,columns=Y.columns)

print "Training voxels..."
for voxel in X.columns:
    train = [x for x in X.index if x not in [pmid1_holdout,pmid2_holdout] and x in X.index]
    if len(X.loc[train,voxel].unique())>0:
        yy = X.loc[train,voxel].tolist() 
        Xtrain = Y.loc[train,:]
        clf = linear_model.LogisticRegression()
        clf.fit(Xtrain,yy)
        regression_params.loc[voxel,:] = clf.coef_[0]

result["regression_params"] = regression_params

print "Making predictions..."
# Use regression parameters to generate predicted images
concept_vector1 =  pandas.DataFrame(holdout1Y)
concept_vector2 =  pandas.DataFrame(holdout2Y)
predicted_nii1 =  regression_params.dot(concept_vector1)
predicted_nii2 =  regression_params.dot(concept_vector2)

# Turn into nifti images
nii1 = numpy.zeros(standard_mask.shape)
nii2 = numpy.zeros(standard_mask.shape)
nii1[standard_mask.get_data()!=0] = predicted_nii1[pmid1_holdout].tolist()
nii2[standard_mask.get_data()!=0] = predicted_nii2[pmid2_holdout].tolist()
nii1 = nibabel.Nifti1Image(nii1,affine=standard_mask.get_affine())
nii2 = nibabel.Nifti1Image(nii2,affine=standard_mask.get_affine())

#result["image1_predicted"] = predicted_nii1
#result["image2_predicted"] = predicted_nii2

# Turn the holdout image data back into nifti
actual1 = X.loc[pmid1_holdout,:]
actual2 = X.loc[pmid2_holdout,:]
actual_nii1 = numpy.zeros(standard_mask.shape)
actual_nii2 = numpy.zeros(standard_mask.shape)
actual_nii1[standard_mask.get_data()!=0] = actual1.tolist()
actual_nii2[standard_mask.get_data()!=0] = actual2.tolist()
actual_nii1 = nibabel.Nifti1Image(actual_nii1,affine=standard_mask.get_affine())
actual_nii2 = nibabel.Nifti1Image(actual_nii2,affine=standard_mask.get_affine())

# Make a dictionary to lookup images based on nifti
lookup = dict()
lookup[actual_nii1] = pmid1_holdout
lookup[actual_nii2] = pmid2_holdout
lookup[nii1] = pmid1_holdout
lookup[nii2] = pmid2_holdout

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

result["comparison_df"] = comparison_df

# Calculate accuracy
correct = 0
acc1 = comparison_df[comparison_df.actual==pmid1_holdout]
acc2 = comparison_df[comparison_df.actual==pmid2_holdout]

# Did we predict image1 to be image1?
if acc1.loc[acc1.predicted==image1_holdout,"cca_score"].tolist()[0] > acc1.loc[acc1.predicted==image2_holdout,"cca_score"].tolist()[0]:
    correct+=1

# Did we predict image2 to be image2?
if acc2.loc[acc2.predicted==image2_holdout,"cca_score"].tolist()[0] > acc2.loc[acc2.predicted==image1_holdout,"cca_score"].tolist()[0]:
    correct+=1

result["number_correct"] = correct
pickle.dump(result,open(output_file,"wb"))
