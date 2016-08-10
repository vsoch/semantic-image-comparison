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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from pybraincompare.mr.transformation import *
from sklearn.metrics import r2_score
import pickle
import pandas
import nibabel
import numpy
import sys
import os

image1_holdout = int(sys.argv[1])
image2_holdout = int(sys.argv[2])
output_file = sys.argv[3]
labels_tsv = sys.argv[4]
image_lookup = sys.argv[5]
contrast_file = sys.argv[6]

# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

# Images data frame with contrast info, and importantly, number of subjects
image_df = pandas.read_csv(contrast_file,sep="\t",index_col=0)
image_df.index = image_df.image_id

# We should standardize cognitive concepts before modeling
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaled = pandas.DataFrame(StandardScaler().fit_transform(X))
scaled.columns = X.columns
scaled.index = X.index
X = scaled

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
  
norm = pandas.DataFrame(columns=mr.columns)

# Normalize the image data by number of subjects
#V* = V/sqrt(S) 
for row in mr.iterrows():
    subid = row[0]
    number_of_subjects = image_df.loc[subid].number_of_subjects.tolist()
    norm_vector = row[1]/numpy.sqrt(number_of_subjects)
    norm.loc[subid] = norm_vector

del mr

# Get the labels for holdout images
holdout1Y = X.loc[image1_holdout,:]
holdout2Y = X.loc[image2_holdout,:]

# what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

regression_params = pandas.DataFrame(0,index=norm.columns,columns=concepts)
predicted_nii1 = pandas.DataFrame(0,index=norm.columns,columns=["nii"])
predicted_nii2 = pandas.DataFrame(0,index=norm.columns,columns=["nii"])

print "Training voxels and building predicted images..."
for voxel in norm.columns:
    train = [x for x in X.index if x not in [image1_holdout,image2_holdout] and x in norm.index]
    Y = norm.loc[train,voxel].tolist()
    Xtrain = X.loc[train,:]
    # Use pls instead of regularized regression
    clf = PLSRegression(n_components=5)
    clf.fit(Xtrain, Y)    
    # Need to find where regression/intercept params are in this model
    regression_params.loc[voxel,:] = [x[0] for x in clf.coef_]
    predicted_nii1.loc[voxel,"nii"] = clf.predict(holdout1Y.reshape(1,-1))[0][0]
    predicted_nii2.loc[voxel,"nii"] = clf.predict(holdout2Y.reshape(1,-1))[0][0]


predicted_nii1 = predicted_nii1['nii'].tolist()
predicted_nii2 = predicted_nii2['nii'].tolist()

# Turn into nifti images
nii1 = numpy.zeros(standard_mask.shape)
nii2 = numpy.zeros(standard_mask.shape)
nii1[standard_mask.get_data()!=0] = predicted_nii1
nii2[standard_mask.get_data()!=0] = predicted_nii2
nii1 = nibabel.Nifti1Image(nii1,affine=standard_mask.get_affine())
nii2 = nibabel.Nifti1Image(nii2,affine=standard_mask.get_affine())

# Turn the holdout image data back into nifti
actual1 = norm.loc[image1_holdout,:]
actual2 = norm.loc[image2_holdout,:]
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
#0     136        136   0.587533
#1     136       3186  -0.384926
#2    3186        136  -0.325092
#3    3186       3186   0.816416

result["comparison_df"] = comparison_df

# Calculate accuracy
correct = 0
acc1 = comparison_df[comparison_df.actual==image1_holdout]
acc2 = comparison_df[comparison_df.actual==image2_holdout]

# Did we predict image1 to be image1?
if acc1.loc[acc1.predicted==image1_holdout,"cca_score"].tolist()[0] > acc1.loc[acc1.predicted==image2_holdout,"cca_score"].tolist()[0]:
    correct+=1

# Did we predict image2 to be image2?
if acc2.loc[acc2.predicted==image2_holdout,"cca_score"].tolist()[0] > acc2.loc[acc2.predicted==image1_holdout,"cca_score"].tolist()[0]:
    correct+=1

result["number_correct"] = correct

# We should have a measure of encoding regression performance (in addition to classification). I like out of sample R^2 i.e. how much variance is explained by the model using only cognitive concepts.
result["r2_%s" %(image1_holdout)] = r2_score(actual1, predicted_nii1)
result["r2_%s" %(image2_holdout)] = r2_score(actual2, predicted_nii2)

pickle.dump(result,open(output_file,"wb"))
