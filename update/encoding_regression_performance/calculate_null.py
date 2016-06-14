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
from sklearn.preprocessing import StandardScaler
from pybraincompare.mr.transformation import *
from sklearn.metrics import r2_score
from sklearn import linear_model
from random import shuffle
import pickle
import pandas
import nibabel
import numpy
import sys
import os

image_pairs = sys.argv[1]
output_file = sys.argv[2]
labels_tsv = sys.argv[3]
contrast_file = sys.argv[4]

##################################################################################
# DATA PREPARATION ###############################################################
##################################################################################

# image_pairs should be string of image pairs 1|2,3|4 for about 1000 (one quarter) the image pairs
# We will calculate an accuracy across the image set, and build a null distribution by doing
# this procedure many times
image_pairs = image_pairs.split(",")
total_correct = 0
total = 0

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

# We will save a data frame of pearson scores (to calculate accuracies later)
comparison_dfs = pandas.DataFrame(columns=["actual","predicted","cca_score"])
confusions = pandas.DataFrame(0,index=norm.index,columns=norm.index)

##################################################################################
# FUNCTIONS ######################################################################
##################################################################################

# A function to generate a predicted image for a heldout image
def generate_predicted_image(uid,holdoutY,regression_params,intercept_params,standard_mask):
    concept_vectorY =  pandas.DataFrame(holdoutY)
    predicted_nii =  regression_params.dot(concept_vectorY)[uid] + intercept_params["intercept"]
    nii = numpy.zeros(standard_mask.shape)
    nii[standard_mask.get_data()!=0] = predicted_nii.tolist()
    return nibabel.Nifti1Image(nii,affine=standard_mask.get_affine())

# A function to reconstruct an actual image (from X matrix)
def generate_actual_image(uid,norm,standard_mask):
    actual = norm.loc[uid,:]
    actual_nii = numpy.zeros(standard_mask.shape)
    actual_nii[standard_mask.get_data()!=0] = actual.tolist()
    return nibabel.Nifti1Image(actual_nii,affine=standard_mask.get_affine())

def assess_similarity(image1_holdout,image2_holdout,predicted_nii1,predicted_nii2,actual_nii1,actual_nii2):
    correct = 0
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
    # Save list of [actual,predicted] to add to confusion
    predictions = []
    # Calculate accuracy
    acc1 = comparison_df[comparison_df.actual==image1_holdout]
    acc2 = comparison_df[comparison_df.actual==image2_holdout]
    # Did we predict image1 to be image1?
    if acc1.loc[acc1.predicted==image1_holdout,"cca_score"].tolist()[0] > acc1.loc[acc1.predicted==image2_holdout,"cca_score"].tolist()[0]:
        correct+=1
        predictions.append([image1_holdout,image1_holdout])  # image 1 predicted to be image 1
    else:
        predictions.append([image1_holdout,image2_holdout])  # image 1 predicted to be image 2
    # Did we predict image2 to be image2?
    if acc2.loc[acc2.predicted==image2_holdout,"cca_score"].tolist()[0] > acc2.loc[acc2.predicted==image1_holdout,"cca_score"].tolist()[0]:
        correct+=1
        predictions.append([image2_holdout,image2_holdout])  # image 2 predicted to be image 2
    else:
        predictions.append([image2_holdout,image1_holdout])  # image 2 predicted to be image 1
    return comparison_df,predictions,correct


##################################################################################
# PERMUTATIONS ###################################################################
##################################################################################

for image_pair in image_pairs:
    
    image1_holdout,image2_holdout = [int(x) for x in image_pair.split("|")]

    # Get the labels for holdout images
    holdout1Y = X.loc[image1_holdout,:]
    holdout2Y = X.loc[image2_holdout,:]

    # what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

    regression_params = pandas.DataFrame(0,index=norm.columns,columns=concepts)
    intercept_params = pandas.DataFrame(0,index=norm.columns,columns=["intercept"])

    # We will use the above df to get the test data, and the below (shuffled) to train
    Xshuffled = X.copy()
    train = [x for x in Xshuffled.index if x not in [image1_holdout,image2_holdout] and x in norm.index]
    Xtrain = Xshuffled.loc[train,:]
    Xshuffled_index = Xtrain.index.tolist()
    shuffle(Xshuffled_index)
    Xtrain.index = Xshuffled_index

    # TRAINING #######################################################################

    for voxel in norm.columns:
        Y = norm.loc[Xshuffled_index,voxel].tolist()
        # Use regularized regression
        clf = linear_model.ElasticNet(alpha=0.1)
        clf.fit(Xtrain,Y)
        regression_params.loc[voxel,:] = clf.coef_.tolist()
        intercept_params.loc[voxel,"intercept"] = clf.intercept_    


    # PREDICTIONS ####################################################################

    print "Making predictions for %s vs %s..." %(image1_holdout,image2_holdout)

    # Use regression parameters to generate predicted images
    nii1 = generate_predicted_image(image1_holdout,holdout1Y,regression_params,intercept_params,standard_mask)
    nii2 = generate_predicted_image(image2_holdout,holdout2Y,regression_params,intercept_params,standard_mask)

    # Turn the holdout image data back into nifti
    actual_nii1 = generate_actual_image(image1_holdout,norm,standard_mask)
    actual_nii2 = generate_actual_image(image1_holdout,norm,standard_mask)

    # Generate confusion_df, predictions, and correct count
    comp_df,predictions,correct = assess_similarity(image1_holdout=image1_holdout,
                                                    image2_holdout=image2_holdout,
                                                    predicted_nii1=nii1,
                                                    predicted_nii2=nii2,
                                                    actual_nii1=actual_nii1,
                                                    actual_nii2=actual_nii2)

    # comp_df
    #   actual  predicted  cca_score
    #0    3186       3186   0.839541
    #1    3186        412   0.765565
    #2     412       3186   0.839541
    #3     412        412   0.765565

    # predictions
    # [[3186, 3186], [412, 3186]]
    # correct
    # 1

    # Add predictions to the confusion matrix
    for prediction in predictions:
        actual = prediction[0]
        predicted = prediction[1]
        confusions.loc[actual,predicted] = confusions.loc[actual,predicted] + 1

    # Add comparison result to data frame
    comparison_dfs = comparison_dfs.append(comp_df)
    total_correct = total_correct + correct
    total = total + 2


result = dict()
result["comparison_df"] = comparison_dfs
result["confusions"] = confusions
result["number_correct"] = total_correct
result["total_comparisons"] = total
result["accuracy"] = float(total_correct)/total
pickle.dump(result,open(output_file,"wb"))
