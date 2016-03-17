#!/usr/bin/python
# Classification framework

# hold out holdouts, generate regression parameter matrix using other images

# for holdout1 in all holdouts:
#    for holdout2 in holdouts:
#        if holdout1 != holdout2:
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

holdout_start = int(sys.argv[1]) #index to start holdout group
holdout_end = int(sys.argv[2])   #index to stop holdout group
X_pickle = sys.argv[3]
Y_pickle = sys.argv[4]
output_file = sys.argv[5]


###################################################################################
# DATA PREPARATION
###################################################################################

X = pickle.load(open(X_pickle,"rb"))
Y = pickle.load(open(Y_pickle,"rb"))

# Only include labels that have at least 10 occurences
counts = Y.sum()
grten = counts[counts>=10].index.tolist()
Y = Y.loc[:,grten]

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# We will save data to dictionary
result = dict()

# Get the holdout imageids
holdouts = Y.index[holdout_start:holdout_end]
   
# Get the labels for holdout images
holdoutsY = Y.loc[holdouts,:]

# how about an extended version of the leave-two-out strategy, in which you train the model while leaving out say 1% of abstracts (which would be ~100 per holdout set), generate the predicted images for those left out abstracts, then randomly pair them and test whether the similarity is higher between the true and predicted for each pair? this would only require 100 training rounds but you would still get to assess accuracy on all of the abstracts. 

###################################################################################
# FUNCTIONS
###################################################################################

# Write a function to generate a predicted image for a heldout image
def generate_predicted_image(pmid,Y,regression_params,standard_mask):
    concept_vectorY = Y.loc[pmid,regression_params.columns]
    predicted_nii =  regression_params.dot(concept_vectorY)
    nii = numpy.zeros(standard_mask.shape)
    nii[standard_mask.get_data()!=0] = predicted_nii.tolist()
    return nibabel.Nifti1Image(nii,affine=standard_mask.get_affine())

# A function to reconstruct an actual image (from X matrix)
def generate_actual_image(pmid,X,standard_mask):
    actual = X.loc[pmid,:]
    actual_nii = numpy.zeros(standard_mask.shape)
    actual_nii[standard_mask.get_data()!=0] = actual.tolist()
    return nibabel.Nifti1Image(actual_nii,affine=standard_mask.get_affine())

# A function to test similarity between two images predicted and actuals
def assess_similarity(pmid1,pmid2,predicted_nii1,predicted_nii2,actual_nii1,actual_nii2):
    lookup = dict()
    lookup[actual_nii1] = pmid1
    lookup[actual_nii2] = pmid2
    lookup[predicted_nii1] = pmid1
    lookup[predicted_nii2] = pmid2
    comparison_df = pandas.DataFrame(columns=["actual","predicted","cca_score"])
    comparisons = [[actual_nii1,predicted_nii1],[actual_nii1,predicted_nii2],[actual_nii2,predicted_nii1],[actual_nii2,predicted_nii2]]
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
    # Calculate accuracy
    correct = 0
    acc1 = comparison_df[comparison_df.actual==pmid1]
    acc2 = comparison_df[comparison_df.actual==pmid2]
    # Save list of [actual,predicted] to add to confusion
    predictions = []
    # Did we predict image1 to be image1?
    if acc1.loc[acc1.predicted==pmid1,"cca_score"].tolist()[0] > acc1.loc[acc1.predicted==pmid2,"cca_score"].tolist()[0]:
        correct+=1
        predictions.append([pmid1,pmid1])
    else:
        predictions.append([pmid1,pmid2])
    # Did we predict image2 to be image2?
    if acc2.loc[acc2.predicted==pmid2,"cca_score"].tolist()[0] > acc2.loc[acc2.predicted==pmid1,"cca_score"].tolist()[0]:
        correct+=1
        predictions.append([pmid2,pmid2])
    else:
        predictions.append([pmid2,pmid1])
    return comparison_df,predictions,correct

###################################################################################
# TRAINING
###################################################################################

regression_params = pandas.DataFrame(0,index=X.columns,columns=Y.columns)

# Keep count of how many voxels we couldn't build a model for (labels of all one type)
missed = 0

print "Training voxels..."
for voxel in X.columns:
    train = [x for x in X.index if x not in holdouts and x in X.index]
    if len(X.loc[train,voxel].unique())>0:
        # We can only build a model for voxels with 0/1 data
        if len(X.loc[train,voxel].unique()) > 1:
            yy = X.loc[train,voxel].tolist() 
            Xtrain = Y.loc[train,:]
            clf = linear_model.LogisticRegression()
            clf.fit(Xtrain,yy)
            regression_params.loc[voxel,:] = clf.coef_[0]
        else:
            missed +=1

result["regression_params"] = regression_params
result["y_singular_voxel_count"] = missed

# generate the predicted images for those left out abstracts, then randomly pair them and test whether the similarity is higher between the true and predicted for each pair? this would only require 100 training rounds but you would still get to assess accuracy on all of the abstracts. 

print "Making predictions..."

# Save a confusion data frame, comparison data frame, as we go
comparison_df = pandas.DataFrame(columns=["actual","predicted","cca_score"])
confusions = pandas.DataFrame(0,index=X.index,columns=X.index)
total_correct = 0
total_comparisons = 0

# Test each holdout against all other holdouts
for holdout1 in holdouts:
    for holdout2 in holdouts:
        if holdout1 != holdout2:
            total_comparisons+=1
            # Generate predicted images
            predicted1 = generate_predicted_image(holdout1,Y,regression_params,standard_mask)
            predicted2 = generate_predicted_image(holdout2,Y,regression_params,standard_mask)   
            actual1 = generate_actual_image(holdout1,X,standard_mask)
            actual2 = generate_actual_image(holdout2,X,standard_mask)
            # Generate confusion_df, predictions, and correct count
            comp_df,predictions,correct = assess_similarity(pmid1=holdout1,
                                                            pmid2=holdout2,
                                                            predicted_nii1=predicted1,
                                                            predicted_nii2=predicted2,
                                                            actual_nii1=actual1,
                                                            actual_nii2=actual2)
            # Add predictions to the confusion matrix
            for prediction in predictions:
                actual = prediction[0]
                predicted = prediction[1]
                confusions.loc[actual,predicted] = confusions.loc[actual,predicted] + 1
            # Add comparison result to data frame
            comparison_df = comparison_df.append(comp_df)
            total_correct = total_correct + correct


result["comparison_df"] = comparison_df
result["confusions"] = confusions
result["number_correct"] = total_correct
result["total_comparisons"] = total_comparisons
result["accuracy"] = float(total_correct)/total_comparisons
pickle.dump(result,open(output_file,"wb"))
