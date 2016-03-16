#!/usr/bin/python
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.compare.mrutils import get_images_df
from glob import glob
import numpy
import pickle
import pandas
import json
import sys
import os

base = sys.argv[1]
results = "%s/results" %(base)
data = "%s/data" %base
node_folder = "%s/groups" %data
scores_folder = "%s/classification" %(base)
output_folder = "%s/classification_final" %results

#####################################################################################
# PRELIMINARIES
#####################################################################################

# Wisdom a la Sanmi Koyejo @sanmik
# Start with the following definitions:
# d: index for brain voxels
# j: index for cognitive processes
# n: index for examples

# And data you should have available for fitting the forward model:

##### Y ( # examples x # processes): Cognitive process labels as a binary vector, where y_{n,j}=1 if example n contains process j, and zero otherwise. 
# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
Ymat = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
concepts = Ymat.columns.tolist()

##### X (# examples x # voxels): brain data [we will call this Xmat]

# We need to load all the data, this is a data frame of images by voxels
node_pickles = glob("%s/*.pkl" %node_folder)

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# Load any group data to get image paths
group = node_pickles[0]
group = pickle.load(open(group,"rb"))

# Change paths in group pickle to point to 4mm folder
group["in"] = [x.replace("resampled_z","resampled_z_4mm") for x in group["in"]]
group["out"] = [x.replace("resampled_z","resampled_z_4mm") for x in group["out"]]

# X (# examples x # voxels): brain data [we will call this Xmat]
Xmat = get_images_df(file_paths=group["in"] + group["out"],mask=standard_mask)
image_paths = group["in"] + group["out"]
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
image_ids = image_ids_in + image_ids_out
Xmat.index = image_ids
   
##### W (# processes x # voxels): Forward model prediction matrix
W = pickle.load(open("%s/regression_params_dfs.pkl" %output_folder,"rb"))['regression_params'] # [28549 rows x 132 columns]
W = W.transpose()

# For each example "n", the encoding model is given by X_{n,:} = Y{n,:} x W

#####################################################################################
# Model 1: Standard Naive Bayes decoder
#####################################################################################

# Given the images X, and the labels Y, fit a naive Bayes model for each cognitive process. The idea is that the results of this initial model will serve as a baseline for any modifications made in the following steps. Thus, we save the cross-validation test results from this model

# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

import numpy
from sklearn.naive_bayes import GaussianNB

predictions = pandas.DataFrame(index=Xmat.index,columns=Ymat.columns)
only_positive_examples = 0
for concept in concepts:
    print "Predicting images for %s" %concept
    for heldout in Xmat.index.tolist():
        clf = GaussianNB()
        Xtrain = Xmat.loc[Xmat.index!=heldout,:]
        Ytrain = Ymat.loc[Xtrain.index,concept].tolist()
        clf.fit(Xtrain, Ytrain)
        # You should fix this to m1_{j, d} = W_{j, d} + b_{j}
        if len(clf.theta_)==2:
            Xtest = Xmat.loc[heldout,:]
            Yhat = clf.predict(Xtest)[0]
            predictions.loc[heldout,concept] = Yhat
        # For 42/12,276, the Y vector is all 0s, so let's assign a value of 2, 
        # we will give no accuracy for this
        else:
            print "Found %s positive examples!" %numpy.sum(Ytrain)
            only_positive_examples +=1
            predictions.loc[heldout,concept] = 2

only_positive_examples
#42

predictions.to_csv("%s/prediction_concept_matrix_base.tsv" %results,sep="\t")


#####################################################################################
# Model 2: Naive Bayes decoder using forward model
#####################################################################################

# Here we will modify the Gaussian naive Bayes model to use the forward model parameters instead of the results of fitting. The modification will require fixing the distribution for each voxel and cognitive process i.e. P(x_{:, d} | y_{:, j}=1), and P(x_{:, d} | y_{:, j}=0). These are fully characterized by four means and covariances for every voxel and every label. 

# Note: The means will be set directly using the forward model. The covariance will be set using model training error.

# m1_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =1
# You should fix this to m1_{j, d} = W_{j, d}
# m0_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =0
# You should fix this to m0_{j, d} = 0.0
# s1_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =1
# We compute this as the variance of the error in the forward model. 
# Using the training data, set index A = {n | Y[:, j] = 1}
# Then s1_{j, d} = (X_{A, d} - W_{j, d}).var()
#s0_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =0
# We compute this as the variance of the error in the forward model. 
# Using the training data, set index B = {n | Y[:, j] = 0}
# Then s0_{j, d} = (X_{B, d}).var()

predictions_forward = pandas.DataFrame(index=Xmat.index,columns=Ymat.columns)
for concept in concepts:
    print "Predicting images for %s" %concept
    for heldout in Xmat.index.tolist():
        clf = GaussianNB()
        Xtrain = X.loc[Xmat.index!=heldout,:]
        Ytrain = Ymat.loc[Xtrain.index,concept].tolist()
        clf.fit(Xtrain, Ytrain)
        # b = average [X_{n,:} - Y{n,:}W]
        #b = (Xmat - Ymat.dot(W)).mean(axis=0)
        # m1_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =1
        # You should fix this to m1_{j, d} = W_{j, d}
        if len(clf.theta_)==2:
            clf.theta_[1] = W.loc[concept,:]
            # s1_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =1
            # We compute this as the variance of the error in the forward model. 
            # Using the training data, set index A = {n | Y[:, j] = 1}
            A = Ymat.loc[Ymat[concept]==1,concept].index.tolist()
            # Then s1_{j, d} = (X_{A, d} - W_{j, d}).var()
            clf.sigma_[1] = (Xmat.loc[A,:] - W.loc[concept]).var()
            # m0_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =0
            # You should fix this to m0_{j, d} = 0.0
            clf.theta_[0] = 0
            #s0_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =0
            # We compute this as the variance of the error in the forward model. 
            # Using the training data, set index B = {n | Y[:, j] = 0}
            B = Ymat.loc[Ymat[concept]==0,concept].index.tolist()
            # Then s0_{j, d} = (X_{B, d}).var()
            clf.sigma_[0] = Xmat.loc[B,:].var()
            Xtest = Xmat.loc[heldout,:]
            Yhat = clf.predict(Xtest)[0]
            predictions_forward.loc[heldout,concept] = Yhat
        # For 42/12,276, the Y vector is all 0s, so let's assign a value of 2, 
        # we will give no accuracy for this
        else:
            print "Found %s positive examples!" %numpy.sum(Ytrain)
            predictions_forward.loc[heldout,concept] = 2

# Combining across all voxels and labels, you have a total of #voxels x # processes x 4 parameters

# Results:
# The results of this modified model will serve as a second baseline for further modifications. Thus, I suggest you save the cross-validation test results from this model.

predictions_forward.to_csv("%s/prediction_concept_matrix_forward.tsv" %results,sep="\t")
#predictions_forward = pandas.read_csv("%s/prediction_concept_matrix_forward.tsv" %results,sep="\t",index_col=0)

#####################################################################################
# Compare Model 2 to Base
#####################################################################################


# Calculate accuracy metrics for concepts
# @poldrack I guess what I would like to know is for every concept, how many times was that concept predicted to be present, and how many of those were accurate (i.e. 1 - false alarm rate).  and then also, how many times was the concept actually present, and how many of those were accurately predicted (i.e. hit rate).

# I would like to see for each concept a count of how many times each of the following outcomes occurred:
# - predicted category present, true category present (“hit”)
# - predicted category present, true category absent (“false alarm”)
# - predicted category absent, true category present (“miss”)
# - predicted category absent, true category absent (“correct rejection”)
# - Aprime 0.5 + (abs(hit-false_alarm) / (hit-false_alarm)) * ((hit - false_alarm)^2 + abs(hit-false_alarm))
# (continued) / (4*max(hit,false_alarm)-4*hit*false_alarm)

# Function to compare two vectors to calculate the above
def calculate_hits(Ya,Yp,Va,Vp):
    # Y: Y values, V: 0 or 1
    group1_images = Yp[Yp==Vp].index.tolist()
    group2_images = Ya[Ya==Va].index.tolist()
    group_overlap = len([x for x in group1_images if x in group2_images])
    normalized = group_overlap / float(len(Yp))   
    return normalized

def get_concept_acc(predictions):
    concept_acc = pandas.DataFrame(index=concepts,columns=["hit","false_alarm","miss","correct_rejection","aprime"])
    for concept in concepts:
        Yp = predictions.loc[:,concept]
        Ya = Ymat.loc[Yp.index,concept]
        # - predicted category present, true category present (“hit”)
        hit = calculate_hits(Ya,Yp,1,1)
        concept_acc.loc[concept,"hit"] = hit
        false_alarm = calculate_hits(Ya,Yp,0,1)
        concept_acc.loc[concept,"false_alarm"] = false_alarm
        aprime = 0 
        if hit-false_alarm != 0:
            aprime = 0.5 + (abs(hit-false_alarm) / (hit-false_alarm)) * (numpy.power(hit - false_alarm,2) + abs(hit-false_alarm)) / (4*max(hit,false_alarm)-4*hit*false_alarm)
        concept_acc.loc[concept,"aprime"] = aprime
        concept_acc.loc[concept,"miss"] = calculate_hits(Ya,Yp,1,0)
        concept_acc.loc[concept,"correct_rejection"] = calculate_hits(Ya,Yp,0,0)
    return concept_acc 

base_acc = get_concept_acc(predictions)
forward_acc = get_concept_acc(predictions_forward)

# Just compare the two for now
diff_acc = forward_acc - base_acc

# Add the concept names to each
from cognitiveatlas.api import get_concept
concept_names = []
for concept in diff_acc.index:
    concept_names.append(get_concept(id=concept).json[0]["name"])

base_acc["name"] = concept_names
forward_acc["name"] = concept_names
diff_acc["name"] = concept_names

# Add the number of images
number_images = []
for concept in diff_acc.index:
    number_images.append(Ymat.loc[:,concept].sum())

base_acc["number_images"] = number_images
forward_acc["number_images"] = number_images
diff_acc["number_images"] = number_images

diff_acc = diff_acc.sort(columns=["hit"],ascending=False)
diff_acc.to_csv("%s/prediction_concept_accs_diff.tsv" %results,sep="\t")
forward_acc.to_csv("%s/prediction_concept_accs_forward.tsv" %results,sep="\t")
base_acc.to_csv("%s/prediction_concept_accs_base.tsv" %results,sep="\t")


#####################################################################################
# Model 3: Naive Bayes decoder using forward model with tuning
#####################################################################################

# This section covers tuning the nave Bayes model to (hopefully) improve performance. We will use the a simple generic approach that works for any probabilistic model, where we keep the model the same, and tune the threshold for deciding if an example is positive or negative..

# Some background:
# The prediction decision for the naive Bayes model depends on the odds i.e.
# predict Y=1 if P(Y=1|X)>P(Y=0|X)
# predict 0 otherwise.

# This is equivalent to the model using log odds given by:
# predict Y=1 if log[ P(Y=1|X)/P(Y=0|X) ] >0
# predict 0 otherwise.

# Modification:
# We are going to modify the threshold for the log odds to some new threshold “c”. the result is the new classifier:
# predict Y=1 if log[ P(Y=1|X)/P(Y=0|X) ] > c
# predict 0 otherwise.

# Note that log [P(Y=1|X)/P(Y=0|X)]  = log [P(Y=1|X)] - log [P(Y=0|X)] , so you can compute the log odds by subtracting the results from "predict_log_proba" function.

import operator

# Note that log [P(Y=1|X)/P(Y=0|X)]  = log [P(Y=1|X)] - log [P(Y=0|X)] , so you can compute the log odds by subtracting the results from "predict_log_proba" function.
# Write our own predictions function to take value of C

def nb_predict(clf,Xtest,cc):
    # Classes returned in order 0,1
    #clf.classes_: array([0, 1])
    log0,log1 = GaussianNB.predict_log_proba(clf,Xtest)[0]
    log_odds = log1 - log0
    if log_odds > cc:
        return 1
    return 0

# We will save a dictionary of the optimal C's used for each concept
optimal_c = pandas.DataFrame(index=concepts,columns=["optimal_c"])

for concept in concepts:
    print "Predicting images for %s" %concept
    # Find the optimal C between 0.001 and 10, for 10 steps
    accuracies = dict()
    actual = Ymat[concept]
    for cc in numpy.logspace(-3, 1, 10):
        print "Testing C value of %s..." %(cc)
        prediction_cc = pandas.DataFrame(index=Xmat.index,columns=["prediction"])
        for heldout in Xmat.index.tolist():
            clf = GaussianNB()
            Xtrain = Xmat.loc[Xmat.index!=heldout,:]
            Ytrain = Ymat.loc[Xtrain.index,concept].tolist()
            clf.fit(Xtrain, Ytrain)
            if len(clf.theta_)==2:
                clf.theta_[1] = W.loc[concept,:]
                A = Ymat.loc[Ymat[concept]==1,concept].index.tolist()
                clf.sigma_[1] = (Xmat.loc[A,:] - W.loc[concept]).var()
                clf.theta_[0] = 0
                B = Ymat.loc[Ymat[concept]==0,concept].index.tolist()
                clf.sigma_[0] = Xmat.loc[B,:].var()
                Xtest = Xmat.loc[heldout,:]
                # USE OUT OWN PREDICT FUNCTION
                Yhat = nb_predict(clf,Xtest,cc)
                prediction_cc.loc[heldout,"prediction"] = Yhat
        correct_count = len([1 for x in actual.index.tolist() if actual.loc[x] == prediction_cc.loc[x,"prediction"]])
        # Now calculate an accuracy for the value of c - how many did we get right?   
        # This is biased by 0s, but they all are, so we can use to select the value of x
        accuracies[cc] = correct_count  / float(actual.shape[0])
    # Find the highest accuracy to determine C
    # tested this - if all are equivalent, 0.5 is returned
    final_c = max(accuracies.iteritems(), key=operator.itemgetter(1))[0]
    print "Found optimal c for concept %s as %s" %(concept,final_c)
    optimal_c.loc[concept,"optimal_c"] = final_c

# You may implement this modification by rolling your own prediction function, or look into hacking the built in naive Bayes prediction function.
# I suggest you search for “c” using 10 values in the range 10^[-3, 1]. 
# I have no apriori reason for this, just suggesting arbitrary small-ish numbers. It would make sense to visually examine the log odds for each cognitive process to see its typical range, then decide on tuning based on what you observe - instead of using my default suggestions.
# You can set the best “c” values either by picking the best “c” on the training set, or picking the best “c” via cross-validation. Cross-validation is better, training set may be adequate.

optimal_c.to_csv("%s/optimal_c_tuned.tsv" %results,sep="\t")

# Now run again using our modified Cs!!
predictions_tuned = pandas.DataFrame(index=Xmat.index,columns=Ymat.columns)
for concept in concepts:
    print "Predicting images for %s" %concept
    for heldout in Xmat.index.tolist():
        clf = GaussianNB()
        Xtrain = X.loc[Xmat.index!=heldout,:]
        Ytrain = Ymat.loc[Xtrain.index,concept].tolist()
        clf.fit(Xtrain, Ytrain)
        if len(clf.theta_)==2:
            clf.theta_[1] = W.loc[concept,:]
            A = Ymat.loc[Ymat[concept]==1,concept].index.tolist()
            clf.sigma_[1] = (Xmat.loc[A,:] - W.loc[concept]).var()
            clf.theta_[0] = 0
            B = Ymat.loc[Ymat[concept]==0,concept].index.tolist()
            clf.sigma_[0] = Xmat.loc[B,:].var()
            Xtest = Xmat.loc[heldout,:]
            Yhat = nb_predict(clf,Xtest,optimal_c.loc[concept,"optimal_c"])
            predictions_tuned.loc[heldout,concept] = Yhat
        else:
            print "Found %s positive examples!" %numpy.sum(Ytrain)
            predictions_tuned.loc[heldout,concept] = 2

predictions_tuned.to_csv("%s/prediction_concept_matrix_tuned.tsv" %results,sep="\t")

tuned_acc = get_concept_acc(predictions_tuned)

# Just compare the two for now
diff_acc = tuned_acc - base_acc

# Add the concept names to each
from cognitiveatlas.api import get_concept
concept_names = []
for concept in diff_acc.index:
    concept_names.append(get_concept(id=concept).json[0]["name"])

tuned_acc["name"] = concept_names
diff_acc["name"] = concept_names

# Add the number of images
number_images = []
for concept in diff_acc.index:
    number_images.append(Ymat.loc[:,concept].sum())

tuned_acc["number_images"] = number_images
diff_acc["number_images"] = number_images

diff_acc = diff_acc.sort(columns=["hit"],ascending=False)
diff_acc.to_csv("%s/prediction_concept_accs_tuned_diff.tsv" %results,sep="\t")
tuned_acc.to_csv("%s/prediction_concept_accs_tuned.tsv" %results,sep="\t")
