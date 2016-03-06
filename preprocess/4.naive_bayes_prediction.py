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

# Wisdom a la Sanmi Koyejo @sanmik
#====================================================
#Encoding model setup
#====================================================
# We represent the cognitive process labels as a binary vector Y ( # examples x # processes), where y_{n,j}=1 if example n contains process j, and zero otherwise. 

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
Ymat = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
concepts = Ymat.columns.tolist()

# Let X (# examples x # voxels) be the brain data, and W (# processes x # voxels) be the prediction matrix, and b (# processes x 1) be the offset (bias) of the linear model. If you didnt fit a bias, then b_j = 0.

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

Xmat = get_images_df(file_paths=group["in"] + group["out"],mask=standard_mask)
image_paths = group["in"] + group["out"]
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
image_ids = image_ids_in + image_ids_out
Xmat.index = image_ids
   
# The prediction matrix is the regression parameters?
W = pickle.load(open("%s/regression_params_dfs.pkl" %output_folder,"rb"))['regression_params'] # [28549 rows x 132 columns]
W = W.transpose()

# For each example "n", the encoding model is given by X_{n,:} = Y{n,:} x W + b

#====================================================
#Explanation
#====================================================
# Background: http://scikit-learn.org/stable/modules/naive_bayes.html. We will be using Gaussian Naive Bayes. Also, I will focus on binary prediction i.e. predicting each cognitive process Y_{:, j} one at a time.

# For Gaussian naive Bayes, you will need to fix the distribution for each voxel and cognitive process i.e. P(x_{:, d} | y_{:, j}=1), and P(x_{:, d} | y_{:, j}=0). These are fully characterized by four numbers:
# 1. m1_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =1
# You should fix this to m1_{j, d} = W_{j, d} + b_{j}

# 2. m0_{j, d}: This is the mean of the Gaussian for voxel = d, and process y_{:, j} =0
# You should fix this to m0_{j, d} = b_{j}

# 3. s1_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =1
# with no other information, might as well fix this to s1_{j, d} = 1.0

# 4. s0_{j, d}: This is the variance of the Gaussian for voxel = d, and process y_{:, j} =0
# with no other information, might as well fix this to s1_{j, d} = 1.0

# Combining across all voxels and labels, you have a total of #voxels x # processes x 4 parameters

# ====================================================
# Implementation
# ====================================================
# The path of least resistance is probably to hack an existing implementation of Naive Bayes and replace the parameters with the ones you compute from the fit model. I would hack http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html and replace the following:

# theta_ : array, shape (n_classes, n_features): mean of each feature per class
# Replace this with the m above

# sigma_ : array, shape (n_classes, n_features): variance of each feature per class
# Replace this with the s above

# To clarify, I am suggesting the following steps:
# FOR EVERY LABEL (cognitive process)
# STEP 1: Fit the Gaussian naive Bayes model from scikit learn using X's and Y's
# STEP 2: Replace the theta_ and sigma_ parameters as suggested above.
# STEP 2b (optional): Tune parameters in cross-validation loop (see below)
# STEP 3: Predict using the predict function of naive Bayes

import numpy
from sklearn.naive_bayes import GaussianNB

predictions = pandas.DataFrame(index=Xmat.index,columns=Ymat.columns)
only_positive_examples = 0
for concept in concepts:
    print "Predicting images for %s" %concept
    for heldout in Xmat.index.tolist():
        clf = GaussianNB()
        # The encoding model is given by X_{n,:} = Y{n,:} x W + b
        Xtrain = X.loc[Xmat.index!=heldout,:]
        Ytrain = Ymat.loc[Xtrain.index,concept].tolist()
        clf.fit(Xtrain, Ytrain)
        # b = average [X_{n,:} - Y{n,:}W]
        #b = (Xtrain - Ymat.loc[Xtrain.index,:].dot(W)).mean()
        #print b.unique()
        # You should fix this to m1_{j, d} = W_{j, d} + b_{j}
        # Why would this only have one dimension? A bad model?
        if len(clf.theta_)==2:
            clf.theta_[1] = W.loc[concept,:] #+ b
            clf.sigma_[1] = numpy.ones(W.shape[1])
            # You should fix this to m0_{j, d} = b_{j}
            clf.theta_[0] = numpy.zeros(W.shape[1])#b
            # with no other information, might as well fix this to s1_{j, d} = 1.0
            clf.sigma_[0] = numpy.ones(W.shape[1])
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

predictions.to_csv("%s/prediction_concept_matrix.tsv" %results,sep="\t")


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
    return len([x for x in group1_images if x in group2_images])

concept_acc = pandas.DataFrame(index=concepts,columns=["hit","false_alarm","miss","correct_rejection","aprime"])
for concept in concepts:
    Yp = predictions.loc[:,concept]
    Ya = Ymat.loc[Yp.index,concept]
    # - predicted category present, true category present (“hit”)
    hit = calculate_hits(Ya,Yp,1,1)
    concept_acc.loc[concept,"hit"] = hit
    false_alarm = calculate_hits(Ya,Yp,0,1)
    concept_acc.loc[concept,"false_alarm"] = false_alarm
    if hit-false_alarm != 0:
        aprime_num = 0.5 + (abs(hit-false_alarm) / (hit-false_alarm)) * ((hit - false_alarm)^2 + abs(hit-false_alarm)) 
        aprime_denom = (4*max(hit,false_alarm)-4*hit*false_alarm)
        if aprime_denom != 0:
            aprime = 0
    else:
        aprime = 0
    concept_acc.loc[concept,"aprime"] = aprime
    concept_acc.loc[concept,"miss"] = calculate_hits(Ya,Yp,1,0)
    concept_acc.loc[concept,"correct_rejection"] = calculate_hits(Ya,Yp,0,0)
 

############# OLD
#concept_acc = pandas.DataFrame(index=concepts,columns=["n_accurate_predictions_1","accurate_predictions_1_outof_predicted_1_total","n_predicted_present","n_actually_present"])
#for concept in concepts:
#    Yp = predictions.loc[:,concept]
#    #how many times was the concept actually present
#    predicted_present = Yp[Yp==1]
#    Ya = Ymat.loc[predicted_present.index,concept]
#    actually_present = Ya[Ya==1]
#    concept_acc.loc[concept,"n_actually_present"] = len(actually_present)
#    concept_acc.loc[concept,"n_predicted_present"] = len(predicted_present)
#    # how many times was that concept predicted to be present, and how many of those were accurate
#    concept_acc.loc[concept,"n_accurate_predictions_1"] = len([x for x in predicted_present.index.tolist() if x in actually_present.index.tolist()])
#    acc = concept_acc.loc[concept,"n_accurate_predictions_1"] / float(len(predicted_present))
#    concept_acc.loc[concept,"accurate_predictions_1_outof_predicted_1_total"] = acc
  
concept_acc = concept_acc.sort(columns=["hit"],ascending=False)

# Add the concept name
from cognitiveatlas.api import get_concept
concept_names = []
for concept in concept_acc.index:
    concept_names.append(get_concept(id=concept).json[0]["name"])

concept_acc["name"] = concept_names

# Add the number of images
number_images = []
for concept in concept_acc.index:
    number_images.append(Ymat.loc[:,concept].sum())

concept_acc["number_images"] = number_images
concept_acc.to_csv("%s/prediction_concepts_accs.tsv" %results,sep="\t")

# What percentage of labels for each image did we get correct?
images_acc = pandas.DataFrame(index=Ymat.index,columns=["percent_correct","total_number_concepts"])
for image in Xmat.index.tolist():
    Ip = predictions.loc[image,:]
    Ia = Ymat.loc[image,Ip.index].tolist()
    Ip = Ip.tolist()
    acc = numpy.sum([1 for x in range(len(Ip)) if Ip[x]==Ia[x]]) / float(len(Ip))
    images_acc.loc[image,:] = [acc,numpy.sum(Ia)]
    
images_acc = images_acc.sort(columns=["percent_correct"],ascending=False)

# Look up contrast and task names
images_tsv = "%s/contrast_defined_images_filtered.tsv" %results
images = pandas.read_csv(images_tsv,sep="\t")
tasks = images.loc[images.image_id.isin(images_acc.index.tolist()),"cognitive_paradigm_cogatlas"]
cons = images.loc[images.image_id.isin(images_acc.index.tolist()),"cognitive_contrast_cogatlas"]
    
images_acc["contrast"] = cons.tolist()
images_acc["task"] = tasks.tolist()
images_acc.to_csv("%s/prediction_images_accs.tsv" %results,sep="\t")

# Count contrasts for Sanmi
contrast_counts = images.cognitive_contrast_cogatlas.value_counts()
contrast_counts.sort(ascending=False,inplace=True)
contrast_counts.to_csv("%s/contrast_counts.tsv" %results,sep="\t")

#====================================================
#Tuning
#====================================================
#If the performance is bad, It may be worth tuning a bit to see if you can improve. The easiest place to tune is to change the m0_{j, d}'s (especially if you didnt fit a bias in the forward model). Next, you can tune s0_{j, d}'s. 

#Since there are so many parameters, I would tune by fixing the same m0_{j, :} = m0_j - same for for all voxels but separate for each process. Similarly, and s0_{j, :} = s0_j - same for for all voxels but separate for each process
