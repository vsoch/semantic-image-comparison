#!/usr/bin/python

from pybraincompare.compare.maths import calculate_correlation
from pybraincompare.compare.mrutils import get_images_df
from pybraincompare.mr.datasets import get_standard_mask
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from pybraincompare.mr.transformation import *
from sklearn.metrics import r2_score, confusion_matrix
from sklearn import linear_model
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

# Save the values for later
holdout1X = X.loc[image1_holdout,:]
holdout2X = X.loc[image2_holdout,:]

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

# Get the holdout brain data normalized (from X)
holdout1X_norm = norm.loc[image1_holdout,:]
holdout2X_norm = norm.loc[image2_holdout,:]

# BUILD THE FORWARD MODEL - elastic net
regression_params = pandas.DataFrame(0,index=norm.columns,columns=concepts)

print "Training voxels..."
for voxel in norm.columns:
    train = [x for x in X.index if x not in [image1_holdout,image2_holdout] and x in norm.index]
    Y = norm.loc[train,voxel].tolist()
    Xtrain = X.loc[train,:] 
    # Use regularized regression
    clf = linear_model.ElasticNet(alpha=0.1)
    clf.fit(Xtrain,Y)
    regression_params.loc[voxel,:] = clf.coef_.tolist()

##### W (# processes x # voxels): Forward model prediction matrix
W = regression_params

# Make predictions for images
# Ypred = Ytest x W
# 1 x 132 = 1 x 28k by 28k x 132
# Y{test,:} = numpy.linalg.lstsq(W.T, X_{test,:}.T)[0].T

holdouts = [image1_holdout,image2_holdout]
predictions = pandas.DataFrame(0,index=holdouts, columns=X.columns)
for heldout in holdouts:
    heldout_X = norm.loc[heldout,:] #28K by 1
    Xtest = numpy.linalg.lstsq(W,heldout_X)[0]
    predictions.loc[heldout,:] = Xtest

result = dict()

# Predict the labels
predicted_labels1 = [numpy.sign(x) for x in predictions.loc[image1_holdout]]
predicted_labels2 = [numpy.sign(x) for x in predictions.loc[image2_holdout]]
    
result["predictions_raw"] = predictions
result["predictions_thresh"] = {image1_holdout:predicted_labels1,
                                image2_holdout:predicted_labels2}

# Accuracy metric one - accuracy as a % of all labels
accuracy1 = numpy.sum([1 for x in range(len(holdout1X)) if holdout1X.tolist()[x] == predicted_labels1[x]])/float(len(holdout1X))
accuracy2 = numpy.sum([1 for x in range(len(holdout2X)) if holdout1X.tolist()[x] == predicted_labels2[x]])/float(len(holdout2X))

result["accuracy_overall"] = {image1_holdout:accuracy1,
                              image2_holdout:accuracy2}


# Confusion Stuffs
def calculate_confusion(predicted,actual):
    res = dict()
    res["TP"] = 0
    res["TN"] = 0
    res["FP"] = 0
    res["FN"] = 0
    for x in range(len(actual)):
        x_pred = predicted[x]
        x_act = actual[x]
        # Map is labeled with concept
        if x_act == 1:
            # We got it right
            if x_act == x_pred:
                res["TP"]+=1
            # We didn't
            else:
                res["FN"]+=1
        # Map isn't labeled
        else:
            # We got it right
            if x_act == x_pred:
                res["TN"]+=1
            # We didn't
            else:
                res["FP"]+=1
    return res

conf1 = calculate_confusion(predicted_labels1,holdout1X.tolist())
# {'TN': 73, 'FP': 43, 'FN': 6, 'TP': 10}

conf2 = calculate_confusion(predicted_labels2,holdout2X.tolist())
# {'TN': 73, 'FP': 56, 'FN': 3, 'TP': 0}

result["confusions"] = {image1_holdout:conf1,
                        image2_holdout:conf2}

# Accuracy as number correct / number labels
number_labels1 = len(holdout1X[holdout1X==1])
number_labels2 = len(holdout1X[holdout2X==1])

accuracy1 = conf1["TP"]/float(number_labels1)
accuracy2 = conf2["TP"]/float(number_labels2)

result["accuracy/number_labels"] = {image1_holdout:accuracy1,
                                    image2_holdout:accuracy2}

pickle.dump(result,open(output_file,'wb'))
