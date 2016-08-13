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
number_components = int(sys.argv[7])

# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

# Transform to be -1 and 1
#for col in X.columns:
#    holder = X[col]
#    holder[holder==0] = -1
#    X[col] = holder

# Save the values for later
holdout1X = X.loc[image1_holdout,:]
holdout2X = X.loc[image2_holdout,:]
# If we need to change 0 to be represented as -1
#holdout1X[holdout1X == 0] = -1
#holdout2X[holdout2X == 0] = -1

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


print "Training voxels and building predicted images..."
for label in regression_params.columns:
    train = [x for x in X.index if x not in [image1_holdout,image2_holdout] and x in norm.index]
    Xtrain = norm.loc[train,:]
    Ytrain = numpy.array(X.loc[train,:]) 
    # Use pls instead of regularized regression
    clf = PLSRegression(n_components=number_components)
    # a. Use PLS to predict brains from labels instead of the elastic net i.e. the forward model pls.fit(Y_train, X_train). Note that X, and Y are flipped for the forward model.
    clf.fit(Ytrain, Xtrain)
    # b) Same as step b above. PLS returns a low rank weight matrix (pls.coef_), which you can use in place of W
    regression_params.loc[voxel,:] = [x[0] for x in clf.coef_]
    label_predictions1.loc[]
        

predictions = pandas.DataFrame(index=Xmat.index,columns=Ymat.columns)
W = pickle.load(open("%s/regression_params_dfs.pkl" %output_folder,"rb"))['regression_params'] # [28549 
for heldout in Xmat.index.tolist():
    heldout_X = Xmat.loc[heldout,:] #28K by 1
    Xtest = numpy.linalg.lstsq(W,heldout_X)[0]
    predictions.loc[heldout,:] = Xtest


# b) Test: estimate label scores using the forward model. With some algebra, you can show that the label scores are given by solving the linear regression problem:
# transpose(X_test - B) = transpose(W)*transpose(Y_predict)
Ypredict1 = clf.predict(holdout1X.reshape(1,-1))

.reshape(1,-1))[0][0]

regression_params.transpose().dot()

#c) You can either plug the label scores directly into something like AUC, or threshold the scores to get labels. I suggested sign, but you can use whatever thresholding makes the most sense given the way the data is setup. Let's discuss this more on a separate thread if it's causing issues.
 transpose(W)*transpose(Y_predict)


# Need to find where regression/intercept params are in this model
    predicted_nii1.loc[voxel,"nii"] = clf.predict(holdout1Y.reshape(1,-1))[0][0]
    predicted_nii2.loc[voxel,"nii"] = clf.predict(holdout2Y.reshape(1,-1))[0][0]

concept_vector1 =  pandas.DataFrame(holdout1Y)
concept_vector2 =  pandas.DataFrame(holdout2Y)
predicted_nii1 =  regression_params.dot(concept_vector1)[image1_holdout] + intercept_params["intercept"] 
predicted_nii2 =  regression_params.dot(concept_vector2)[image2_holdout] + intercept_params["intercept"]


c) Same as step c above using the PLS weights


# OLD
train = [x for x in X.index if x not in [image1_holdout,image2_holdout] and x in norm.index]
Xtrain = norm.loc[train,:]
Ytrain = numpy.array(X.loc[train,:]) 
# Use pls instead of regularized regression
clf = PLSRegression(n_components=number_components)
clf.fit(Xtrain, Ytrain)    

# Predict the labels
predicted_labels1 = [numpy.sign(x) for x in clf.predict(holdout1X_norm.reshape(1,-1))[0]]
predicted_labels2 = [numpy.sign(x) for x in clf.predict(holdout2X_norm.reshape(1,-1))[0]]
    
# Accuracy
accuracy1 = numpy.sum([1 for x in range(len(holdout1X)) if holdout1X.tolist()[x] == predicted_labels1[x]])/float(len(holdout1X))
accuracy2 = numpy.sum([1 for x in range(len(holdout2X)) if holdout1X.tolist()[x] == predicted_labels2[x]])/float(len(holdout2X))

# When normalized labels are derived from [0,1]
# accuracy1
# 0.43181818181818182
# accuracy2
# 0.32575757575757575

# When normalized labels are derived from [-1,1]
# accuracy1
# 0.46969696969696972
# accuracy2
# 0.51515151515151514

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
# {'TN': 62, 'FP': 58, 'FN': 12, 'TP': 0}

conf2 = calculate_confusion(predicted_labels2,holdout2X.tolist())
# {'TN': 62, 'FP': 56, 'FN': 0, 'TP': 14}

# Accuracy as number correct / number labels
number_labels1 = len(holdout1X[holdout1X==1])
number_labels2 = len(holdout1X[holdout2X==1])

accuracy1 = conf1["TP"]/float(number_labels1)
accuracy2 = conf2["TP"]/float(number_labels2)

# 0.0 and 1.0
