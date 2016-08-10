#!/usr/bin/python

from pybraincompare.compare.maths import calculate_correlation
from pybraincompare.compare.mrutils import get_images_df
from pybraincompare.mr.datasets import get_standard_mask
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from pybraincompare.mr.transformation import *
from sklearn.metrics import r2_score, confusion_matrix
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

# Save the values for later, but 0 must be represented as -1
holdout1X = X.loc[image1_holdout,:]
holdout2X = X.loc[image2_holdout,:]
holdout1X[holdout1X == 0] = -1
holdout2X[holdout2X == 0] = -1

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

# accuracy1
# 0.43181818181818182
# accuracy2
# 0.32575757575757575
