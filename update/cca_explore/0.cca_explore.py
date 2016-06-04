#!/usr/bin/python

from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from pybraincompare.compare.maths import calculate_correlation
from pybraincompare.compare.mrutils import get_images_df
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.mr.transformation import *
from sklearn import linear_model
import matplotlib.pyplot as plt
from glob import glob
import pickle
import pandas
import nibabel
import sys
import os

base = '/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison'
update = "%s/update" %base
results = "%s/results" %base  # any kind of tsv/result file

# Images by Concepts data frame (YES including all levels of ontology)
labels_tsv = "%s/images_contrasts_df.tsv" %results
image_lookup = "%s/image_nii_lookup.pkl" %update

# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

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
   
## CCA ANALYSIS ############################################################################
# Dataset based latent variables model

# Let's choose a random voxel
mr = mr.fillna(0)

output_folder = '%s/results/cca' %update
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

print "Training voxels..."
for voxel in mr.columns:
    print "Training voxel %s of %s" %(voxel,len(mr.columns))
    holdout = numpy.random.choice(mr.index.tolist(),5).tolist()
    train = [x for x in X.index if x not in holdout and x in mr.index]
    test = [image1_holdout,image2_holdout]
    Ytrain = mr.loc[train,voxel].tolist()
    Ytest = mr.loc[test,voxel].tolist()
    Xtrain = numpy.array(X.loc[train,:]) 
    Xtest = X.loc[test,:]
    plsca = PLSCanonical(n_components=2)
    plsca.fit(Xtrain, Ytrain)
    X_train_r, Y_train_r = plsca.transform(Xtrain, Ytrain)
    X_test_r, Y_test_r = plsca.transform(Xtest, Ytest)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.plot(X_train_r[:, 0], Y_train_r[:], "ob", label="train")
    plt.plot(X_test_r[:, 0], Y_test_r[:], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('PLSCA: Comp. 1: X vs Y (test corr = %.2f)' %numpy.corrcoef(X_test_r[:, 0], Y_test_r[:])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")
    plt.subplot(222)
    cca = CCA(n_components=2)
    cca.fit(Xtrain, Ytrain)
    plt.plot(X_train_r[:, 0], Y_train_r[:], "ob", label="train")
    plt.plot(X_test_r[:, 0], Y_test_r[:], "or", label="test")
    plt.xlabel("x scores")
    plt.ylabel("y scores")
    plt.title('CCA: Comp. 1: X vs Y (test corr = %.2f)' %numpy.corrcoef(X_test_r[:, 0], Y_test_r[:])[0, 1])
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc="best")
    plt.savefig('%s/plsca_%s.pdf' %(output_folder,voxel))
    plt.close()
