#!/usr/bin/python

# This script will prepare a node by concept data frame, to be easily used for labels in the 
# classification framework.

from glob import glob
import pandas
import pickle
import re
import os

#base = sys.argv[1]
base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison"
data = "%s/data" %base
node_pickles = glob("%s/likelihood/*.pkl" %data)
results = "%s/results" %base
images_tsv = "%s/contrast_defined_images_filtered.tsv" %results
images = pandas.read_csv(images_tsv,sep="\t")

unique_contrasts = images.cognitive_contrast_cogatlas_id.unique().tolist()

# Images that do not match the correct identifier will not be used (eg, "Other")
expression = re.compile("cnt_*")
concepts = [os.path.basename(x).split("group_")[-1].replace(".pkl","") for x in node_pickles]
image_ids = images.image_id.unique().tolist()

# This is the model that we will want to build for each voxel
# Y (n images X 1) = X (n images X n concepts) * beta (n concepts X 1)
# Y is the value of the voxel (we are predicting)
# X is images by concepts, with values corresponding to concept labels
# beta are the coefficients we get from building the model
X = pandas.DataFrame(0,index=image_ids,columns=concepts)
for group_pkl in node_pickles:
    group = pickle.load(open(group_pkl,"rb"))
    concept_id = os.path.basename(group_pkl).split("group_")[-1].replace(".pkl","")
    print "Parsing concept %s" %(concept_id)
    image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
    image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
    X.loc[image_ids_in,concept_id] = 1 

X = X[X.sum(axis=1)!=0]
X.to_csv("%s/images_contrasts_df.tsv" %results,sep="\t")
