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
from pybraincompare.mr.transformation import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from glob import glob
import pickle
import pandas
import nibabel
import sys
import os

base = sys.argv[1]
data = "%s/data" %base
node_folder = "%s/likelihood" %data
results = "%s/results" %base  # any kind of tsv/result file

# Images by Concepts data frame
labels_tsv = "%s/images_contrasts_df.tsv" %results
images = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
output_folder = "%s/classification_final" %results

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Images by Concept data frame, our X
X = pandas.read_csv(labels_tsv,sep="\t",index_col=0)

# Node pickles
node_pickles = glob("%s/*.pkl" %node_folder)

# Get standard mask, 4mm
standard_mask=get_standard_mask(4)

# We will save data to dictionary
result = dict()

# Load any group data to get image paths
group = node_pickles[0]
group = pickle.load(open(group,"rb"))

# Change paths in group pickle to point to 4mm folder
group["in"] = [x.replace("resampled_z","resampled_z_4mm") for x in group["in"]]
group["out"] = [x.replace("resampled_z","resampled_z_4mm") for x in group["out"]]

concepts = X.columns.tolist()

# We will go through each voxel (column) in a data frame of image data
mr = get_images_df(file_paths=group["in"] + group["out"],mask=standard_mask)
image_paths = group["in"] + group["out"]
image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
image_ids = image_ids_in + image_ids_out
mr.index = image_ids
   
# what we can do is generate a predicted image for a particular set of concepts (e.g, for a left out image) by simply multiplying the concept vector by the regression parameters at each voxel.  then you can do the mitchell trick of asking whether you can accurately classify two left-out images by matching them with the two predicted images. 

regression_params = pandas.DataFrame(0,index=mr.columns,columns=concepts)

# Build voxelwise models using all data
print "Training voxels..."
for voxel in mr.columns:
    Y = mr.loc[:,voxel].tolist() 
    # Use regularized regression
    clf = linear_model.ElasticNet(alpha=0.1)
    clf.fit(X,Y)
    regression_params.loc[voxel,:] = clf.coef_.tolist()

result["regression_params"] = regression_params

# Convert regression params into Z scores (to make maps)
print "Generating regression parameter z scores..."
regression_params_z = pandas.DataFrame(0,index=mr.columns,columns=concepts)
for single_concept in regression_params.columns.tolist():
    print "Calculating Z for and saving %s" %(single_concept)
    z = (regression_params.loc[:,single_concept] - regression_params.loc[:,single_concept].mean()) / regression_params.loc[:,single_concept].std()
    regression_params_z.loc[:,single_concept] = z
    # Turn into nifti images
    nii = numpy.zeros(standard_mask.shape)
    nii[standard_mask.get_data()!=0] = z.tolist()
    nii = nibabel.Nifti1Image(nii,affine=standard_mask.get_affine())
    nibabel.save(nii,"%s/%s_regparam_z.nii.gz" %(output_folder,single_concept))

result["regression_params_z"] = regression_params_z

# Save result to file
pickle.dump(result,open("%s/regression_params_dfs.pkl" %output_folder,"wb"))
#result = pickle.load(open("%s/regression_params_dfs.pkl" %output_folder,"rb"))

# DECODING
print "Starting neurosynth decoding..."
'''
# Here is how to do with neurosynth API - it's slow and annoying. Do not recommend
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import decode
from nilearn.image import resample_img

neurosynth_data = "%s/neurosynth-data" %base
dataset = Dataset('%s/database.txt' %neurosynth_data)
dataset.add_features('%s/features.txt' %neurosynth_data)
decoder = decode.Decoder(dataset) # select all features
pickle.dump(decoder,open("%s/decoder.pkl" %output_folder,"wb"))

concept_maps = glob("%s/*.nii.gz" %output_folder)
brain_2mm = get_standard_mask(2)

# Decoder needs 2mm images
concept_maps_2mm = []
for concept_map in concept_maps:
    twomm = concept_map.replace("regparam_z","regparam_z_2mm")
    nii = resample_img(concept_map,target_affine=brain_2mm.get_affine())
    nibabel.save(nii,twomm)
    concept_maps_2mm.append(twomm)

decode_result_file = "%s/concept_regparam_decoding.txt" %results
decode_result = decoder.decode(concept_maps_2mm, save=decode_result_file)
'''

# URL-based decoding (possible since all images are in NeuroVault) - much faster!

from urllib2 import Request, urlopen, HTTPError
import pandas
import json

def get_url(url):
    request = Request(url)
    response = urlopen(request)
    return response.read()

# First, just get all the terms
ns = json.loads(get_url("http://neurosynth.org/decode/data/?neurovault=308"))
terms = []
for term in  ns["data"]:
    terms.append(term["analysis"])

# Now decode each of our images
unique_images = images.image_id.unique().tolist()
decode = pandas.DataFrame(index=unique_images,columns=terms)

for unique_image in unique_images:
    print "Decoding image %s" %(unique_image)
    ns = json.loads(get_url("http://neurosynth.org/decode/data/?neurovault=%s" %unique_image))
    for term in  ns["data"]:
        decode.loc[unique_image,term["analysis"]] = term["r"]

decode_result_file = "%s/concept_regparam_decoding.tsv" %results
decode.to_csv(decode_results_file,sep="\t")
