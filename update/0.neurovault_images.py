#!/usr/bin/env python2

import os
import pandas
import sys
import numpy
from glob import glob
from pyneurovault import api
from cognitiveatlas.api import get_task, get_concept

# ** Numbers in parens reflect number of images left AFTER opertion performed! **

# Get all collections
collections = api.get_collections()

# Filter images to those that have a DOI
collections = collections[collections.DOI.isnull()==False]

# Get image meta data for collections (N=7308)
images = api.get_images(collection_pks=collections.collection_id.tolist())

# Get rid of any not in MNI (N=7002)
images = images[images.not_mni == False]

# Get rid of thresholded images (N=6398)
images = images[images.is_thresholded == False]

# Remove single subject maps (N=1724)
images = images[images.analysis_level!='single-subject']
images = images[images.number_of_subjects!=1]

# Remove non fmri-BOLD (N=910)
images = images[images.modality=='fMRI-BOLD']

# How many collections left? (N=93)
unique_collections = images.collection_id.unique().tolist()

# Which are missing number of subjects?
images[images.number_of_subjects.isnull()].collection_id.unique().tolist()

#[1054, 866, 1037, 555, 418, 419, 1035, 830, 1325, 1055, 917, 648, 160, 422, 421, 110, 1072, 1126, 693, 1029, 1351, 507, 474, 445, 599, 1370, 1066]

# How many collections have at least one non-single subject map with a contrast other than None/Other?
keepers = []
for unique_collection in unique_collections:
    collection_images = images[images.collection_id==unique_collection]
    other_contrasts = collection_images.cognitive_contrast_cogatlas[collection_images.cognitive_contrast_cogatlas.isin([None,"None / Other"])==False].unique()
    print "Collection %s has %s tagged maps" %(unique_collection,len(other_contrasts))
    if len(other_contrasts) > 0:
        keepers.append(unique_collection)

len(keepers)
#39

# This will allow us to know how many papers are there that would potentially have to be read and interpreted.

# Retrieve list of collections to find number_of_subjects
collections = api.get_collections(images.collection_id.unique().tolist())

