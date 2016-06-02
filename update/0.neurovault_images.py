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

# How many collections left? (N=123)
unique_collections = images.collection_id.unique().tolist()

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
