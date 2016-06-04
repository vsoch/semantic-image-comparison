#!/usr/bin/python

from cognitiveatlas.api import get_concept
import pandas
import numpy
import pickle
import os
import re
import sys

# For the VM: these paths will be environmental variables
base = sys.argv[1]
results = "%s/results" %base  # any kind of tsv/result file
update = "%s/update" %base    # second round analyses
data = "%s/data" %base        # mostly images
web = "%s/web" %base

if not os.path.exists(update):
    os.mkdir(update)

# Read in images metadata
images = pandas.read_csv("%s/contrast_defined_images_filtered.tsv" %results,sep="\t")

unique_concepts = dict()
for row in images.iterrows():
    idx = row[1].image_id
    # Bug with getting contrasts for images:
    if idx == 109:
        unique_concepts[idx] = ["trm_567982752ff4a","trm_4a3fd79d0afcf","trm_5534111a8bc96",
                                "trm_557b48a224b95","trm_557b4a81a4a17","trm_4a3fd79d0b64e","trm_4a3fd79d0a33b",
                                "trm_557b4a7315f1b","trm_4a3fd79d0af71","trm_557b4b56de455","trm_557b4add1837e"]
    elif idx == 118:
        unique_concepts[idx] = ["trm_4a3fd79d0b642","trm_4a3fd79d0a33b","trm_557b4a7315f1b","trm_4a3fd79d0af71",
                                "trm_557b4b56de455"]
    else:
        contrast = row[1].cognitive_contrast_cogatlas_id
        concepts = get_concept(contrast_id=contrast)
        concepts = numpy.unique(concepts.pandas.id).tolist() 
        unique_concepts[idx] = concepts
    
all_concepts = []
for image_id,concepts in unique_concepts.iteritems():
    for concept in concepts:
        if concept not in all_concepts:
            all_concepts.append(concept)


res = {"all_concepts":all_concepts,"unique_concepts":unique_concepts,"images":images}

## STEP 1: GENERATE IMAGE BY CONCEPT DATA FRAME
concept_df = pandas.DataFrame(0,columns=all_concepts,index=images.image_id.unique().tolist())
for image_id,concepts in unique_concepts.iteritems():
    concept_df.loc[image_id,concepts] = 1   

res["concept_df"] = concept_df
pickle.dump(res,open("%s/concepts.pkl" %update,"wb"))
concept_df.to_csv("%s/concepts_binary_df.tsv" %update,sep="\t")

## STEP 2: Generate image lookup
image_folder = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/data/resampled_z_4mm"
files = glob("%s/*.nii.gz" %image_folder)

lookup = dict()
for f in files:
    image_id = int(os.path.basename(f).strip(".nii.gz"))
    if image_id in concept_df.index:
        lookup[image_id] = f
    else:
        print "Cannot find image %s in concept data frame" %(image_id)

pickle.dump(lookup,open("%s/image_nii_lookup.pkl" %update,"wb"))
