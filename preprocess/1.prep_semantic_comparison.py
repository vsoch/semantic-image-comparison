#!/usr/bin/python
### REVERSE INFERENCE

import pandas
import os
import re
import sys

# For the VM: these paths will be environmental variables
base = sys.argv[1]
results = "%s/results" %base  # any kind of tsv/result file
data = "%s/data" %base        # mostly images
web = "%s/web" %base

if not os.path.exists(web):
    os.mkdir(web)

# Read in images metadata
images = pandas.read_csv("%s/contrast_defined_images_filtered.tsv" %results,sep="\t")

## STEP 1: GENERATE TRIPLES DATA STRUCTURE
from cognitiveatlas.datastructure import concept_node_triples, get_concept_categories

'''
  id    parent  name
  1 none BASE                   # there is always a base node
  2 1   MEMORY                  # high level concept groups
  3 1   PERCEPTION              
  4 2   WORKING MEMORY          # concepts
  5 2   LONG TERM MEMORY
  6 4   image1.nii.gz           # associated images (discovered by way of contrasts)
  7 4   image2.nii.gz
'''

# We need a dictionary to look up image lists by contrast ids
unique_contrasts = images.cognitive_contrast_cogatlas_id.unique().tolist()

# Images that do not match the correct identifier will not be used (eg, "Other")
expression = re.compile("cnt_*")
unique_contrasts = [u for u in unique_contrasts if expression.match(u)]

image_lookup = dict()
for u in unique_contrasts:
   image_lookup[u] = images.image_id[images.cognitive_contrast_cogatlas_id==u].tolist()

output_triples_file = "%s/task_contrast_triples.tsv" % results

# Create a data structure of tasks and contrasts for our analysis
concept_node_triples(image_dict=image_lookup,output_file=output_triples_file)
relationship_table = pandas.read_csv(output_triples_file,sep="\t")

# If you want to include concept categories
#meta_data = get_concept_categories()

## STEP 2: VISUALIZATION WITH PYBRAINCOMPARE
from pybraincompare.template.templates import save_template
from pybraincompare.ontology.tree import named_ontology_tree_from_tsv, make_ontology_tree_d3

# First let's look at the tree structure - here is with categories
#tree = named_ontology_tree_from_tsv(relationship_table,output_json=None,meta_data=meta_data)

# And without
tree = named_ontology_tree_from_tsv(relationship_table,output_json=None)

html_snippet = make_ontology_tree_d3(tree)
save_template(html_snippet,"%s/index.html" %web)

## STEP 3: DERIVATION OF CONCEPT NODE GROUPS
# The following steps should be run in a cluster environment
# this will show an example in a single batch script
from pybraincompare.ontology.inference import groups_from_tree
from pybraincompare.mr.datasets import get_standard_mask

standard_mask = get_standard_mask()
input_folder = "%s/resampled_z" %data # Images folder
output_folder = "%s/groups" %data
os.mkdir(output_folder)
# Take a look at "image_pattern" and "node_pattern" inputs if not using NeuroVault and pybraincompare tree

###### 3.1 First generate node groups
groups = groups_from_tree(tree,standard_mask,input_folder,output_folder=output_folder)

# Bug that image 109 is not included in groups, here are the contrasts
concept_109 =["trm_567982752ff4a","trm_4a3fd79d0afcf","trm_5534111a8bc96","trm_557b48a224b95","trm_557b4a81a4a17","trm_4a3fd79d0b64e","trm_4a3fd79d0a33b","trm_557b4a7315f1b","trm_4a3fd79d0af71","trm_557b4b56de455","trm_557b4add1837e"]

pickle_folder = "%s/groups" %data
image109 = "%s/resampled_z/000109.nii.gz" %data
for single_concept in concept_109:
    concept_pickle = pickle.load(open("%s/pbc_group_%s.pkl" %(pickle_folder,single_concept),"rb"))
    concept_pickle["in"].append(image109)  
    pickle.dump(concept_pickle,open("%s/pbc_group_%s.pkl" %(pickle_folder,single_concept),"wb"))

concept_out = [x for x in concepts if x not in concept_109]
for single_concept in concept_out:
    concept_pickle = pickle.load(open("%s/pbc_group_%s.pkl" %(pickle_folder,single_concept),"rb"))
    concept_pickle["out"].append(image109)  
    pickle.dump(concept_pickle,open("%s/pbc_group_%s.pkl" %(pickle_folder,single_concept),"wb"))
