#!/usr/bin/python

# We want to investigate how changing the size of the set for comparison (the "in" set) 
#     influences the outcome. We will select ?(a subset) of concepts with many tagged images, 
#     and at each size of the "in" set (from 1..N) calculate a score for each held out image.

# For each concept, (N=132):
#    Select a number G from 1...[total "in" group] as the size of the set to investigate
#        For each image in (entire) "in"set:
#            For some number of iterations:
#                Randomly select G other images for "in" set, calculate P(concept|image)
#                Take mean score of iterations as P(concept|image)

from pybraincompare.ontology.inference import calculate_reverse_inference_distance
from pybraincompare.mr.datasets import get_standard_mask
from pybraincompare.compare.mrutils import get_images_df
from numpy.random import choice
import pickle
import pandas
import sys
import os

image = sys.argv[1]             # Full path of input image
node = sys.argv[2]              # pickle with groups for concept node
output_pkl = sys.argv[3]        # Path to save output file pickle
group_size = int(sys.argv[4])   # Number of other images to randomly select
                                # Note that as group_size approaches size of "in" set,
                                # we are randomly sampling to get the same image set

group = pickle.load(open(node,"rb"))
standard_mask = get_standard_mask()
image_id = os.path.split(image)[1].replace(".nii.gz","")

# Remove image from the in and out groups
in_group = [x for x in group["in"] if x != image]
out_group = [x for x in group["out"] if x != image]

# Randomly sample "group_size" from the in set
in_group = choice(in_group,group_size,replace=False).tolist()

# We will save to a result object
result = dict()

# Calculate reverse inference (posterior) for query image
# P(node mental process|activation) = P(activation|mental process) * P(mental process)
# divided by
# P(activation|mental process) * P(mental process) + P(A|~mental process) * P(~mental process)
# P(activation|mental process): my voxelwise prior map

# This is a reverse inference score, the p(cognitive process | query)
ri = calculate_reverse_inference_distance(image,in_group,out_group,standard_mask)
result["ri_query"] = ri
result["bayes_factor"] = ri / 0.5

# Save rest of varibles to result object
result["in_count"] = len(in_group)
result["out_count"] = len(group["out"])
result["image"] = image
result["nid"] = group["nid"]
result["concept_node"] = node
result["image_id"] = image_id

# Save result to file
pickle.dump(result,open(output_pkl,"wb"))
