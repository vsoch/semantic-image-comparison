#!/usr/bin/python
from glob import glob
import numpy
import pickle
import pandas
import sys
import os

base = sys.argv[1]

data = "%s/data" %base        # mostly images
scores_folder = "%s/individual_scores" %(data)   # output folder for individual scores
likelihood_pickles = glob("%s/likelihood/*.pkl" %(data))
scores = glob("%s/*.pkl" %scores_folder)

nodes = []

# First make a list of all the nodes, images
for i in range(0,len(likelihood_pickles)):
    node = likelihood_pickles[i]
    group = pickle.load(open(node,"rb"))
    all_images = group["in"] + group["out"]
    nodes.append(group["nid"])

# Parse image IDS
image_ids = [os.path.split(x)[1].replace(".nii.gz","") for x in all_images]

# We will save a flat data frame
binary_columns = ["ri_binary_%s" %(thresh) for thresh in range(0,14)]
ri_score = pandas.DataFrame(columns=["image_id","node","ri_distance","ri_range","in_count","out_count"] + binary_columns)

count=1
for s in range(0,len(scores)):
    print "Parsing %s of %s" %(s,len(scores))
    # Read in each score table, we will save to one master data frame
    result = pickle.load(open(scores[s],"rb"))
    ri_score.loc[count,"image_id"] = result["image_id"]
    ri_score.loc[count,"node"] = result["nid"]
    ri_score.loc[count,"ri_range"] = result["ri_ranges_query"]
    ri_score.loc[count,"ri_distance"] = result["ri_distance_query"]
    for thresh in range(0,14):
        ri_score.loc[count,"ri_binary_%s" %(thresh)] = result["ri_binary_%s_query" %(thresh)]
    ri_score.loc[count,"in_count"] = result["in_count"]
    ri_score.loc[count,"out_count"] = result["out_count"]
    count+=1
  
      
# Save table to file
ri_score.to_csv("%s/reverse_inference_scores.tsv" %data,sep="\t")
