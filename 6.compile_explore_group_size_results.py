#!/usr/bin/python
from glob import glob
import numpy
import pickle
import pandas
import re
import os

base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison/"
data = "%s/data" %base        # mostly images
scores_folder = "%s/group_size_vary_scores" %(data)     # output folder for group size scores
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

# We will parse the scores for node separately, there is too much data to put into one
for n in range(len(nodes)):
    node = nodes[n]
    print "Parsing %s of %s" %(n,len(nodes))
    nodere = re.compile("/%s_size" %(node))
    node_scores = [s for s in scores if nodere.search(s)]
    # Let's make a flat data frame this time
    ri_scores = pandas.DataFrame(columns=["image_id","node","ri_score","bayes_factor","in_count","out_count"])
    for s in range(0,len(node_scores)):
        r = pickle.load(open(node_scores[s],"rb"))
        result_id = node_scores[s].replace(".pkl","")
        ri_scores.loc[result_id] = [r["image_id"],r["nid"],r["ri_query"],r["bayes_factor"],r["in_count"],r["out_count"]]    
    # Save tables with result to file
    ri_scores.to_csv("%s/ri_explore_size_results_%s.tsv" %(data,node),sep="\t")
