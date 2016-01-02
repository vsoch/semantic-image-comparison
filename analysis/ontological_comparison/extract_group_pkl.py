#!/usr/bin/python

# We want to do some basic analysis in R (sorry world, I really like R for analyses!) so let's extract groups from the python pickles.

from glob import glob
import pandas
import pickle

groups = glob("groups/*.pkl")
output_tsv = "groups/all_groups.tsv"

df = pandas.DataFrame(columns=["image","group","direction","name"])

count = 1
for g in range(0,len(groups)):
    group_pkl = groups[g]
    print "Processing group %s of %s" %(g,len(groups))
    group = pickle.load(open(group_pkl,"rb"))
    for image in group["in"]:
        df.loc[count] = [image,group["nid"],"in",group["name"]]
        count+=1
    for image in group["out"]:
        df.loc[count] = [image,group["nid"],"out",group["name"]]
        count+=1

df.to_csv(output_tsv,sep="\t",index=None)
