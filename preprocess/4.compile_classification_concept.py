#!/usr/bin/python
from glob import glob
import numpy
import pickle
import pandas
import sys
import os

#base = sys.argv[1]
base = "/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison"
results = "%s/results" %(base)
scores_folder = "%s/classification" %(base)
scores = glob("%s/*.pkl" %scores_folder)

# Let's save a big data frame of the prediction scores
comparison_df = pandas.DataFrame(columns=["actual","predicted","cca_score"])

total = 0
correct = 0
for i in range(0,len(scores)):
    print "Parsing score %s of %s" %(i,len(scores))
    result = pickle.load(open(scores[i],"rb"))
    total=total+2
    correct=correct+result["number_correct"]
    comparison_df = comparison_df.append(result["comparison_df"])

accuracy = correct/float(total)
result = dict()
result["comparison_df"] = comparison_df
result["total"] = total
result["correct"] = correct
result["accuracy"] = accuracy
pickle.dump(result,open("%s/classification_results_binary.tsv" %results,"wb"))

