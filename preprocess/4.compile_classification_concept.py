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
    single_result = pickle.load(open(scores[i],"rb"))
    total=total+2
    correct=correct+single_result["number_correct"]
    comparison_df = comparison_df.append(single_result["comparison_df"])

# Use file names for index
index_names = []
for score in scores:
    fname = os.path.basename(score)
    index_names = index_names + [fname,fname,fname,fname]

comparison_df.index = index_names
accuracy = correct/float(total)
result = dict()
result["comparison_df"] = comparison_df
result["total"] = total
result["correct"] = correct
result["accuracy"] = accuracy
pickle.dump(result,open("%s/classification_results_binary_4mm.tsv" %results,"wb"))


# Parse results for weighted (ontology based) classification
scores_folder = "%s/classification_weighted" %(base)
scores = glob("%s/*.pkl" %scores_folder)

# Let's save a big data frame of the prediction scores
comparison_weighted = pandas.DataFrame(columns=["actual","predicted","cca_score"])

total = 0
correct = 0
for i in range(0,len(scores)):
    print "Parsing score %s of %s" %(i,len(scores))
    single_result = pickle.load(open(scores[i],"rb"))
    total=total+2
    correct=correct+single_result["number_correct"]
    comparison_weighted = comparison_weighted.append(single_result["comparison_df"])

accuracy = correct/float(total)
result = dict()
result["comparison_df"] = comparison_weighted
result["total"] = total
result["correct"] = correct
result["accuracy"] = accuracy
pickle.dump(result,open("%s/classification_results_weighted_4mm.pkl" %results,"wb"))

# Remove data frames from saved files (to make space)
for i in range(7,len(scores)):
    print "Parsing score %s of %s" %(i,len(scores))
    try:
        single_result = pickle.load(open(scores[i],"rb"))
        single_result.pop("regression_params",None)
        pickle.dump(scores[i],open(scores[i],"wb"))
    except:
        os.remove(scores[i])
