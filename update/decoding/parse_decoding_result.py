#!/usr/bin/python

from glob import glob
import numpy
import pickle
import pandas

base = sys.argv[1]
update = "%s/update" %base
output_folder = "%s/decoding" %update  # any kind of tsv/result file
results = "%s/results" %update  # any kind of tsv/result file

# Save accuracies, confusions, etc.
result_df = pandas.DataFrame(columns=["TP","TN","FP","FN","image1","image2"])
accuracy_overall = []

result_files = glob("%s/*" %(output_folder))
for r in range(len(result_files)):
    result_file = result_files[r]
    print("Parsing %s of %s..." %(r,len(result_files)))
    res = pickle.load(open(result_file,'rb'))
    image1,image2 = [int(x) for x in os.path.basename(result_file).split("_")[0:2]]
    rowname = "%s_%s" %(image1,image2)
    TP = res['confusions'][image1]["TP"] + res['confusions'][image2]["TP"]
    TN = res['confusions'][image1]["TN"] + res['confusions'][image2]["TN"]
    FP = res['confusions'][image1]["FP"] + res['confusions'][image2]["FP"]
    FN = res['confusions'][image1]["FN"] + res['confusions'][image2]["FN"]
    result_df.loc[rowname] = [TP,TN,FP,FN,image1,image2]
    accuracy_overall = accuracy_overall + [res["accuracy_overall"][image1] + res["accuracy_overall"][image2]]

# Mean and SD for accuracy
numpy.mean(accuracy_overall)
# 0.10
numpy.std(accuracy_overall)
# 0.067390565957290011

# Totals for confusion
result_df.sum()
# TP          10612
# TN             16
# FP         174540
# FN           3328
