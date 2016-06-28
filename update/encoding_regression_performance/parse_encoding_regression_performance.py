#!/usr/bin/python
from cognitiveatlas.api import get_concept
from repofish.utils import save_json
from glob import glob
import numpy
import pickle
import pandas
import json
import sys
import os
import re

base = sys.argv[1]
update = "%s/update" %(base)
results = "%s/results" %(update)
results_old = "%s/results" %(base)
scores_folder = "%s/performance" %(update)
scores = glob("%s/*.pkl" %scores_folder)

len(scores)
# 4278

# Let's save a big data frame of the prediction scores
comparison_df = pandas.DataFrame(columns=["actual","predicted","cca_score"])

# If images are the same contrast, don't include in accuracy calculation
images_tsv = "%s/contrast_defined_images_filtered.tsv" %results_old
images = pandas.read_csv(images_tsv,sep="\t")

# Save total and correct
total = 0
correct = 0

# Save a vector of r2 scores for each image
r2s = dict()

# Use file names for index
index_names = []
for i in range(0,len(scores)):
    print "Parsing score %s of %s" %(i,len(scores))
    single_result = pickle.load(open(scores[i],"rb"))
    fname = os.path.basename(scores[i])
    cdf = single_result["comparison_df"]
    two_images = cdf["actual"].unique().tolist()
    contrast_ids = images.cognitive_contrast_cogatlas_id[images.image_id.isin(two_images)].unique()
    # Only include for images with different contrast
    if len(contrast_ids) == 2:
        total=total+2
        correct=correct+single_result["number_correct"]
        comparison_df = comparison_df.append(single_result["comparison_df"])
        index_names = index_names + [fname,fname,fname,fname]
        # Save the r2s
        r2_fields = [x for x in single_result.keys() if re.search("r2_",x)]
        for r2_field in r2_fields:
            image_id = int(r2_field.split("_")[1])
            if image_id in r2s:
                r2s[image_id].append(single_result[r2_field])
            else:
                r2s[image_id] = [single_result[r2_field]]

# Write r2s to json so we can make javascript for it
save_json(r2s,"%s/performance_r2s_norm.json" %results)

# Write text for html page select box
for image in images.iterrows():
    image_id = image[1].image_id
    image_contrast = image[1].cognitive_contrast_cogatlas
    image_task = image[1].cognitive_paradigm_cogatlas
    print "<option value=%s>%s: %s</option>" %(image_id,image_task,image_contrast)

comparison_df.index = index_names
accuracy = correct/float(total)
result = dict()
result["comparison_df"] = comparison_df
result["total"] = total
result["correct"] = correct
result["accuracy"] = accuracy
pickle.dump(result,open("%s/classification_results_binary_4mm_perform_norm.pkl" %results,"wb"))

#accuracy without norm
# 0.75578676642506426
# correct
# 6465
# total
# 8554

# accuracy
# 0.84276361935936406
# total
# 8554
# correct
# 7209


# Create a confusion matrix
binary_df = result["comparison_df"]
unique_images = [int(x) for x in binary_df.actual.unique().tolist()]
confusion = pandas.DataFrame(0,index=unique_images,columns=unique_images)

# We cannot evaluate images with NaN - meaning the predicted image vector was empty
nanimages = []

print "Generating confusion matrix..."
for i in range(0,len(scores)):
    print "Parsing confusion matrix for %s of %s" %(i,len(scores))
    single_result = pickle.load(open(scores[i],"rb"))    
    cdf = single_result["comparison_df"]
    two_images = cdf["actual"].unique().tolist()
    contrast_ids = images.cognitive_contrast_cogatlas_id[images.image_id.isin(two_images)].unique()
    # Only include for images with different contrast
    if len(contrast_ids) == 2:
        actuals = cdf.actual.unique().tolist()
        cdf.index = cdf.predicted.tolist()
        if cdf["cca_score"].isnull().any():
            predicted_nulls = cdf["predicted"][cdf["cca_score"].isnull()].unique()
            for predicted_null in predicted_nulls:
                if predicted_null not in nanimages:
                    nanimages.append(predicted_null)
        else:
            for actual in actuals:
                predictions = cdf.cca_score[cdf.actual==actual]
                predictions.sort_values(ascending=False,inplace=True)
                predicted = int(predictions.index[0])
                actual = int(actual)  
                confusion.loc[actual,predicted] = confusion.loc[actual,predicted] + 1

confusion.to_csv("%s/classification_confusion_binary_4mm_perform.tsv" %results,sep="\t")

# Normalize confusion 
confusion = confusion.divide(confusion.sum(1).tolist())
confusion.to_csv("%s/classification_confusion_binary_4mm_perform_norm.tsv" %results,sep="\t")


# CONCEPT CONFUSION ####################################################################
# generate a RIGHT/WRONG by concepts data frame, to see how often different concepts are associated with correct or incorrect prediction
confusion = pandas.read_csv("%s/classification_confusion_binary_4mm_perform.tsv" %results,sep="\t",index_col=0)

# Read in concept labels
labels_tsv = "%s/images_contrasts_df.tsv" %results_old
labels = pandas.read_csv(labels_tsv,sep="\t",index_col=0)
concepts = labels.columns.tolist()
concepts_df = pandas.DataFrame(0,columns=["correct","incorrect"],index=concepts)

for row in confusion.iterrows():
    actual = str(row[0])
    actual_concepts = labels.loc[int(actual)][labels.loc[int(actual)]==1].index.tolist()
    predictions = row[1]
    for predicted,predicted_count in predictions.iteritems():
        if predicted == actual:
            concepts_df.loc[actual_concepts,"correct"] = concepts_df.loc[actual_concepts,"correct"] + predicted_count
        else:
            concepts_df.loc[actual_concepts,"incorrect"] = concepts_df.loc[actual_concepts,"incorrect"] + predicted_count

# Add the number of images
for concept_name in labels.columns:
    number_images = labels[concept_name][labels[concept_name]==1].shape[0]
    concepts_df.loc[concept_name,"number_images"] = number_images

concepts_df.to_csv("%s/classification_concept_confusion_cogatid_perform.tsv" %results,sep="\t")

# Replace concept ids with concept names
conceptnames = []
for conceptname in concepts_df.index:
    conceptnames.append(get_concept(id=conceptname).json[0]["name"])

concepts_df.index = conceptnames        
concepts_df.to_csv("%s/classification_concept_confusion_perform.tsv" %results,sep="\t")

# Normalize by the row count (to see what percentage of the time we get it wrong/right)
concepts_df_norm = pandas.DataFrame(columns=["correct","incorrect","number_images"])
for row in concepts_df.iterrows():
   rowsum = row[1][0:2].sum()
   if rowsum != 0:
       norm_values = [float(x)/rowsum for x in row[1].tolist()[0:2]]
       norm_values.append(concepts_df.loc[row[0],"number_images"])
       concepts_df_norm.loc[row[0]] = norm_values

concepts_df_norm.sort(columns=["correct"],ascending=False,inplace=True)
concepts_df_norm.to_csv("%s/classification_concept_confusion_norm_perform.tsv" %results,sep="\t")

# COLLECTION / TASK CONFUSION ###########################################################
unique_images = confusion.index.tolist()
collections = images.collection_id[images.image_id.isin(unique_images)].tolist()
tasks = images.cognitive_paradigm_cogatlas_id[images.image_id.isin(unique_images)].tolist()

confusion_categories = pandas.DataFrame(index=unique_images,columns=unique_images)
confusion.columns = [int(x) for x in confusion.columns.tolist()]

for i in range(len(confusion)):
    image1_name = confusion.index[i]
    row_counts = confusion.loc[image1_name]
    row_counts[image1_name] = 0 # get rid of diagonal
    for j in range(len(row_counts)):
        image2_name = row_counts.index[j]
        count = row_counts[image2_name]
        if image1_name==image2_name:
            continue
        if collections[i]==collections[j]:
            if tasks[i]==tasks[j]:
                confusion_categories.loc[image1_name,image2_name] = 0
            else:
                confusion_categories.loc[image1_name,image2_name] = 1
        else:
            confusion_categories.loc[image1_name,image2_name] = 2
            
confusion_result = dict()
# within-task (0), within-collection (between-task) (1), bw-collection (2)
confusion_result["confusion_categories_df"] = confusion_categories
confusion_result["confusion_categories"] = {0:"within-task",1:"within-collection (between-task)",2:"between-collection"}
pickle.dump(confusion_result,open("%s/confusion_categories.pkl" %results,"wb"))
confusion_result = pickle.load(open("%s/confusion_categories.pkl" %results,"rb"))

# Now calculate final answers!
value_counts = confusion_categories.apply(pandas.value_counts).fillna(0).sum(axis=1)
# First column shows the index (the value in the DataFrame) and second shows the count
value_counts
#0     322
#1     450
#2    7784

confusion = pandas.read_csv("%s/classification_confusion_binary_4mm_perform.tsv" %results,sep="\t",index_col=0)
confusion.columns = [int(x) for x in confusion.columns.tolist()]

confusion_labels = ["within-task","within-collection (between-task)","between-collection"]
normalized_confusion = dict()
for i in range(3):
    confusions_for_i = float(numpy.sum(confusion.values[confusion_categories.values==i]))
    print "%s confusions out of %s for the category" %(confusions_for_i,value_counts.loc[i])
    normalized_value = confusions_for_i / value_counts.loc[i]
    normalized_confusion[confusion_labels[i]] = normalized_value

# >>> normalized_confusion, not standard images
# {'within-collection (between-task)': 0.26222222222222225, 'within-task': 0.25465838509316768, 'between-collection': 0.24036485097636176}
 
# 82.0 confusions out of 322.0 for the category
# 118.0 confusions out of 450.0 for the category
# 1871.0 confusions out of 7784.0 for the category


# >>> normalized_confusion, standard images
# {'within-collection (between-task)': 0.20666666666666667, 'within-task': 0.2360248447204969, 'between-collection': 0.23098663926002055}
 
# 76.0 confusions out of 322.0 for the category within-task
# 93.0 confusions out of 450.0 for the category  within-collection
# 1798.0 confusions out of 7784.0 for the category between-collection


# COMPILE NULL ######################################################################
scores_folder = "%s/null" %(update)
scores = glob("%s/*.pkl" %scores_folder)

len(scores)
#1000
comparison_null = []

for i in range(0,len(scores)):
    print "Parsing score %s of %s" %(i,len(scores))
    single_result = pickle.load(open(scores[i],"rb"))
    comparison_null.append(single_result["accuracy"])

accuracy = numpy.mean(comparison_null)
result = dict()
result["comparison_null"] = comparison_null
result["accuracy"] = accuracy
pickle.dump(result,open("%s/classification_results_null_4mm.pkl" %results,"wb"))
#>>> result["accuracy"]
#0.49753903881477762

# What percentile is the actual accuracy in the null distribuition?
# se scipy.stats.percentileofscore to find the percentile in the null distribution for the observed value
# Do two sample T test against actual vs null accuracies
result_binary = pickle.load(open("%s/classification_results_binary_4mm_perform_norm.pkl" %results,"rb"))
from scipy.stats import percentileofscore

# Default kind="rank" - multiple matches, average the percentage rankings of all matching scores. percentileofscore(a, score, kind='rank')
binary_score = percentileofscore(comparison_null, result_binary["accuracy"])

# the p value is 100 minus that percentile, divided by number of samples in null
# p_cutoff=(100-percentile)/100 +1/<number of samples in null distribution>
pval_binary = ((100.0-binary_score)/100.0) + 1.0/len(comparison_null)
# 0.001
# @russpold you want to add one because you want to say that the p is less than the most extreme value - thus, if you are 100 percent and there were 1000 samples in the null distribution, you want to say that p<0.001 rather than giving an exact p value of p=0.000 

# weighted not the appropriate null distribution, but tested anyway
result_binary["pval"] = pval_binary

pickle.dump(result_binary,open("%s/classification_results_binary_4mm.pkl" %results,"wb"))
