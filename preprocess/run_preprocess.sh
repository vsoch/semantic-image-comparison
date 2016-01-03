#!/bin/sh

# You should first define a base directory for your data outputs
base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison

# The following scripts should be run from the directory where they are located.
# We will keep jobs and output in hidden folders, .job, and .out
mkdir .jobs
mkdir .out

# Step 0: obtaining NeuroVault images, saving data, and spatial image comparison
python 0.neurovault_comparison.py $base

#* This step has files that are moved from cluster into analysis
cp $base/results/contrast_defined_images_filtered.tsv ../analysis/wang/data
cp $base/results/contrast_defined_images_pearsonpd_similarity.tsv ../analysis/wang/data

# Step 1. Cognitive Atlas Prepartion for Reverse Inference
python 1.prep_reverse_inference.py $base
cp $base/web/index.html ../index.html

# Step 2: Generate groups of images associated with each concept node
python 2.run_calculate_likelihood.py $base
cp $base/data/likelihood/pbc_likelihood_group*.pkl ../analysis/reverse_inference/groups

# Step 3: Generate priors tables, and calculate reverse inference
python 3.run_calculate_reverse_inference.py

# Step 4. Compile reverse inference scores
python 4.compile_reverse_inferences_results.py
cp $base/data/reverse_inference_scores.tsv ../analysis/reverse_inference_data

# Step 5: Impact of group size on reverse inference scores
python 5.run_explore_group_size.py $base

# Step 6: STOPPED HERE I am so tired.
