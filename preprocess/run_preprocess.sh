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

# Step 6: Compile group size results (typically ~60K script runs)
python 6.compile_explore_group_size_results.py $base
cp $base/data/ri_explore_size_results*.tsv ../analysis/reverse_inference/data/size_results

# Step 7: Run calculations for Wang Graph metric, compile results
RSCRIPT 7.run_wang_graph_comparison_sherlock.R $base
# This has a lot of jobs, and should be done manually
RSCRIPT 7.compile_wang_comparison.R $base
cp $base/data/wang_scores/contrast_defined_images_wang.tsv ../analysis/reverse_inference/data/size_results/contrast_defined_images_wang.tsv

# Optional web interface to produce data files for https://github.com/vsoch/semantic-image-comparison-web/tree/master/data
python 8.web_interface.py $base

# Now move on to analysis under ../analysis



