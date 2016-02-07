#!/bin/sh

# You should first define a base directory for your data outputs
base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison

# The following scripts should be run from the directory where they are located.
# We will keep jobs and output in hidden folders, .job, and .out
mkdir .jobs
mkdir .out

# Download neurosynth data for later
git clone https://github.com/neurosynth/neurosynth-data
mv neurosynth-data $base/
tar -xzvf $base/neurosynth-data/current_data.tar.gz

# Step 0: obtaining NeuroVault images, saving data, and spatial image comparison
python 0.neurovault_comparison.py $base

#* This step has files that are moved from cluster into analysis
cp $base/results/contrast_defined_images_filtered.tsv ../analysis/wang/data
cp $base/results/contrast_defined_images_pearsonpd_similarity.tsv ../analysis/wang/data

# Step 1. Cognitive Atlas Prepartion for Semantic Comparison Analyses
python 1.prep_semantic_comparison.py $base

# groups of images associated with each concept node
cp $base/data/groups/pbc_group*.pkl ../analysis/classification/groups

# simple graph to show images associated with concept nodes
cp $base/web/index.html ../index.html

# Step 2: Make a node by concept data frame, weighted and non weighted
python 2.make_node_concept_df.py $base
cp $base/results/images_contrasts_df_weighted.tsv ../analysis/classification/data
cp $base/results/images_contrasts_df.tsv ../analysis/classification/data

# Step 3: Classification framework (run these separately!)
python 3.run_calculate_concept_map.py $base
python 3.run_calculate_concept_map_ontology.py $base
python 3.run_calculate_null.py $base # This can take weeks if not done with launcher

# Step 4. Compile classification results
python 4.compile_classification_concept.py
cp $base/results/classification_results_binary_4mm.tsv ../analysis/classification/data
cp $base/results/classification_results_weighted_4mm.tsv ../analysis/classification/data
cp $base/results/classification_results_null_4mm.tsv ../analysis/classification/data
mkdir ../analysis/classification/data/concept_maps
cp $base/results/classification_final/*.nii.gz ../analysis/classification/data/concept_maps

# Comparison with neurosynth (decoding of Z score concept maps of regression parameters)
python 4.comparison_with_neurosynth.py $base
cp $base/results/concept_regparam_decoding.txt ../analysis/classification/data

# Step 5: Run calculations for Wang Graph metric, compile results
RSCRIPT 5.run_wang_graph_comparison_sherlock.R $base
# This has a lot of jobs, and should be done manually
RSCRIPT 5.compile_wang_comparison.R $base
cp $base/data/wang_scores/contrast_defined_images_wang.tsv ../analysis/wang/data/contrast_defined_images_wang.tsv

# Optional web interface to produce data files for https://github.com/vsoch/semantic-image-comparison-web/tree/master/data
python 6.web_interface.py $base

# Now move on to analysis under ../analysis
