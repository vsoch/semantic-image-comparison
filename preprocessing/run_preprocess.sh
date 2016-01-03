#!/bin/sh

# You should first define a base directory for your data outputs
base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison

# The following scripts should be run from the directory where they are located.
# We will keep jobs and output in hidden folders, .job, and .out
mkdir .jobs
mkdir .out

# Step 0: obtaining NeuroVault images and saving data
python 0.neurovault_comparison.py $base


