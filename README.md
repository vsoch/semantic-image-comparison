# Semantic Image Comparison

## Overview
This work aims to assess the value of semantic (ontological) image comparison. We conduct the following analyses:

- [classification value](analysis/classification): Our main analysis is testing the ability of semantic image comparison scores on our ability to classify images. We show that semantic annotations can be used succesfully in a classification framework, and thus that there is value in performing semantic image comparison.
- [wang graph similarity](analysis/wang): The Wang Graph metric is a commonly used strategy to assess the similarity of gene sets, and we apply this to compare images based on their Cognitive Atlas tags. We compare an image similarity matrix of this metric against traditional spatial image comparison to assess if the method captures something that we understand about images.

## Tools
All analyses are driven by the [NeuroVault](http://www.neurovault.org) and [Cognitive Atlas](http://www.cognitiveatlas.org) APIs, meaning that they can be reproduced as these databases are updated. While this particular workflow requires a cluster environment for performing many of the tests, the resources are all publicly available by way of the [NeuroVault](http://www.github.com/NeuroVault/pyneurovault), [Cognitive Atlas](http://www.github.com) APIs (`data`), and the [pybraincompare](http://www.github.com/vsoch/pybraincompare) and [CogatSimilaR](https://github.com/CognitiveAtlas/cogat-similaR) packages (`methods`). This means that the data and algorithms, if useful to you, can be easily integrated into tools for development of applications that could use semantic image comparison.

## Dependencies

While this is not (yet) set up to be a completely, one-click reproducible workflow, I intend to give this a shot. For now, assume that everything in "preprocessing" should be run in a cluster environment, and analysis (can be) run locally. All compiled outputs from the preprocessing steps are provided in this repo, and detailed where they come from in the documentation. For now, please use the requirements.txt file to install python dependencies, locally and on your cluster. 

      pip install -r requirements.txt --user

For the R analyses, you will need R version 3.02 or higher, and to install the plyr, dplyr, and [cogat-similaR](https://github.com/CognitiveAtlas/cogat-similaR) package.
      
It is recommended to perform the [preprocessing](preprocessing) analyses in a cluster environment (SLURM was the cluster use to generate these results, and a local machine is only appropriate for the [analyses](analyses).


## Installation
Installation (for now) means cloning the repo in your cluster environment:

      git clone http://www.github.com/vsoch/semantic-image-comparison

and then following the [instructions](preprocess) in the README.md, which coincide with example commands in [run_analyses.sh](preprocess/run_analyses.sh). It is unfortunately still recommended to work through the steps in sequence, manually using the scripts, as many rely on submitting jobs, waiting for completion, and then compiling output. Let's get started with [preprocessing](preprocess)!

## Moving Between Cluster and Local
You will notice that the cluster output is all relative to a `base` path, which is not necessarily relative to the repo you clone with scripts. You will need to move compiled outputs into their appropriate spots in the [analysis](analysis) directory (likely you will want to clone the repo and run this locally). Detailed instructions are provided in both the scripts and [README.md](preprocess/README.md) in the preprocess folder.
