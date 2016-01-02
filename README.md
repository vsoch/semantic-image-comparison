# Semantic Image Comparison

## Overview
This work aims to assess the value of semantic (ontological) image comparison. We conduct the following analyses:

- [ontological comparison](analysis/reverse_inference): Our main analysis is testing different variations of reverse inference image comparison scores on our ability to classify images based on experimental contrast. Showing that the scores have value toward this task demonstrates value in semantic image comparison.
- [wang graph similarity](analysis/wang): The Wang Graph metric is a commonly used strategy to assess the similarity of gene sets, and we apply this to compare images based on their Cognitive Atlas tags. We compare an image similarity matrix of this metric against traditional spatial image comparison to assess if the method captures something that we understand about images.
- [ontological comparison](analysis/group_set_sizes): It was not clear how the number of images defined under each of our Cognitive Atlas concept groups influenced the resulting scores, and this analysis demonstrated that a minimum size of approximately 20 images is appropriate to conduct analysis.

## Preprocessing

The entire pipeline in this folder is numbered in sequential order, and requires a cluster environment to complete in any reasonable amount of time. Detailed steps are provided below for each.

### 1.

## Analysis

### Reverse Inference
[folder](analysis/reverse_inference): Our main analysis is testing different variations of reverse inference image comparison scores on our ability to classify images based on experimental contrast. Showing that the scores have value toward this task demonstrates value in semantic image comparison.

### Influence of group set sizes on Reverse Inference
[ontological comparison](analysis/group_set_sizes): It was not clear how the number of images defined under each of our Cognitive Atlas concept groups influenced the resulting scores, and this analysis demonstrated that a minimum size of approximately 20 images is appropriate to conduct analysis.

### Pairwise Graph Similarity
[wang graph similarity](analysis/wang): The Wang Graph metric is a commonly used strategy to assess the similarity of gene sets, and we apply this to compare images based on their Cognitive Atlas tags. We compare an image similarity matrix of this metric against traditional spatial image comparison to assess if the method captures something that we understand about images.


