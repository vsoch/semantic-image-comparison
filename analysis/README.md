# Analysis

## Classification
[classification](classification): Our main analysis is testing the ability of semantic image comparison scores on our ability to classify images. We show that semantic annotations can be used succesfully in a classification framework, and thus that there is value in performing semantic image comparison.

All analyses are completed with the preprocess scripts, but if you want to generate a heatmap of decoded concept regression parameter z maps to all neurosynth terms:


     RSCRIPT ../classification/generate_decode_pdf.R

The output of this step is [provided](classification/conept_regparam_decoding.pdf) and recommended to download and view in browser, as the matrix is very wide.


### Pairwise Graph Similarity
[wang graph similarity](wang): The Wang Graph metric is a commonly used strategy to assess the similarity of gene sets, and we apply this to compare images based on their Cognitive Atlas tags. We use representational similarity analysis to compare an image similarity matrix of this metric against traditional spatial image comparison to assess if the method captures something that we understand about images.

