# Analysis

## Classification
[classification](classification): Our main analysis is testing the ability of semantic image comparison scores on our ability to classify images. We show that semantic annotations can be used succesfully in a classification framework, and thus that there is value in performing semantic image comparison.

All analyses are completed with the preprocess scripts, but if you want to generate a heatmap of decoded concept regression parameter z maps to all neurosynth terms:


     RSCRIPT ../classification/generate_decode_pdf.R

The output of this step is [provided](classification/conept_regparam_decoding.pdf) and recommended to download and view in browser, as the matrix is very wide.


## Comparison with NeuroSynth
We decode each of the Z score regression parameter maps as a "soft validation," to supplement the classification tests. Since the complete cognitive atlas terms are not added to the neurosynth database at the time of the analysis, we obtain the entire set of 11K+ abstracts, and parse the text for the exact terms. Out of ~800 Cognitive Atlas terms, we find 613 in the abstracts, and normalize this data into an equivalent "features.txt" file to use the Neurosynth decoder. We generate an entire decoder result (a data frame with cognitive concept regression maps in rows, and the neurosynth terms in columns), and visualize. We also generate a file with the top ten (no absolute value taken) concepts for each regression parameter map. What we see is that the regression parameter maps map to Cognitive Atlas terms that make sense, as parsed from the neuroscience literature.

  
      python decode_classification_regparam_maps.py $base


This script requires a step to add a column name to the output feature file, and should be run manually. Note that the decoding also takes some time. Note that this script includes a second section that shows how to do the decoding with the neurosynth API when the regression parameter maps are uploaded to neurovault, which might be a faster and easier alternative if NeuroSynth is updated to include these terms.

### Outputs from this step moved into analysis folder:
 - [$base/results/concept_regparam_decoding_named.csv](decode/data/concept_regparam_decoding_named.csv) is the complete decoding result.
 - [$base/results/concept_regparam_decoding_named_topten.tsv](decode/data/concept_regparam_decoding_named_topten.tsv) is the version produced with the python API (takes a lot longer, but over 3K terms)

Finally, if you want to generate a heatmap to show terms, you can use the [R script provided](decode/view_decode_result.R)


### Pairwise Graph Similarity
[wang graph similarity](wang): The Wang Graph metric is a commonly used strategy to assess the similarity of gene sets, and we apply this to compare images based on their Cognitive Atlas tags. We use representational similarity analysis to compare an image similarity matrix of this metric against traditional spatial image comparison to assess if the method captures something that we understand about images.

