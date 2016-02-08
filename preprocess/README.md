# Preprocessing

The entire pipeline in this folder is numbered in sequential order, and requires a cluster environment to complete in any reasonable amount of time. As previously stated, you should download the scripts somewhere on your cluster, likely on your home folder:

      git clone http://www.github.com/vsoch/semantic-image-comparison

and see [here](../README.md) for instructions on installing Python and R dependencies. You can then use the [run_preprocess.sh](run_preprocess.sh) to walk through script submission, compilation, and saving of results. Detailed steps are provided below for each.


# Part I: Classification

## 0. NeuroVault Comparison
The initial image set comes by way of the NeuroVault database. For this work, we did extensive development to allow for the tagging of statistical brain maps with the cognitive concepts that they measure. This first script sets up an analysis output directory (called `base`) that will have the following structure:

      BASE
          data
          results

The base directory should be the first argument for running the script:

      base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison
      python 0.neurovault_comparison.py $base

You will notice that the script also creates `.out` and `.jobs` directories, these are for storing output and job files, respectively.

#### NeuroVault Images
We first use the NeuroVault API to obtain images from all collections with DOIs, and from that set filter down to those that are not thresholded, have contrast labels defined in the Cognitive Atlas, are either Z or T maps, and are in MNI. We then download these images to $base/data resulting in two folders, `original` and `resampled` to our standard space.

#### Conversion to Z Score Maps
We convert all resampled images to Z score maps, and the output is put into a new `resampled_z` folder, additionally for a 4mm transformation (`resampled_z_4mm`). To convert T score maps into Z maps we use TtoZ that is implemented in the [pybraincompare](http://www.github.com/vsoch/pybraincompare) package, however a standalone package [TtoZ](https://github.com/vsoch/TtoZ) is also available.

#### Spatial Image Similarity
We will want to compare spatially derived similarity (meaning Pearson scores using voxelwise data, for complete case analysis within a brain map, see [this work](https://github.com/vsoch/image-comparison-thresholding)) and so we are extracting that matrix at this step.

### Outputs from this step moved into analysis folder:
- $base/results/contrast_defined_images_filtered.tsv: is a data frame of meta data about images, including contrast/task IDs and labels, along with collection and image meta data should be moved to [../analysis/wang/data](../analysis/wang/data).
 - $base/results/contrast_defined_images_pearsonpd_similarity.tsv: Is a spatial similarity matrix (pearson score with complete case analysis, see [this work](https://github.com/vsoch/image-comparison-thresholding)) also should be moved to [../analysis/wang/data](../analysis/wang/data).


## 1. Prep Semantic Comparison
To perform classification using semantic annotations (concepts) for images, we need to know the cognitive processes! This script uses the [cognitiveatlas](http://cognitiveatlas.readthedocs.org) API to look up contrasts for the images that are tagged in NeuroVault. It again takes the base directory as the only argument:

      python 1.prep_semantic_comparison.py $base


### Graph of Contrasts
We first want to generate a graph of Cognitive Atlas concepts, with each node having a certain set of NeuroVault images that are tagged to it. This graph structure assume that an image tagged lower in the tree is thus defined for nodes that are parents to it. In order to do this, we first use the API to look up contrasts for our images, and then generate a list of triples, meaning child --> parent --> name relationships. Pybraincompare has a function that understands this structure, and can produce an [interactive graph to view it](http://www.vsoch.github.io/semantic-image-comparison). While the file of triples is not used beyond this step and does not need to leave the cluster, it is [provided for inspection](https://github.com/vsoch/semantic-image-comparison/blob/master/doc/task_contrast_triples.tsv).

This step adds a new folder to the hierarchy, `web` to output [this](http://vsoch.github.io/semantic-image-comparison) d3 visualization of the current Cognitive Atlas concept hierarchy:

      BASE
          data
          results
          web

### Concept Groups
In order to perform classification, we will need to know (many times during the analysis) the images that are tagged with each concept. Since we will be running components of the classification analyses many times in a cluster environment, it makes sense to save these lists in advance. Further, since we are going to want to remember (and easily load) the images that were relevant to each group, we should keep this handy. This is the last step that is done by this script, and we save our group information pickles (a pickle is a python object) to `$base/data/groups`. 

### Outputs from this step moved into analysis folder:
- $base/web/index.html: is a d3 visualization of the Cognitive Atlas concept hierarchy, and was moved to the base of the repo, viewable [here](http://www.vsoch.github.io/semantic-image-comparison)
- $base/data/groups/pbc_group*.pkl: These are the pickled group files that we will want to use in our analysis, and we copy them to [../analysis/classification/groups](../analysis/classification/groups)
 

## 2. Make a node by concept data frame
We will want to easily generate a vector, for each image, with binary indicators (or weights) that indicate a particular concept being tagged to the image. This script will perform that step. In the case of the weights, we use the Wang algorithm that (high level) represents distance in the Cognitive Atlas concept tree (see our manuscript for more details).

      python 2.make_node_concept_df.py $base

### Outputs from this step moved into analysis folder:
 - $base/results/images_contrasts_df_weighted.tsv:  This is the [weighted data frame](../analysis/classification/data/images_contrasts_df_weighted.tsv)
 - $base/results/images_contrasts_df.tsv This is the [binary data frame](../analysis/classification/data/images_contrasts_df.tsv)


## 3. Classification Framework
This is the meat of the analysis. Here we will test the ability of the concept annotations to predict held out images (see manuscript for details). We will need to run three batches of jobs: first to run the classification procedure across ~4K pairs of images using binary labels, then using weighted (Wang) labels, and finally, the entire procedure needs to be run over 1000 times to produce a null distrubution of accuracies (for comparison of the actual accuracies).

      python 3.run_calculate_concept_map.py $base
      python 3.run_calculate_concept_map_ontology.py $base
      python 3.run_calculate_null.py $base # This can take weeks if not done with launcher

The actual null distribution was generated in a slurm environment, and it took many weeks. An example is provided to run in a slurm (launch) environment, which kikely will be much faster. You will need to find the line that specifies the queue and change it to your cluster queue (normally removing the parameter defaults to the "normal" or default queue:

      os.system("sbatch -p russpold " + ".job/oc_%s_%s.job" %(i,image_id))

You may also want to adjust other cluster specific variables in the job files being written to specify memory, etc.


## 4. Compile Classification Results
The output of the above is over 10K files, each containing python (pickled) scores for an image against a particular concept node. We will want to compile these scores into nice (single) data frames, and move into the analysis directory, and that is what this script [4.compile_classification_concept.py](4.compile_classification_concept.py) accomplishes, for each of the null, weighted, and standard procedures detailed above. Finally, since the script generates a final statistical test and a Z score map that represents each concept (from the regression parameter matrix) we will want to move these maps as well.

### Outputs from this step moved into analysis folder:
 - [$base/results/classification_results_binary_4mm.tsv](../analysis/classification/data/classification_results_binary_4mm.tsv)
 - [$base/results/classification_results_weighted_4mm.tsv](../analysis/classification/data/classification_results_weighted_4mm.tsv)
 - [$base/results/classification_results_null_4mm.tsv](../analysis/classification/data/classification_results_null_4mm.tsv)
 - $base/results/classification_final/*.nii.gz: these are the [concept maps](../analysis/classification/data/concept_maps)


## Comparison with NeuroSynth
We decode each of the Z score regression parameter maps as a "soft validation," to supplement the classification tests. We would want to see the maps, for each concept, be "decoded" as similar to relevant terms in the NeuroSynth database. Note that to make this easy, I uploaded the images to neurovault, and then used the collection_id to do the decoding programatically. If you instead want to use the neurosynth API, see the commented out portion of the script.

   
      neurovault_collection=1170
      python 4.comparison_with_neurosynth.py $base $neurovault_collection


### Outputs from this step moved into analysis folder:
 - [$base/results/concept_regparam_decoding.tsv](../analysis/classification/data/concept_regparam_decoding.tsv) is the version produced with the REST API (fewer terms)
 - [$base/results/concept_regparam_decoding.csv](../analysis/classification/data/concept_regparam_decoding.tsv) is the version produced with the python API (takes a lot longer, but over 3K terms)

Finally, if you want to generate a heatmap to show terms, you can use the R script in the classification folder.


# Part II: Graph Comparison

## Step 5. Wang Similarity

Wang Similarity is a method that has been used in bioinformatics to assess the similarity of gene sets. [This method](http://bioinformatics.oxfordjournals.org/content/23/10/1274.full)
 aggregates the semantic contributions of ancestor terms (including this specific term), and works as follows:

### Generating a list of concepts and weights for each contrast
 1. We start with concepts associated with the contrast
 2. We walk up the tree and append associated "is_a" and "part_of" concepts
 3. The weight for each concept is determined by multiplying the last (child node) weight by:
       0.8 for "is_a" relationships
       0.6 for "part_of" relationships
       This means that weights decrease as we move up the tree toward the root
 3. We stop at the root node

### Calculating similarity between contrasts
 1. We take the weights at the intersection of each list from above
 2. The similarity score is sum(intersected weights) / sum(all weights)

This step has four parts, and is the only R code that must be run in a cluster environment:
- [5.run_wang_graph_comparison_sherlock.R](5.run_wang_graph_comparison_sherlock.R) runs instances of
- [5.wang_graph_comparison_sherlock.R](5.wang_graph_comparison_sherlock.R) uses the [cogat-similaR](https://github.com/CognitiveAtlas/cogat-similaR) package to calculate the metric for all images in our set.
- [5.compile_wang_comparison.R](5.compile_wang_comparison.R): compiles all individual result files for analysis.
- [5.wang_graph_comparison_local.R](5.wang_graph_comparison_local.R): was not used in the analysis, but is provided as an example to run the algorithm locally.


      mkdir $base/data/wang_scores
      RSCRIPT 5.run_wang_graph_comparison_sherlock.R $base


As before, the [5.run_wang_graph_comparison_sherlock.R](5.run_wang_graph_comparison_sherlock.R) should be modified for your cluster environment. This script has two sections - the top is intended to run initial comparisons, and the bottom is intended to run for a subset of missing comparisons (see below).


      RSCRIPT 5.compile_wang_comparison.R $base


Use the [5.compile_wang_comparison.R](5.compile_wang_comparison.R) to do exactly that. This script has two sections - the top is intended to compile results into a "similarities" data frame, and then the bottom checks for missing results. This script makes over 60K calls to the Cognitive Atlas API, and it's likely that a small number of those calls fail. The bottom section of [5.run_wang_graph_comparison_sherlock.R](5.run_wang_graph_comparison_sherlock.R) has code for loading the `missing.Rda` file, and then [5.compile_wang_comparison.R](5.compile_wang_comparison.R) can be run again to assess for missing. It's not ideal that the Cognitive Atlas API is buggy at sometimes, but it's the best we have for now. For this analysis, I used this routine twice, and was able to run for all initial missing.


### Outputs from this step moved into analysis folder:
- $base/data/wang_scores/contrast_defined_images_wang.tsv: This is a data frame of images by images, with wang scores for image I against image J.


### Generation of Web Interface
Optionally, you can generate an interactive web interface [as seen here](http://vsoch.github.io/semantic-image-comparison-web) by using the script [6.web_interface.py](6.web_interface.py) to produce data files in JSON that are uploaded to the [data folder](https://github.com/vsoch/semantic-image-comparison-web/tree/master/data) and then deployed to a web server. 

      python 6.web_interface.py $base

A simpler version of this interface is deployed via [continuous integration](http://www.github.com/vsoch/semantic-image-comparison-ci). For this deployment, the reverse inference scores are not included, and rather a visualization to show Cognitive Atlas tasks and associated tagged NeuroVault images is produced when the continuous integration (on Circle-CI) is re-run.

# Analysis
You've finished preprocessing! Now move on to [../analysis](../analysis)
