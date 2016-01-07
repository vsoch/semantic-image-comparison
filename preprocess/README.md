# Preprocessing

The entire pipeline in this folder is numbered in sequential order, and requires a cluster environment to complete in any reasonable amount of time. As previously stated, you should download the scripts somewhere on your cluster, likely on your home folder:

      git clone http://www.github.com/vsoch/semantic-image-comparison

and see [here](../README.md) for instructions on installing Python and R dependencies. You can then use the [run_preprocess.sh](run_preprocess.sh) to walk through script submission, compilation, and saving of results. Detailed steps are provided below for each.


# Part I: Reverse Inference

## 0. NeuroVault Comparison
The initial image set comes by way of the NeuroVault database. For this work, we did extensive development to allow for the tagging of statistical brain maps with the cognitive concepts that they measure. This first script sets up an analysis output directory (called `base`) that will have the following structure:

      BASE
          data
          results

The base directory should be the first argument for running the script:

      base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison
      python 0.neurovault_comparison.py $base

#### NeuroVault Images
We first use the NeuroVault API to obtain images from all collections with DOIs, and from that set filter down to those that are not thresholded, have contrast labels defined in the Cognitive Atlas, are either Z or T maps, and are in MNI. We then download these images to $base/data resulting in two folders, `original` and `resampled` to our standard space.

#### Conversion to Z Score Maps
We convert all resampled images to Z score maps, and the output is put into a new `resampled_z` folder. To convert T score maps into Z maps we use TtoZ that is implemented in the [pybraincompare](http://www.github.com/vsoch/pybraincompare) package, however a standalone package [TtoZ](https://github.com/vsoch/TtoZ) is also available.

#### Spatial Image Similarity
We will want to compare spatially derived similarity (meaning Pearson scores using voxelwise data, for complete case analysis within a brain map, see [this work](https://github.com/vsoch/image-comparison-thresholding)) and so we are extracting that matrix at this step.

### Outputs from this step moved into analysis folder:
- $base/results/contrast_defined_images_filtered.tsv: is a data frame of meta data about images, including contrast/task IDs and labels, along with collection and image meta data should be moved to [../analysis/wang/data](../analysis/wang/data).
 - $base/results/contrast_defined_images_pearsonpd_similarity.tsv: Is a spatial similarity matrix (pearson score with complete case analysis, see [this work](https://github.com/vsoch/image-comparison-thresholding)) also should be moved to [../analysis/wang/data](../analysis/wang/data).


## 1. Prep Reverse Inference
To perform reverse inference, meaning calculating the P(cognitive process | pattern of activation), we need to know the cognitive processes! This script uses the [cognitiveatlas](http://cognitiveatlas.readthedocs.org) API to look up contrasts for the images that are tagged in NeuroVault. It again takes the base directory as the only argument:

      python 1.prep_reverse_inference.py $base

### Graph of Contrasts
We first want to generate a graph of Cognitive Atlas concepts, with each node having a certain set of NeuroVault images that are tagged to it. This graph structure assume that an image tagged lower in the tree is thus defined for nodes that are parents to it. In order to do this, we first use the API to look up contrasts for our images, and then generate a list of triples, meaning child --> parent --> name relationships. Pybraincompare has a function that understands this structure, and can produce an [interactive graph to view it](http://www.vsoch.github.io/semantic-image-comparison). While the file of triples is not used beyond this step and does not need to leave the cluster, it is [provided for inspection](https://github.com/vsoch/semantic-image-comparison/blob/master/doc/task_contrast_triples.tsv).

This step adds a new folder to the hierarchy, `web` to output [this](http://vsoch.github.io/semantic-image-comparison) d3 visualization of the current Cognitive Atlas concept hierarchy:

      BASE
          data
          results
          web

### Likelihood Groups
In order to perform reverse inference, we need to have a brain map that, derived from images with the tag, has the P(activation | cognitive concept) for each voxel. Since we will be running the reverse inference calculations many times in a cluster environment, it makes sense to save these data frames in advance. Further, since we are going to want to remember (and easily load) the images that were relevant to each group, we should keep this handy. This is the last step that is done by this script, and we save our group information pickles (a pickle is a python object) to `$base/data/likelihood`. 

### Outputs from this step moved into analysis folder:
- $base/web/index.html: is a d3 visualization of the Cognitive Atlas concept hierarchy, and was moved to the base of the repo.
 

## 2. Calculate Likelihoods
We can then use our likelihood groups to generate data frames for each node (a Cognitive Atlas concept that is associated with a set of images). The "correct" term for these tables are the priors.


      python 2.run_calculate_likelihood.py $base


It is `essential` that you manually edit this script to work for your particular cluster environment. It works by writing job files into the folders .jobs and .out in the present working directory, and submitting them with the final command `sbatch`. I assure you that you do not have access to a `russpold` SLURM queue, so it is essential that you review this submission script and make appropriate changes.

This step will generate likelihood data tables for each concept node associated with images. While the script will produce the tables for several different approaches (range, binary thresholds, and distance), we found that the scores were almost indistinguishable, and chose to use the fastest (distance) for this work. These priors can then be used en-masse to calculate reverse inference scores for all of our concept nodes mapped with images!

## 3. Calculate Reverse Inference
Finally, we can perform reverse inference!

      P(node mental process|activation) = P(activation|mental process) * P(mental process)
      divided by
      P(activation|mental process) * P(mental process) + P(A|~mental process) * P(~mental process)
      P(activation|mental process): my voxelwise prior map

This is another submission script that has been modified to submit jobs dynamically, depending on your userid. Again, please find the line that details checking the queue status:

      queue_count = int(os.popen("squeue -u vsochat | wc -l").read().strip("\n"))

along with the submission command:

      os.system("sbatch -p russpold " + ".job/oc_%s_%s.job" %(i,image_id))

and modify to work for your username and cluster environment. You may also want to adjust other cluster specific variables at the head of the script (3.run_calculate_reverse_inference.py)[3.run_calculate_reverse_inference.py].

### Outputs from this step moved into analysis folder:
- $base/data/likelihood/pbc_likelihood_group*.pkl: These are the pickled group files that we will want to use in our analysis, and we copy them to [../analysis/reverse_inference/groups](../analysis/reverse_inference/groups)

## 4. Compile Reverse Inference
The output of the above is over 10K files, each containing python (pickled) scores for an image against a particular concept node. We will want to compile these scores into a nice (single) data frame, and move into the analysis directory, and that is what this script [4.compile_reverse_inference_results.py](4.compile_reverse_inference_results.py) accomplishes.

### Outputs from this step moved into analysis folder:
- $base/data/reverse_inference_scores.tsv ../analysis/reverse_inference/data

## 5. Compile Reverse Inference
We were worried about reverse inference scores changing depending on the number of images defined for the node, and wanted to find some threshold to establish stability in scores. This was the aim of this portion, and the workflow is as follows:

      # For each concept, (N=132):
      #    Select a number G from 1...[total "in" group] as the size of the set to investigate
      #        For each image in (entire) "in"set:
      #            For some number of iterations:
      #                Randomly select G other images for "in" set, calculate P(concept|image)
      #                Take mean score of iterations as P(concept|image) - and we also calculate confidence intervals, etc.

This will power the analysis to produce [this analysis](../analysis/reverse_inference/img/explore_concept_set_sizes.pdf). The [5.run_explore_group_size.py](5.run_explore_reverse_inference.py) also has assumptions about cluster size and paths, and you should edit it to work on your cluster.


      python 5.run_explore_group_size.py $base


## Step 6: Compile Explore Group Size Results
There are no outputs from the previous step to move into the analysis folder, because [6.compile_explore_group_size_results.py](6.compile_explore_group_size_results.py) will compile those results and we will move the compiled file.

      
      python 6.compile_explore_group_size_results.py $base


Each of these files is a data frame with reverse inference calculated for each image, using each possible "in" set size given the contrast. For the concepts with only one tagged image (for example, [here](../analysis/reverse_inference/data/size_results/ri_explore_size_results_trm_4a3fd79d09849.tsv)) we can only calculate a score for other images given an "in" set size of 1, so we cannot say anything about how set sizes influence the scores using these data.

### Outputs from this step moved into analysis folder:
- $base/data/ri_explore_size_results*.tsv: These are ../analysis/reverse_inference/data/size_results


# Part II: Graph Comparison

## Step 7. Wang Similarity

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
- [7.run_wang_graph_comparison_sherlock.R](7.run_wang_graph_comparison_sherlock.R) runs instances of
- [7.wang_graph_comparison_sherlock.R](7.wang_graph_comparison_sherlock.R) uses the [cogat-similaR](https://github.com/CognitiveAtlas/cogat-similaR) package to calculate the metric for all images in our set.
- [7.compile_wang_comparison.R](7.compile_wang_comparison.R): compiles all individual result files for analysis.
- [7.wang_graph_comparison_local.R](7.wang_graph_comparison_local.R): was not used in the analysis, but is provided as an example to run the algorithm locally.


      mkdir $base/data/wang_scores
      RSCRIPT 7.run_wang_graph_comparison_sherlock.R $base


As before, the [7.run_wang_graph_comparison_sherlock.R](7.run_wang_graph_comparison_sherlock.R) should be modified for your cluster environment. This script has two sections - the top is intended to run initial comparisons, and the bottom is intended to run for a subset of missing comparisons (see below).


      RSCRIPT 7.compile_wang_comparison.R $base


Use the [7.compile_wang_comparison.R](7.compile_wang_comparison.R) to do exactly that. This script has two sections - the top is intended to compile results into a "similarities" data frame, and then the bottom checks for missing results. This script makes over 60K calls to the Cognitive Atlas API, and it's likely that a small number of those calls fail. The bottom section of [7.run_wang_graph_comparison_sherlock.R](7.run_wang_graph_comparison_sherlock.R) has code for loading the `missing.Rda` file, and then [7.compile_wang_comparison.R](7.compile_wang_comparison.R) can be run again to assess for missing. For this analysis, I used this routine twice, and was able to run for all initial missing.

### Outputs from this step moved into analysis folder:
- $base/data/wang_scores/contrast_defined_images_wang.tsv: This is a data frame of images by images, with wang scores for image I against image J saved to ../analysis/reverse_inference/data/size_results

### Generation of Web Interface
Optionally, you can generate an interactive web interface [as seen here](http://vsoch.github.io/semantic-image-comparison-web) by using the script [8.web_interface.py](8.web_interface.py) to produce data files in JSON that are uploaded to the [data folder](https://github.com/vsoch/semantic-image-comparison-web/tree/master/data) and then deployed to a web server. 

      python 8.web_interface.py $base

A simpler version of this interface is deployed via [continuous integration](http://www.github.com/vsoch/semantic-image-comparison-ci). For this deployment, the reverse inference scores are not included, and rather a visualization to show Cognitive Atlas tasks and associated tagged NeuroVault images is produced when the continuous integration (on Circle-CI) is re-run.

# Analysis
You've finished preprocessing! Now move on to [../analysis](../analysis)
