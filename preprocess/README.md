# Preprocessing

The entire pipeline in this folder is numbered in sequential order, and requires a cluster environment to complete in any reasonable amount of time. As previously stated, you should download the scripts somewhere on your cluster, likely on your home folder:

      git clone http://www.github.com/vsoch/semantic-image-comparison

and see [here](../README.md) for instructions on installing Python and R dependencies. You can then use the [run_preprocess.sh](run_preprocess.sh) to walk through script submission, compilation, and saving of results. Detailed steps are provided below for each.

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


It also adds a new folder to the hierarchy, `web` to output a d3 visualization of the current Cognitive Atlas concept hierarchy. You can see an example of this visualization [here](http://vsoch.github.io/semantic-image-comparison)

      BASE
          data
          results
          web


