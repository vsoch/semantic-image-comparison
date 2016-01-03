# Preprocessing

The entire pipeline in this folder is numbered in sequential order, and requires a cluster environment to complete in any reasonable amount of time. As previously stated, you should download the scripts somewhere on your cluster, likely on your home folder:

      git clone http://www.github.com/vsoch/semantic-image-comparison

and see [here](../README.md) for instructions on installing Python and R dependencies. You can then use the [run_preprocess.sh](run_preprocess.sh) to walk through script submission, compilation, and saving of results. Detailed steps are provided below for each.

## 1. NeuroVault Images
The initial image set comes by way of the NeuroVault database. For this work, we did extensive development to allow for the tagging of statistical brain maps with the cognitive concepts that they measure. This first script sets up an analysis output directory (called `base`) that will have the following structure:

      BASE
          data
          results

The base directory should be the first argument for running the script:

      base=/scratch/users/vsochat/DATA/BRAINMETA/ontological_comparison
      python 0.neurovault_comparison.py $base

To convert T score maps into Z maps we use TtoZ that is implemented in the [pybraincompare](http://www.github.com/vsoch/pybraincompare) package, however a standalone package [TtoZ](https://github.com/vsoch/TtoZ) is also available.
