# RSA ANALYSIS ################################################################################
# We can assess differences between image comparison calculated with spatial vs. semantic similarity

import pandas
import numpy
from scipy.stats import pearsonr

def read_wang(file_path):
    version1 = pandas.read_csv(file_path,sep=",",index_col=0)
    version1.columns = [int(x) for x in version1.columns]
    return version1

# Read in wang versions 1 and 2
version1 = read_wang("data/contrast_defined_images_wang.tsv")
version2 = read_wang("data/contrast_defined_images_wang_v2.tsv")
images = pandas.read_csv("data/contrast_defined_images_filtered.tsv",sep="\t")

# These should be empty results
[x for x in range(version1.shape[0]) if version1.index[x] != version2.index[x]]
[x for x in range(version1.shape[1]) if version1.columns[x] != version2.columns[x]]

# Function for RSA analysis of two matrices
def rsa(spatial,graph):
    # This will take the diagonal of each matrix (and the other half is changed to nan) and flatten to vector
    vector_spatial = spatial.mask(numpy.triu(numpy.ones(spatial.shape)).astype(numpy.bool)).values.flatten()
    vector_graph = graph.mask(numpy.triu(numpy.ones(graph.shape)).astype(numpy.bool)).values.flatten()
    # Now remove the nans
    vector_spatial_defined = numpy.where(~numpy.isnan(vector_spatial))[0]
    vector_graph_defined = numpy.where(~numpy.isnan(vector_graph))[0]
    idx = numpy.intersect1d(vector_spatial_defined,vector_graph_defined)
    return pearsonr(vector_spatial[idx],vector_graph[idx])[0]

rsas = dict()

# Perform RSA between two matrices
rsa_value = rsa(version1,version2)
#  0.93464445530745577
