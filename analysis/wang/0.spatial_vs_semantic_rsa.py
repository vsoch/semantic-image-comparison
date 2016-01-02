# RSA ANALYSIS ################################################################################
# We can assess differences between image comparison calculated with spatial vs. semantic similarity

import pandas
import numpy
from scipy.stats import pearsonr

# Read in spatial and wang (graph-based) similarities, row and colnames are images.image_id
spatial = pandas.read_csv("data/contrast_defined_images_pearsonpd_similarity.tsv",sep="\t",index_col=0)
spatial.columns = [int(x) for x in spatial.columns]
graph = pandas.read_csv("data/contrast_defined_images_wang.tsv",sep=",",index_col=0)
graph.columns = [int(x) for x in graph.columns]
images = pandas.read_csv("data/contrast_defined_images_filtered.tsv",sep="\t")

if spatial.shape[0] != graph.shape[0]:
   print("ERROR: matrices not equally sized!")

# These should be empty results
[x for x in range(spatial.shape[0]) if spatial.index[x] != graph.index[x]]
[x for x in range(spatial.shape[1]) if spatial.columns[x] != graph.columns[x]]

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

# Perform RSA across all contrasts
rsa_df = pandas.DataFrame(columns=["RSA"])
rsa_df.loc["all","RSA"] = rsa(spatial,graph)
# 0.51420919533486686

# Now let's calculate for each task
tasks = images["cognitive_paradigm_cogatlas_id"].unique().tolist()
for task in tasks:
   task_images = images.image_id[images.cognitive_paradigm_cogatlas_id==task].tolist()
   if len(task_images) > 2:
       df1 = spatial.loc[spatial.index.isin(task_images),spatial.columns.isin(task_images)]
       df2 = graph.loc[graph.index.isin(task_images),graph.columns.isin(task_images)]
       rsa_df.loc[task,"RSA"] = rsa(df1,df2)

# Now let's calculate for contrasts (note - we have no images tagged with same contrast)
contrasts = images["cognitive_contrast_cogatlas_id"].unique().tolist()
for contrast in contrasts:
   contrast_images = images.image_id[images.cognitive_contrast_cogatlas_id==task].tolist()
   if len(contrast_images) > 2:
       df1 = spatial.loc[spatial.index.isin(contrast_images),spatial.columns.isin(contrast_images)]
       df2 = graph.loc[graph.index.isin(contrast_images),graph.columns.isin(contrast_images)]
       rsa_df.loc[task,"RSA"] = rsa(df1,df2)

# However - we can look at RSA for concept images! Let's make a df of contrast_id by images
from cognitiveatlas.api import get_concept
concepts = []
for contrast in contrasts:
    tmp = get_concept(contrast_id=contrast).json
    concepts = concepts + [t["id"] for t in tmp if "id" in t]

concepts = numpy.unique(concepts).tolist()
contrast_df = pandas.DataFrame(0,index=contrasts,columns=concepts)

# Now fill in the data frame
for contrast in contrasts:
    tmp = get_concept(contrast_id=contrast).json
    contrast_concepts = [t["id"] for t in tmp if "id" in t]
    contrast_df.loc[contrast,contrast_concepts] = 1

# Save if we want it later
contrast_df.to_csv("data/contrast_by_concept_binary_df.tsv",sep="\t")

# Now let's perform RSA by concept
for concept in concepts:
    contrast_ids = contrast_df[concept][contrast_df[concept]==1].index.tolist()
    contrast_images = images.image_id[images.cognitive_contrast_cogatlas_id.isin(contrast_ids)].tolist()
    if len(contrast_images) > 2:
       df1 = spatial.loc[spatial.index.isin(contrast_images),spatial.columns.isin(contrast_images)]
       df2 = graph.loc[graph.index.isin(contrast_images),graph.columns.isin(contrast_images)]
       rsa_df.loc[concept,"RSA"] = rsa(df1,df2)

# Now let's add some meta data (names) for tasks and contrasts, save result to file
tasks_in_cogat = images.cognitive_paradigm_cogatlas[images.cognitive_paradigm_cogatlas_id.isin(rsa_df.index)].tolist()
tasks_in_cogat_id = images.cognitive_paradigm_cogatlas_id[images.cognitive_paradigm_cogatlas_id.isin(rsa_df.index)].tolist()
rsa_df.loc[tasks_in_cogat_id,"name"] = tasks_in_cogat

# The rest are concepts
for concept in concepts:
    if concept in rsa_df.index:
        rsa_df.loc[concept,"name"] = get_concept(id=concept).json[0]["name"]

rsa_df.loc["all","name"] = "all"
rsa_df.to_csv("data/spatial_semantic_rsa_df.tsv",sep="\t")
