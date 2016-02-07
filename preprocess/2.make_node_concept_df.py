#!/usr/bin/python

# This script will prepare a node by concept data frame, to be easily used for labels in the 
# classification framework.

from glob import glob
import pandas
import pickle
import re
import os

base = sys.argv[1]
data = "%s/data" %base
node_pickles = glob("%s/groups/*.pkl" %data)
results = "%s/results" %base
images_tsv = "%s/contrast_defined_images_filtered.tsv" %results
images = pandas.read_csv(images_tsv,sep="\t")

unique_contrasts = images.cognitive_contrast_cogatlas_id.unique().tolist()

# Images that do not match the correct identifier will not be used (eg, "Other")
expression = re.compile("cnt_*")
concepts = [os.path.basename(x).split("group_")[-1].replace(".pkl","") for x in node_pickles]
image_ids = images.image_id.unique().tolist()

# This is the model that we will want to build for each voxel
# Y (n images X 1) = X (n images X n concepts) * beta (n concepts X 1)
# Y is the value of the voxel (we are predicting)
# X is images by concepts, with values corresponding to concept labels
# beta are the coefficients we get from building the model
X = pandas.DataFrame(0,index=image_ids,columns=concepts)
for group_pkl in node_pickles:
    group = pickle.load(open(group_pkl,"rb"))
    concept_id = os.path.basename(group_pkl).split("group_")[-1].replace(".pkl","")
    print "Parsing concept %s" %(concept_id)
    image_ids_in = [int(os.path.basename(x).split(".")[0]) for x in group["in"]]
    image_ids_out = [int(os.path.basename(x).split(".")[0]) for x in group["out"]]
    X.loc[image_ids_in,concept_id] = 1 

# Looked this up manually - bug in API right now
X.loc[109,["trm_567982752ff4a","trm_4a3fd79d0afcf","trm_5534111a8bc96","trm_557b48a224b95","trm_557b4a81a4a17","trm_4a3fd79d0b64e","trm_4a3fd79d0a33b","trm_557b4a7315f1b","trm_4a3fd79d0af71","trm_557b4b56de455","trm_557b4add1837e"]] = 1
#X = X[X.sum(axis=1)!=0]
X.to_csv("%s/images_contrasts_df.tsv" %results,sep="\t")


# Now we want to make a design matrix that takes ontology into account

# image1 --> concept3 ---> concept2 --> concept1  BASE
# image2 --> concept4 ---> concept2 --> concept1  BASE

#                  the design matrix would be:

#             concept1 concept2 concept3 concept4
# image1   0.64           0.8            1              0
# image2   0.64           0.8            0              1
# image3   0                0            0              0

# Read in the list of triples, this will give us concepts/parent relationships
output_triples_file = "%s/task_contrast_triples.tsv" % results
relationship_table = pandas.read_csv(output_triples_file,sep="\t")
Xweighted = X.copy()

# is_a relationship weight is 0.8
weight = 0.8

# For each row (image) find related concepts
for image_id in Xweighted.index.tolist():
    print "Parsing image %s" %(image_id)
    concept_series = Xweighted.loc[image_id,Xweighted.loc[image_id,:]!=0]
    for concept in concept_series.index.tolist():
        current_node = relationship_table.loc[relationship_table.id==concept,:]
        # These are the direct parents of the node, at the end we change these back to 1
        parents = [x for x in current_node.parent.tolist() if x != "1"]  
        direct_parents = parents[:]
        # We will save a list of lists of all parents, indexed by distance from original term
        treewalk = []
        treewalk.append(parents)
        while len(parents) > 0:
            parents = relationship_table.parent[relationship_table.id.isin(parents)].tolist()                        
            parents = [x for x in parents if x!="1"] # remove base node
            if len(parents) > 0:
                treewalk.append(parents)
        # Now we assign weights to current_node vector, work backwards in the tree so that
        # repeated parents get credit closer in the tree
        # In retrospect we didn't need to reverse (because we give credit for closer parents --> higher weights)
        treewalk.reverse()
        current_value = concept_series.loc[concept]
        concepts_seen = []
        for l in range(len(treewalk)):
            # index[len(treewalk)-1] --> len(treewalk) away, farthest away, weight is weight to the len(treewalk)-l power
            # The weight is the distance from the node (len(treewalk)-l), l is the index
            adjusted_weight = numpy.power(weight,len(treewalk)-l-1)
            # weights should be (closest to farthest): 1, 0.8, 0.8*0.8=0.64 with corresponding indices [2][1][0]
            # So when l==2, when len(treewalk)==3, treewalk[2] are parents with distance ==1, numpy.power(weight,3-2-1) is 1
            # So when l==1, when len(treewalk)==3, treewalk[1] are middle distance == 2 numpy.power(weight,3-1-1) is 0.8
            # So when l==0, when len(treewalk)==3, treewalk[0] are farthest away with distance ==3 numpy.power(weight,3-0-1) is 0.64
            for concept_id in treewalk[l]:
                if concept_id in concept_series.index.tolist():
                    if concept_id not in concepts_seen:
                        concept_series.loc[concept_id] = adjusted_weight
                        concepts_seen.append(concept_id)
                    else:
                        # Update the weight if it isn't original 1, and is greater than old (meaning closer relationship)
                        if adjusted_weight > concept_series.loc[concept_id] and concept_series.loc[concept_id] != 1:
                            concept_series.loc[concept_id] = adjusted_weight  
    # Make sure original parents are stil 1 (should be)
    concept_series.loc[direct_parents] = 1
    # Update the Xweighted matrix with the series! 
    Xweighted.loc[image_id,Xweighted.loc[image_id,:]!=0] = concept_series

Xweighted.to_csv("%s/images_contrasts_df_weighted.tsv" %results,sep="\t")
