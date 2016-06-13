# Calculate similarity for different graphs
from similarity import cosine, ascos, jaccard, \
  katz, lhn, rss2, dice, inverse_log_weighted, rsa
import pickle
import pandas

### VERTEX SIMILARITY

graphs = pickle.load(open('data/graphs_networkx.pkl','rb'))
results = dict()

for graph_type,G in graphs.iteritems():
    print "Calculating for %s" %(graph_type)
    results[graph_type] = dict()
    print "ASCOS ---------------------------"
    results[graph_type]["ascos"] = ascos(G)
    print "COSINE ---------------------------"
    results[graph_type]["cosine"] = cosine(G)
    print "JACCARD --------------------------"
    results[graph_type]["jaccard"] = jaccard(G)
    print "KATZ -----------------------------"
    results[graph_type]["katz"] = katz(G)
    print "LHN ------------------------------"
    results[graph_type]["lhn"] = lhn(G)
    print "RSS2 -----------------------------"
    results[graph_type]["rss2"] = rss2(G)
    print "DICE -----------------------------"
    results[graph_type]["dice"] = dice(G)
    print "INVERSE LOG WEIGHTED --------------"
    results[graph_type]["inverse_log_weighted"] = inverse_log_weighted(G)

pickle.dump(results,open("data/sim_metrics.pkl","wb"))

### IMAGE SIMILARITY

# Let's make a concept by concept data frame
contrast_lookup = pandas.read_csv("data/contrast_by_concept_binary_df.tsv",sep="\t",index_col=0)
images = pandas.read_csv("data/contrast_defined_images_filtered.tsv",sep="\t",index_col=0)

def get_concepts(image1):
    image1_concepts = images['cognitive_contrast_cogatlas_id'][images['image_id']==image1].tolist()[0]
    image1_concepts = contrast_lookup.loc[image1_concepts].transpose()
    return image1_concepts[image1_concepts!=0].index.tolist()

# For the vertex metrics, let's calculate a score for the image as the mean across its concepts scores
def get_score(image1,image2,df):
    image1_concepts = get_concepts(image1)
    image2_concepts = get_concepts(image2)
    # subset to concepts we have in the df
    image1_concepts = [x for x in image1_concepts if x in df.index]
    image2_concepts = [x for x in image2_concepts if x in df.index]
    if len(image1_concepts) == 0 or len(image2_concepts) == 0:
        return 0
    image1_sum = df.loc[image1_concepts].sum()
    image2_sum = image1_sum.loc[image2_concepts].sum()
    image2_sum = image2_sum / (len(image1_concepts) + len(image2_concepts))
    return image2_sum    

sims = dict()

for result_type,metric in results.iteritems():
    print "Parsing %s" %(result_type)
    sims[result_type] = dict()
    for metric_name,df in metric.iteritems():
        print "Parsing %s" %(metric_name)
        sim = pandas.DataFrame(index=images.image_id.tolist(),columns=images.image_id.tolist())
        for image1 in images.image_id.tolist():
            for image2 in images.image_id.tolist():
                score = get_score(image1,image2,df)
                sim.loc[image1,image2] = score
                sim.loc[image2,image1] = score
        sims[result_type][metric_name] = sim


pickle.dump(sims,open("data/sim_contrasts.pkl","wb"))

# Finally, do Wang
df_kindof = pandas.read_csv('data/concept_kindof_df.tsv',sep="\t",index_col=0)
df_partof = pandas.read_csv('data/concept_partof_df.tsv',sep="\t",index_col=0)

# Compare each new metric to spatial similarity

# Read in spatial and wang (graph-based) similarities, row and colnames are images.image_id
spatial = pandas.read_csv("data/contrast_defined_images_pearsonpd_similarity.tsv",sep="\t",index_col=0)
spatial.columns = [int(x) for x in spatial.columns]
graph = pandas.read_csv("data/contrast_defined_images_wang_v2.tsv",sep=",",index_col=0)
graph.columns = [int(x) for x in graph.columns]

# Perform RSA across all contrasts
rsa_df = pandas.DataFrame(columns=["RSA"])
rsa_df.loc["wang","RSA"] = rsa(spatial,graph)

for result_type,metrics in sims.iteritems():
    for metric_name,df in metrics.iteritems():
        name = "%s_%s" %(result_type,metric_name)
        df = df.fillna(0)
        rsa_df.loc[name,"RSA"] = rsa(spatial,df)

rsa_df = rsa_df.dropna()
rsa_df = rsa_df.sort_values(by="RSA",ascending=False)
rsa_df.to_csv("data/spatial_semantic_rsa_df.tsv",sep="\t")
