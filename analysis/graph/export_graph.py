# Export graph will export the cognitive atlas concept relationships (is-a,kind-of)

from cognitiveatlas.api import get_concept
import networkx as nx
import pickle
import pandas

# Let's make a concept by concept data frame
contrast_lookup = pandas.read_csv("data/contrast_by_concept_binary_df.tsv",sep="\t",index_col=0)
images = pandas.read_csv("data/contrast_defined_images_filtered.tsv",sep="\t",index_col=0)

# source will be in rows, target in columns
concept_kindof = pandas.DataFrame()
concept_partof = pandas.DataFrame()
concepts = contrast_lookup.columns.tolist()
seen = []

while len(concepts) > 0:
    concept = concepts.pop(0)
    seen.append(concept)
    try:
        tmp = get_concept(id=concept).json[0]
        if 'relationships' in tmp:
            for relation in tmp["relationships"]:
                if relation['id'] not in seen and relation['id'] not in concepts:
                    print "Adding concept %s" %(relation['id'])
                    concepts.append(relation['id'])
                if relation['direction'] == "parent":
                    if relation['relationship'] == 'kind of':
                        concept_kindof.loc[tmp['id'],relation['id']] = 1
                    elif relation['relationship'] == 'part of':
                        concept_partof.loc[tmp['id'],relation['id']] = 1
                elif relation['direction'] == "child":
                    if relation['relationship'] == 'kind of':
                        concept_kindof.loc[relation['id'],tmp['id']] = 1
                    elif relation['relationship'] == 'part of':
                        concept_partof.loc[relation['id'],tmp['id']] = 1
    except:
        print "cannot find %s in the Cognitive Atlas!" %(concept)

concept_kindof = concept_kindof.fillna(0)
concept_partof = concept_partof.fillna(0)
concept_kindof.to_csv('data/concept_kindof_df.tsv',sep="\t")
concept_partof.to_csv('data/concept_partof_df.tsv',sep="\t")
concept_both = concept_kindof + concept_partof
concept_both = concept_both.fillna(0)
concept_both.to_csv('data/concept_both_df.tsv',sep="\t")

# Now, create a graph for each.

def make_graph(df):
    G = nx.Graph()
    for node in df.index.tolist():
        G.add_node(node)
        edges = df.loc[node][df.loc[node]!=0].index.tolist()
        for edge in edges:
            G.add_edge(node,edge)
    return G

Gpartof = make_graph(concept_partof)
Gkindof = make_graph(concept_kindof)
Gboth = make_graph(concept_both)

graphs = {"partof":Gpartof,"kindof":Gkindof,"both":Gboth}
pickle.dump(graphs,open('data/graphs_networkx.pkl','wb'))
