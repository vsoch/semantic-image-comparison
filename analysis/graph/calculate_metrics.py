# Calculate similarity for different graphs
from similarity import simrank, ascos
import pickle

graphs = pickle.load(open('data/graphs_networkx.pkl','rb'))
results = dict()

for graph_type,G in graphs.iteritems():
    results[graph_type] = dict()
    results[graph_type]["ascos"] = ascos(G)
