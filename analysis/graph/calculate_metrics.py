# Calculate similarity for different graphs
from similarity import cosine, ascos, jaccard, katz, lhn, rss2
import pickle

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

pickle.dump(results,open("data/sim_metrics.pkl","wb"))
