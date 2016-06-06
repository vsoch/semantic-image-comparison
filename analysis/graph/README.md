# Similarity Metrics

Our goal is to assess the similiarity of cognitive concepts, which are vertices in the graph, along with contrast images, which are tagged with multiple vertices (concepts). For this reason, we want to explore similarity metrics for contrast vs. contrast comparison, along with concept vs. concept comparison. Both families of methods will generally work by operating over graphs, but comparison of contrasts requires comparing **sets** of nodes.

## Vertex Comparison

### ASCOS:
infers the similarity between nodes based solely on structure context, i.e., the patterns of the edges, because "structurally similar nodes on a social network are more likely to have similar node attributes and have similar behavior." This is similar to a well known algorithm called SimRank, but "empirically, ASCOS is shown to return a better score than SimRank because ASCOS considers all paths between two target nodes, whereas SimRank considers only the paths of even lengths." [ref](http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=6785743)

### COSINE
The i-th and j-th rows/columns of the adjecency matrix are regarded as two vectors, and use the cosine of the angle between them is the  similarity measure. The cosine similarity of i and j is the number of common neighbors divided by the geometric mean of their degrees. [ref](https://en.wikipedia.org/wiki/Similarity_(network_science)#Cosine_similarity)

### JACCARD
Number of common neighbors divided by number of vertices that are neighbors of at least one of the two vertices being considered.

### KATZ
Katz centrality is used to measure the degree of influence of an actor in a social network, it computes the relative influence of a node within a network by measuring the number of the immediate neighbors (first degree nodes) and also all other nodes in the network that connect to the node under consideration through these immediate neighbors. Connections made with distant neighbors are penalized by an attenuation factor [ref](https://en.wikipedia.org/wiki/Katz_centrality)

### LHN
Means "Leicht-Holme-Newman" - is like Jaccard, but "punishes" the high degree nodes even more. [ref](http://www.nature.com/articles/srep11404) and [original ref](https://arxiv.org/pdf/physics/0510143v1.pdf)

### RSS2
relational strength similarity: "allows users to explicitly specify the relation strength between neighboring vertices for initialization; and offers a discovery range parameter could be adjusted by users for extended network degree search" [ref](https://clgiles.ist.psu.edu/pubs/SAC2012-discovering-missing-links.pdf)

### DICE
twice the number of common neighbors divided by the sum of the degrees of the vertices.

### INVERSE LOG WEIGHTED
the number of common neighbors weighted by the inverse logarithm of their degrees. It is based on the assumption that two vertices should be considered more similar if they share a low-degree common neighbor, since high-degree common neighbors are more likely to appear even by pure chance.  Lada A. Adamic and Eytan Adar: Friends and neighbors on the Web. Social Networks, 25(3):211-230, 2003.

## Vertex Set Comparison

### WANG
