# Similarity metrics for networkx graphs
# credit/thanks to networkx_addon package

from collections import defaultdict
import networkx as nx
import scipy.linalg
import itertools
import pandas
import numpy
import copy


def rss2(G, remove_neighbors=False, disregard_weight=True):
    """Return the rss2 similarity between nodes"""

    weighted_deg = G.degree(weight='weight')
    rss2 = pandas.DataFrame(0,index=G.nodes(),columns=G.nodes())
    cur_iter = 0
    total_iter = G.number_of_nodes()
    for a in G.nodes():
        for b in G.neighbors(a):
            for c in G.neighbors(b):
                if a == c:
                    continue
                if remove_neighbors and c in G.neighbors(a):
                    continue
                if disregard_weight:
                    t1 = float(1)
                    t2 = float(1)
                    s1 = len(set(G.neighbors(a)))
                    s2 = len(set(G.neighbors(b)))
                else:
                    t1 = float(G[a][b]['weight'])
                    t2 = float(G[b][c]['weight'])
                    s1 = weighted_deg[a]
                    s2 = weighted_deg[b]
                rss2.loc[a,c] = rss2.loc[a,c] + (t1/s1 * t2/s2)

    return rss2


def lhn(G, c=0.9, remove_neighbors=False, inv_method=0):

    A = nx.adjacency_matrix(G, nodelist=G.nodes(), weight=None)
    S = katz(G,c=c,remove_neighbors=remove_neighbors, inv_method=inv_method)
    deg = numpy.array(sum(A.todense())).reshape(-1,)

    for i in range(S.shape[0]):
        row_idx = S.index[i]
        for j in range(S.shape[1]):
            col_idx = S.columns[j]
            S.loc[row_idx,col_idx] = S.loc[row_idx,col_idx] / (deg[i]*deg[j])
  
    return S


def katz(G, c=0.9, remove_neighbors=False, inv_method=0):

    A = nx.adjacency_matrix(G, nodelist=G.nodes(), weight=None)
    w, v = numpy.linalg.eigh(A.todense()) # should use numpy.linalg.matrix_rank
    lambda1 = max([abs(x) for x in w])
    I = numpy.eye(A.shape[0])
    if inv_method == 1:
        S = scipy.linalg.pinv(I - c/lambda1 * A)
    elif inv_method == 2:
        S = numpy.linalg.inv(I - c/lambda1 * A)
    else:
        S = numpy.linalg.pinv(I - c/lambda1 * A)
    S = pandas.DataFrame(S)
    S.index = G.nodes()
    S.columns = G.nodes()
    return S

def ascos(G, c=0.9, max_iter=100, is_weighted=False, remove_neighbors=False, remove_self=False):
    """Return the ASCOS similarity between nodes
    #[1] ASCOS: an Asymmetric network Structure COntext Similarity measure.
    # Hung-Hsuan Chen and C. Lee Giles.  ASONAM 2013
    """
    node_ids = G.nodes()
    node_lookup = dict()
    for i, n in enumerate(node_ids):
        node_lookup[n] = i

    neighbor_ids = [G.neighbors(n) for n in node_ids]
    neighbors = []
    for neighbor_id in neighbor_ids:
        neighbors.append([node_lookup[n] for n in neighbor_id])

    n = G.number_of_nodes()
    sim = numpy.eye(n)
    sim_old = numpy.zeros(shape = (n, n))

    for iter_ctr in range(max_iter):
        if _is_converge(sim, sim_old, n, n):
            break
        sim_old = copy.deepcopy(sim)
        for i in range(n):
            for j in range(n):
                if not is_weighted:
                    if i == j:
                        continue
                    s_ij = 0.0
                    for n_i in neighbors[i]:
                        s_ij += sim_old[n_i, j]
                    sim[i, j] = c * s_ij / len(neighbors[i]) if len(neighbors[i]) > 0 else 0
                else:
                    if i == j:
                        continue
                    s_ij = 0.0
                    for n_i in neighbors[i]:
                        w_ik = G[node_ids[i]][node_ids[n_i]]['weight'] if 'weight' in G[node_ids[i]][node_ids[n_i]] else 1
                        s_ij += float(w_ik) * (1 - math.exp(-w_ik)) * sim_old[n_i, j]

                    w_i = G.degree(weight='weight')[node_ids[i]]
                    sim[i, j] = c * s_ij / w_i if w_i > 0 else 0

    if remove_self:
        for i in range(n):
            sim[i,i] = 0

    if remove_neighbors:
        for i in range(n):
           for j in nbs[i]:
               sim[i,j] = 0

    sim = pandas.DataFrame(sim)
    sim.index = node_ids
    sim.columns = node_ids
    return sim

def _is_converge(sim, sim_old, nrow, ncol, eps=1e-4):
  for i in range(nrow):
    for j in range(ncol):
      if abs(sim[i,j] - sim_old[i,j]) >= eps:
        return False
  return True


def cosine(G, remove_neighbors=False):

    cos = pandas.DataFrame()
    total_iter = G.number_of_nodes()
    for i, a in enumerate(G.nodes()):
        for b in G.neighbors(a):
            for c in G.neighbors(b):
                if a == c:
                    continue
                if remove_neighbors and c in G.neighbors(a):
                    continue
                s1 = set(G.neighbors(a))
                s2 = set(G.neighbors(c))
                cos.loc[a,c] = float(len(s1 & s2)) / (len(s1) + len(s2))
    cos = cos.fillna(0)
    return cos



def jaccard(G, remove_neighbors=False, dump_process=False):
    jac = pandas.DataFrame()
    total_iter = G.number_of_nodes()
    for i, a in enumerate(G.nodes(), 1):
        for b in G.neighbors(a):
            for c in G.neighbors(b):
                if a == c:
                    continue
                if remove_neighbors and c in G.neighbors(a):
                    continue
                s1 = set(G.neighbors(a))
                s2 = set(G.neighbors(c))
                jac.loc[a,c] = float(len(s1 & s2)) / len(s1 | s2)
    jac = jac.fillna(0)
    return jac
