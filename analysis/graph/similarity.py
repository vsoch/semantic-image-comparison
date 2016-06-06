# Similarity metrics for networkx graphs

import itertools
import pandas
import numpy
import copy

#[1] ASCOS: an Asymmetric network Structure COntext Similarity measure.
# Hung-Hsuan Chen and C. Lee Giles.  ASONAM 2013
def ascos(G, c=0.9, max_iter=100, is_weighted=False, remove_neighbors=False, remove_self=False):
    """Return the ASCOS similarity between nodes"""
 
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

# http://stackoverflow.com/questions/9767773/calculating-simrank-using-networkx
def simrank(G, r=0.8, max_iter=100, eps=1e-4):

    nodes = G.nodes()
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = numpy.zeros(len(nodes))
    sim = numpy.identity(len(nodes))

    for i in range(max_iter):
        if numpy.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = numpy.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)

            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0: 
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:                    
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))


    return sim


