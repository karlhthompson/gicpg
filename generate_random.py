import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np

Graph = nx.read_graphml('adapterPattern.graphml')
options = {
    'line_color': 'grey',
    'font_size': 10,
    'node_size': 10,
    'with_labels': False
    }
n = 340

# G1 = nx.circular_ladder_graph(n, create_using=None)
# nx.draw(G1, **options)
# plt.show()

# G2 = nx.circulant_graph(n, [1], create_using=None)
# nx.draw(G2, **options)
# plt.show()

# G3 = nx.cycle_graph(n, create_using=G)
# nx.draw(G3, **options)
# plt.show()

# G4 = nx.dorogovtsev_goltsev_mendes_graph(n, create_using=None) # n is the number of generation # not random
# print(nx.info(G4))
# nx.draw(G4, **options)
# plt.show()

# G5 = nx.full_rary_tree(r=8, n=n, create_using=Graph)
# G5 = G5.to_undirected()
# print(nx.info(G5))
# nx.draw(G5, **options)
# plt.show()

# G6 = nx.lollipop_graph(m=3, n=n, create_using=None)
# print(nx.info(G6))
# nx.draw(G6, **options)
# plt.show()

# G7 = nx.path_graph(n, create_using=Graph)
# print(nx.info(G7))
# nx.draw(G7, **options)
# plt.show()

# G8 = nx.star_graph(n, create_using=None)
# print(nx.info(G8))
# nx.draw(G8, **options)
# plt.show()

# G9 = nx.turan_graph(n, r=100)
# print(nx.info(G9))
# nx.draw(G9, **options)
# plt.show()

# G10 = nx.wheel_graph(n, create_using=None)
# print(nx.info(G10))
# nx.draw(G10, **options)
# plt.show()

# G11 = nx.margulis_gabber_galil_graph(n, create_using=None)
# print(nx.info(G11))
# nx.draw(G11, **options)
# plt.show()

# G12 = nx.grid_2d_graph(m=1, n=n, periodic=False, create_using=None)
# print(nx.info(G12))
# nx.draw(G12, **options)
# plt.show()

# G13 = nx.fast_gnp_random_graph(n, p=0.003, seed=None, directed=True)
# G13 = G13.to_undirected()
# G13 = max(nx.connected_component_subgraphs(G13), key=len)
# print(nx.info(G13))
# nx.draw(G13, **options)
# plt.show()

# G14 = nx.gnp_random_graph(n, p=0.003, seed=None, directed=True)
# G14 = G14.to_undirected()
# G14 = max(nx.connected_component_subgraphs(G14), key=len)
# print(nx.info(G14))
# nx.draw(G14, **options)
# plt.show()

# G = nx.dense_gnm_random_graph(n, m=500, seed=None)

# G = nx.gnm_random_graph(n, m=500, seed=None, directed=True)

# G = nx.erdos_renyi_graph(n, p=0.005, seed=None, directed=True)

# G = nx.binomial_graph(n, p=0.005, seed=None, directed=True)

# G = nx.newman_watts_strogatz_graph(n, k=2, p=0.5, seed=None)

# G = nx.watts_strogatz_graph(n, k=3, p=0.5, seed=None)

# G = nx.connected_watts_strogatz_graph(n, k=3, p=0.5, tries=100, seed=None)

# G = nx.gn_graph(n, kernel=None, create_using=Graph, seed=None)

# G = nx.gnr_graph(n, p=0.5, create_using=None, seed=None)

# G = nx.random_k_out_graph(n, k=2, alpha=0.1, self_loops=True, seed=None)

# G = nx.scale_free_graph(n, alpha=0.41, beta=0.24, gamma=0.35, delta_in=0.2, delta_out=0, create_using=None, seed=None)

# G = nx.stochastic_graph(Graph, copy=False, weight='weight')

# G = nx.ego_graph(Graph, n='Class:Aircraft', radius=5, center=True, undirected=False, distance=None)

G = nx.spectral_graph_forge(Graph.to_undirected(), alpha=0.8, transformation='identity', seed=None)

# G = G.to_undirected()
G = max(nx.connected_component_subgraphs(G), key=len)

print(nx.info(G))
print("Radius: %d" % nx.radius(G))
print("Diameter: %d" % nx.diameter(G))
print("Density: %s" % nx.density(G))
print("Average clustering: %s" % nx.average_clustering(G))
nx.draw(G, **options)
plt.show()