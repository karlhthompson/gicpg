import networkx as nx
import matplotlib.pyplot as plt 

# Read graph
G = nx.read_graphml('fighterModel.graphml')
G = G.to_undirected()
G = G.subgraph(max(nx.connected_component_subgraphs(G), key=len))

# # Use random reference function
# G = nx.random_reference(G, niter=50, connectivity=True, seed=None)

# # Use lattice reference function
# G = nx.lattice_reference(G, niter=2, connectivity=True, seed=None)

# # Use sigma function
# sig = nx.sigma(G, niter=1, nrand=10, seed=None)
# print(sig)

# # Use omega function
# omg = nx.omega(G, niter=1, nrand=10, seed=None)
# print(omg)

# Print new graph stats
print(nx.info(G))
print("radius: %d" % nx.radius(G))
print("diameter: %d" % nx.diameter(G))
print("density: %s" % nx.density(G))
print("average clustering: %s" % nx.average_clustering(G))
print("local efficiency: %s" % nx.algorithms.local_efficiency(G))
print("global efficiency: %s" % nx.algorithms.global_efficiency(G))

# Plot new graph
options = {
    'line_color': 'grey',
    'font_size': 10,
    'node_size': 10,
    'with_labels': False
    }
nx.draw(G, **options)
plt.show()