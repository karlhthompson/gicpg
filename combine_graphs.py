import networkx as nx 
import matplotlib.pyplot as plt 

# Input graph 1
archname = 'FighterModel'
G1 = nx.read_graphml('architectures/' + archname + '.graphml')
# G1 = G1.to_undirected()
# G1 = G1.subgraph(max(nx.connected_component_subgraphs(G1), key=len))

# # Input graph 2
pattname = 'toolboxadapterPattern'
G2 = nx.read_graphml('patterns/' + pattname + '.graphml')
# G2 = G2.to_undirected()

# Use the composition function
# G3 = nx.compose(G1, G2)

# # Use the union function
G3 = nx.disjoint_union(G1, G2)

# Display all graphs information
print()
print('Architecture graph information:')
print(nx.info(G1))
print("radius: %d" % nx.radius(G1))
print("diameter: %d" % nx.diameter(G1))
print("density: %s" % nx.density(G1))
print("average clustering: %s" % nx.average_clustering(G1))
print("local efficiency: %s" % nx.algorithms.local_efficiency(G1))
print("global efficiency: %s" % nx.algorithms.global_efficiency(G1))
print()

print('Pattern graph information:')
print(nx.info(G2))
print()

print('Combined graph information:')
print(nx.info(G3))
print("radius: %d" % nx.radius(G3))
print("diameter: %d" % nx.diameter(G3))
print("density: %s" % nx.density(G3))
print("average clustering: %s" % nx.average_clustering(G3))
print("local efficiency: %s" % nx.algorithms.local_efficiency(G3))
print("global efficiency: %s" % nx.algorithms.global_efficiency(G3))
print()

# Check isomorphism
GM = nx.algorithms.isomorphism.GraphMatcher(G1=G3, G2=G2, node_match=None, edge_match=None)
isomorph = GM.subgraph_is_isomorphic()
print("Is the generated and pattern graphs isomorphic: %s" % isomorph)
print()

# Check connectivity
connected = nx.is_connected(G3)
print("Is the generated graph connected: %s" % isomorph)
print()

# Plot the new old and new graphs
options = {'line_color': 'grey', 'font_size': 10, 'node_size': 10, 'with_labels': False}
# plt.figure(1)
# nx.draw(G1, **options)
# plt.figure(2)
# nx.draw(G2, **options)
plt.figure(3)
nx.draw(G3, **options)
plt.show()

# # Save generated graph
# path = 'composites/' + archname + '_' + pattname + '.graphml'
# if isomorph and connected:
#     nx.write_graphml(G3, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False)