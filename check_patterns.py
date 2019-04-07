import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt 
import pickle

# Load graphs
G1 = nx.read_graphml('architectures/DistillerSystem.graphml')
G2 = nx.read_graphml('patterns/facadePattern.graphml')

# Transform to undirected
G1 = G1.to_undirected()
G2 = G2.to_undirected()

# # Write node match function
# def node_match(n1,n2):
#     if n1==G1.nodes['ROOT:Distiller'] and n2==G2.nodes['ROOT:FacadePattern']:
#         value = True
#     else:
#         value = False
#     return value

# # Check isomorphism
# GM = GraphMatcher(G1=G1, G2=G2, node_match=None, edge_match=None)
# isomorph = GM.subgraph_is_isomorphic()
# print(isomorph)

# # List all isomorphisms between the two graphs
# isomorph_list = list(GM.subgraph_isomorphisms_iter())
# # print(isomorph_list)

# # Save isomorphism list
# pickling_on = open('pickles/distiller2facade.pickle',"wb")
# pickle.dump(isomorph_list, pickling_on)
# pickling_on.close()

# Read pickle file
pickle_off = open('pickles/distiller2facade.pickle',"rb")
isomorph_list = pickle.load(pickle_off)
pickle_off.close()

# Plot a sample isomorph
options = {'line_color': 'grey', 'font_size': 10, 'node_size': 10, 'with_labels': True}
G3 = G1.subgraph(isomorph_list[0])
plt.figure(1)
nx.draw(G3, **options)
plt.figure(2)
nx.draw(G2, **options)
plt.show()