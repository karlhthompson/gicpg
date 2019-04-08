import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Architecture Graphs
G1 = nx.read_graphml('dataset/arch_1.graphml')
G1 = G1.to_undirected()
G1 = max(nx.connected_component_subgraphs(G1), key=len)

print("Properties for the original graph:")
G1.name = 'Original Graph'
print(nx.info(G1))
# print("Radius: %d" % nx.radius(G1))
# print("Diameter: %d" % nx.diameter(G1))
print("Density: %s" % nx.density(G1))
print("Average clustering: %s" % nx.average_clustering(G1))
# print("local efficiency: %s" % nx.algorithms.local_efficiency(G1))
# print("global efficiency: %s" % nx.algorithms.global_efficiency(G1))

# Generated Graphs
fname = 'graphs/GraphRNN_MLP_arch_4_128_pred_10000_3.dat'
with open(fname, "rb") as f:
    graph_list = pickle.load(f)
# for i in range(len(graph_list)):
#     # edges_with_selfloops = graph_list[i].selfloop_edges()
#     # if len(edges_with_selfloops)>0:
#     #     graph_list[i].remove_edges_from(edges_with_selfloops)
#     graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
#     # graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])

options = {
        'line_color': 'grey',
        'font_size': 10,
        'node_size': 10,
        'with_labels': False,
        }
for G2 in graph_list:
    G2 = G2.reverse(copy=True)
    G2 = G2.to_undirected()
    G2 = max(nx.connected_component_subgraphs(G2), key=len)
    print("Properties for the generated graph:")
    G2.name = 'Generated Graph'
    print(nx.info(G2))
    # print("Radius: %d" % nx.radius(G2))
    # print("Diameter: %d" % nx.diameter(G2))
    print("Density: %s" % nx.density(G2))
    print("Average clustering: %s" % nx.average_clustering(G2))
    # print("local efficiency: %s" % nx.algorithms.local_efficiency(G2))
    # print("global efficiency: %s" % nx.algorithms.global_efficiency(G2))

    plt.figure(1)
    nx.draw(G1, **options)
    plt.figure(2)
    nx.draw(G2, **options)
    plt.show()