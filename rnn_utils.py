import networkx as nx
import numpy as np
import pickle


def create_graphs(args):
    graphs=[]

    graphlist = [
        'dataset/DistillerSystem_adapterPattern.graphml',
        'dataset/DistillerSystem_bridgePattern.graphml',
        'dataset/DistillerSystem_compositePattern.graphml',
        'dataset/DistillerSystem_pipeandfilterPattern.graphml',
        'dataset/DistillerSystem_proxyPattern.graphml',
        'dataset/DistillerSystem_publishandsubscribePattern.graphml',
        'dataset/DistillerSystem_toolboxadapterPattern.graphml',
        'dataset/DistillerSystem.graphml',
        'dataset/FighterModel_adapterPattern.graphml',
        'dataset/FighterModel_bridgePattern.graphml',
        'dataset/FighterModel_compositePattern.graphml',
        'dataset/FighterModel_pipeandfilterPattern.graphml',
        'dataset/FighterModel_proxyPattern.graphml',
        'dataset/FighterModel_publishandsubscribePattern.graphml',
        'dataset/FighterModel_toolboxadapterPattern.graphml',
        'dataset/fighterModel.graphml'
        ]

    for name in graphlist:
        G = nx.read_graphml(name)
        G = G.to_undirected()
        G = max(nx.connected_component_subgraphs(G), key=len)
        nodes = G.nodes._nodes
        G_sub = G.subgraph(nodes)
        graphs.append(G_sub)

    args.max_prev_node = 200 # Changed from 300 to 250

    return graphs


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    # adj = adj[~np.all(adj == 0, axis=1)]
    # adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph) ############ VERY IMPORTANT ##############
    return G


# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)