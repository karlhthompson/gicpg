#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""infer_attributes_ml - Infer attributes of an architecture graph using
the following machine learning approaches: k-nearest-neighbors, decision
trees, gradient boosting, and random forest"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

def infer_attributes_ml(Gnx, archname, savepred=True):
    # Load the architecture graph(s)
    archG = nx.read_graphml("dataset/" + archname + ".graphml")
    nodes = list(archG.nodes)
    archG.node[nodes[0]]['data'] = 'Package'
    archG.node[nodes[-1]]['data'] = 'Package'

    # Create node type vector for the arch graph
    node_type_s = np.asarray([v for k, v in nx.get_node_attributes(archG, 'data').items()])
    node_type = np.copy(node_type_s)
    unique_types = np.unique(node_type)
    number = 0
    for uni_type in unique_types:
        for node in range(len(node_type)):
            if node_type[node] == uni_type:
                node_type[node] = number
        number += 1

    # Create node data matrix for the arch graph
    indeg = np.asarray([[v] for k, v in archG.in_degree])
    outdeg = np.asarray([[v] for k, v in archG.out_degree])
    clustering_co = np.asarray([[v] for k, v in nx.algorithms.clustering(archG).items()])
    core_number = np.asarray([[v] for k, v in nx.algorithms.core_number(archG).items()])
    indeg_cent = np.asarray([[v] for k, v in nx.algorithms.in_degree_centrality(archG).items()])
    outdeg_cent = np.asarray([[v] for k, v in nx.algorithms.out_degree_centrality(archG).items()])
    close_cent = np.asarray([[v] for k, v in nx.algorithms.closeness_centrality(archG).items()])
    between_cent = np.asarray([[v] for k, v in nx.algorithms.betweenness_centrality(archG).items()])
    sq_clustering = np.asarray([[v] for k, v in nx.algorithms.square_clustering(archG).items()])
    pagerank = np.asarray([[v] for k, v in nx.algorithms.pagerank(archG).items()])

    node_data = np.hstack((indeg, outdeg, clustering_co, core_number, indeg_cent, 
                        outdeg_cent, close_cent, between_cent, sq_clustering, pagerank))

    # Create node data matrix for the generated graph
    indeg2 = np.asarray([[v] for k, v in Gnx.in_degree])
    outdeg2 = np.asarray([[v] for k, v in Gnx.out_degree])
    clustering_co2 = np.asarray([[v] for k, v in nx.algorithms.clustering(Gnx).items()])
    core_number2 = np.asarray([[v] for k, v in nx.algorithms.core_number(Gnx).items()])
    indeg_cent2 = np.asarray([[v] for k, v in nx.algorithms.in_degree_centrality(Gnx).items()])
    outdeg_cent2 = np.asarray([[v] for k, v in nx.algorithms.out_degree_centrality(Gnx).items()])
    close_cent2 = np.asarray([[v] for k, v in nx.algorithms.closeness_centrality(Gnx).items()])
    between_cent2 = np.asarray([[v] for k, v in nx.algorithms.betweenness_centrality(Gnx).items()])
    sq_clustering2 = np.asarray([[v] for k, v in nx.algorithms.square_clustering(Gnx).items()])
    pagerank2 = np.asarray([[v] for k, v in nx.algorithms.pagerank(Gnx).items()])

    node_data2 = np.hstack((indeg2, outdeg2, clustering_co2, core_number2, indeg_cent2, 
                        outdeg_cent2, close_cent2, between_cent2, sq_clustering2, pagerank2))

    # Define the classifier functions
    clf1 = DecisionTreeClassifier()
    clf2 = GradientBoostingClassifier()
    clf3 = KNeighborsClassifier(n_neighbors=10, weights='distance')
    clf4 = RandomForestClassifier(n_estimators=100)
    clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), 
            ('clf3', clf3), ('clf4', clf4)], voting='soft', weights=[1,2,1,1])

    # Predict node labels
    X = normalize(node_data)
    Xdash = normalize(node_data2)
    y = node_type.astype(np.int)
    y_pred = clf.fit(X, y).predict(Xdash)

    # Print model validation results
    if len(archG) == len(Gnx):
        print("Number of mislabeled points out of a total %d points : %d"
            % (node_data.shape[0],(y != y_pred).sum()))
        print("Percentage of correctly labeled points : %f %%"
            % np.divide((y == y_pred).sum()*100, (node_data.shape[0])))

    # Return node labels to strings
    y_pred = y_pred.tolist()
    for pred in range(len(y_pred)):
        y_pred[pred] = unique_types[int(y_pred[pred])]

    # Merge the predicted node labels into the graph
    node_ids = [k for k, v in Gnx.degree()]
    for node in range(len(Gnx)):
        Gnx.node[node_ids[node]]['Type'] = y_pred[node]

    # Save the predicted node labels in excel format
    if savepred:
        if len(archG) == len(Gnx):
            output = pd.DataFrame({'Original': node_type_s, 'Predicted': y_pred})
            output.to_excel(".temp/output/ml_classification_output.xlsx")
        else:
            output = pd.DataFrame({'Predicted': y_pred})
            output.to_excel(".temp/output/ml_classification_output.xlsx")

    return Gnx

if __name__ == '__main__':
    # Load the generated, unlabeled graph
    import pickle
    fname = 'graphs/GraphRNN_RNN_arch_4_128_pred_6000_1.dat'
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    Gnx = graph_list[0]
    # Select which architecture graphs to learn from
    archname = 'arch_1'
    # Call the inference function
    Gnx = infer_attributes_ml(Gnx, archname, savepred=True)