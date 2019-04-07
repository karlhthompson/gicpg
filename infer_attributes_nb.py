#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""infer_attributes_nb - Infer attributes of an architecture graph using
a naive bayes clasifier"""
# =============================================================================
# Imports
# =============================================================================
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn import datasets
import networkx as nx
import pandas as pd
import numpy as np 

# Specify the naive bayes distribution to fit the data to
nbd = GaussianNB()
# nbd = MultinomialNB()
# nbd = ComplementNB()

# Load the graph
graphname = "FighterModel"
Gnx = nx.read_graphml("architectures/" + graphname + ".graphml")
nodes = list(Gnx.nodes)
Gnx.node[nodes[0]]['data'] = 'Package'
Gnx.node[nodes[-1]]['data'] = 'Package'
for node in nodes:
    if Gnx.node[node]['data'] == 'Port':
        Gnx.node[node]['data'] = 'Package'
    elif Gnx.node[node]['data'] == 'Profile':
        Gnx.node[node]['data'] = 'Package'
    elif Gnx.node[node]['data'] == 'Signal':
        Gnx.node[node]['data'] = 'Package'
    elif Gnx.node[node]['data'] == 'Extension':
        Gnx.node[node]['data'] = 'Dependency'
    elif Gnx.node[node]['data'] == 'Stereotype':
        Gnx.node[node]['data'] = 'Dependency'
    elif Gnx.node[node]['data'] == 'Parameter':
        Gnx.node[node]['data'] = 'Dependency'
    elif Gnx.node[node]['data'] == 'Operation':
        Gnx.node[node]['data'] = 'Dependency'

# Create node type vector
node_type_s = np.asarray([v for k, v in nx.get_node_attributes(Gnx, 'data').items()])
node_type = np.copy(node_type_s)
unique_types = np.unique(node_type)
number = 0
for uni_type in unique_types:
    for node in range(len(node_type)):
        if node_type[node] == uni_type:
            node_type[node] = number
    number += 1

# Create node data matrix
indeg = np.asarray([[v] for k, v in Gnx.in_degree])
outdeg = np.asarray([[v] for k, v in Gnx.out_degree])
clustering_co = np.asarray([[v] for k, v in nx.algorithms.clustering(Gnx).items()])
core_number = np.asarray([[v] for k, v in nx.algorithms.core_number(Gnx).items()])

# indeg_cent = np.asarray([[v] for k, v in nx.algorithms.in_degree_centrality(Gnx).items()])
# outdeg_cent = np.asarray([[v] for k, v in nx.algorithms.out_degree_centrality(Gnx).items()])
# close_cent = np.asarray([[v] for k, v in nx.algorithms.closeness_centrality(Gnx).items()])
# between_cent = np.asarray([[v] for k, v in nx.algorithms.betweenness_centrality(Gnx).items()])
# sq_clustering = np.asarray([[v] for k, v in nx.algorithms.square_clustering(Gnx).items()])
# pagerank = np.asarray([[v] for k, v in nx.algorithms.pagerank(Gnx).items()])

node_data = np.hstack((indeg, outdeg, clustering_co, core_number))

# Predict node labels
y_pred = nbd.fit(node_data, node_type).predict(node_data)
print("Number of mislabeled points out of a total %d points : %d"
      % (node_data.shape[0],(node_type != y_pred).sum()))
print("Percentage of correctly labeled points : %f %%"
      % np.divide((node_type == y_pred).sum(), (node_data.shape[0])))

# Return node labels to strings
y_pred = y_pred.tolist()
for pred in range(len(y_pred)):
    y_pred[pred] = unique_types[int(y_pred[pred])]

# Save the output in excel format
output = pd.DataFrame({'Original': node_type_s, 'Predicted': y_pred})
output.to_excel(".temp/output/naive_bayes_output.xlsx")
print()