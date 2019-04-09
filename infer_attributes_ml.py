#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""infer_attributes_ml - Infer attributes of an architecture graph using
the following machine learning approaches: k-nearest-neighbors, decision
trees, gaussian processes, and multi-layer perceptron"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx
import pandas as pd
import numpy as np
from sklearn import neighbors, tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier


def infer_attributes_ml(Gnx, learnlist, clf, savepred=True, plot=False):
    # Load the architecture graph(s)
    archG = nx.DiGraph()
    for graphname in learnlist:
        graph = nx.read_graphml("dataset/" + graphname + ".graphml")
        nodes = list(graph.nodes)
        graph.node[nodes[0]]['data'] = 'Package'
        graph.node[nodes[-1]]['data'] = 'Package'
        archG = nx.compose(archG, graph)

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

    # Predict node labels
    X = node_data
    Xdash = node_data2
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

    if savepred:
        if len(archG) == len(Gnx):
            # Save the output in excel format
            output = pd.DataFrame({'Original': node_type_s, 'Predicted': y_pred})
            output.to_excel(".temp/output/ml_classification_output.xlsx")
        else:
            output = pd.DataFrame({'Predicted': y_pred})
            output.to_excel(".temp/output/ml_classification_output.xlsx")

    if plot:
        # Create color maps
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        h = .01  # step size in the mesh
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
        X = X[:, :2]
        clf.fit(X, y)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')" % (10, 'distance'))
        plt.show()

    return y_pred

if __name__ == '__main__':
    # Select which architecture graphs to learn from
    learnlist = ['arch_2', 'arch_1']

    # Load the generated, unlabeled graph
    Gnx = nx.read_graphml("dataset/" + "arch_2" + ".graphml")

    # Select the classifier
    clf = tree.DecisionTreeClassifier()
    # clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
    # clf = GaussianProcessClassifier()
    # clf = MLPClassifier(activation='tanh', solver='lbfgs')

    # Call the inference function
    y_pred = infer_attributes_ml(Gnx, learnlist, clf, savepred=True, plot=False)