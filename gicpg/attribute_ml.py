#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""infer_attributes_ml - Infer attributes of an architecture graph using an
ensemble of random forests (an averaging method) and gradient tree boosting 
(a boosting method)."""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

def infer_attributes_ml(Gnx, archname, savepred=False, testnodes=False):
    # Load the architecture graph(s)
    archG = nx.read_graphml("dataset/" + archname + ".graphml")
    nodes = list(archG.nodes)
    archG.node[nodes[0]]['data'] = 'Package'
    archG.node[nodes[-1]]['data'] = 'Package'

    # Create node type vector for the arch graph
    node_type_s = np.asarray([v for k, v in nx.get_node_attributes(archG, 'data').items()])
    unique_types = np.unique(node_type_s)
    le = LabelEncoder()
    le.fit(unique_types)
    node_type = le.transform(node_type_s)

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
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GradientBoostingClassifier()
    clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2)], voting='soft')

    # Predict node labels
    X = node_data
    y = node_type.astype(np.int)
    Xdash = node_data2
    y_pred = clf.fit(X, y).predict(Xdash)
    if testnodes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        y_pred_part = clf.fit(X_train, y_train).predict(X_test)
    
    # Print model validation results
    if len(archG) == len(Gnx) and testnodes == False:
        print("Number of mislabeled nodes out of a total %d nodes : %d"
            % (node_data.shape[0],(y != y_pred).sum()))
        print("Percentage of correctly labeled nodes : %f %%"
            % np.divide((y == y_pred).sum()*100, (node_data.shape[0])))
    elif len(archG) == len(Gnx) and testnodes == True:
        print("Number of mislabeled nodes out of a total %d nodes : %d"
            % (X_test.shape[0],(y_test != y_pred_part).sum()))
        print("Percentage of correctly labeled nodes : %f %%"
            % np.divide((y_test == y_pred_part).sum()*100, (X_test.shape[0])))

    # Return node labels to strings
    y_pred = y_pred.tolist()
    for pred in range(len(y_pred)):
        y_pred[pred] = unique_types[int(y_pred[pred])]

    # Merge the predicted node labels into the graph
    node_ids = [k for k, v in Gnx.degree()]
    for node in range(len(Gnx)):
        Gnx.node[node_ids[node]]['Type'] = y_pred[node]

    # Create edge type vector for the arch graph
    archG = nx.convert_node_labels_to_integers(archG)
    edge_type_s = np.array(list(nx.get_edge_attributes(archG, 'data').items()))[:, 1]
    unique_types = np.unique(edge_type_s)
    le = LabelEncoder()
    le.fit(unique_types)
    edge_type = le.transform(edge_type_s)

    # Create edge data matrix for the arch graph
    edgelist = list(nx.edges(archG))
    att_1 = []
    att_2 = []
    for n in range(len(edgelist)):
        att_1.append(archG.node[int(edgelist[n][0])]['data'])
        att_2.append(archG.node[int(edgelist[n][1])]['data'])
    edge_data = np.column_stack((att_1, att_2))

    # Create edge data matrix for the generated graph
    Gnx = nx.convert_node_labels_to_integers(Gnx)
    edgelist2 = list(nx.edges(Gnx))
    att_1 = []
    att_2 = []
    for n in range(len(edgelist2)):
        att_1.append(Gnx.node[int(edgelist2[n][0])]['Type'])
        att_2.append(Gnx.node[int(edgelist2[n][1])]['Type'])
    edge_data2 = np.column_stack((att_1, att_2))

    # Encode edge data for both graphs using a one-hot encoder
    enc = OneHotEncoder(handle_unknown='ignore').fit(edge_data)
    edge_data_bi = enc.transform(edge_data).toarray()
    edge_data_bi2 = enc.transform(edge_data2).toarray()

    # Predict edge labels
    clsf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    ye_pred = clsf.fit(edge_data_bi, edge_type).predict(edge_data_bi2)

    # Print model validation results
    if len(archG) == len(Gnx):
        print("Number of mislabeled edges out of a total %d edges : %d"
            % (edge_data.shape[0],(edge_type != ye_pred).sum()))
        print("Percentage of correctly labeled edges : %f %%"
            % np.divide((edge_type == ye_pred).sum()*100, (edge_data.shape[0])))

    # Return edge labels to strings
    ye_pred = ye_pred.tolist()
    for pred in range(len(ye_pred)):
        ye_pred[pred] = unique_types[int(ye_pred[pred])]

    # Merge the predicted edge labels into the graph
    n = 0
    for edge in Gnx.edges:
        Gnx.edges[edge]['EType'] = ye_pred[n]
        n += 1

    # Save the predicted node and edge labels in excel format
    if savepred:
        if len(archG) == len(Gnx):
            output = pd.DataFrame.from_dict({'Orig. Node Labels': node_type_s, 
                'Pred. Node Labels': y_pred,'Orig. Edge Labels': edge_type_s, 
                'Pred. Edge Labels': ye_pred}, orient='index')
            output = output.transpose()
            output.to_excel(".temp/output/ml_classification_output.xlsx")
        else:
            output = pd.DataFrame.from_dict({'Pred. Node Labels': y_pred, 
                'Pred. Edge Labels': ye_pred}, orient='index')
            output = output.transpose()
            output.to_excel(".temp/output/ml_classification_output.xlsx")

    return Gnx

if __name__ == '__main__':
    # Select which architecture graphs to learn from
    archname = 'arch_2'
    # Load the generated, unlabeled graph
    # import pickle
    # fname = 'graphs/GraphRNN_RNN_arch_4_128_pred_6000_1.dat'
    # with open(fname, "rb") as f:
    #     graph_list = pickle.load(f)
    # Gnx = graph_list[0]
    # Optional: load the same architecture graph for validation
    Gnx = nx.read_graphml('dataset/' + archname + '.graphml')
    # Call the inference function
    Gnx = infer_attributes_ml(Gnx, archname, savepred=True, testnodes=False)