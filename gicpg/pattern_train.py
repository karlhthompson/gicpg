#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""cluster_train - """
# =============================================================================
# Imports
# =============================================================================
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from gicpg.pattern_model import *
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def embed_graphs(graph, FLAGS):
    # Load node type data
    node_type_s = np.asarray([[v] for k, v in nx.get_node_attributes(graph, 
                             'Type').items()])
    unique_types = np.unique(node_type_s)
    le = LabelEncoder().fit(unique_types)
    node_type = le.transform(node_type_s.ravel())
    node_type = node_type.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore').fit(node_type)
    node_type = enc.transform(node_type).toarray()

    # Load node property data
    idg = np.asarray([[v] for k, v in graph.in_degree])
    idg = Binarizer(threshold=np.median(idg)).fit(idg).transform(idg)
    odg = np.asarray([[v] for k, v in graph.out_degree])
    odg = Binarizer(threshold=np.median(odg)).fit(odg).transform(odg)
    cco = np.asarray([[v] for k, v in nx.algorithms.clustering(graph).items()])
    cco = Binarizer(threshold=np.median(cco)).fit(cco).transform(cco)
    cnm = np.asarray([[v] for k, v in nx.algorithms.core_number(graph).items()])
    cnm = Binarizer(threshold=np.median(cnm)).fit(cnm).transform(cnm)
    idc = np.asarray([[v] for k, v in nx.algorithms.in_degree_centrality(graph).items()])
    idc = Binarizer(threshold=np.median(idc)).fit(idc).transform(idc)
    odc = np.asarray([[v] for k, v in nx.algorithms.out_degree_centrality(graph).items()])
    odc = Binarizer(threshold=np.median(odc)).fit(odc).transform(odc)
    clc = np.asarray([[v] for k, v in nx.algorithms.closeness_centrality(graph).items()])
    clc = Binarizer(threshold=np.median(clc)).fit(clc).transform(clc)
    btc = np.asarray([[v] for k, v in nx.algorithms.betweenness_centrality(graph).items()])
    btc = Binarizer(threshold=np.median(btc)).fit(btc).transform(btc)
    sqc = np.asarray([[v] for k, v in nx.algorithms.square_clustering(graph).items()])
    sqc = Binarizer(threshold=np.median(sqc)).fit(sqc).transform(sqc)
    pgr = np.asarray([[v] for k, v in nx.algorithms.pagerank(graph).items()])
    pgr = Binarizer(threshold=np.median(pgr)).fit(pgr).transform(pgr)
    node_data = np.hstack((idg, odg, cco, cnm, idc, odc, clc, btc, sqc, pgr))

    # Define adjacency and feature matrices
    adj = nx.adjacency_matrix(graph)
    features = sp.csr_matrix(np.hstack((node_data, node_type)))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], 
                                         [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = GCNModelAE(placeholders, num_features, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0]*adj.shape[0]/float((adj.shape[0]*adj.shape[0]-adj.sum())*2)

    # Optimizer
    with tf.name_scope('optimizer'):
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders[
                          'adj_orig'], validate_indices=False), [-1]), pos_weight=
                          pos_weight, norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Get current embedding
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    return emb
