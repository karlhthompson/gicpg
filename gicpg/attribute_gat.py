#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""infer_attributes_gat - Infer attributes of an architecture graph using
a graph attention network"""
# =============================================================================
# Imports
# =============================================================================
import os
import datetime
import numpy as np
import pandas as pd
import networkx as nx
import stellargraph as sg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from itertools import combinations
from stellargraph.layer import GAT
from sklearn.decomposition import PCA
from stellargraph.mapper import FullBatchNodeGenerator
from sklearn import feature_extraction, model_selection
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers, optimizers, losses, metrics, Model

def infer_attributes_gat(Gnx, savepred=True, plot=False):
    # Define node data
    feature_names = [
                    "in_degree", 
                    "out_degree", 
                    # "in_degree_centrality", 
                    # "out_degree_centrality",
                    # "closeness_centrality", 
                    # "betweenness_centrality",
                    "clustering_coefficient", 
                    # "square_clustering", 
                    "core_number", 
                    # "pagerank", 
                    # "constraint", 
                    # "effective_size"
                    ]
    node_type = [v for k, v in nx.get_node_attributes(Gnx, 'data').items()]
    d = {"node_type": node_type}
    if "in_degree" in feature_names:
        indeg = [v for k, v in Gnx.in_degree]
        indeg = np.divide(indeg, max(indeg))
        indeg[indeg>=0.5] = 1
        indeg[indeg<0.5] = 0
        d["in_degree"] = indeg
    if "out_degree" in feature_names:
        outdeg = [v for k, v in Gnx.out_degree]
        outdeg = np.divide(outdeg, max(outdeg))
        outdeg[outdeg>=0.5] = 1
        outdeg[outdeg<0.5] = 0
        d["out_degree"] = outdeg
    if "in_degree_centrality" in feature_names:
        indeg_cent = [v for k, v in nx.algorithms.in_degree_centrality(Gnx).items()]
        indeg_cent = np.divide(indeg_cent, max(indeg_cent))
        indeg_cent[indeg_cent>=0.5] = 1
        indeg_cent[indeg_cent<0.5] = 0
        d["in_degree_centrality"] = indeg_cent
    if "out_degree_centrality" in feature_names:
        outdeg_cent = [v for k, v in nx.algorithms.out_degree_centrality(Gnx).items()]
        outdeg_cent = np.divide(outdeg_cent, max(outdeg_cent))
        outdeg_cent[outdeg_cent>=0.5] = 1
        outdeg_cent[outdeg_cent<0.5] = 0
        d["out_degree_centrality"] = outdeg_cent
    if "closeness_centrality" in feature_names:
        close_cent = [v for k, v in nx.algorithms.closeness_centrality(Gnx).items()]
        close_cent = np.divide(close_cent, max(close_cent))
        close_cent[close_cent>=0.5] = 1
        close_cent[close_cent<0.5] = 0
        d["closeness_centrality"] = close_cent
    if "betweenness_centrality" in feature_names:
        between_cent = [v for k, v in nx.algorithms.betweenness_centrality(Gnx).items()]
        between_cent = np.divide(between_cent, max(between_cent))
        between_cent[between_cent>=0.5] = 1
        between_cent[between_cent<0.5] = 0
        d["betweenness_centrality"] = between_cent
    if "clustering_coefficient" in feature_names:
        clustering_co = [v for k, v in nx.algorithms.clustering(Gnx).items()]
        clustering_co = np.divide(clustering_co, max(clustering_co))
        clustering_co[clustering_co>=0.5] = 1
        clustering_co[clustering_co<0.5] = 0
        d["clustering_coefficient"] = clustering_co
    if "square_clustering" in feature_names:
        sq_clustering = [v for k, v in nx.algorithms.square_clustering(Gnx).items()]
        sq_clustering = np.divide(sq_clustering, max(sq_clustering))
        sq_clustering[sq_clustering>=0.5] = 1
        sq_clustering[sq_clustering<0.5] = 0
        d["square_clustering"] = sq_clustering
    if "core_number" in feature_names:
        core_number = [v for k, v in nx.algorithms.core_number(Gnx).items()]
        core_number = np.divide(core_number, max(core_number))
        core_number[core_number>=0.5] = 1
        core_number[core_number<0.5] = 0
        d["core_number"] = core_number
    if "pagerank" in feature_names:
        pagerank = [v for k, v in nx.algorithms.pagerank(Gnx).items()]
        pagerank = np.divide(pagerank, max(pagerank))
        pagerank[pagerank>=0.5] = 1
        pagerank[pagerank<0.5] = 0
        d["pagerank"] = pagerank
    if "constraint" in feature_names:
        constraint = [v for k, v in nx.algorithms.constraint(Gnx).items()]
        constraint = np.divide(constraint, max(constraint))
        constraint[np.isnan(constraint)] = 0
        constraint[constraint>=0.5] = 1
        constraint[constraint<0.5] = 0
        d["constraint"] = constraint
    if "effective_size" in feature_names:
        effective_size = [v for k, v in nx.algorithms.effective_size(Gnx).items()]
        effective_size = np.divide(effective_size, max(effective_size))
        effective_size[np.isnan(effective_size)] = 0
        effective_size[effective_size>=0.5] = 1
        effective_size[effective_size<0.5] = 0
        d["effective_size"] = effective_size
    node_data = pd.DataFrame(data=d, index=nodes)
    node_data = shuffle(node_data)

    # Split the data
    train_data, test_data = model_selection.train_test_split(node_data, train_size=int(0.80*len(Gnx)), test_size=None, stratify=node_data['node_type'])
    val_data, test_data = model_selection.train_test_split(test_data, train_size=int(0.15*len(Gnx)), test_size=None, stratify=test_data['node_type'])

    # Convert to numeric arrays
    target_encoding = feature_extraction.DictVectorizer(sparse=False)

    train_targets = target_encoding.fit_transform(train_data[["node_type"]].to_dict('records'))
    val_targets = target_encoding.transform(val_data[["node_type"]].to_dict('records'))
    test_targets = target_encoding.transform(test_data[["node_type"]].to_dict('records'))

    node_features = node_data[feature_names]

    # Create the GAT model in Keras
    G = sg.StellarDiGraph(Gnx, node_features=node_features)
    print(G.info())

    generator = FullBatchNodeGenerator(G)

    train_gen = generator.flow(train_data.index, train_targets)

    gat = GAT(
        layer_sizes=[8, train_targets.shape[1]],
        attn_heads=8,
        generator=generator,
        bias=True,
        in_dropout=0.5,
        attn_dropout=0.5,
        activations=["elu","softmax"],
        normalize=None,
    )

    # Expose the input and output tensors of the GAT model for node prediction, via GAT.node_model() method:
    x_inp, predictions = gat.node_model()

    # Train the model
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        weighted_metrics=["acc"],
    )

    val_gen = generator.flow(val_data.index, val_targets)

    if not os.path.isdir(".temp/logs"):
        os.makedirs(".temp/logs")
    if not os.path.isdir(".temp/output"):
        os.makedirs(".temp/output")

    es_callback = EarlyStopping(
        monitor="val_weighted_acc", 
        patience=100   # patience is the number of epochs to wait before early stopping in case of no further improvement
    )  

    mc_callback = ModelCheckpoint(
        ".temp/logs/best_model.h5",
        monitor="val_weighted_acc",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit_generator(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )

    # Reload the saved weights
    model.load_weights(".temp/logs/best_model.h5")

    # Evaluate the best nidek in the test set
    test_gen = generator.flow(test_data.index, test_targets)

    test_metrics = model.evaluate_generator(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Make predictions with the model
    all_nodes = node_data.index
    all_gen = generator.flow(all_nodes)
    all_predictions = model.predict_generator(all_gen)

    node_predictions = target_encoding.inverse_transform(all_predictions)

    results = pd.DataFrame(node_predictions, index=G.nodes()).idxmax(axis=1)
    df = pd.DataFrame({"Predicted": results, "True": node_data['node_type']})
    print(df.head)

    if savepred:
        df.to_excel(".temp/output/output" + str(datetime.datetime.now()).replace(':','-') + ".xlsx")

    if plot:
        # Node embeddings
        emb_layer = model.layers[3]
        print("Embedding layer: {}, output shape {}".format(emb_layer.name, emb_layer.output_shape))
        embedding_model = Model(inputs=x_inp, outputs=emb_layer.output)
        emb = embedding_model.predict_generator(all_gen)

        X = emb
        y = np.argmax(target_encoding.transform(node_data.reindex(G.nodes())[["node_type"]].to_dict('records')), axis=1)

        if X.shape[1] > 2:
            transform = TSNE #PCA 
            trans = transform(n_components=2)
            emb_transformed = pd.DataFrame(trans.fit_transform(X), index=list(G.nodes()))
            emb_transformed['label'] = y
        else:
            emb_transformed = pd.DataFrame(X, index=list(G.nodes()))
            emb_transformed = emb_transformed.rename(columns = {'0':0, '1':1})

        def plot_emb(transform, emb_transformed):
            fig, ax = plt.subplots(figsize=(7,7))
            ax.scatter(emb_transformed[0], emb_transformed[1], c=emb_transformed['label'].astype("category"), 
                        cmap="jet", alpha=0.7)
            ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
            plt.title('{} visualization of GAT embeddings for the fighter graph'.format(transform.__name__))

        # Plot the training history
        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]

        def plot_history(history):
            metrics = sorted(set([remove_prefix(m, "val_") for m in list(history.history.keys())]))
            for m in metrics:
                # summarize history for metric m
                plt.figure()
                plt.plot(history.history[m])
                plt.plot(history.history['val_' + m])
                plt.title(m)
                plt.ylabel(m)
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='best')

        plot_history(history)
        plot_emb(transform, emb_transformed)
        plt.show()

    return df

if __name__ == '__main__':
    # Load the architecture graph
    graphname = "arch_1"
    Gnx = nx.read_graphml("dataset/" + graphname + ".graphml")
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

    # Increase number of input data points
    # Gnx = nx.disjoint_union_all([Gnx, Gnx, Gnx, Gnx, Gnx, Gnx, Gnx, Gnx, Gnx, Gnx])
    Gnx = nx.disjoint_union_all([Gnx])
    nodes = list(Gnx.nodes)

    # Call the inference function
    df = infer_attributes_gat(Gnx, savepred=True, plot=True)