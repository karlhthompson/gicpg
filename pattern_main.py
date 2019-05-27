#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""pattern_main - Identify design patterns in architecture graphs"""
# =============================================================================
# Imports
# =============================================================================
import pickle
import pandas as pd
from os import listdir
import tensorflow as tf
from os.path import isfile, join
from gicpg.pattern_train import *
from sklearn.cluster import AgglomerativeClustering

# Embedding settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

# Load attributed graphs
graph_file = [f for f in listdir("./attributed_graphs") if 
              isfile(join("./attributed_graphs",f))][0]
with open('attributed_graphs/'+graph_file, "rb") as f:
    graph_list = pickle.load(f)

# Start the main loop
subgraph_list = []
for n in range(10): #range(len(graph_list))

    # Print processing status
    print('Processing graph %i out of %i' %(n+1, len(graph_list)))

    # Embed each attributed graph
    emb = embed_graphs(graph_list[n], FLAGS)

    # Cluster the graph embedding
    clr = AgglomerativeClustering(n_clusters=16,
          connectivity=nx.adjacency_matrix(graph_list[n])).fit(emb)

    # Extract individual clusters
    nodes = list(graph_list[n].nodes)
    for label in np.unique(clr.labels_):
        subgraph_nodes = [nodes[i] for i in range(len(graph_list[n])) if (
                          clr.labels_==label)[i]]
        subgraph = graph_list[n].subgraph(subgraph_nodes)
        subgraph_list.append(subgraph)
    
# # Save extracted subgraphs to file
# if not os.path.isdir('./pickles/'):
#     os.makedirs('./pickles/')
# filename = 'extracted_subgraphs_' + str(len(subgraph_list)) + '.pkl'
# with open('pickles/' + filename, 'wb') as f:
#     pickle.dump(subgraph_list, f)
# print('Saved extracted subgraphs to file: ' + filename)

# # Load extracted subgraphs
# with open('pickles/' + filename, "rb") as f:
#     subgraph_list = pickle.load(f)

# Calculate subgraph density
density_list = []
for m in range(len(subgraph_list)):
    density = nx.density(subgraph_list[m])
    density_list.append(density)
    print('Processing subgraph %i out of %i' %(m+1, len(subgraph_list)))

# Find frequently occuring subgraphs
density_series = pd.Series(density_list)
frequency = density_series.value_counts(normalize=False)

for den in frequency.index:
    subgraph_ind = [i for i, x in enumerate(str(density_list==den)) if x]
    nx.draw(subgraph_list[subgraph_ind[0]])
    import matplotlib.pyplot as plt
    plt.show()
