#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""pattern_main - Identify design patterns in architecture graphs"""
# =============================================================================
# Imports
# =============================================================================
import pickle
from os import listdir
import tensorflow as tf
from os.path import isfile, join
from gicpg.pattern_train import *

# Embedding settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')

# Load attributed graphs
graph_file = [f for f in listdir("./attributed_graphs") if 
              isfile(join("./attributed_graphs",f))][0]
with open('attributed_graphs/'+graph_file, "rb") as f:
    graph_list = pickle.load(f)

# Embed attributed graphs
for graph in graph_list:
    emb = embed_graphs(graph, FLAGS)

    print()