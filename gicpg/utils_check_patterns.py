#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""utils_check_patterns - Check whether a given graph contains a given pattern"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt 
import pickle
import os

def check_patterns(G1, G2, saveisolist=False, readisolist=False, plot=False):
    # Check isomorphism
    GM = GraphMatcher(G1=G1, G2=G2, node_match=None, edge_match=None)
    isomorph = GM.subgraph_is_isomorphic()

    if saveisolist:
        # Check if the pickles folder exists
        if not os.path.isdir("./pickles/"):
            os.makedirs("./pickles/")

        # List all isomorphisms between the two graphs
        isomorph_list = list(GM.subgraph_isomorphisms_iter())

        # Save isomorphism list
        pickling_on = open('pickles/arch2_patt8.pickle',"wb")
        pickle.dump(isomorph_list, pickling_on)
        pickling_on.close()

    if readisolist:
        # Read pickle file
        pickle_off = open('pickles/arch2_patt8.pickle',"rb")
        isomorph_list = pickle.load(pickle_off)
        pickle_off.close()

    if plot:
        # Plot a sample isomorph
        options = {'line_color': 'grey', 'font_size': 10, 'node_size': 10, 'with_labels': True}
        G3 = G1.subgraph(isomorph_list[0])
        plt.figure(1)
        nx.draw(G3, **options)
        plt.figure(2)
        nx.draw(G2, **options)
        plt.show()

    return isomorph

if __name__ == '__main__':
    # Load graphs
    G1 = nx.read_graphml('dataset/arch_1.graphml')
    G2 = nx.read_graphml('dataset/patt_8.graphml')

    # # Transform to undirected
    # G1 = G1.to_undirected()
    # G2 = G2.to_undirected()

    # Check whether the graphs contain the pattern
    isomorph = check_patterns(G1, G2)
    print("Is the generated and pattern graphs isomorphic: %s" % isomorph)