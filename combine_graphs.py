#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""combine_graphs - Combine two graphs and save the resulting graph"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx 
import matplotlib.pyplot as plt 
from check_patterns import check_patterns

def combine_graphs(G1, G2, plot=False):
    # Use the union function
    # G3 = nx.disjoint_union(G1, G2)

    # Use the composition function
    G3 = nx.compose(G1, G2)

    # Display all graphs information
    print()
    print('Architecture graph information:')
    print(nx.info(G1))
    if nx.is_directed(G1) == False:
        print("radius: %d" % nx.radius(G1))
        print("diameter: %d" % nx.diameter(G1))
        print("local efficiency: %s" % nx.algorithms.local_efficiency(G1))
        print("global efficiency: %s" % nx.algorithms.global_efficiency(G1))
    print("density: %s" % nx.density(G1))
    print("average clustering: %s" % nx.average_clustering(G1))
    
    print()

    print('Pattern graph information:')
    print(nx.info(G2))
    print()

    print('Combined graph information:')
    print(nx.info(G3))
    if nx.is_directed(G3) == False:
        if nx.is_connected(G3):
            print("radius: %d" % nx.radius(G3))
            print("diameter: %d" % nx.diameter(G3))
        print("local efficiency: %s" % nx.algorithms.local_efficiency(G3))
        print("global efficiency: %s" % nx.algorithms.global_efficiency(G3))
    print("density: %s" % nx.density(G3))
    print("average clustering: %s" % nx.average_clustering(G3))
    print()

    # Check isomorphism
    isomorph = check_patterns(G3, G2)
    print("Is the generated and pattern graphs isomorphic: %s" % isomorph)
    print()

    # Check connectivity
    connected = nx.is_connected(G3.to_undirected())
    print("Is the generated graph connected: %s" % connected)
    print()

    if plot:
        # Plot the new old and new graphs
        options = {'line_color': 'grey', 'font_size': 10, 'node_size': 10, 'with_labels': False}
        plt.figure(1)
        nx.draw(G1, **options)
        plt.figure(2)
        nx.draw(G2, **options)
        plt.figure(3)
        nx.draw(G3, **options)
        plt.show()
    
    return G3

if __name__ == '__main__':
    # Input graph 1
    archname = 'arch_1'
    G1 = nx.read_graphml('dataset/' + archname + '.graphml')
    # G1 = G1.to_undirected()
    # G1 = G1.subgraph(max(nx.connected_component_subgraphs(G1), key=len))

    # Input graph 2
    pattname = 'patt_8'
    G2 = nx.read_graphml('dataset/' + pattname + '.graphml')
    # G2 = G2.to_undirected()

    # Combine the two graphs
    G3 = combine_graphs(G1, G2, plot=True)
    # # Save generated graph
    # path = 'dataset/' + archname + '_' + pattname + '.graphml'
    # if isomorph and connected:
    #     nx.write_graphml(G3, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False)