#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""utils_combine_graphs - Combine two graphs and save the resulting graph"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx 
from os import listdir
import matplotlib.pyplot as plt 
from os.path import isfile, join
from gicpg.utils_check_patterns import check_patterns

def combine_graphs(archgraph, pattgraphs, print_info=False, plot=False):
    # Use the networkx composition function
    allgraphs = [archgraph] + pattgraphs
    combgraph = nx.compose_all(allgraphs)

    if print_info:
        # Display all graphs information
        print()
        print('Architecture graph information:')
        print(nx.info(archgraph))
        if nx.is_directed(archgraph) == False:
            print("radius: %d" % nx.radius(archgraph))
            print("diameter: %d" % nx.diameter(archgraph))
            print("local efficiency: %s" % nx.algorithms.local_efficiency(archgraph))
            print("global efficiency: %s" % nx.algorithms.global_efficiency(archgraph))
        print("density: %s" % nx.density(archgraph))
        print("average clustering: %s" % nx.average_clustering(archgraph))
        
        print()

        print('Pattern graphs information:')
        for patt in pattgraphs:
            print(nx.info(patt))
            print()

        print('Combined graph information:')
        print(nx.info(combgraph))
        if nx.is_directed(combgraph) == False:
            if nx.is_connected(combgraph):
                print("radius: %d" % nx.radius(combgraph))
                print("diameter: %d" % nx.diameter(combgraph))
            print("local efficiency: %s" % nx.algorithms.local_efficiency(combgraph))
            print("global efficiency: %s" % nx.algorithms.global_efficiency(combgraph))
        print("density: %s" % nx.density(combgraph))
        print("average clustering: %s" % nx.average_clustering(combgraph))
        print()

        # Check isomorphism
        isolist = []
        for patt in pattgraphs:
            isomorph = check_patterns(combgraph.to_undirected(), patt.to_undirected())
            isolist.append(isomorph)
        print("Are the generated and pattern graphs isomorphic: %s" % all(isolist))
        print()

        # Check connectivity
        connected = nx.is_connected(combgraph.to_undirected())
        print("Is the generated graph connected: %s" % connected)
        print()

    if plot:
        # Plot the old and new arch graphs
        options = {'line_color': 'grey', 'font_size': 10, 'node_size': 10, 'with_labels': False}
        plt.figure(1)
        nx.draw(archgraph, **options)
        plt.figure(2)
        nx.draw(combgraph, **options)
        plt.show()
    
    return combgraph

if __name__ == '__main__':
    # Input the arch graph
    archname = 'arch_1'
    G1 = nx.read_graphml('dataset/' + archname + '.graphml')
    G1und = max((G1.to_undirected().subgraph(c) for c in nx.connected_components(G1.to_undirected())), key=len)
    archgraph = G1.subgraph(G1und.nodes).copy()
    # archgraph = archgraph.to_undirected()

    # Input all the pattern graphs
    pattfiles = [f for f in listdir("./dataset") if isfile(join("./dataset",f)) and 'patt' in f]
    pattgraphs = []
    for pattname in pattfiles:
        G2 = nx.read_graphml('dataset/' + pattname)
        # G2 = G2.to_undirected()
        pattgraphs.append(G2)

    # Combine all the graphs
    combgraph = combine_graphs(archgraph, pattgraphs, print_info=True, plot=True)

    # Save generated graph
    path = 'dataset/' + archname + '_' + 'plus' + '.graphml'
    nx.write_graphml(combgraph, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False)