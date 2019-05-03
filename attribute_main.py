#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
"""attribute_main - Infer attributes of the generated architecture graphs using
a machine learning classification ensemble, defined in gicpg/attribute_ml.py"""
# =============================================================================
# Imports
# =============================================================================
import os
import pickle
import numpy as np
import networkx as nx
from os import listdir
from os.path import isfile, join
from gicpg.attribute_ml import infer_attributes_ml

if __name__ == '__main__':
    
    # Create a list of generated graph files
    graphfiles = [f for f in listdir("./graphs") if isfile(join("./graphs",f))]
    graphfiles.reverse()
        
    # Load the original architecture graph
    archname = 'arch_1'
    ArchG = nx.read_graphml('dataset/'+archname+'.graphml')
    ArchGund = max((ArchG.to_undirected().subgraph(c) for c in 
            nx.connected_components(ArchG.to_undirected())), key=len)
    ArchG = ArchG.subgraph(ArchGund.nodes).copy()

    # Load the generated graph files one by one
    evalgraphlist = []
    fcount = 1
    for graphfile in graphfiles:
        with open('./graphs/' + graphfile, "rb") as f:
            graphlist = pickle.load(f)

        # Evaluate generated graphs to filter out outliers and identicals
        gcount = 1
        for graph in graphlist:
            graph = graph.reverse(copy=True)
            graphund = max((graph.to_undirected().subgraph(c) for c in 
                    nx.connected_components(graph.to_undirected())), key=len)
            graph = graph.subgraph(graphund.nodes).copy()
            ll = 0.01     # lower limit
            ul = 0.25     # upper limit
            el = 4        # eccentricity limit 
            sl = 1000     # sample limit
            if (
                abs((nx.density(graph)/nx.density(ArchG))-1) >= ll and
                abs((nx.density(graph)/nx.density(ArchG))-1) <= ul and
                abs((nx.average_clustering(graph)/
                    nx.average_clustering(ArchG))-1) >= ll and
                abs((nx.average_clustering(graph)/
                    nx.average_clustering(ArchG))-1) <= ul and 
                abs((nx.algorithms.local_efficiency(graphund)/
                    nx.algorithms.local_efficiency(ArchGund))-1) >= ll and 
                abs((nx.algorithms.local_efficiency(graphund)/
                    nx.algorithms.local_efficiency(ArchGund))-1) <= ul and 
                abs((nx.radius(graphund)-nx.radius(ArchGund))) <= el and 
                abs((nx.diameter(graphund)-nx.diameter(ArchGund))) <= el
                ): 
                    evalgraphlist.append(graph)
                    print('Graph added. Current length of evaluated graphs list is: %i' 
                        %(len(evalgraphlist)))
            if len(evalgraphlist) == sl: break
            print('Evaluation Progress: file %i out of %i: graph %i out of %i' 
                %(fcount, len(graphfiles), gcount, len(graphlist)))
            gcount += 1
        if len(evalgraphlist) == sl: break
        fcount += 1
    print('Evaluation complete. Final length of evaluated graphs list is: %i' 
        %(len(evalgraphlist)))

    # Infer attributes of all evaluated graphs
    archname = 'arch_1'
    attrgraphlist = []
    count = 1
    for graph in evalgraphlist:
        graph = infer_attributes_ml(graph, archname, savepred=False)
        attrgraphlist.append(graph)
        print('Attribution Progress: graph %i out of %i' 
            %(count, len(evalgraphlist)))
        count += 1

    # Save attributed graphs to file
    if not os.path.isdir('./attributed_graphs/'):
        os.makedirs('./attributed_graphs/')
    filename = 'att_graphs_' + str(len(graphfiles)) + '_' + str(len(evalgraphlist)) + '.pkl'
    with open('attributed_graphs/' + filename, 'wb') as f:
        pickle.dump(attrgraphlist, f)
    print('Attribution complete. Saved attributed graphs to file: ' + filename)