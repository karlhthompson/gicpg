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
from os import listdir
from os.path import isfile, join
from gicpg.attribute_ml import infer_attributes_ml

if __name__ == '__main__':
    
    # Create a list of generated graph files
    graphfiles = [f for f in listdir("./graphs") if isfile(join("./graphs",f))]

    # Input desired epoch and sampling time of the generated graphs
    epoch = 3000
    sampling = 3

    # Load the generated graphs file  with the desired attributes
    att = str(epoch) + '_' + str(sampling)
    for graphfile in graphfiles:
        if att in graphfile:
            fname = graphfile
    with open('./graphs/' + fname, "rb") as f:
        graphlist = pickle.load(f)

    # infer attributes of all graphs in graphlist
    archname = 'arch_1'
    newgraphlist = []
    count = 1
    for graph in graphlist:
        graph = infer_attributes_ml(graph, archname, savepred=False)
        newgraphlist.append(graph)
        print('Progress: graph', count)
        count += 1

    # Save attributed graphs to file
    if not os.path.isdir('./attributed_graphs/'):
        os.makedirs('./attributed_graphs/')
    with open('attributed_graphs/att_graphs_'+str(epoch)+'_'+str(sampling)+
            '.pkl', 'wb') as f:
        pickle.dump(newgraphlist, f)

    print()