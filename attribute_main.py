#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Thu April 18 12:12:00 CDT 2019
# =============================================================================
"""infer_attributes_gat - Infer attributes of an architecture graph using
a graph attention network"""
# =============================================================================
# Imports
# =============================================================================
import pickle
from os import listdir
from os.path import isfile, join
from gicpg.attribute_ml import infer_attributes_ml

def main():
    # Create a list of generated graph files
    graphfiles = [f for f in listdir("./graphs") if isfile(join("./graphs", f))]

    # Input desired epoch and sampling time
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
    for graph in graphlist:
        graph = infer_attributes_ml(graph, archname, savepred=False)
        newgraphlist.append(graph)

    print()

main()