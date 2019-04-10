#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Karl Thompson
# Created Date: Mon March 25 17:34:00 CDT 2019
# =============================================================================
"""infer_attributes_rule - Infer attributes of an architecture graph using
a rule-based probabilistic clasifier"""
# =============================================================================
# Imports
# =============================================================================
import networkx as nx 
import random
import pickle
import pandas as pd

def node_classifier(G):
    # A function to infer node type based on its degree
    node_ids = [k for k, v in G.degree()]
    in_deg = [v for k, v in G.in_degree()]
    out_deg = [v for k, v in G.out_degree()]

    for node in range(len(node_ids)):
        G.node[node_ids[node]]['Type'] = {}
        if out_deg[node] > 1:
            if in_deg[node] > 1:
                G.node[node_ids[node]]['Type'] = 'Class'
            if in_deg[node] < 1:
                G.node[node_ids[node]]['Type'] = 'Package'
            if in_deg[node] == 1 and out_deg[node] > 10:
                num = random.uniform(0, 1)
                if num <= 0.33:
                    G.node[node_ids[node]]['Type'] = 'Class'
                else:
                    G.node[node_ids[node]]['Type'] = 'Package'
            if in_deg[node] == 1 and out_deg[node] < 4:
                num = random.uniform(0, 1)
                if num <= 0.57:
                    G.node[node_ids[node]]['Type'] = 'Class'
                else:
                    G.node[node_ids[node]]['Type'] = 'Package'
            if in_deg[node] == 1 and  G.node[node_ids[node]]['Type'] == {}:
                G.node[node_ids[node]]['Type'] = 'Package'
        if out_deg[node] == 0:
            if in_deg[node] == 1:
                num = random.uniform(0, 1)
                if num <= 0.8:
                    G.node[node_ids[node]]['Type'] = 'Dependency'
                elif num <= 0.9:
                    G.node[node_ids[node]]['Type'] = 'Class'
                else:
                    G.node[node_ids[node]]['Type'] = 'Package'
            if in_deg[node] > 1:
                G.node[node_ids[node]]['Type'] = 'Class'
        if out_deg[node] == 1:
            if in_deg[node] == 0:
                G.node[node_ids[node]]['Type'] = 'Package'
            if in_deg[node] == 1:
                num = random.uniform(0, 1)
                if num <= 0.92:
                    G.node[node_ids[node]]['Type'] = 'Property'
                elif num <= 0.96:
                    G.node[node_ids[node]]['Type'] = 'Parameter'
                else:
                    G.node[node_ids[node]]['Type'] = 'Operation'
            if in_deg[node] > 1:
                num = random.uniform(0, 1)
                if num <= 0.67:
                    G.node[node_ids[node]]['Type'] = 'Class'
                else:
                    G.node[node_ids[node]]['Type'] = 'Property'
    return G

def edge_classifier(G):
    # A function to infer edge type based on its node types
    node_ids = [k for k, v in G.in_degree()]
    for node1 in range(len(node_ids)):
        for node2 in range(len(node_ids)):          
            if G.has_edge(node_ids[node1], node_ids[node2]):             
                node1type = G.node[node_ids[node1]]['Type']
                node2type = G.node[node_ids[node2]]['Type']
                G[node_ids[node1]][node_ids[node2]]['EType'] = {}
                if node1type == 'Class' and node2type == 'Property':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'prop'
                if node1type == 'Class' and node2type == 'Operation':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'prop'
                if node1type == 'Class' and node2type == 'Parameter':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'prop'
                if node1type == 'Class' and node2type == 'Class':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'generalization'
                if node1type == 'Operation' and node2type == 'Parameter':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'prop'
                if node1type == 'Package':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'part-of'
                if node1type == 'Parameter' and node2type == 'Class':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'is-type'
                if node1type == 'Class' and node2type == 'Dependency':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'is-type'
                if node1type == 'Property' and node2type == 'Class':
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'is-type'
                if G[node_ids[node1]][node_ids[node2]]['EType'] == {}:
                    G[node_ids[node1]][node_ids[node2]]['EType'] = 'prop'
    return G

def graph_classifier(G):
    # A function calling the node and edge classifier functions successively
    G1 = node_classifier(G)
    G2 = edge_classifier(G1)

    # Save predicted node labels to an excel file
    node_att = nx.get_node_attributes(G1, 'Type')
    output = pd.DataFrame({'Predicted': node_att})
    output.to_excel(".temp/output/rule_classification_output.xlsx")
    return G2

if __name__ == '__main__':
    # Load the generated, unlabeled graph
    fname = 'graphs/GraphRNN_RNN_arch_4_128_pred_6000_1.dat'
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    G = graph_classifier(graph_list[0])
    # G = graph_classifier(nx.read_graphml('dataset/arch_1.graphml'))