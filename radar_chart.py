import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
import networkx as nx

def radarplot(G1,G2):

    # Extract G1 data
    n1 = len(G1.nodes)
    e1 = len(G1.edges)
    dg1 = np.mean([v for k, v in G1.degree()])
    dn1 = nx.density(G1)
    ac1 = nx.average_clustering(G1)
    le1 = nx.algorithms.local_efficiency(G1)
    ge1 = nx.algorithms.global_efficiency(G1)

    # Extract G2 data
    n2 = len(G2.nodes)
    e2 = len(G2.edges)
    dg2 = np.mean([v for k, v in G2.degree()])
    dn2 = nx.density(G2)
    ac2 = nx.average_clustering(G2)
    le2 = nx.algorithms.local_efficiency(G2)
    ge2 = nx.algorithms.global_efficiency(G2)

    # Set data
    df = pd.DataFrame({
    'group': ['Original','Generated'],
    'Nodes': [n1/max(n1,n2), n2/max(n1,n2)],
    'Edges': [e1/max(e1,e2), e2/max(e1,e2)],
    'Degree': [dg1/max(dg1,dg2), dg2/max(dg1,dg2)],
    'Density': [dn1/max(dn1,dn2), dn2/max(dn1,dn2)],
    'Clustering': [ac1/max(ac1,ac2), ac2/max(ac1,ac2)],
    'Local Efficiency': [le1/max(le1,le2), le2/max(le1,le2)],
    'Global Efficiency': [ge1/max(ge1,ge2), ge2/max(ge1,ge2)]
    })

    # Number of variable
    categories=list(df)[1:]
    N = len(categories)

    # Set angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Set the first axis to be on top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Plot first graph
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Original")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Plot second graph
    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Generated")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

# Test the function
if __name__ == '__main__':

    G1 = nx.read_graphml('dataset/arch_1.graphml')
    G1 = G1.to_undirected()
    G1 = G1.subgraph(max(nx.connected_component_subgraphs(G1), key=len))

    G2 = nx.read_graphml('dataset/arch_2.graphml')
    G2 = G2.to_undirected()
    G2 = G2.subgraph(max(nx.connected_component_subgraphs(G2), key=len))

    radarplot(G1,G2)