"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


G = nx.read_edgelist(
    "datasets/CA-HepTh.txt",
    delimiter="\t",    
    comments="#",      
    nodetype=int       
)
print("task 1")
print(f"number of nodes: {G.number_of_nodes()}")
print(f"number of edges: {G.number_of_edges()}")


print("task 2")
num_components = nx.number_connected_components(G)
print(f"\nNumber of connected components: {num_components}")
    
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
    
num_nodes_lcc = G_lcc.number_of_nodes()
num_edges_lcc = G_lcc.number_of_edges()
    
num_nodes_total = G.number_of_nodes()
num_edges_total = G.number_of_edges()
    
fraction_nodes = num_nodes_lcc / num_nodes_total
fraction_edges = num_edges_lcc / num_edges_total
    
print(f"Number of nodes: {num_nodes_lcc} ({fraction_nodes} of total)")
print(f"Number of edges: {num_edges_lcc} ({fraction_edges} of total)")




