"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    A = nx.adjacency_matrix(G)
    
    degrees = np.array([G.degree(node) for node in G.nodes()]).flatten()
    D_inv = diags(1.0 / degrees)
    
    n = A.shape[0]
    I = eye(n)
    L_rw = I - D_inv @ A
    
    eigenvalues, eigenvectors = eigs(L_rw.astype(float), k=k, which='SM')
    U = eigenvectors.real
    
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(U)
    
    nodes = list(G.nodes())
    clustering = {nodes[i]: int(labels[i]) for i in range(len(nodes))}

    return clustering





G = nx.read_edgelist('datasets/CA-HepTh.txt', delimiter='\t', comments='#', create_using=nx.Graph())

largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print("k=50:")
k=50
clustering = spectral_clustering(G_lcc, k)
print(f"Number of clusters: {len(set(clustering.values()))}")


# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    

    
    m = G.number_of_edges()
    clusters = set(clustering.values())
    Q = 0.0
    
    for c in clusters:
        nodes_in_c = [node for node, cluster in clustering.items() if cluster == c]
        subgraph_c = G.subgraph(nodes_in_c)
        lc = subgraph_c.number_of_edges()
        dc = sum(G.degree(node) for node in nodes_in_c)
        Q += (lc / m) - (dc / (2 * m)) ** 2
    
    return Q

print("\nTask 6 - Computing Modularity")
Q_spectral = modularity(G_lcc, clustering)
print(f"Modularity Q = {Q_spectral}")

import random

random.seed(0) 

clusters_random = {
    n: random.randint(0, k-1) for n in G_lcc.nodes()
}

Q_random = modularity(G_lcc, clusters_random)
print("Modularity of a random clustering :", Q_random)