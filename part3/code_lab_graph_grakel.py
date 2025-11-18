import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = 'datasets/train_5500_coarse.label'
path_to_test_set = 'datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()
        
        for word in doc:
            if word in vocab:
                G.add_node(word)
        
        for i, word in enumerate(doc):
            if word in vocab:
                for j in range(i + 1, min(i + window_size + 1, len(doc))):
                    if doc[j] in vocab:
                        G.add_edge(word, doc[j])
        
        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)


for G in G_train_nx:
    nx.set_node_attributes(G, {n: n for n in G.nodes()}, 'label')

for G in G_test_nx:
    nx.set_node_attributes(G, {n: n for n in G.nodes()}, 'label')


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score




# Transform networkx graphs to grakel representations
G_train = list(graph_from_networkx(G_train_nx, node_labels_tag='label'))
G_test  = list(graph_from_networkx(G_test_nx, node_labels_tag='label'))

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=1, normalize=False, base_graph_kernel=VertexHistogram)# your code here #

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)




# Train an SVM classifier and make predictions
print("Task 13 - WL Kernel with SVM")

clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)

accuracy_wl = accuracy_score(y_test, y_pred)
print(f"Accuracy with WL kernel: {accuracy_wl}")

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


print("Task 14 - Experimenting with Different Kernels")

from grakel.kernels import ShortestPath, RandomWalk, GraphletSampling

print("\n1. Shortest Path Kernel")
gk_sp = ShortestPath(normalize=True)
K_train_sp = gk_sp.fit_transform(G_train)
K_test_sp = gk_sp.transform(G_test)

clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)
y_pred_sp = clf_sp.predict(K_test_sp)
accuracy_sp = accuracy_score(y_test, y_pred_sp)
print(f"Accuracy: {accuracy_sp}")

print("\n2. Random Walk Kernel")
gk_rw = RandomWalk(normalize=True)
K_train_rw = gk_rw.fit_transform(G_train)
K_test_rw = gk_rw.transform(G_test)

clf_rw = SVC(kernel='precomputed')
clf_rw.fit(K_train_rw, y_train)
y_pred_rw = clf_rw.predict(K_test_rw)
accuracy_rw = accuracy_score(y_test, y_pred_rw)
print(f"Accuracy: {accuracy_rw}")

print("\n3. Graphlet Sampling Kernel")
gk_gs = GraphletSampling(normalize=True, sampling={'n_samples': 200})
K_train_gs = gk_gs.fit_transform(G_train)
K_test_gs = gk_gs.transform(G_test)

clf_gs = SVC(kernel='precomputed')
clf_gs.fit(K_train_gs, y_train)
y_pred_gs = clf_gs.predict(K_test_gs)
accuracy_gs = accuracy_score(y_test, y_pred_gs)
print(f"Accuracy: {accuracy_gs}")