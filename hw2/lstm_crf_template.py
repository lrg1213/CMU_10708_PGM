"""Template code for HW2 LSTM-CRF."""

import torch
import torch.nn as nn
from torch.autograd import Function

import time

from torch.utils.data import Dataset
from collections import defaultdict
from nltk.tree import Tree
import numpy as np
import copy

####################################################################
## Loading the data
####################################################################

tag_dict = defaultdict(lambda:len(tag_dict))

class MyNode():
    def __init__(self, treeNode, parent, idx, idx_set):
        if isinstance(treeNode,tuple):
            self.true_label = np.array([tag_dict[treeNode[1]]])
            self.word = treeNode[0]
        else:
            self.true_label = np.array([tag_dict[treeNode._label]])

        self.true_label = torch.tensor(self.true_label).long()

        self.children = []
        self.unary_potential = self.belief = 1
        self.parent = parent
        self.idx = idx  # give each node in the tree a unique index
        idx_set.add(idx)

        for child in treeNode:
            if not isinstance(child, str):
                self.children.append(MyNode(child, self, max(idx_set) + 1, idx_set))


def build_tree(tree):
    idx_set = set([])
    my_tree = MyNode(tree, None, 0, idx_set)
    return my_tree, len(idx_set)


class MyDataset():
    def __init__(self, path_to_file, w2i=None):
        self.trees = []
        with open(path_to_file) as f:
            lines = [l.strip() for l in f.readlines()]

        for line in lines:
            try:
                self.trees.append(Tree.fromstring(line))
            except:
                continue

        self.sentences = list([tree.leaves() for tree in self.trees])
        self.tree_size = list([len(tree.treepositions()) for tree in self.trees])
        self.len = len(self.trees)

        self.tree_tups = list([build_tree(tree) for tree in self.trees])
        self.my_trees = list([t[0] for t in self.tree_tups])
        self.tree_lens = list([t[1] for t in self.tree_tups])

        self.w2i = w2i
        self.sentences = self.build_vocab()
        if w2i is None:  # initialized vocab for the first time
            self.w2i['<UNK>'] = 0
            self.w2i.default_factory = lambda: 0  # all future unknown tokens will be mapped to this index

        self.vocab_size = max(list(self.w2i.values())) + 1
        self.tag2idx = tag_dict

        self.tag_size = len(self.tag2idx)
        self.batch_size = 1

        self.ptr = 0
        self.reverse_dict = {v: k for k, v in self.tag2idx.items()}

    def reset(self):
        self.ptr = 0

    def build_vocab(self):
        if self.w2i is None:
            self.w2i = defaultdict(lambda: len(self.w2i) + 1)
        sentence_idxs = [[self.w2i[x.lower()] for x in sent] for sent in self.sentences]
        return sentence_idxs

    def get_next_batch(self):
        current_batch = (torch.LongTensor(np.array(self.sentences[self.ptr])),
                         self.tree_size[self.ptr],
                         self.my_trees[self.ptr],
                         self.trees[self.ptr],
                         self.tree_lens[self.ptr])
        self.ptr += 1
        return self.ptr == self.len, current_batch


####################################################################
## Helper functions
####################################################################

def get_leaves(node, leaves):
    if len(node.children) == 0:
        leaves += [node.idx]
    for child in node.children:
        leaves = get_leaves(child, leaves)
    return leaves


def count_leaves(node, ct=0):
    if len(node.children) == 0:
        ct += 1
    for child in node.children:
        ct = count_leaves(child, ct)
    return ct


def count_nodes(node, ct=0):
    ct += 1
    for child in node.children:
        ct = count_nodes(child, ct)
    return ct

####################################################################
## POSTagger
####################################################################

class POSTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(POSTagger,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.build_intermediate_hidden = nn.Linear(2 * hidden_dim, hidden_dim)

        self.build_unary_potentials = nn.Linear(hidden_dim, num_tags)
        self.build_unary_potentials_intermediate_node = nn.Linear(2 * hidden_dim, num_tags)
        self.build_edge_potentials = nn.Linear(2 * hidden_dim, num_tags * num_tags)

        self.criterion = TreeNLLLoss()
        self.num_tags = num_tags

    def forward(self):
        # ...implement the forward function...
        return

####################################################################
## Belief propagation
####################################################################

def belief_propagation():
    # ...implement belief propagation and return the beliefs for every node...
    return

####################################################################
## Loss Function
####################################################################

class TreeNLLLoss(nn.Module):
    def __init__(self):
        super(TreeNLLLoss,self).__init__()

    def forward(self):
        # ...implement the forward function for negative log likelihood loss...
        return


def train(train_dataset, validation_dataset, embedding_dim, hidden_dim):
    myTagger = POSTagger(train_dataset.vocab_size, embedding_dim, hidden_dim, train_dataset.tag_size)
    
    # ...define training loop...
    done, train_example = train_dataset.get_next_batch()
    sentence, _, my_tree, _, tree_len = train_example

    # ...every 1,000 batches, evaluate on validation set...

if __name__ == "__main__":
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    
    # tip for debugging: if loading the data takes too long, 
    # you can move your code to a python notebook 
    # or Google Colab so you only have to run the 
    # data-loading cell once. 
    # Colab isn't meant for long-running jobs, however, 
    # so it may be better to run your final version elsewhere.
    # If you have to run on colab, then be sure to mount your 
    # google drive (https://colab.research.google.com/notebooks/io.ipynb)
    # so that you can pick up where you left off.
    train_dataset = MyDataset('ptb-train-10000.txt')
    validation_dataset = MyDataset('ptb-dev-1000.txt', w2i=train_dataset.w2i)
    train(train_dataset, validation_dataset, EMBEDDING_DIM, HIDDEN_DIM)
    
    # ...add additional code...
    
    # evaluate final model on test dataset
    # test_dataset = MyDataset('ptb-test-2000.txt', w2i=train_dataset.w2i)


