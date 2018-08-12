#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 18:57:18 2018

@author: thiago
"""

# We gonna built a machine to predict if an user would like a movie or not
# Data --> https://grouplens.org/datasets/movielens/

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importing the dataset
# sep to separator of columns
# encoding for especial caracters
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# later, we should do cross validation in all .base.
# Which has different splits of the data. For now, just one
# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Why we need this numbers?
# Because we are gonna convert test and training to a matrix
# Where lines = user
# Columns = movies
# Cells = Ratings(no rating?, just put a 0)
# With this matrix, we can calculate dot products and others.

# Converting the data into an matrix with users in lines and movies in columns, and ratings in cells
# This has to be done, because RBM expects an matrix
# Where the columns represents the features to be analysed
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users] # to get all movies rated from this user
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies) # create a list of size nb_movies, with zeros
        ratings[id_movies - 1] = id_ratings # For all element in id_movies, assign the next in id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# tensors is a mult dimension array. We convert our list of list to torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
# Why do we have to?
# Because our output is just 0 or 1.
# Also,RBM needs consistency, output and input has to have the same format
training_set[training_set == 0] = -1 # not rated yet
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1 # good enough review
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the arquitecure of RBM
# Creating the architecture of the Neural Network
class RBM():
    # nv --> num of visiable nodes
    # nh --> numb hidden nodes
    def __init__(self, nv, nh): # initializer
        # randn initialize the matrix, with random numbers, according to a normal distribution
        self.W = torch.randn(nh, nv) # weights - Matrix of size nh x nv
        # Bias is a vector of 2 dimension --> first dimension for bach and second for bias
        self.a = torch.randn(1, nh) # Bias for the probability of the hidden nodes, given visiable nodes
        self.b = torch.randn(1, nv)  # Bias for the probability of the visiable nodes, given hidden nodes


    # Calculate the probabilites of hidden nodes, given visiable nodes
    # Function to sample the hiden nodes, giving the probability of
    # P(h)/v --> This is basically the sigmoid activation function
    # Basically, for each hidden nodes, it will activate them, base on the probability
    # P(h)/V that we will calculate it.
    # We need this function to calculates the probabilities
    # which is gonna be used on the Gibs sample, to approximate the likehood gradients
    def sample_h(self, x): # X --> visible nodes V in the probability P(h)/V
        # First, we have to compute P(h)/V --> Basically the sigmoid function
        # torch.mm --> Product of 2 tensors
        # self.W.t() --> get the weights, but we need the transpose. The formula requires
        # self.a.expand_as(wx) --> To make sure the bias is applied to all baches
        wx = torch.mm(x, self.W.t()) # Product of the weights X neuros(x)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation) # activation function for the neuron
        #torch.bernoulli(p_h_given_v)--> sample of all Hidden nodes, according to this probability p(h)/v
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # It's basically the same as sample_h. But in this case, for visible nodes
    # At the end, based on the probabilty of the hidden nodes, we calculate
    # the probability of the visible nodes
    def sample_v(self, y): # y --> hidden nodes
        wy = torch.mm(y, self.W) # no need for transpose
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # Approximating the RBM log-Likelihood Gradient
    # Contrastive Divergence --> Gibbs sampling. More details in the article
    # Arguments
        # v0 --> input vector, contaning the rating of all moviews, by one user
        # vk --> visible nodes obtained after K samplings(roudtrip from visiable to hidden nodes)
        # ph0 --> Vector of probabilities at the first iteration
        # phk --> Vector of probabilities of hidden node after k samplings, giving the Visiable nodes VK
    def train(self, v0, vk, ph0, phk):
        # based on the formulas and algorithm, from the articl
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0]) # This is the number of data to be trained/how many movies
nh = 100 # we can play with this numner. This is the features we RBM is gonna detect
batch_size = 100
rbm = RBM(nv, nh)


# Training the RBM
nb_epoch = 10
k_steps = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # to check our loss. We can use different ways of calculating the loss
    s = 0. # counter. Used to normalize our train_loss. incremented after each batch trained
    # to train for all use. but we change the weights only after a bach size
    for id_user in range(0, nb_users - batch_size, batch_size):
        # input. it goes to the Gibbs sampling, and it's update after each iteration
        vk = training_set[id_user:id_user+batch_size] # get all user from the range

        # target. We don't wanna change. Gonna use to compare at the end, to calculate the loss
        v0 = training_set[id_user:id_user+batch_size]

        # Initial probabilities
        # P(hn at start) = 1 / real ratings(already rated)
        # Variable,_ gets only the first element return by the function and assign to variable
        # _,Variable gets only the last element return by the function and assign to variable
        ph0,_ = rbm.sample_h(v0)

        # Execute K steps for Contrastive Divergence - random walk, but with some direction
        # Idea of Gibbs sampling
            # makes several round trips from the visible nodes to hidden nodes
            # & from the hidden nodes to visible nodes
            # At each trip:
                # visible nodes are updated
        for k in range(k_steps): # random walk
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk) # ratings are updated
            # don't wanna -1 for the training. Then, put -1 again for the original
            # because rating do not exist
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk) # after k iterations
        rbm.train(v0, vk, ph0, phk) # update the weights

        # to our loss, we are using the Mean function
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    # we use the training set to get the activation
    v = training_set[id_user:id_user+1] # test 1 by one user
    vt = test_set[id_user:id_user+1] # the original test set, to compare at tge end
    if len(vt[vt>=0]) > 0: # just one step now, for the test. And make sure is not a -1 rating
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))



