#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
## Exercise: CIFAR-10 Classification
# Access and unpack the data
CIFAR_DIR = 'data/CIFAR_data/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

all_data = [0,1,2,3,4,5,6]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)
    
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

#===================================================================================================
# # inspect an image
# X = data_batch1[b'data']
# X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
# plt.imshow(X[0])
#===================================================================================================

# Helper functions
def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

        
    def next_batch(self, batch_size):
        for idx in range(0, len(self.training_images), batch_size):
            yield (self.training_images[idx:idx+batch_size],
                   self.training_labels[idx:idx+batch_size])
            
    
ch = CifarHelper()
ch.set_up_images()

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

# INIT WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return init_bias_vals

# CONV2D
def conv2D(x, W):
    # x = [batch, H, W, channels]
    # W = [filter H, filter W, ch_in, ch_out]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
 
# POOLING
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# CONVOLUTION LAYER
def convolution_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2D(input_x, W) + b)

# DENSE LAYER
def dense_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# # LAYERS
convo_1 = convolution_layer(x, shape=[3,3,3,32])
convo_1_pool = max_pool_2x2(convo_1)
convo_2 = convolution_layer(convo_1_pool, shape=[3,3,32,64])
convo_2_pool = max_pool_2x2(convo_2)
convo_2_flat = tf.reshape(convo_2_pool, [-1, 8*8*64])
dense_1 = tf.nn.relu(dense_layer(convo_2_flat, 1024))

# DROPOUT
dense_1_dropout = tf.nn.dropout(dense_1, keep_prob=hold_prob)
y_pred = dense_layer(dense_1_dropout, 10)

# LOSS
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=y_pred))
# OPTIMIZER
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train = opt.minimize(loss)

init = tf.global_variables_initializer()

steps = 5000
batch_size=64

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
#         batch = ch.next_batch(batch_size)
        for batch in ch.next_batch(batch_size):
            sess.run(train, feed_dict={x:batch[0], y_true:batch[1], hold_prob:0.5})
        
        if i%100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            print('On Step: {}'.format(i))
            print("Accuracy: {}".format(sess.run(acc, feed_dict={x:ch.test_images,
                                                                  y_true:ch.test_labels,
                                                                  hold_prob:1.0})))
# 