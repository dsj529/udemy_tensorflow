#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:43:39 2020

@author: dsj529
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
## MNIST Classification -- DNN approach

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

# VARIABLES
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# GRAPH OPS
y = tf.matmul(x, W) + b

# LOSS
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=y))

# OPTIMIZER
opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = opt.minimize(loss)

# SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(5000):
        if step%100 == 0:
            print('Current epoch: {}'.format(step))
        batch_x, batch_y = mnist.train.next_batch(batch_size=128)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})
         
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
    print(sess.run(acc, feed_dict={x: mnist.test.images,
                                   y_true: mnist.test.labels}))
# 92% accuracy

#%%
## MNIST Classification -- CNN approach
# HELPER
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
hold_prob = tf.placeholder(tf.float32)

# LAYERS
x_image = tf.reshape(x, [-1, 28, 28, 1])
convo_1 = convolution_layer(x_image, shape=[5,5,1,32])
convo_1_pool = max_pool_2x2(convo_1)
convo_2 = convolution_layer(convo_1_pool, shape=[5,5,32,64])
convo_2_pool = max_pool_2x2(convo_2)
convo_2_flat = tf.reshape(convo_2_pool, [-1, 7*7*64])
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

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(64)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y, hold_prob:0.5})
        
        if i%100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            
            print('On Step: {}'.format(i))
            print("Accuracy: {}".format(sess.run(acc, feed_dict={x:mnist.test.images,
                                                                 y_true:mnist.test.labels,
                                                                 hold_prob:1.0})))
# 98.9% accuracy