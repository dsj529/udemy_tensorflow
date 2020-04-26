#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:38:58 2020

@author: dsj529
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

#%%
# Example 1: simple linear autoencoder imitating PCA to reduce 30 dimensions to 2

data = pd.read_csv('./data/anonymized_data.csv')

sc = MinMaxScaler()
scaled_data = sc.fit_transform(data.drop('Label', axis=1))
label = data['Label'].values

from tensorflow.contrib.layers import fully_connected

num_inputs = 30
num_outputs = num_inputs
num_hidden = 2
learning_rate = 0.003

X = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

loss = tf.reduce_mean(tf.square(outputs-X))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_steps = 1500

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train, feed_dict={X:scaled_data})
    
    out_2d = hidden.eval(feed_dict={X:scaled_data})
    
plt.scatter(out_2d[:,0], out_2d[:,1], c=label)

#%%
# Example 2: Multiple layers of autoencoding with MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
tf.reset_default_graph()

# basic model values
num_inputs = 784 # 28 x 28
neurons_hid1 = 441
neurons_hid2 = 196 # final layer of encoding
neurons_hid3 = neurons_hid1
num_outputs = num_inputs

learning_rate = 0.003

act_f = tf.nn.relu
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# weights and biases
initializer = tf.variance_scaling_initializer() # adapts the weights based on the sizes of the layers

w1 = tf.Variable(initializer([num_inputs, neurons_hid1]), dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hid1, neurons_hid2]), dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hid2, neurons_hid3]), dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hid3, num_outputs]), dtype=tf.float32)

b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))            
b4 = tf.Variable(tf.zeros(num_outputs)) 

# define the layers
hid_lay1 = act_f(tf.matmul(X, w1) + b1)
hid_lay2 = act_f(tf.matmul(hid_lay1, w2) + b2)
hid_lay3 = act_f(tf.matmul(hid_lay2, w3) + b3)
output_layer = tf.matmul(hid_lay3, w4) + b4

## loss and optimizer
loss = tf.reduce_mean(tf.square(output_layer-X))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train = optimizer.minimize(loss)  

init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_epochs = 10
batch_size = 64

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
        num_batches = mnist.train.num_examples // batch_size
        
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X:X_batch})
            
        train_loss = loss.eval(feed_dict={X:X_batch})
        print('Epoch: {} || Loss: {}'.format(epoch, train_loss))
        
    saver.save(sess, './Models/Autoencoder_example.ckpt')
    
num_test_imgs = 10

with tf.Session() as sess:
    saver.restore(sess, './Models/Autoencoder_example.ckpt')
    
    results = []
    for lyr in [hid_lay1, hid_lay2, hid_lay3, output_layer]:
        results.append(lyr.eval(feed_dict={X:mnist.test.images[:num_test_imgs]}))
    
f, a = plt.subplots(5, 10, figsize=(20, 12))
for i in range(num_test_imgs):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(results[0][i], (21,21)))
    a[2][i].imshow(np.reshape(results[1][i], (14,14)))
    a[3][i].imshow(np.reshape(results[2][i], (21,21)))
    a[4][i].imshow(np.reshape(results[3][i], (28,28)))
    
#%%
## personal experiment: convolutional autoencoder
from tensorflow.examples.tutorials.mnist import input_data

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                    minval=0,
                                                    maxval=2,
                                                    dtype=tf.int32),
                                  tf.float32))

def autoencoder(input_shape=[None, 784],
                n_filters=[1,128,128,128],
                filter_size=[3,3,3,3],
                corrupted=False):

    initializer = tf.variance_scaling_initializer()

    
    X = tf.placeholder(tf.float32, shape=input_shape, name='X')
    if len(X.get_shape()) == 2:
        X_dim = np.sqrt(X.get_shape().as_list()[1])
        if X_dim != int(X_dim):
            raise ValueError('Unsupported tensor dimensions: {}'.format(X.shape))
        X_dim = int(X_dim)
        X_tensor = tf.reshape(X, [-1, X_dim, X_dim, n_filters[0]])
    elif len(X.get_shape()) == 4:
        X_tensor = X
    else:
        raise ValueError('Unsupported tensor dimensions: {}'.format(X.shape))
    current_input = X_tensor
    
    if corrupted:
        current_input = corrupt(current_input)
        

    # start the encoder half
    encoder = []
    shapes = []
    for layer_n, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        
        W = tf.Variable(initializer([filter_size[layer_n],
                                     filter_size[layer_n],
                                     n_input,
                                     n_output]))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(tf.add(tf.nn.conv2d(current_input, W, 
                                           strides=[1,2,2,1], 
                                           padding='SAME'),
                              b))
        current_input = output
        
    # capture the latent representation
    Z = current_input
    encoder.reverse()
    shapes.reverse()

    # start the decoder
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W,
                                                     tf.stack([tf.shape(X)[0], 
                                                               shape[1], 
                                                               shape[2], 
                                                               shape[3]]),
                                                     strides=[1, 2, 2, 1],
                                                     padding='SAME'),
                              b))
        current_input = output

    y =  current_input
    
    cost = tf.reduce_sum(tf.square(y - X_tensor))
    
    return {'x': X, 'z': Z, 'y': y, 'cost': cost}
##

mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
mean_img =np.mean(mnist.train.images, axis=0)
tf.reset_default_graph()

cae = autoencoder()

learning_rate = 0.003
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize((cae['cost']))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    batch_size = 128
    epochs = 15
    
    for ep_i in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={cae['x']: train})
            
            print(ep_i, sess.run(cae['cost'], feed_dict={cae['x']: train}))
            
            
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(cae['y'], feed_dict={cae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()