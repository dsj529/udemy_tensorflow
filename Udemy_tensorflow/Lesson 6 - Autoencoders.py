# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

data = make_blobs(n_samples=500, n_features=3, centers=2)

sc = MinMaxScaler()
scaled_data = sc.fit_transform(data[0])
scaled_x = scaled_data[:,0]
scaled_y = scaled_data[:,1]
scaled_z = scaled_data[:,2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(scaled_x, scaled_y, scaled_z, c=data[1])

from tensorflow.contrib.layers import fully_connected

num_inputs = 3
num_hidden = 2
num_outputs = num_inputs
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
    
plt.scatter(out_2d[:,0], out_2d[:,1], c=data[1])
