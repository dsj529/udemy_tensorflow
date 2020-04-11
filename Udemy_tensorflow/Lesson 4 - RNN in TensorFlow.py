#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


#%%
## RNN in TF
class TimeSeriesData():
    def __init__(self, num_pts, x_min, x_max):
        self.num_pts = num_pts
        self.x_min = x_min
        self.x_max = x_max
        self.resolution = (x_max - x_min)/num_pts
        self.x_data = np.linspace(x_min, x_max, num_pts)
        self.y_true = np.sin(self.x_data)
        
    def ret_true(self, x_series):
        return np.sin(x_series)
    
    def next_batch(self, batch_size, steps, return_batch_ts=False):
        rand_start = np.random.rand(batch_size,1)
        ts_start = rand_start * (self.x_max - self.x_min - (steps * self.resolution))
        batch_ts = ts_start + np.arange(0.0, steps+1) * self.resolution
        batch_y = np.sin(batch_ts)
        
        if return_batch_ts:
            return (batch_y[:,:-1].reshape(-1, steps, 1),
                    batch_y[:,1:].reshape(-1, steps, 1),
                    batch_ts)
        else:
            return (batch_y[:,:-1].reshape(-1, steps, 1),
                    batch_y[:,1:].reshape(-1, steps, 1))

tsd = TimeSeriesData(250, 0, 10)
# plt.plot(ts.x_data, ts.y_true)
# plt.show()

# NUM_TIME_STEPS=30
# y1, y2, ts = tsd.next_batch(1, NUM_TIME_STEPS, True)
# plt.plot(tsd.x_data, tsd.y_true, label='Sin(t)')
# plt.plot(ts.flatten()[1:], y2.flatten(), '*', label='Single training instance')
# plt.show()

# START BUILDING MODEL HERE
tf.reset_default_graph()

# MODEL CONSTANTS
NUM_INPUTS = 1
NUM_NEURONS = 100
NUM_OUTPUTS = 1
LEARNING_RATE = 0.0003
NUM_TRAIN_EPOCHS = 5000
NUM_TIME_STEPS = 30
BATCH_SIZE = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, NUM_TIME_STEPS, NUM_INPUTS])
y = tf.placeholder(tf.float32, [None, NUM_TIME_STEPS, NUM_OUTPUTS])

# RNN CELL LAYER
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=NUM_NEURONS,
                                                                           activation=tf.nn.relu),
                                              output_size=NUM_OUTPUTS)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# TRAIN SESSION
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for ep in range(NUM_TRAIN_EPOCHS):
        X_batch, y_batch = tsd.next_batch(BATCH_SIZE, NUM_TIME_STEPS)
        sess.run(train, feed_dict={X:X_batch, y:y_batch})
        
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print('Epoch: {}, MSE: {}'.format(ep, mse))
    saver.save(sess, './Models/RNN_example')
    
#%%
# CREATE TEST SET
test_inst = np.linspace(5, 5 + tsd.resolution * (NUM_TIME_STEPS + 1), NUM_TIME_STEPS + 1)

# TEST SESSION
with tf.Session() as sess:
    saver.restore(sess, './Models/RNN_example')
    
    X_new = np.sin(np.array(test_inst[:-1].reshape(-1, NUM_TIME_STEPS, NUM_INPUTS)))
    y_pred = sess.run(outputs, feed_dict={X:X_new})
    
# RESULTS PLOT
plt.title("TEST RESULTS")
plt.plot(test_inst[:-1], tsd.ret_true(test_inst[:-1]),
          'bo', markersize=15, alpha=0.5, label='TRAIN')
plt.plot(test_inst[1:], tsd.ret_true(test_inst[1:]),
          'ko', markersize=10, alpha=0.5, label='TARGET')
plt.plot(test_inst[1:], y_pred[0, :, 0], 'r.', markersize=10, label='PREDICTIONS')
plt.xlabel("TIME")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# PREDICTING MORE THAN ONE STEP OUT
with tf.Session() as sess:
    saver.restore(sess, './Models/RNN_example')
    
    seed_data = list(tsd.y_true[:30])
    
    for ep in range(len(tsd.x_data)-NUM_TIME_STEPS):
        X_batch = np.array(seed_data[-NUM_TIME_STEPS:]).reshape(1, NUM_TIME_STEPS, 1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        seed_data.append(y_pred[0,-1,0])
        
plt.plot(tsd.x_data, seed_data, 'b-')
plt.plot(tsd.x_data[:NUM_TIME_STEPS], seed_data[:NUM_TIME_STEPS], 'r', linewidth=3)
plt.xlabel('Time')
plt.show()
