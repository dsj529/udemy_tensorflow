#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
## The coding exercise uses a proprietary dataset included with the lecture materials.
## I have chosen to do a similar process using a publicly available dataset instead.
 
# Load and prepare the data
seaIce = pd.read_csv('https://timeseries.weebly.com/uploads/2/1/0/8/21086414/sea_ice.csv')
seaIce['Time'] = pd.to_datetime(seaIce['Time'], format='%YM%m')
seaIce = seaIce.set_index('Time')
 
# test-train split
train = seaIce.iloc[:-12]
test = seaIce.iloc[-12:]
 
# scale the data for better analysis
cols = ['Arctic', 'Antarctica']
sc = MinMaxScaler().fit(train[cols])
train_sc = sc.transform(train[cols])
test_sc = sc.transform(test[cols])
 
# create a batch function
def next_batch(train_data, batch_size, steps):
    rand_start = np.random.randint(0, len(train_data) - steps)
    y_batch = train_data[rand_start: rand_start+steps+1].reshape(2, steps+1)
    return (y_batch[:, :-1].reshape(-1, steps, 2),
            y_batch[:, 1:].reshape(-1, steps, 2))
 
tf.reset_default_graph()
# MODEL CONSTANTS
NUM_INPUTS = 2
NUM_TS = 12
NUM_NEURONS = 2000
NUM_OUTPUTS = 2
LEARNING_RATE = 0.0003
NUM_TRAIN_EPOCHS = 7500
BATCH_SIZE = 1
 
# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, NUM_TS, NUM_INPUTS])
y = tf.placeholder(tf.float32, [None, NUM_TS, NUM_OUTPUTS])
  
# RNN LAYER
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=NUM_NEURONS,
                                                                     activation=tf.nn.relu),
                                              output_size=NUM_OUTPUTS)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
  
# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(loss)
  
init = tf.global_variables_initializer()
  
# TRAIN SESSION
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
config = tf.ConfigProto(gpu_options=gpu_options)
saver = tf.train.Saver()
  
with tf.Session(config=config) as sess:
    sess.run(init)
       
    for ep in range(NUM_TRAIN_EPOCHS):
        X_batch, y_batch = next_batch(train_sc, BATCH_SIZE, NUM_TS)
        sess.run(train, feed_dict={X:X_batch, y:y_batch})
           
        if (ep+1) % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
            print('Epoch: {}, MSE: {}'.format(ep+1, mse))
    saver.save(sess, './Models/RNN_exercise')

# TEST SESSION
with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./Models/RNN_exercise")

    # CODE HERE!
    seed_data = list(train_sc[-12:])
    
    for ep in range(12):
        X_batch = np.array(seed_data[-NUM_TS:]).reshape(1, NUM_TS, 2)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        seed_data.append(y_pred[0,-1,:])
        
results = sc.inverse_transform(seed_data[12:])

test['Pred_Arctic'] = results[:,0]
test['Pred_Antarctica'] = results[:,1]

# test.plot()