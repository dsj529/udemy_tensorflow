#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## common setup to all APIs
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

wine_data = load_wine()
features = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

sc = MinMaxScaler().fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

#%%
## Estimator API
# from tensorflow import estimator

# feat_cols = [tf.feature_column.numeric_column('x', shape=[13])]

# deep_model = estimator.DNNClassifier(hidden_units=[13,13,13],
#                                      feature_columns=feat_cols,
#                                      n_classes=3,
#                                      optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01))
# input_fn = estimator.inputs.numpy_input_fn(x={'x':X_train_sc},
#                                            y=y_train,
#                                            shuffle=True,
#                                            num_epochs=10)
# deep_model.train(input_fn=input_fn, steps=500)

# input_fn_eval = estimator.inputs.numpy_input_fn(x={'x': X_test_sc},
#                                                 shuffle=False)
# pred = list(deep_model.predict(input_fn=input_fn_eval))
# preds = [p['class_ids'][0] for p in pred]

# print(classification_report(y_test, preds))

#%%
# ## tf.keras API
# from tensorflow.contrib.keras import models
# from tensorflow.contrib.keras import layers
# # from tensorflow.contrib.keras import losses, optimizers, metrics, activations

# dnn_keras_model = models.Sequential()
# dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
# dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
# dnn_keras_model.add(layers.Dense(units=13, activation='relu'))
# dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))
# dnn_keras_model.compile(optimizer='rmsprop',
#                         loss='sparse_categorical_crossentropy',
#                         metrics=['accuracy'])
# dnn_keras_model.fit(X_train_sc, y_train, epochs=50)

# preds = dnn_keras_model.predict_classes(X_test_sc)
# print(classification_report(y_test, preds))

#%%
## TF.layers
import pandas as pd
y_train_onehot = pd.get_dummies(y_train).as_matrix()
y_test_onehot = pd.get_dummies(y_test).as_matrix()

num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01

from tensorflow.contrib.layers import fully_connected
X = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, 3])
actf = tf.nn.relu

hidden1 = fully_connected(X, num_hidden1, activation_fn=actf)
hidden2 = fully_connected(hidden1, num_hidden2, activation_fn=actf)
output = fully_connected(hidden2, num_outputs)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init=tf.global_variables_initializer()

training_steps = 1500

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(training_steps):
        sess.run(train, feed_dict={X:X_train_sc,
                                    y_true: y_train_onehot})
    logits = output.eval(feed_dict={X:X_test_sc})
    preds = tf.argmax(logits, axis=1)
    results = preds.eval()

print(classification_report(results, y_test))