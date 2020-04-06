from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

#%%
## Exercise 1 -- Regression with real data
## The lectures used a set of California Housing data.  To challenge myself, I've chosen instead
## to regress on the Million Song Dataset from the UCI machine learning archives.
names = (['year'] +
         ['tim_{:0d}'.format(x+1) for x in range(12)] + # 12 columns of timbre-range averages
         ['tim_var_{:0d}'.format(x+1) for x in range(78)]) # 78 columns of timbre variances

msd = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
                  compression='zip', header=None, names=names)

# train-test split as defined in dataset description
X_train = msd.iloc[:463715, 1:13]
y_train = msd.iloc[:463715, 0].astype('float32')
X_test = msd.iloc[463715:, 1:13]
y_test = msd.iloc[463715:, 0].astype('float32')

cols = X_train.columns

sc_x = StandardScaler().fit(X_train[cols])
X_train[cols] = sc_x.transform(X_train[cols])
X_test[cols] = sc_x.transform(X_test[cols])

def rmse(labels, predictions):
    pred_values = predictions['predictions']
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

def acc(labels, predictions):
    pred_values = predictions['predictions']
    return {'acc': tf.metrics.accuracy(labels, pred_values)}

batch_size = 8
data_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns]
model = tf.estimator.DNNRegressor(feature_columns=data_cols,
                                  hidden_units=[16, 256, 256, 256, 256],
                                  optimizer='Adagrad')

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=batch_size,
                                                 num_epochs=10000,
                                                 shuffle=True)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
                                                      batch_size=batch_size, 
                                                      num_epochs=1,
                                                      shuffle=False)
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      batch_size=batch_size,
                                                      num_epochs=1,
                                                      shuffle=False)

model = tf.contrib.estimator.add_metrics(model, rmse)
model = tf.contrib.estimator.add_metrics(model, acc)

model.train(input_fn=input_func, steps=10000)
base_results = model.evaluate(eval_input_func)
print(base_results)

#%%
## Exercise 2 -- Classification with real data
## Evaluating sensor positions to determine wearer's posture

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00405/Postures.zip',
                   skiprows=[1])
data = data.drop('User', axis=1)
data = data.replace('?', np.nan)
data['Class'] -= 1
cols = data.drop('Class', axis=1).columns

X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.3)
sc = StandardScaler().fit(X_train[cols])
X_train[cols] = sc.transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])

data_cols = [tf.feature_column.numeric_column(col) for col in cols]
model = tf.estimator.DNNClassifier(hidden_units=[128, 128, 128, 128, 128],
                                   feature_columns=data_cols,
                                   n_classes=5)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=16, 
                                                 num_epochs=10000,
                                                 shuffle=True)
model.train(input_fn=input_func, steps=10000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
                                                      batch_size=16,
                                                      num_epochs=1,
                                                      shuffle=False)
results = model.evaluate(eval_input_func)
print(results)