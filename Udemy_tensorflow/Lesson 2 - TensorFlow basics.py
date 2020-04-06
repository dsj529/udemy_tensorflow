import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


#  first example: toy exercise
#  regressing to solve y = mx + b with noisy x,y data

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, '*')
plt.show()

seeds = np.random.rand(2)
m = tf.Variable(seeds[0])
b = tf.Variable(seeds[1])

error = 0
for x, y in zip(x_data, y_label):
    y_hat = m*x + b
    error += (y - y_hat)**2
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 10
    for i in range(training_steps):
        sess.run(train)
        print('estimated m: {}, b: {}'.format(*sess.run([m, b])))

#%%
## Regression in TF
## estimate y = mx +b, where m = 0.5, b = 5
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

X_df = pd.DataFrame(data=x_data, columns=['X'])
y_df = pd.DataFrame(data=y_true, columns=['y'])
my_data = pd.concat([X_df, y_df], axis=1)

batch_size = 8
seeds = np.random.rand(2)
m = tf.Variable(seeds[0], dtype=tf.float32)
b = tf.Variable(seeds[1], dtype=tf.float32)
xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_model = (m * xph) + b
error = tf.reduce_sum(tf.square(yph - y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    batches = 1000

    for i in range(batches):
        rand_idxs = np.random.randint(len(x_data), size=batch_size)
        feed = {xph:x_data[rand_idxs], yph:y_true[rand_idxs]}
        sess.run(train, feed)
        print('Current estimates: m={}, b={}'.format(*sess.run([m,b])))
        
    final_m, final_b = sess.run([m, b])
    
y_hat = x_data * final_m + final_b
my_data.sample(300).plot(kind='scatter', x='X', y='y')
plt.plot(x_data, y_hat, 'r')
plt.show()

#%%
# Regression in TF -- using the tf.estimator api
from sklearn.model_selection import train_test_split

feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


X_train, X_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3)

input_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train,
                                                batch_size=10,
                                                num_epochs=None,
                                                shuffle=True)
train_in_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train,
                                                    batch_size=10,
                                                    num_epochs=1000,
                                                    shuffle=False)
eval_in_func = tf.estimator.inputs.numpy_input_fn({'x': X_eval}, y_eval,
                                                    batch_size=10,
                                                    num_epochs=1000,
                                                    shuffle=False)

estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_in_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_in_func, steps=1000)
print('TRAIN:\n{}\n\nEVAL:\n{}'.format(train_metrics, eval_metrics))

new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': new_data},
                                                        shuffle=False)
preds = [pred['predictions'] for pred in estimator.predict(input_fn=input_fn_predict)]

my_data.sample(250).plot(kind='scatter', x='X', y='y')
plt.plot(new_data, preds, 'r')
plt.show()

#%%
## Classification in TF
##
## The course lecture uses the UCI Pima Indians Diabetes dataset, which has been withdrawn from
## public access.  I have chosen to do a similar classification exercise using the UCI Online Shoppers
## dataset instead

# load and preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv')
X = X.replace({True: 1, False:0})

cont_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
             'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 
             'SpecialDay']
cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
            'VisitorType', 'Weekend']
target = 'Revenue'

X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :-1],
                                                    X.iloc[:, -1],
                                                    test_size=0.3)

sc = StandardScaler()
sc.fit(X_train[cont_cols])
X_train[cont_cols] = sc.transform(X_train[cont_cols])
X_test[cont_cols] = sc.transform(X_test[cont_cols])

tf_base_cols = [tf.feature_column.numeric_column(col) for col in cont_cols]
tf_cat_cols =[tf.feature_column.categorical_column_with_vocabulary_list(col, X[col].unique())
                    for col in cat_cols]

# define and test the model
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=16, num_epochs=1000,
                                                 shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=tf_base_cols + tf_cat_cols,
                                      n_classes=2)
model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
                                                      batch_size=16, num_epochs=1,
                                                      shuffle=False)
base_results = model.evaluate(eval_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      batch_size=16,
                                                      num_epochs=1,
                                                      shuffle=False)
pred = list(model.predict(pred_input_func))

# define and test a DNN model
embedded_cols = [tf.feature_column.embedding_column(col, dimension=len(col.vocabulary_list)) 
                     for col in tf_cat_cols]

dnn_model = tf.estimator.DNNClassifier(hidden_units=[32,32,32],
                                        feature_columns=tf_base_cols + embedded_cols,
                                        n_classes=2)
dnn_model.train(input_func, steps=1000)
dnn_results = dnn_model.evaluate(eval_input_func)

print('\nBasic Classifier:\n{}\n\nDNN Classifier: {}'.format(base_results, dnn_results))