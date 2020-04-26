# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

import collections
import math
import os
import errno
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf

data_dir = "./data/word2vec_data/words"
data_url = 'http://mattmahoney.net/dc/text8.zip'

def fetch_words_data(url=data_url, words_data=data_dir):
    
    # Make the Dir if it does not exist
    os.makedirs(words_data, exist_ok=True)
    
    # Path to zip file 
    zip_path = os.path.join(words_data, "words.zip")
    
    # If the zip file isn't there, download it from the data url
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        
    # Now that the zip file is there, get the data from it
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    
    # Return a list of all the words in the data source.
    return data.decode("ascii").split()

# Use Defaults (this make take awhile!!)
words = fetch_words_data()

#%%
# ## Explore data set
# # Total words
# print(len(words))
# # Random slice of words
# for w in words[9000:9040]:
#     print(w,end=' ')

#%%
## Build word counts
from collections import Counter

def create_counts(vocab_size=50000):

    # Begin adding vocab counts with Counter
    vocab = [] + Counter(words).most_common(vocab_size )
    
    # Turn into a numpy array
    vocab = np.array([word for word, _ in vocab])
    
    
    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data,vocab

vocab_size = 50000
data,vocabulary = create_counts(vocab_size=vocab_size)

# # Explore the word/vocab mappings:
# (words[100], vocabulary[100])
# Out[9]: ('interpretations', 'city')

# (words[100], data[100])
# Out[10]: ('interpretations', 4186)

# vocabulary.shape
# Out[11]: (50000,)


#%%
def generate_batch(batch_size, num_skips, skip_window):
    
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# # Look at a test batch
# data_index=0
# batch, labels = generate_batch(8,2,1)
# batch
# Out[15]: array([3080, 3080, 3080, 3080, 3080, 3080, 3080, 3080], dtype=int32)

# labels
# Out[16]: 
# array([[5233],
#        [  11],
#        [  11],
#        [5233],
#        [5233],
#        [  11],
#        [  11],
#        [5233]], dtype=int32)

#%%
## MODEL CONSTANTS
BATCH_SIZE = 128
EMBEDDING_SIZE = 150
SKIP_WINDOW = 1
NUM_SKIPS = 2

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.

# Random set of words to evaluate similarity on.
valid_size = 16   

# Only pick dev samples in the head of the distribution.
valid_window = 100  
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Number of negative examples to sample.
num_sampled = 64   

# Model Learning Rate
learning_rate = 0.01

# How many words in vocab
vocabulary_size = 50000

#%%
## PLACEHOLDERS AND CONSTANTS
tf.reset_default_graph()

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Look up embeddings for inputs.
init_embeds = tf.random_uniform([vocabulary_size, EMBEDDING_SIZE], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# NCE Loss function
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, EMBEDDING_SIZE],
                                              stddev=1.0 / np.sqrt(EMBEDDING_SIZE)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

# OPTIMIZER
# Construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()

#%%
## TRAINING SESSION
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# Usually needs to be quite large to get good results, 
# training takes a long time!
num_steps = 200001

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    average_loss = 0
    data_index = 0
    for step in range(num_steps):
         
        batch_inputs, batch_labels = generate_batch(BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        empty, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if (step+1) % 1000 == 0:
            average_loss /= 1000
            # The average loss is an estimate of the loss over the last 1000 batches.
            print("Average loss at step {}: {}".format(step+1, average_loss))
            average_loss = 0    

    final_embeddings = normalized_embeddings.eval()
    
#%%
## Visualize the results
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

plot_only = 2000
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [vocabulary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

#%%
np.save('./data/trained_embeddings_200k_steps',final_embeddings)