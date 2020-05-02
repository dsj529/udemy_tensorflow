from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


mnist = input_data.read_data_sets('./data/MNIST_data/', one_hot=True)
tf.reset_default_graph()

## build the image generator
def generator(z, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hidden_1 = tf.layers.dense(inputs=z, units=128) 
        hidden_1 = tf.nn.leaky_relu(hidden_1, alpha=0.01)
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=128) 
        hidden_ = tf.nn.leaky_relu(hidden_2, alpha=0.01)
        output = tf.layers.dense(inputs=hidden_2, units=784,
                                 activation=tf.nn.tanh)
        return output
    
## build the image discriminator
def discriminator(X, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hidden_1 = tf.layers.dense(inputs=X, units=128, 
                                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01))
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=128, 
                                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01))
        logits = tf.layers.dense(inputs=hidden_2, units=1)
        output = tf.sigmoid(logits)
        return output, logits
    
real_images = tf.placeholder(tf.float32, shape=[None, 784])
z = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(z)
D_output_real, D_logits_real = discriminator(real_images)
D_output_fake, D_logits_fake = discriminator(G, reuse=True)

## define loss fn
def loss_fn(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,
                                                                  labels=labels_in))
    
D_real_loss = loss_fn(D_logits_real, tf.ones_like(D_logits_real) * 0.9)
D_fake_loss = loss_fn(D_logits_fake, tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss
G_loss = loss_fn(D_logits_fake, tf.ones_like(D_logits_fake))

learning_rate = 0.003

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

D_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 128
epochs = 500
samples = []
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
    sess.run(init)
    
    for e in range(epochs):
        num_batches = mnist.train.num_examples // batch_size
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images * 2 - 1 # rescale for tanh activation fn
            
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            
            _ = sess.run(D_trainer, feed_dict={real_images:batch_images, 
                                               z:batch_z})
            _ = sess.run(G_trainer, feed_dict={z:batch_z})
            
        print('Finished epoch {}/{}'.format(e+1, epochs))
        
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True),
                              feed_dict={z:sample_z})
        samples.append(gen_sample)
        
        saver.save(sess, './Models/GAN_example')
        
new_samples=[]
with tf.Session() as sess:
    saver.restore(sess, './Models/GAN_example')
    
    for x in range(5):
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = sess.run(generator(z, reuse=True),
                              feed_dict={z:sample_z})
        new_samples.append(gen_sample)