#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:57:26 2020

@author: dsj529
"""


#%%
# Simple rule-based game play
# =============================================================================
# import gym
# env = gym.make('CartPole-v0')
# 
# obs = env.reset()
# 
# for t in range(1000):
#     env.render()
#     cart_pos, cart_v, pole_ang, ang_v = obs
#     if pole_ang > 0:
#         action = 1
#     else:
#         action = 0
#         
#     obs, reward, done, info = env.step(action)
#     if done:
#         break
#             
# env.close()
# =============================================================================
    
#%%
# =============================================================================
# # DNN-informed game play
# import tensorflow as tf
# import gym
# import numpy as np
# 
# num_inputs = 4
# num_hidden = 4
# num_outputs = 1
# 
# initializer = tf.contrib.layers.variance_scaling_initializer()
# 
# X = tf.placeholder(tf.float32, shape=[None, num_inputs])
# 
# hidden_1 = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
# hidden_2 = tf.layers.dense(hidden_1, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
# output = tf.layers.dense(hidden_2, num_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)
# 
# probs = tf.concat(axis=1, values=[output, 1-output])
# action = tf.multinomial(probs, num_samples=1)
# 
# init = tf.global_variables_initializer()
# 
# eps = 50
# step_limit = 500
# env = gym.make('CartPole-v0')
# avg_steps = []
# 
# with tf.Session() as sess:
#     init.run()
#     
#     for i_episode in range(eps):
#         obs = env.reset()
#         
#         for step in range(step_limit):
#             action_val=action.eval(feed_dict={X:obs.reshape(1, num_inputs)})
#             obs, reward, done, info = env.step(action_val[0][0])
#             
#             if done:
#                 avg_steps.append(step)
#                 break
#         print('ep: {} || {} || {:.2f} (+/- {:.2f})'
#                   .format(i_episode, step, np.mean(avg_steps), np.std(avg_steps)))
# env.close()
# =============================================================================

#%%
# Policy gradient-informed game play
import tensorflow as tf
import gym
import numpy as np

tf.reset_default_graph()

num_inputs = 4
num_hidden = 4
num_outputs = 1
learning_rate = 0.03

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_l = tf.layers.dense(X, num_hidden, 
                           activation=tf.nn.elu, 
                           kernel_initializer=initializer)
logits = tf.layers.dense(hidden_l, num_outputs)
outputs = tf.sigmoid(logits)

probs = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(probs, num_samples=1)

y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.RMSPropOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(cross_entropy)
grads = []
grad_phs = []
grads_and_vars_feed = []

for grad, var in grads_and_vars:
    grads.append(grad)
    grad_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    grad_phs.append(grad_placeholder)
    grads_and_vars_feed.append((grad_placeholder, var))
    
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def helper_discount_rewards(rewards, rate):
    return np.cumsum(rate * np.array(rewards)[::-1])[::-1].tolist()

def discount_and_normalize(all_rewards, rate):
    all_discounted = []
    for rewards in all_rewards:
        all_discounted.append(helper_discount_rewards(rewards, rate))
    flat_rewards = np.concatenate(all_discounted)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std 
                for discounted_rewards in all_discounted]

env = gym.make('CartPole-v0')

num_game_rounds = 10
max_game_steps = 1000
num_iterations = 650
discount_rate = 0.9

with tf.Session() as sess:
    init.run()
    
    for iteration in range(num_iterations):
        print('Current iter: {}'.format(iteration+1))
        all_rewards = []
        all_grads = []

        for game in range(num_game_rounds):
            current_rewards=[]
            current_grads=[]
            
            obs = env.reset()
        
            for step in range(max_game_steps):
                action_val, grad_val = sess.run([action, grads],
                                                feed_dict={X:obs.reshape(1,num_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                
                current_rewards.append(reward)
                current_grads.append(grad_val)
                
                if done:
                    break
            all_rewards.append(current_rewards)
            all_grads.append(current_grads)
        all_rewards = discount_and_normalize(all_rewards, discount_rate)
        feed_dict = {}
        
        for var_idx, grad_ph in enumerate(grad_phs):
            mean_grads = np.mean([reward * all_grads[game_idx][step][var_idx]
                                      for game_idx, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)],
                                 axis=0)
            feed_dict[grad_ph] = mean_grads
        
        sess.run(training_op, feed_dict=feed_dict)
        
    print('Saving graph and session')
    meta_graph_def = tf.train.export_meta_graph(filename='./Models/policy_grad.meta')
    saver.save(sess, './Models/policy_grad')
    
    
observations = env.reset()
with tf.Session() as sess:
    # https://www.tensorflow.org/api_guides/python/meta_graph
    new_saver = tf.train.import_meta_graph('./Models/policy_grad.meta')
    new_saver.restore(sess,'./Models/policy_grad')

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([action, grads], 
                                             feed_dict={X: observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])

        if done:
            break
    env.close()