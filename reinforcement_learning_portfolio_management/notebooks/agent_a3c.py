import numpy as np
import math
import sys
import myenv
sys.path.append('../src/features/')
sys.path.append('../src/data/')
import data_preprocessing_from_yahoo_finance as dp
import myenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import cuda

result_np = dp.data_preprocessing_1(ticker_list_input=['AAPL', 'PG'], path='../data/raw/')
c = myenv.myenv(result_np)

class Agent:
    def __init__(self, name, mode, env=c, reuse=True, lr=0.01):
        self.gamma = 0.99
#         self.sess = sess
        self.lr = lr
        self.actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output/new_weight')
            
        self.critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output/state_value')
        
        #############################################################
        # Inputs
        with tf.name_scope('Inputs'):
            self.price_tensor = tf.placeholder(dtype=tf.float64,
                              shape=[None]+env.observation_space_dimension['price_tensor'], name='price_tensor')
            self.weight = tf.placeholder(dtype=tf.float64,
                        shape=[None]+env.observation_space_dimension['weight'], name='weight')

        
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            # Conv Layers
            conv1 = tf.layers.conv2d(inputs=self.price_tensor,
                         filters=32,
                         kernel_size=[100,3],  # WRONG!
                         strides=2,
                         padding='same', name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=24,
                         kernel_size=[10,3],
                         strides=2,
                         padding='same', name='conv2')
            F = tf.reshape(conv2, [-1,np.prod(conv2.shape[1:])]) # Flatten

            # Dense Layer
            dense1 = tf.layers.dense(inputs=F, units=100,
                        activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=50,
                        activation=tf.nn.relu)
            # Concat weight tensor
            concat = tf.concat([dense2, self.weight], 1)

            dense3 = tf.layers.dense(inputs=concat, units=100,
                        activation=tf.nn.relu)

            logits = tf.layers.dense(dense3, units=env.action_space_dimension)
        
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            new_weight = tf.nn.softmax(logits, name='new_weight') # Actor output
        
            state_value = dense2 = tf.layers.dense(inputs=dense2, units=1, # Critic Output
                        activation='linear', name='state_value')
        
        self.output = tf.concat([new_weight, state_value], 1)
        self.init = tf.global_variables_initializer()
    #######################################################
    
    
a = Agent(name='testing', mode='testing')
price_tensor, weight,_,done = c.reset()
with tf.Session() as sess:
    sess.run(a.init)
    agent_output = sess.run(a.output, 
                  feed_dict={a.price_tensor:[price_tensor], 
                             a.weight:[weight]})
    
    print(agent_output)