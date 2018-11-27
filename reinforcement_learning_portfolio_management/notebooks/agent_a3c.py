import numpy as np
import math
import sys
# import myenv
sys.path.append('../src/features/')
sys.path.append('../src/data/')
import data_preprocessing_from_yahoo_finance as dp
import myenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from numba import cuda

class Agent(object):
    def __init__(self, sess,  scope, no_of_stocks, no_of_trading_days, lr=0.01, entropy_param = 0.01, clip=True):
        self.sess = sess
        self.gamma = np.float64(0.99)
        self.lr = lr
        self.no_of_stocks = no_of_stocks
        self.no_of_trading_days = no_of_trading_days 
        self.entropy_param =  entropy_param
        if scope == 'global':
            self.price_tensor, self.weight, self.state_value, self.new_weight, self.actor_params, self.critic_params = self.build_net(scope)    
        elif scope == 'local':
            self.price_tensor, self.weight, self.state_value, self.new_weight, self.actor_params, self.critic_params = self.build_net(scope) 
            
            self.state_value_target = tf.placeholder(tf.float64, [None, 1])
            temporal_difference = self.state_value_target - self.state_value 
            self.critic_loss = tf.reduce_mean(tf.square(temporal_difference))
            
            ##########################
            # A=Q-V
            self.entropy = -tf.reduce_sum(self.new_weight*tf.log(self.new_weight))
            # We want higher entropy as we want to explore
            self.actor_loss = -tf.log(self.new_weight)* tf.stop_gradient(temporal_difference) - self.entropy_param*self.entropy
 #           self.loss = 0.5*self.critic_loss + self.actor_loss - self.entropy*0.01   
            self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)
            self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
            
#             if clip:
#                 self.critic_grads = [tf.clip_by_value(grad, -1., 1.) for grad in self.critic_grads]
#                 self.actor_grads = [tf.clip_by_value(grad, -1., 1.) for grad in self.actor_grads]                
        else:
            print("Please select 'global' or 'local'")     
            
    def build_net(self, scope, reuse=tf.AUTO_REUSE):
        #############################################################
        # Inputs
#         initializer = tf.random_normal_initializer(mean=1, stddev=1)
        initializer = None
        with tf.variable_scope(scope+'_inputs', reuse=reuse):
            price_tensor = tf.placeholder(dtype=tf.float64,
                              shape=[None]+[4, self.no_of_trading_days,self.no_of_stocks+1] , name='price_tensor')
            weight = tf.placeholder(dtype=tf.float64,
                        shape=[None]+[self.no_of_stocks+1], name='weight')

        with tf.variable_scope(scope+'_network', reuse=reuse):
            # Conv Layers
            conv1 = tf.layers.conv2d(inputs=price_tensor,
                         filters=32,
                         kernel_size=[100,self.no_of_stocks], # Consider all stocks at once
                         kernel_initializer= initializer, 
                         strides=2,
                         padding='same', name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                         filters=24,
                         kernel_size=[10,self.no_of_stocks],
                                     kernel_initializer= initializer, 
                         strides=2,
                         padding='same', name='conv2')
            
            F = tf.reshape(conv2, [-1, np.prod(conv2.shape[1:])], name='flattern') # Flatten

            # Dense Layer
            dense1 = tf.layers.dense(inputs=F, units=100,
                        activation=tf.nn.relu, name='dense1',kernel_initializer= initializer)
            dense2 = tf.layers.dense(inputs=dense1, units=50,
                        activation=tf.nn.relu, name='dense2',kernel_initializer= initializer)
            # Concat weight tensor
            concat = tf.concat([dense2, weight], 1, name='concat')

            dense3 = tf.layers.dense(inputs=concat, units=100,
                        activation=tf.nn.relu, name='dense3', kernel_initializer= initializer)

        with tf.variable_scope(scope+'_policy', reuse=reuse):
            logits = tf.layers.dense(dense3, units=self.no_of_stocks+1, name='logit', kernel_initializer= initializer)
            new_weight = tf.nn.softmax(logits, name='new_weight') # Actor output
            with tf.variable_scope('train', reuse=True):
                self.OPT_A = tf.train.RMSPropOptimizer(0.01,name='RMSProp_A')

        with tf.variable_scope(scope+'_value', reuse=reuse):
            state_value = tf.layers.dense(inputs=dense3, units=1, # Critic Output
                        activation='linear', name='state_value')
            critic_train_op = tf.train.AdamOptimizer(self.lr)
            with tf.variable_scope('train', reuse=True):
                self.OPT_C = tf.train.RMSPropOptimizer(0.01,name='RMSProp_C')

        output = tf.concat([new_weight, state_value], 1)            

        actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'_network')+ \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'_policy')
        critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'_network')+ \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'_value')
                   
        return price_tensor, weight, state_value, new_weight, actor_params, critic_params
    
    def step(self, price_tensor, weight):
        return self.sess.run(output, feed_dict={self.price_tensor:price_tensor, self.weight:weight})
        
    def get_value(self, S):
        return self.sess.run(self.state_value, feed_dict={self.price_tensor:S[0], self.weight:S[1]})
  
    def get_critic_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'_input') + \
               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope+'_network') + \
               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+'_value/state_value')

            
class Memory(): # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/9_Deep_Deterministic_Policy_Gradient_DDPG/DDPG.py
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.env = env
        self.pointer = 0
        dim = 1+1+1+1# S,A,R,S_
        self.data = np.zeros(([capacity]+[dim])) # state, action, reward, next_state

    def store_transition(self, state_number, action_number, r, next_state_number):
        transition = [state_number, action_number, r, next_state_number]
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1
        
    def sample(self, n):
        assert self.pointer >= self.capacity, 'Please fill the memory'
        indices = np.random.choice(self.capacity, size = n)
        sampled_data = self.data[indices, :]
        
        S=[]
        A=[]
        R=[]
        S_=[]
        for row in sampled_data:
            s = [self.env.all_prices_normalized[:,int(row[0]):int(row[0]+self.env.price_window),:], self.env.weights[int(row[0])] ]
            S.append(s) # current step -1
            A.append(self.env.weights[int(row[1]-1)])
            R.append(row[2])
            s_ = [self.env.all_prices_normalized[:,int(row[3]):int(row[3]+self.env.price_window),:], self.env.weights[int(row[3])] ]
            S_.append(s_)
        
        S = np.array(S)
        S_ = np.array(S_)
        
        result_S = [np.stack(S[:,0], axis=0), np.stack(S[:,1], axis=0)]
        result_A = A
        result_R = R
        result_S_ = [np.stack(S_[:,0], axis=0), np.stack(S_[:,1], axis=0)]
        return result_S, result_A, result_R, result_S_
#         return S,A,R,S_