import numpy as np
import pandas as pd
import os, sys
# sys.path.append('../src/data/')
import data_preprocessing_from_yahoo_finance as dp
import matplotlib.pyplot as plt

class myenv:
    def make(self):
        # working
        pass
        
    def __init__(self, result_np,
                 price_window = 2000,
                 total_steps=100,
                 reward_type='log_return',
                 commission_rate = 0.05, 
                 initial_weight = None,
                 stocks_name = None): #reward_type can choose log_return
        self.current_step = 0 # start from step 0
        self.total_steps = total_steps
#         self.price_window = int(result_np.shape[1]-self.total_steps)-1 # How many data points to look at in each step
        self.price_window = price_window
        self.reward_type = reward_type
        self.commission_rate = commission_rate
                
        # Initialize Price Tensor
        self.cash_price = np.ones(result_np.shape[0:2])
        self.all_prices = np.concatenate((self.cash_price[:,:,None], result_np), axis=2) 
        self.all_prices_normalized = self.all_prices[:,1:,:]/self.all_prices[:,:-1,:]
        
        self.initial_weight = initial_weight
        
        # Action Space is the number of asset in result_np + cash
        self.action_space_dimension = result_np.shape[2]+1
        
        # Observation Space
        self.observation_space_dimension = {'price_tensor':[4,self.price_window,result_np.shape[2]+1], 'weight':[self.all_prices.shape[2]]}

        # Backtest
        self.portfolio_size = []
        self.portfolio_return = []
        self.sharpe_ratio = 0
        self.weights = []
        self.total_commision = 0
                
    def reset(self):   
        self.total_steps = min(self.total_steps, self.all_prices_normalized.shape[1] - self.price_window) # how many days to trade on
        if self.initial_weight is not None:
            weight = self.initial_weight # initialize weight
        else:
            weight = np.zeros(self.all_prices.shape[2]) # add cash dimension 
            weight[0] = 1
            
        self.weights.append(weight)
        
        # Keep track of how many stock units are in portfolio
        self.units_list = []
        self.units_list.append(weight)
        
        self.current_step = 0 
        start_prices = self.all_prices_normalized[:,self.current_step:self.current_step+self.price_window]
        
        start_portfolio_size = 1 
        self.portfolio_size.append(start_portfolio_size)
        return start_prices, weight, 0, False # new_prices_toagent, weight, reward, done    

    def step(self, weight, verbose=False):
        assert round(np.sum(weight),5)==1, "Sum of input weight is not equal to 1, %s" %weight  # make sure input weight intact 
        assert ~(np.sign(weight) == -1).any(), "Negative weight is not allowed, %s" %weight
        
        old_units = self.units_list[-1]
        portfolio_size = self.portfolio_size[-1]
        current_prices = self.all_prices[3,self.current_step+self.price_window,:] # +1?
        units = portfolio_size*weight/current_prices
        self.units_list.append(units)
        self.weights.append(weight)
        
        # Price change
        new_prices = self.all_prices[3,self.current_step+self.price_window+1,:] # +2?
        new_portfolio_size = np.sum(units*new_prices)
        reward = (new_portfolio_size/portfolio_size)-1
        
        # Commission
        commission = np.absolute(old_units-units) * self.commission_rate
        self.total_commision += commission
        
        # Keep track
        self.portfolio_size.append(new_portfolio_size)
        self.portfolio_return.append(reward) # Update reward before turning it into log scale
        
        # next step
        self.current_step += 1
        
        done = False
        if self.current_step == self.total_steps:
            done = True

        new_prices_toagent = self.all_prices_normalized[:,self.current_step:self.current_step+self.price_window,:] 
        
        if self.reward_type=='log_return':
            reward = np.log(reward+1)
        
        return new_prices_toagent, weight, reward, done    
    
    def render_psize(self):
        assert self.sharpe_ratio !=0, 'Have you end game?'
        p1 = plt.plot(self.portfolio_size)
        p2 = plt.plot(self.all_prices[3,-self.total_steps-1:,:]/self.all_prices[3,-self.total_steps-1,:])
        return p1, p2
        # Plot the weight, return, and the price together. 
        
    def render_weights(self, include_start=True):
        if include_start:
            i = 1
        else:
            i=0
        return plt.plot(self.weights[i:])
    
    def end_game(self):
        expected_return = np.mean(self.portfolio_return)
        std_return = np.std(self.portfolio_return)
        self.sharpe_ratio = expected_return / std_return