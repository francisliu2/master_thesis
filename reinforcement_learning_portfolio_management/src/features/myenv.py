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
                 total_steps=100,
                 reward_type=None,
                 commission_rate = 0.05, 
                 initial_weight = None): #reward_type can choose log_return
        self.current_step = 0 # start from step 0
        self.total_steps = total_steps # how many days to trade on
        self.price_window = int(result_np.shape[1]-self.total_steps)-1 # How many data points to look at in each step
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
    
    def reset(self):
        # Backtest
        self.portfolio_size = []
        self.portfolio_return = []
        self.sharpe_ratio = 0
        self.weights = []
        
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
        
        # Keep track
        self.portfolio_size.append(new_portfolio_size)
        self.portfolio_return.append(reward)
        
        # next step
        self.current_step += 1
        
        done = False
        if self.current_step == self.total_steps:
            done = True

        new_prices_toagent = self.all_prices_normalized[:,self.current_step:self.current_step+self.price_window,:] 
        
        if self.reward_type=='log_return':
            reward = np.log(reward+1)
        
        return new_prices_toagent, weight, reward, done    
    
    def render(self):
        pass
        # Plot the weight, return, and the price together. 
    
    def end_game(self):
        expected_return = np.mean(self.portfolio_return)
        std_return = np.std(self.portfolio_return)
        self.sharpe_ratio = expected_return / std_return