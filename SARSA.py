#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent
from Helper import LearningCurvePlot

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        if not done:
            G_t = r + self.gamma * (self.Q_sa[s_next,a_next])
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (G_t - self.Q_sa[s,a])       
        else: 
            G_t = r 
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (G_t - self.Q_sa[s,a])      
        
        

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=1000):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!
    done = False
    steps = 0
    # sample initial state
    
    s = env.reset()
    while steps < n_timesteps:
        a = pi.select_action(s,epsilon=epsilon,policy='egreedy')
        s_next,r,done = env.step(a)
        a_next = pi.select_action(s_next,epsilon=epsilon,policy='egreedy')
        pi.update(s, a= a, r=r, s_next=s_next, a_next=a_next,done=done)
        if done:
            s = env.reset()
            
            
        else:
            s = s_next
            
        
        steps += 1
        if steps % eval_interval == 0:
            eval_ret = pi.evaluate(eval_env=eval_env)
            eval_timesteps.append(steps)
            eval_returns.append(eval_ret)
            
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 50001
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    for i in range(len(eval_timesteps)):
        print(f"Timestep {eval_timesteps[i]} = {eval_returns[i]} mean return ")  
      
    
if __name__ == '__main__':
    test()
