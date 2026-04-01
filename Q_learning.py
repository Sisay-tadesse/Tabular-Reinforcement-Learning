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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        if not done:
            G_t = r + self.gamma * max(self.Q_sa[s_next])
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (G_t - self.Q_sa[s,a])       
        else: 
            G_t = r 
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (G_t - self.Q_sa[s,a])      
        

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    
    done = False
    steps = 0
    # sample initial state
    
    s = env.reset()
    while steps < n_timesteps:
        a = agent.select_action(s,epsilon=epsilon,policy='egreedy')
        s_next,r,done = env.step(a)
        agent.update(s, a= a, r=r, s_next=s_next, done=done)
        if done:
            s = env.reset()
        else:
            s = s_next
        
        steps += 1
        if steps % eval_interval == 0:
            eval_ret = agent.evaluate(eval_env=eval_env)
            eval_timesteps.append(steps)
            eval_returns.append(eval_ret)
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 50001
    eval_interval=1000
    gamma = 1
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.3
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    for i in range(len(eval_timesteps)):
        print(f"Timestep {eval_timesteps[i]} = {eval_returns[i]} mean return ")
    
    # when evaluating, at the evaluation timesteps, we get a certain policy,
    #  we test that policy by running it through several episodes
    # , finding the return of each episode, then averaging by the total number of episode
    # this will let us find the average return or mean
    # then we can calcuate the standard error , that is standard deviation over square root of the number of episodes for that policy,
    # confidence = mean +- SE
    # 95 percent confidence bound = mean  +- 1.96 * SE 

if __name__ == '__main__':
    test()
