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
class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done):
        """
        states:  length T + 1
        actions: length T
        rewards: length T
        done: whether the final state in states is terminal
        """
        # First visit MonteCarlo
        keys = list(zip(states,actions))
        t = len(rewards) - 1 
        target = 0
        while t >= 0:
            target = rewards[t] + self.gamma * (target) 
            if (states[t],actions[t]) not in list(zip(states[:t],actions[:t])):
                self.Q_sa[states[t],actions[t]] = self.Q_sa[states[t],actions[t]] + self.learning_rate * (target - self.Q_sa[states[t],actions[t]])
            t = t - 1
def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=1000):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    t_steps = 0
    next_eval = eval_interval

    s = env.reset()
    allrewards = []
    while t_steps < n_timesteps:
        actions = []
        states = [s]
        rewards = []
        
        done = False

        for _ in range(max_episode_length):
            if t_steps >= n_timesteps:
                break

            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)
            
            actions.append(a)
            states.append(s_next)
            rewards.append(r)
            allrewards.append(r)
            
            t_steps += 1

            if done:
                s = env.reset()
                break
            s = s_next

        pi.update(states=states, actions=actions, rewards=rewards, done=done )

        while t_steps >= next_eval:
           
            eval_ret = pi.evaluate(eval_env=eval_env)
            eval_timesteps.append(next_eval)
            eval_returns.append(eval_ret)
            next_eval += eval_interval

    return np.array(eval_returns), np.array(eval_timesteps)

    
def test():
    n_timesteps = 50001
    max_episode_length = 1000
    gamma = .9
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps= monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
    for i in range(len(eval_timesteps)):
        print(f"Timestep {eval_timesteps[i]} = {eval_returns[i]} mean return") 
  
       
if __name__ == '__main__':
    test()
