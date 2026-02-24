#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland|?
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        qs=self.Q_sa[s]
        pi_s = argmax(qs)
        return pi_s
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        # p_sas and r_sas are returns from env.model(s,a)
        Q_sa_update = 0
       
        for s_prime,p in enumerate(p_sas):
           
           max_Q_s_prime_a_prime = np.max(self.Q_sa[s_prime])
           Q_sa_update +=  p * (r_sas[s_prime] + self.gamma * max_Q_s_prime_a_prime ) 
        
        self.Q_sa[s,a] = Q_sa_update
        return 1
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    print (f"running Q_value_iteration with gamma = {gamma}, and threshold = {threshold}")
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
 
     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
    stop = False
    iterated = 0
    while not stop: 
        iterated += 1
        error = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s,a)
                x = QIagent.Q_sa[s,a]
                QIagent.update(s,a,p_sas,r_sas)
                # used to render the updates of Q_sa 
                env.render(QIagent.Q_sa,step_pause=0.5)
                error = max(error,abs(x - QIagent.Q_sa[s,a]))
        print(f"Maximum Absolute error {error} : iteration {iterated}")
        if error < threshold: 
            stop = True
    
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    print( "environment created")
    # env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    print("Q value iteration agent implemented")

    # view optimal policy
    print("view optimal policy")
    done = False
    s = env.reset()
    
    while not done:
        a = QIagent.select_action(s)
        
        s_next, r, done = env.step(a)
        # visualizes the Q - state- action value used by the Agent as to decide the optimal policy
        # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        print(f"state {s} action {a} reward {r} done {done} \n")
        s = s_next
        

    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
