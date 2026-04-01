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

class NstepQLearningAgent(BaseAgent):

    def update(self, states, actions, rewards, done, n):
        """
        states:  length T + 1
        actions: length T
        rewards: length T
        done: whether the final state in states is terminal
        """
        T = len(rewards)

        for t in range(T):
            target = 0.0
            m = min(n, T - t)

            for k in range(m):
                target += (self.gamma ** k) * rewards[t + k]

            # bootstrap only if we still have n full steps before the end
            if (t + m < T) or ((t + m == T) and not done):
                target += (self.gamma ** m) * np.max(self.Q_sa[states[t + m]])

            s = states[t]
            a = actions[t]
            self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])


def n_step_Q(
    n_timesteps,
    max_episode_length,
    learning_rate,
    gamma,
    policy='egreedy',
    epsilon=None,
    temp=None,
    plot=True,
    n=5,
    eval_interval=1000,
):
    """Runs a single repetition of an n-step Q-learning agent."""
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    eval_timesteps = []
    eval_returns = []

    t_steps = 0
    next_eval = eval_interval

    s = env.reset()
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

            s = s_next
            t_steps += 1

            if done:
                s = env.reset()
                break
            else:
                s = s_next

        pi.update(states=states, actions=actions, rewards=rewards, done=done, n=n)

        while t_steps >= next_eval:
    
            eval_ret= pi.evaluate(eval_env=eval_env)
            eval_timesteps.append(next_eval)
            eval_returns.append(eval_ret)
            next_eval += eval_interval

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 50001
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.05
    n = 5

    policy = 'egreedy'
    epsilon = 0.1
    temp = 1.0

    plot = True

    eval_returns, eval_timesteps = n_step_Q(
        n_timesteps,
        max_episode_length,
        learning_rate,
        gamma,
        policy,
        epsilon,
        temp,
        plot,
        n=n,
    )

    
    for i in range(len(eval_timesteps)):
        print(f"Timestep {eval_timesteps[i]} = {eval_returns[i]} mean return ")
    


if __name__ == '__main__':
    test()