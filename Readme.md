## Description
This project implements several tabular reinforcement learning algorithms, including 
Q-learning, SARSA, and n-step Q-learning, in the Stochastic Windy Gridworld environment.

The goal of the experiments is to compare learning performance and analyze the effect
of different algorithms on Tabular RL.

## File Structure

Streamlit_app.py
    script to run a GUI for Stochastic Windy Grid world
    use:  streamlit run streamlit_app.py

experiment.py
    Script used to run training experiments and generate results

Q_learning.py
    Implementation of the Q-learning algorithm

SARSA.py
    Implementation of the SARSA algorithm

DynamicProgramming.py
    Implementation of the DP algorithm
    
MonterCarlo.py
    Implementation of the MC algorithm

Nstep.py
    Implementation of the Nstep q-learning algorithm

Environment.py
    Implementation of the Stochastic Windy Gridworld environment

Agent.py
    Base class for reinforcement learning agents

