import streamlit as st
import sys
import os
sys.path.insert(0, '/vol/home/s4184343/Desktop/Reinforcement_learning/Assignment1_2026/Code_assignment')

from Environment import StochasticWindyGridworld
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from DynamicProgramming import QValueIterationAgent
from Helper import argmax
st.title("Stochastic Windy Gridworld RL Environment")


# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = StochasticWindyGridworld(initialize_model=True)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0
    st.session_state.Q_sa = np.zeros((st.session_state.env.n_states, st.session_state.env.n_actions))
    st.session_state.iteration_count = 0

env = st.session_state.env
state = st.session_state.state
done = st.session_state.done
step_count = st.session_state.step_count
Q_sa = st.session_state.Q_sa
iteration_count = st.session_state.iteration_count

st.write(f"Current State: {state}, Steps: {step_count}, Done: {done}")
# Action buttons
optimal, col1, col2, col3, col4 = st.columns(5)
with col1:
    if st.button("Up (0)"):
        if not done:
            next_state, reward, done = env.step(0)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.write(f"Action: Up, Reward: {reward}, Next State: {next_state}")
with col2:
    if st.button("Right (1)"):
        if not done:
            next_state, reward, done = env.step(1)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.write(f"Action: Right, Reward: {reward}, Next State: {next_state}")
with col3:
    if st.button("Down (2)"):
        if not done:
            next_state, reward, done = env.step(2)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.write(f"Action: Down, Reward: {reward}, Next State: {next_state}")
with col4:
    if st.button("Left (3)"):
        if not done:
            next_state, reward, done = env.step(3)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.write(f"Action: Left, Reward: {reward}, Next State: {next_state}")

# Optimal action button
with optimal:
    if st.button("Take Optimal Action"):
        if not done:
            optimal_action = argmax(Q_sa[state])
            next_state, reward, done = env.step(optimal_action)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.step_count += 1
            st.write(f"Optimal Action: {optimal_action}, Reward: {reward}, Next State: {next_state}")
            

# Reset button
if st.button("Reset"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0
    st.rerun()

# Q-Value Iteration button
if st.button("Run One Q-Value Iteration"):
    gamma = 1.0
    error = 0
    for s in range(env.n_states):
        for a in range(env.n_actions):
            p_sas, r_sas = env.model(s, a)
            old_q = Q_sa[s, a]
            Q_sa_update = 0
            for s_prime, p in enumerate(p_sas):
                max_Q_s_prime = np.max(Q_sa[s_prime])
                Q_sa_update += p * (r_sas[s_prime] + gamma * max_Q_s_prime)
            Q_sa[s, a] = Q_sa_update
            error = max(error, abs(old_q - Q_sa[s, a]))
    st.session_state.Q_sa = Q_sa
    st.session_state.iteration_count += 1
    st.write(f"Iteration {st.session_state.iteration_count} completed. Max error: {error}")
    st.rerun()



# Render the environment
st.write("Rendering environment...")
fig = env.render(Q_sa,plot_optimal_policy=True,step_pause=0.5)
# st.write(f"Figure type: {type(fig)}")
if fig:
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # st.write(f"Buffer size: {len(buf.getvalue())} bytes")
    if len(buf.getvalue()) > 0:
        # st.write("Image below:")
        st.image(buf)
    else:
        st.write("Buffer is empty, savefig failed")
else:
    st.write("Figure is None")
  

# Optional: Show Q-values if available
if st.checkbox("Show Q-values "):
    fig_q = env.render(Q_sa=Q_sa, plot_optimal_policy=True, step_pause=0)
    buf_q = BytesIO()
    fig_q.savefig(buf_q, format='png')
    buf_q.seek(0)
    st.image(buf_q)