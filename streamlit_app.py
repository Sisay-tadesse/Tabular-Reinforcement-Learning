import streamlit as st
import sys
import os
sys.path.insert(0, '/vol/home/s4184343/Desktop/Reinforcement_learning/Assignment1_2026/Code_assignment')

from Environment import StochasticWindyGridworld
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

st.title("Stochastic Windy Gridworld RL Environment")

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = StochasticWindyGridworld(initialize_model=True)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0

env = st.session_state.env
state = st.session_state.state
done = st.session_state.done
step_count = st.session_state.step_count

st.write(f"Current State: {state}, Steps: {step_count}, Done: {done}")

# Action buttons
col1, col2, col3, col4 = st.columns(4)
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

# Reset button
if st.button("Reset"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.session_state.step_count = 0
    st.rerun()

# Render the environment
st.write("Rendering environment...")
fig = env.render(step_pause=0)
st.write(f"Figure type: {type(fig)}")
if fig:
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.write(f"Buffer size: {len(buf.getvalue())} bytes")
    if len(buf.getvalue()) > 0:
        st.write("Image below:")
        st.image(buf)
    else:
        st.write("Buffer is empty, savefig failed")
else:
    st.write("Figure is None")
  

# # Optional: Show Q-values if available
# if st.checkbox("Show Q-values (random for demo)"):
#     Q_sa = np.random.rand(env.n_states, env.n_actions)
#     fig_q = env.render(Q_sa=Q_sa, plot_optimal_policy=True, step_pause=0)
#     buf_q = BytesIO()
#     fig_q.savefig(buf_q, format='png')
#     buf_q.seek(0)
#     st.image(buf_q)