#!/usr/bin/env python
# coding: utf-8

# # Banana Collector
# 
# ---
# 
# ### 1. Load the necessary packages
# 
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import numpy as np

import random
import torch
from collections import deque
import matplotlib.pyplot as plt

# ### 2. Load the necessary packages
# 
# Next, we will start the environment! If you want to run the code on a different executable, change the path below!:

# In[3]:


env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")


# Load the external brain of the environment to run it from Python

# In[4]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 3. Launch a DQN Agent to learn the environment
# 
# Instantiate the Agent with the input & ouput spaces required for this environment & launch the dqn algorithm to learn the weights for the fully connected networks

# In[ ]:


from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)


# In[ ]:


def dqn(n_episodes=10000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    dones = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True: #for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                dones.append(1)
                break 
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'final_model_weights.pth')
            break
    print(np.sum(dones))
    return scores

scores = dqn()


# When finished, you close the environment.

# In[ ]:


env.close()

