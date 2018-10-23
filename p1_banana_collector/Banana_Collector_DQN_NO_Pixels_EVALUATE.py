#!/usr/bin/env python
# coding: utf-8

# # Banana Collector - EVALUATE
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


# In[ ]:


env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")


# In[ ]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# In[ ]:


from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)


# In[ ]:


# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('final_model_weights.pth'))

for i in range(100):
    print("Start of episode")
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    for j in range(10000):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        if done:
            break 


# In[ ]:


env.close()

