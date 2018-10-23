[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project, illustrates how to train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

Requirements to run the environment:
- Python 3
- unityagents (see below to install)
- numpy
- torch
- random
- collections
- matplotlib

### Code organization

The code to train & evaluate the agent is divided into 3 main scripts:
-**model.py**: Defines the QNetwork to use, with desired # of inputs, outputs & neurons per layer
-**dqn_agent.py**: Defines the class *ReplayBuffer* to store the experiences & sample them for training. Also defines the class *Agent*, in charge of simulating & training the response of the environment through the QNetworks. Parameters like *BUFFER_SIZE*, *BATCH_SIZE*, *GAMMA*, *TAU*, *LR* or *UPDATE_EVERY* should be changed here.
-**Banana_Collector_DQN_NO_Pixels_TRAIN.py**: Defines dqn function to train the Agent to navigate through the environment. Parameters like *n_episodes*, *eps_start*, *eps_end*, *eps_decay* should be changed here.


### Instructions

