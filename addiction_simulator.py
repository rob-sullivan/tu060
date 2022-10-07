# %% [markdown]
# # Addiction Simulator

# %% [markdown]
# ## Setup Virtual Environment
# 1. make sure you have a python virtual environment setup
# 2. python -m pip install --upgrade pip setuptools virtualenv
# 3. python -m venv venv (this creates a virtual environment called venv)
# 4. add /venv/ to .gitignore.
# 5. activate virtual environment with \venv\Scripts\activate.bat on windows or source kivy_venv/bin/activate on mac+linux

# %% [markdown]
# # Install Packages
# When inside python virtual enironment install the following packages:
# 1. pip install numpy
# 2. pip install matplotlib
# 3. pip install rich
# 4. pip install torch

# %% [markdown]
# ## Import Libraries

# %%
#basic
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

#pytorch for gpu processing of ML model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# %% [markdown]
# ## Deep Q-Learning Agent

# %% [markdown]
# ### Neural Network Model

# %%
class Network(nn.Module):  
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# %% [markdown]
# ### Experience Replay Model
# This model is used for training our DQN model. It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from it randomly, the transitions that build up a batch are decorrelated. It has been shown that this greatly stabilizes and improves the DQN training procedure.

# %%
class ReplayMemory(): 
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# %% [markdown]
# ### DQN Ensemble

# %%
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

# %% [markdown]
# ## Environment
class Environment():
    def __init__(self, input_size, nb_action, gamma):
        # Getting our AI, which we call "agent", that contains 
        # our neural network that represents our Q-function
        self.agent_actions = [0,20,-20] #3 actions
        self.agent = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9 
        
        # the mean score curve (sliding window of the rewards) with 
        # respect to time.
        self.scores = []

        #the agent's environment
        self.end_day = 365 #goal
        self.end_hour = 24 #goal
        self.current_day = 1
        self.current_time = 1

        # temporary. this will change
        self.reward_received = 0

        #ref https://fourminutebooks.com/habits-of-a-happy-brain-summary/
        self.serotonin = 0 #feeling of self achievement
        self.oxytocin = 0 #rewarding you for being social
        self.dopamine = 0 #going after a reward
        self.endorphins = 0 #pushing through physical pain at different times

        #unhappy chemicals protect us from harm by warning us of potential threats.
        self.cortisol = 0 #stress hormone which makes you feel uncomfortable and wants you to do something

        #habituation is not an experience that makes you most happy because it’s new
        self.habituation = np.zeros((self.end_day,self.end_hour)) # initializing the habituation array with only zeros.

    def progress_time(self):
        day = self.end_day - self.current_day
        hour = self.end_hour - self.current_time