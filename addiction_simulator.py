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

#rich library for Terminal UI
from rich import print
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.prompt import Prompt
from rich.prompt import IntPrompt
from rich.prompt import Confirm]

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
        # Agent Brain - a neural network that represents our Q-function
        self.agent = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
        #Agent's Actions
        self.agent_actions = ['Binge on Internet', 'Work', 'Exercise', 'Socialising', 'Drink Alcohol', 'Smoke'] #6 actions
        
        #Agent Sensors
        ##ref habits of a happy brain
        ## Happy Chemicals
        self.agent_serotonin = 0 #agent feeling of self achievement
        self.agent_oxytocin = 0 #rewarding agent for being social
        self.agent_dopamine = 0 #agent gets going after a reward
        self.agent_endorphins = 0 #agent gets for pushing through physical pain at different times

        ### Unhappy Chemicals
        self.agent_cortisol = 0 #stress hormone which makes agent feel uncomfortable and wants to do something
        
        # the mean score curve (sliding window of the rewards) with 
        # respect to time.
        self.scores = []

        #the agent's environment
        self.end_day = 365 # day simulation ends
        self.end_hour = 24 # hour simulation ends
        self.current_day = 1 # day agent starts
        self.current_hour = 1 # hour agent starts

        # temporary. this will change
        self.reward_received = 0 # agent wants to maximise this score

        #habituation will reduce the experience that makes the agent happy because the action is new
        self.habituation = np.zeros((self.end_day,self.end_hour)) # initializing the habituation array with only zeros.

    def next_time_interval(self):
        ## get the time left
        day = self.end_day - self.current_day #difference in current day and end day
        hour = self.end_hour - self.current_hour #difference in current hour and end hour
        time_left = day*24 + hour

        ## agent input state vector, composed of the five brain signals received by being in the environment
        current_state = [self.agent_serotonin, self.agent_oxytocin, self.agent_dopamine, self.agent_endorphins, self.agent_cortisol]
        action_to_take = self.agent.update(self.reward_received, current_state) # playing the action from the ai (dqn class)
        self.scores.append(self.agent.score()) # appending the score (mean of the last 100 rewards to the reward window)

        #This is the limbic system to say what action to take.
        #we can take the agent's NN and call forward to output q values for each state.
        suggested_action = self.agent_actions[action_to_take]
        sa = int(1 if suggested_action=="Bing on Internet" else 2 if suggested_action=="Work" else 3 if suggested_action=="Exercise" else 4 if suggested_action=="Socialise" else 5 if suggested_action=="Drink Alcohol" else 6 if suggested_action=="Smoke" else 0)
        #give user project options
        print("** Current Day: [bold dark_violet]" + int(self.current_day) + "[/bold dark_violet], Current Hour: " + int(self.current_hour) + " ** \n")
        
        print("\n1. [bold dark_violet]Binge on Internet[/bold dark_violet]\n2. [bold dark_violet]Work[/bold dark_violet]\n3. [bold dark_violet]Exercise[/bold dark_violet]\n4. [bold dark_violet]Socialise[/bold dark_violet]\n5. [bold dark_violet]Drink Alcohol[/bold dark_violet]\n6. [bold dark_violet]Smoke[/bold dark_violet]\n")
        action_taken = 0
        action_taken = IntPrompt.ask("Choose from 1 to 6", default=sa)

        #update agent brain chemicals after action taken
        if(action_taken == 'Binge on Internet'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1
        elif(action_taken == 'Work'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
        elif(action_taken == 'Exercise'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1
        elif(action_taken == 'Socialise'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1
        elif(action_taken == 'Drink Alcohol'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1
        elif(action_taken == 'Smoke'):
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1
        else:
            self.agent_serotonin = 0
            self.agent_oxytocin = 0
            self.agent_dopamine = 0
            self.agent_endorphins = 0
            self.agent_cortisol = 0
            self.reward_received = -1 # agent gets bad reward -1

        if(time_left <= 0): #end simulation
            pass 

        #reward and punishment conditions
        if(self.habituation[0,0] > 0):
            self.reward_received = -1 # and reward = -1
        else:
            #otherwise
            self.reward_received = -0.2 # and it gets bad reward (-0.2)
            if current_hours_elapsed < previous_hours_elapsed: # however if it getting close to the goal
                self.reward_received = 0.1 # it still gets slightly positive reward 0.1

        # Updating the last time from the agent to the end time (goal)
        self.current_day += 1  #update to next day interval
        self.current_time += 1 #update to next hour interval
        previous_hours_elapsed = current_hours_elapsed