# -*- coding: utf-8 -*-
"""hrl_env.ipynb
Original file is located at
    https://colab.research.google.com/drive/1thnWxbviGs7KOVZqSCiPb8ZaGHxVMPoY

# Addiction Simulator
Homeostatic Reinforcement Learning Simulator 
created by Robert S. Sullivan in 2023 using Python 3.8.10.
Adapted from Mehdi Keramati's HomeoSim. Keramati's simulator 
was called *escalation of cocaine-seeking in the homeostatic 
reinforcement learning framework*. It was created by Keramati 
in March 2013 using Python 2.6 to mimic experimental cocaine data.
This simulator incoporates open ai gym to allow deep Q learning, 
which is a value based off policy reinforcement learning technique
that makes use of deep neural networks for value approximation.

## Import Libraries
"""

#basic
from os import system, name
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

#rich library for Terminal UI
#from rich.jupyter import print
from rich import print
#from rich.prompt import IntPrompt

from IPython.display import clear_output

"""## Simulation"""

class HRLSim(gym.Env):
    metadata = {"render_modes": ["notebook", "script"]}

    def __init__(self, render_mode=None, time_hours=4):
        #Agent's Actions
        #self.actions = ['Sleep', 'Binge on Internet', 'Work', 'Exercise', 'Socialise', 'Drink Alcohol', 'Smoke', 'Take Cocaine'] #8 actions
        self.actions = ['Do Nothing', 'In-Active Lever', 'Active Lever']
        self.action_space = spaces.Discrete(3) #'Do Nothing', 'In-Active Lever', 'Active Lever'
       
        # Mehdi Keramati's definition of an animal ref Mehdi Keramati's Homeostatic RL sim of addiction
        ## fatigue or lever/action cost
        self.fatigue = -1

        #Keramati's outcome, e.g cocaine = 50, representing the dose of self-administered drug,
        self.outcome = [0,0,50] #all others were scaled from cocaine as their assumed impact on the brain. #[0,10,5,12.5,12.5,10,25,50]#self.outcome = [0,0,0,0,0,0,0,50] #all others were scaled from cocaine as their assumed impact on the brain. #[0,10,5,12.5,12.5,10,25,50]
        self.outcomeBuffer = 0
        self.epochs_inactive = 4 # python index starts at 0 so (1+4)* 4 = 5*4 = 20sec lever will be active for 4 seconds then disabled for 20secs, on the 20th second it will be active.
        self.outcome_to_disable = []

        ## Homeostatic System
        self.initialInState = 0
        self.initialSetpoint = 200
        self.inStateLowerBound = 0
        self.outcomeDegradationRate = 0.007 # e.g dose of cocaine that the animal loses in every time-step
        self.outcomeAbsorptionRatio = 0.12 # e.g proportion of the injected cocaine that affects the brain right after infusion
        self.estimatedNonHomeostaticReward = 0.0

        ## Allostatic (Stress) System
        self.setpointShiftRate = 0.0018
        self.setpointRecoveryRate = 0.00016
        self.optimalInStateLowerBound = 100
        self.optimalInStateUpperBound = 200

        ## Drive Function
        self.m = 3 # Parameter of the drive function : m-th root
        self.n = 4 # Parameter of the drive function : n-th pawer
        self.driveReductionReward = 0.0

        ## Goal-directed system
        self.updateRewardRate = 0.2  # Learning rate for updating the non-homeostatic reward function
        self.updateOutcomeRate = 0.2  # Learning rate for updating the outcome function
        
        #agent's sensors or observation space
        self.last_action = 0 #c(external state /exited)
        self.internal_state = float(self.initialInState) #internal variable that moves homeostatic setpoint
        self.setpoint_S = float(self.initialSetpoint) #homeostatic setpoint

        
        # the mean score curve (sliding window of the rewards) with 
        # respect to time.
        self.total_reward = []

        #Environment: the agent's grid world is elapsing time from 1 to endpoint
        self.total_time = time_hours # total time in hours (size of the world) e.g 1 year or 8760hrs, or 4 (1hr pretraining and 3 hours seeking cocaine)
        self.total_epochs = (self.total_time*3600)/4 #e.g 20 secs is 5 trials or epochs of time or 1 hour = 3600secs/4secs = 900 epochs, 8760hrs is 7,884,000 epochs
        self.current_epoch = 0 # e.g 1=4secs
        self.epochs_left = 0.0

        # reward agent wants to maximise this will be the homeostatic reward
        self.reward = 0.0

        self.observation_space = spaces.Box(self.internal_state, self.setpoint_S, dtype=np.float32)
        #to fix WARN: The obs returned by the `reset()` method is not within the observation space.
        #self.observation_space = Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))

        #create data table
        self.df = pd.DataFrame(columns=['epoch', 'action', 'epochs_inactive', 'internal_variable', 'homeostatic_setpoint', 'reward', 'score'])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        next_state = np.array([self.internal_state, self.setpoint_S], dtype=np.float32)
        if self.render_mode == "notebook":
            print("Simulation Reset.\n" + 
                  "Total Time: " + str(self.total_time) + " hrs.\n" + 
                  "Total Epochs: " + str(self.total_epochs))
        return next_state, {}

    def step(self, action):
        ## get the time left
        self.epochs_left = int(self.total_epochs - self.current_epoch)

        #openai gym
        action_taken = action
        self.last_action = action_taken #automatic
            
        #update agent brain chemicals after action taken
        if(action_taken == 0): #Do Nothing/Sleep
            #1. update internal sensors or observations of the environment

            ## Update internal state upon consumption  
            interS = self.internal_state + (self.outcome[action_taken] * self.outcomeAbsorptionRatio) - self.outcomeDegradationRate * (self.internal_state - self.inStateLowerBound)
            if interS < self.inStateLowerBound:
                interS = self.inStateLowerBound    
            self.internal_state = interS

            ## Update homeostatic setpoint
            optInS = self.setpoint_S + self.outcome[action_taken]  * self.setpointShiftRate - self.setpointRecoveryRate
            if optInS < self.optimalInStateLowerBound:
                optInS = self.optimalInStateLowerBound
            if optInS > self.optimalInStateUpperBound:
                optInS = self.optimalInStateUpperBound
            self.setpoint_S = optInS

        elif((action_taken != 0) or (action_taken != -1)): #In-Active Lever/Active Lever #Binge on Internet, Work, Exercise, Socialise, Drink Alcohol, Smoke
            #1. determine the next state that the agent fell into from taking action
            #n/a
            
            #2. get Non-Homeostatic reward e.g energy cost or fatigue of doing an action
                ## if the agent is not sleeping then it is doing something that costs energy
                #nonHomeoRew = -self.fatigue
            self.estimatedNonHomeostaticReward = (1.0 - self.updateRewardRate) * self.estimatedNonHomeostaticReward + self.updateRewardRate * (self.fatigue)
            
            #3. get Homeostatically-regulated Reward of doing action (drive reduction)
            d1 = math.pow(math.fabs(math.pow(self.setpoint_S - self.internal_state, self.n*1.0)),(1.0/self.m))
            d2 = math.pow(math.fabs(math.pow(self.setpoint_S - self.internal_state - self.outcome[action_taken], self.n*1.0)),(1.0/self.m))
                #HomeoRew = d1 - d2
            self.driveReductionReward = (1.0 - self.updateOutcomeRate) * self.driveReductionReward + self.updateOutcomeRate * (d1 - d2)
            
            #4. Now calculate agent reward
            #self.reward = values[action] +  transitionProb * ( self.driveReductionReward + self.estimatedNonHomeostaticReward )
            #self.reward = qValue +  transitionProb * ( self.driveReductionReward + self.estimatedNonHomeostaticReward )
            self.reward = self.driveReductionReward + self.estimatedNonHomeostaticReward
            
            #5. update estimated next state
            # n/a
            
            #6. update internal sensors or observations of the environment
            
            ## Update internal state upon consumption
            self.outcomeBuffer = self.outcomeBuffer + self.outcome[action_taken]   
            interS = self.internal_state + (self.outcomeBuffer * self.outcomeAbsorptionRatio) - self.outcomeDegradationRate * (self.internal_state - self.inStateLowerBound)
            if interS < self.inStateLowerBound:
                interS = self.inStateLowerBound    
            self.internal_state = interS

            ## Update homeostatic setpoint
            optInS = self.setpoint_S + (self.outcome[action_taken]  * self.setpointShiftRate) - self.setpointRecoveryRate
            if optInS < self.optimalInStateLowerBound:
                optInS = self.optimalInStateLowerBound
            if optInS > self.optimalInStateUpperBound:
                optInS = self.optimalInStateUpperBound
            self.setpoint_S = optInS 

            ## Update outcome buffer
            self.outcomeBuffer = self.outcomeBuffer * (1 - self.outcomeAbsorptionRatio)
        elif(action_taken == -1):#quit
            return True   #end simulation
        
        #record cumulative reward
        self.total_reward.append(self.reward + self.total_reward[-1] if len(self.total_reward)>0 else self.reward)
        #save step to dataset log
        self.df.loc[self.current_epoch] = [self.current_epoch, self.actions[action], self.epochs_inactive, self.internal_state, self.setpoint_S, self.reward, self.total_reward[-1]]
            
        #check if this is the last round otherwise continue
        if(self.current_epoch >= self.total_epochs):
            done = True   #end simulation
        else:    
            # Updating the last time from the agent to the end time (goal)
            self.current_epoch += 1  #update to next day interval

            #set time when active lever outcome 50 goes to zero
            if(self.epochs_inactive == 4):#lever is active
                self.epochs_inactive = 0 #off for 16 seconds then on the 20th second activate
                self.outcome_to_disable = self.outcome #remember what values were
                self.outcome = [0] * len(self.outcome)
            else:
                self.outcome = self.outcome_to_disable #restore outcome values
                self.epochs_inactive += 1 #count down by 4secs
                
            done = False
        next_state = np.array([self.internal_state, self.setpoint_S], dtype=np.float32)
        return next_state, self.reward, done, False, {}#info

    def render(self, mode='notebook'):
        if self.render_mode == "notebook":
            clear_output(wait=True)
            print("** Current Time: [bold dark_violet]" + str(round((1/900)*(self.current_epoch),2)) + 
                "[/bold dark_violet] hrs, Epoch Left: [bold dark_violet]" + str(round((1/900)*(self.epochs_left),2)) + 
                "[/bold dark_violet] hrs, **\n[bold dark_green]Last Action: " + str(self.actions[self.last_action]) + 
                "[/bold dark_green],\n[bold dark_green]Current Homeostatic Variable: " + str(round(self.internal_state, 4)) + 
                "[/bold dark_green], [bold dark_green]Current Homeostatic Setpoint: " + str(round(self.setpoint_S, 4)) +
                "[/bold dark_green],\n [bold dark_green]Reward Received: " + str(round(self.reward,2)) + 
                "[/bold dark_green], [bold dark_green]Total Score: " + str(round(self.total_reward[-1] if len(self.total_reward)>0 else self.reward,2)))

    def clear(self): 
        """
        This function was taken from https://www.geeksforgeeks.org/clear-screen-python/ to
        allow the terminal to be cleared when changing menus or showing the user important
        messages. It checks what operating system is being used and uses the correct 
        clearing command.
        """
        # for windows 
        if name == 'nt': 
            _ = system('cls') 

        # for mac and linux(here, os.name is 'posix')
        else: 
            _ = system('clear')

    def close(self):
        #plot our results

        # Filter the data to only include 'Active Lever' actions
        active_lever_data = self.df[self.df['action'] == 'Active Lever']
        active_lever_data['active_lever'] = active_lever_data['action'].apply(lambda x: 1 if x == 'Active Lever' else 0)

        # Convert epoch values to seconds
        active_lever_data['time_sec'] = active_lever_data['epoch'] * 4 / 60

        # Create a bar chart of the 'Active Lever' action occurrence as a time series
        plt.bar(active_lever_data['time_sec'], active_lever_data['active_lever'])

        #plot
        plt.xlabel('Time (min)')
        plt.ylabel('Infusion')
        plt.title('20 sec lever time out')

        # Set x-axis limits to 0 and 40 minutes
        #plt.xlim([0, 35])

        plt.ylim([0, 1.5])
        # Set y-tick values and labels
        plt.yticks([0, 1], ['0', '1'])

        plt.show()

        plt.title("Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.plot(self.df["reward"])
        plt.show()

        plt.title("Cumulative Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.plot(self.df["score"])
        plt.show()

        plt.title("Internal_variable [e.g Dopamine]")
        plt.xlabel("Epoch")
        plt.ylabel("Internal Variable")
        plt.plot(self.df["internal_variable"])
        plt.show()

        plt.title("Homeostatic Setpoint")
        plt.xlabel("Epoch")
        plt.ylabel("Homeostatic Setpoint")
        plt.plot(self.df["homeostatic_setpoint"])
        plt.show()