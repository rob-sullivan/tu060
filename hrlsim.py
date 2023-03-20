"""Homeostatic Reinforcement Learning Simulator 
created by Robert S. Sullivan in 2023 using Python 3.8.10.
Adapted from Mehdi Keramati's HomeoSim. Keramati's simulator 
was called escalation of cocaine-seeking in the homeostatic 
reinforcement learning framework. It was created by Keramati 
in March 2013 using Python 2.6 to mimic experimental cocaine data.
This simulator incoporates object orientated programming and deep
Q learning, a value based off policy reinforcement learning technique
that makes use of deep neural networks for value approximation.
"""

import os
from os import system, name
import random
import time
import numpy as np #Keramati's simulator imported as numpy. changed for consistency
import matplotlib.pyplot as plt

#pytorch for gpu processing of ML model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#rich library for Terminal UI
#from rich.jupyter import print
#from rich import print
#from rich.prompt import IntPrompt

#used for Keramati's simulator
import scipy
#import numpy
import pylab
import cmath

#hide pytorch warnings (should eventually be resolved)
import warnings
warnings.filterwarnings("ignore")

## Deep Q-Learning Agent Architecture
### Neural Network Value Approximator
class Network(nn.Module):
    """This model is a feed forward neural network"""
    def __init__(self, input_size, nb_action):
        #ref: https://discuss.pytorch.org/t/super-model-in-init/97426
        #super(Network, self).__init__()
        super().__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

### Experience Replay
class ReplayMemory():
    """This model is used for training our DQN model. 
    It stores the transitions that the agent observes, 
    allowing us to reuse this data later. By sampling 
    from it randomly, the transitions that build up a 
    batch are decorrelated. 
    It has been shown that this greatly stabilizes 
    and improves the DQN training procedure.
    """
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

### DQN Ensemble
class Dqn():
    """This is our Deep Q-Learning Agent Architecture"""
    def __init__(self, input_size, nb_action, gamma, alpha=0.001, memory=100000, temp=100, samp=100, reward_max=1000):
        self.gamma = gamma
        self.temperature = temp
        self.experience_sample_size = samp
        self.reward_window = []
        self.reward_window_max_size = reward_max
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(memory)
        self.optimizer = optim.Adam(self.model.parameters(), lr = alpha)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    
    def select_action(self, state):
        """This uses softmax to convert outputted q value 
        numbers from the neural network into a probability distribution.
        Temperature value = 100. Closer to zero the less sure the NN will 
        be to taking the action. e.g actions[1,2,3] = prob[0.04, 0.11, 0.85]. 
        Temperature increases 0.85 value to be selected"""
        probs = F.softmax(self.model(Variable(state, volatile = True))*self.temperature) # T=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]#convert from pytorch tensor to action
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """To train the Agent, we use forward propagation then backpropagation.
        We forward input data through the network to get our output and target then 
        we compare our output to the target to compute the loss error.
        This error is backproagated into the nn and we use stochastic gradient descent 
        to update the weights according to how much they contributed to that loss error.
        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #gathers samples of output q-values from each row of the nn along a dimension
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # this detaches values into a sample that doesn't receive gradient or backpropagation.
        target = self.gamma * next_outputs + batch_reward #create a target q value using the bellman equation
        td_loss = F.smooth_l1_loss(outputs, target) #calculate our losses from what the nn thinks q value should be vs actual
        self.optimizer.zero_grad() #zero gradient is a pytorch parameter to ensure we don't use old gradient with our new one in backpropagation and get wrong minimums.
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > self.experience_sample_size: #e.g size 100
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.experience_sample_size) #e.g size 100
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > self.reward_window_max_size: #e.g size 1000
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
            print("saved !")
        else:
            print("no checkpoint found...")


class HomeoRLEnv():
    """Homeostatic Reinforcement Learning Simulator 
    created by Robert S. Sullivan in 2023 using Python 3.8.10.
    Adapted from Mehdi Keramati's HomeoSim. Keramati's simulator 
    was called escalation of cocaine-seeking in the homeostatic 
    reinforcement learning framework. It was created by Keramati 
    in March 2013 using Python 2.6 to mimic experimental cocaine data.
    """
    def __init__(self):

        """
        #ros - Create a ShA DQL Agent
        s = self.state.shape[0]#get shape of np array sensors. i.e external state, internal state, setpoint, and trial
        a = self.actionsNum #get number of actions
        g = self.gamma #get discount rate
        l = 0.2 #learning rate taken from goal directed system
        t = 100 #temperature value. e.g if actions[1,2,3] = prob[0.04, 0.11, 0.85] then temperature will increase change of 0.85 value to be selected
        r = 1000 #reward window max. collecting rewards as agent progresses through environment but window is emptied if over this value
        #ros - Agent Brain, a neural network that represents our Q-function
        self.ShA_DQN_agent = Dqn(s, a, g, l, t , r) # e.g 5 sensors, 6 actions, gamma = 0.9, temperature
        """
        """
        #ros - Create a LgA DQL Agent
        s = self.state.shape[0]#get shape of np array sensors. i.e external state, internal state, setpoint, and trial
        a = self.actionsNum #get number of actions
        g = self.gamma #get discount rate
        l = 0.2 #learning rate taken from goal directed system
        t = 100 #temperature value. e.g if actions[1,2,3] = prob[0.04, 0.11, 0.85] then temperature will increase change of 0.85 value to be selected
        r = 1000 #reward window max. collecting rewards as agent progresses through environment but window is emptied if over this value

        #ros - Agent Brain, a neural network that represents our Q-function
        self.LgA_DQN_agent = Dqn(s, a, g, l, t, r) # e.g 5 sensors, 6 actions, gamma = 0.9, temperature = 100
        """

        #set initial addiction values, leverpress values, number of states, actions, etc.
        self.setupEnvironment()
        #setup Homeostatic System, Allostatic (Stress) System, Drive Function and Goal-directed system
        self.setupAnimal()

    #Simulated Experiments
    def pretraining(self, ratType):
        """Pre-training Sessions"""

        """
        #ros - set cumulated score and rewards so far
        # used to plot performance. The mean score curve (sliding window of the rewards) with respect to time.
        scores = []
        #score agent wants to maximise
        reward_received = 0.0
        """

        exState = self.state[0]
        inState = self.state[1]
        setpointS = self.state[2]
        trialCount = self.state[3]

        """
        #ros - get current state external state, internal state, setpoint, and trial
        current_state = [exState, inState, setpointS, trialCount]
        """

        cocBuffer = 0
        
        trialsNum = self.pretrainingTrialsNum
        
        for trial in range(0,trialsNum):
            """
            # ros - playing the action from the ai (dqn class)
            if ratType=='ShA':
                action = self.ShA_DQN_agent.update(reward_received, current_state)
            elif ratType=='LgA':
                action = self.LgA_DQN_agent.update(reward_received, current_state)
            """

            estimatedActionValues = self.valueEstimation(exState,inState,setpointS, self.searchDepth)
            action = self.actionSelectionSoftmax(exState,estimatedActionValues) #ros - removed non DQL agent
            nextState = self.getRealizedTransition(exState,action)
            out = self.getOutcome(exState,action,nextState)
            nonHomeoRew = self.getNonHomeostaticReward(exState,action,nextState)
            HomeoRew = self.driveReductionReward(inState,setpointS,out)

            if ratType=='ShA':  
                self.loggingShA (trial,action,inState,setpointS,out)    
                print("ShA rat number: %d / %d \t Pre-training session \t trial: %d / %d \t animal seeking cocaine" %(self.animal+1,self.animalsNum,trial+1,trialsNum))
            elif ratType=='LgA':  
                self.loggingLgA (trial,action,inState,setpointS,out)    
                print("LgA rat number: %d / %d \t Pre-training session \t trial: %d / %d \t animal seeking cocaine" %(self.animal+1, self.animalsNum,trial+1,trialsNum))

            self.updateOutcomeFunction(exState, action, nextState, out)
            self.updateNonHomeostaticRewardFunction(exState, action, nextState, nonHomeoRew)
            self.updateTransitionFunction(exState, action, nextState)            
            
            cocBuffer = cocBuffer + out                
            
            inState     = self.updateInState(inState, cocBuffer * self.cocAbsorptionRatio)
            setpointS   = self.updateSetpoint(setpointS,out)

            cocBuffer = cocBuffer*(1 - self.cocAbsorptionRatio)

            exState   = nextState

            """
            #ros - add scores and update to next state
            # appending the score for DQN agent(mean of the last 100 rewards to the reward window)
            if ratType=='ShA':
                scores.append(self.ShA_DQN_agent.score())
            elif ratType=='LgA':
                scores.append(self.LgA_DQN_agent.score())
            
            #reward or punish DQN agent
            reward_received = HomeoRew + nonHomeoRew

            #get next state for DQN agent
            current_state = [exState, inState, setpointS, trialCount+trialsNum]
            """

        """
        #ros - get current state external state, internal state, setpoint, and trial
        current_state = [exState, inState, setpointS, trialCount+trialsNum]
        """
        
        self.state[0] = exState
        self.state[1] = inState
        self.state[2] = setpointS
        self.state[3] = trialCount+trialsNum

        """
        #ros show results of DQN agent
        if ratType=='ShA':  
            plt.title("Pretrained ShA DQL, Cocaine Seeking: Scores")   
        if ratType=='LgA':  
            plt.title("Pretrained LgA DQL, Cocaine Seeking: Scores")
        plt.xlabel("Epochs (Trials)")
        plt.ylabel("Non-Homeostatic Reward")
        plt.plot(scores)
        plt.show()
        """
        return 

    def cocaineSeeking(self, sessionNum, ratType):
        """Cocaine Seeking Session"""
        """
        #ros - used to plot performance. The mean score curve (sliding window of the rewards) with respect to time.
        scores = []

        #ros - score agent wants to maximise
        reward_received = 0.0
        #ros - external state, internal state, setpoint, and trial
        current_state = [state[0], state[1], state[2], state[3]]
        """

        exState     = self.state[0]
        inState     = self.state[1]
        setpointS   = self.state[2]
        trialCount  = self.state[3]
        cocBuffer   = 0
        
        if ratType=='ShA':  
            trialsNum = self.seekingTrialsNumShA    
        if ratType=='LgA':  
            trialsNum = self.seekingTrialsNumLgA    
        
        for trial in range(trialCount,trialCount+trialsNum):
            """
            # ros - playing the action from the ai (dqn class)
            action = dqn_agent.update(reward_received, current_state)
            """
            estimatedActionValues= self.valueEstimation(exState,inState,setpointS, self.searchDepth)
            action = self.actionSelectionSoftmax(exState,estimatedActionValues)
            nextState = self.getRealizedTransition(exState,action)
            out = self.getOutcome(exState,action,nextState)
            nonHomeoRew = self.getNonHomeostaticReward(exState,action,nextState)
            HomeoRew = self.driveReductionReward(inState,setpointS,out)

            if ratType=='ShA':  
                self.loggingShA(trial,action,inState,setpointS,out)    
                print("ShA rat number: %d / %d \t Session Number: %d / %d \t trial: %d / %d \t animal seeking cocaine" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum, trial-trialCount+1, trialsNum))
            if ratType=='LgA':  
                self.loggingLgA(trial,action,inState,setpointS,out)    
                print("LgA rat number: %d / %d \t Session Number: %d / %d \t trial: %d / %d \t animal seeking cocaine" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum, trial-trialCount+1, trialsNum))

            self.updateOutcomeFunction(exState,action,nextState,out)
            self.updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
            self.updateTransitionFunction(exState,action,nextState)            
            
            cocBuffer = cocBuffer + out                
            
            inState     = self.updateInState(inState, cocBuffer * self.cocAbsorptionRatio)
            setpointS   = self.updateSetpoint(setpointS,out)

            cocBuffer = cocBuffer * (1 - self.cocAbsorptionRatio)

            exState   = nextState

            """
            #ros - appending the score for DQN agent(mean of the last 100 rewards to the reward window)
            scores.append(dqn_agent.score())

            #ros - get next state for DQN agent
            current_state = [state[0], state[1], state[2], state[3]]
            
            #ros - reward or punish DQN agent
            reward_received = HomeoRew + nonHomeoRew
            """
        self.state[0] = exState
        self.state[1] = inState
        self.state[2] = setpointS
        self.state[3] = trialCount+trialsNum

        """
        #ros - get current state external state, internal state, setpoint, and trial
        current_state = [state[0], state[1], state[2], state[3]]

        #show results of DQN agent
        if ratType=='ShA':  
            plt.title("Pretrained ShA DQL, Cocaine Seeking: Scores")   
        if ratType=='LgA':  
            plt.title("Pretrained LgA DQL, Cocaine Seeking: Scores")
        plt.xlabel("Epochs (Trials)")
        plt.ylabel("Non-Homeostatic Reward")
        plt.plot(scores)
        plt.show()
        """
        return

    def homeCage(self, sessionNum, ratType):
        """Home-cage Sessions"""
        exState = self.state[0]
        inState = self.state[1]
        setpointS = self.state[2]
        trialCount = self.state[3]
    
        if ratType=='ShA':  
            trialsNum = self.restTrialsNumShA    
            print("ShA rat number: %d / %d \t Session Number: %d / %d \t animal rests in home cage" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum))
        elif ratType=='LgA':  
            trialsNum = self.restTrialsNumLgA
            print("LgA rat number: %d / %d \t Session Number: %d / %d \t animal rests in home cage" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum))
        elif ratType=='afterPretrainingShA':  
            trialsNum = self.restAfterPretrainingTrialsNum    
            print("ShA rat number: %d / %d \t After pretraining \t animal rests in home cage" %(self.animal+1, self.animalsNum))
        elif ratType=='afterPretrainingLgA':  
            trialsNum = self.restAfterPretrainingTrialsNum    
            print("LgA rat number: %d / %d \t After pretraining \t animal rests in home cage" %(self.animal+1, self.animalsNum))
        
        for trial in range(trialCount,trialCount+trialsNum):

            inState     = self.updateInState(inState,0)
            setpointS   = self.updateSetpoint(setpointS,0)

            if ratType=='ShA':  
                self.loggingShA(trial,0,inState,setpointS,0)    
            elif ratType=='LgA':  
                self.loggingLgA(trial,0,inState,setpointS,0)    
            elif ratType=='afterPretrainingShA':  
                self.loggingShA(trial,0,inState,setpointS,0)    
            elif ratType=='afterPretrainingLgA':  
                self.loggingLgA(trial,0,inState,setpointS,0)    

        self.state[0]    = exState
        self.state[1]    = inState
        self.state[2]    = setpointS
        self.state[3]    = trialCount+trialsNum

        return

    def extinction(self, trialsNum, ratsType):
        """Extinction Sessions. No drugs given 
        no matter what lever is pressed"""
        exState = self.state[0]
        inState = self.state[1]
        setpointS = self.state[2]
        trialCount = self.state[3]
        cocBuffer = 0
        
        for trial in range(trialCount,trialCount+trialsNum):
            
            estimatedActionValues = self.valueEstimation(exState, inState, setpointS, self.searchDepth)
            action = self.actionSelectionSoftmax( exState, estimatedActionValues)
            nextState = self.getRealizedTransition( exState, action)
            out = 0
            nonHomeoRew = self.getNonHomeostaticReward( exState, action, nextState)
            HomeoRew = self.driveReductionReward( inState, setpointS, out)
            
            if ratsType == 'ShA':
                self.loggingShA(trial,action,inState,setpointS,out)    
                print("ShA rat number: %d / %d \t Extinction session \t trial: %d / %d \t Extinction of cocaine seeking" %(self.animal+1, self.animalsNum,trial-trialCount+1,trialsNum))
            if ratsType == 'LgA':
                self.loggingLgA(trial,action,inState,setpointS,out)    
                print("LgA rat number: %d / %d \t Extinction session \t trial: %d / %d \t Extinction of cocaine seeking" %(self.animal+1, self.animalsNum,trial-trialCount+1,trialsNum))

            self.updateOutcomeFunction( exState, action, nextState, out)
            self.updateNonHomeostaticRewardFunction( exState, action, nextState, nonHomeoRew)
            self.updateTransitionFunction( exState, action, nextState)            
            
            cocBuffer = cocBuffer + out                
            
            inState = self.updateInState( inState, cocBuffer * self.cocAbsorptionRatio )
            setpointS = self.updateSetpoint( setpointS, out )

            cocBuffer = cocBuffer * ( 1- self.cocAbsorptionRatio )

            exState   = nextState

        self.state[0] = exState
        self.state[1] = inState
        self.state[2] = setpointS
        self.state[3] = trialCount+trialsNum

        return

    def noncontingentInfusion(self, sessionNum,ratType):
        """Non-contingent cocaine infusion"""
        exState = self.state[0]
        inState = self.state[1]
        setpointS = self.state[2]
        trialCount = self.state[3]

        inState = self.updateInState(inState, self.cocaine)
        setpointS = self.updateSetpoint(setpointS, self.cocaine)
        if ratType == 'ShA':
            self.loggingShA(trialCount,0,inState,setpointS,self.cocaine)    
            print("ShA rat number: %d / %d \t Session Number: %d / %d \t animal receives non-contingent cocaine infusion" %(self.animal+1, self.animalsNum,sessionNum+1, self.sessionsNum))
        if ratType == 'LgA':
            self.loggingLgA(trialCount,0,inState,setpointS,self.cocaine)    
            print("LgA rat number: %d / %d \t Session Number: %d / %d \t animal receives non-contingent cocaine infusion" %(self.animal+1, self.animalsNum,sessionNum+1, self.sessionsNum))
            

        self.state[0] = exState
        self.state[1] = inState
        self.state[2] = setpointS
        self.state[3] = trialCount + 1

        return

    #Utility Functions
    def setTransition(self, state, action, nextState, transitionProbability):
        """Setting the transition function of the MDP"""
        self.transition [state][action][nextState] = transitionProbability
        return 

    def getTransition(self, s,a,nextS):
        """Return the probability of the transitions to 
        the next state s-a->s'
        used in getRealizedTransition() and isActionAvailable()
        """
        return self.transition[s][a][nextS]

    def getRealizedTransition(self, state,action):
        """Return the next state that the animal fell into"""
        index = np.random.uniform(0,1)
        probSum = 0
        for nextS in range(0, self.statesNum):
            probSum = probSum + self.getTransition(state,action,nextS)
            if index <= probSum:
                return nextS    
            
    def updateTransitionFunction(self, state,action,nextState):
        '''Update the expected-transition function'''
        #---- First inhibit all associations
        for i in range(0, self.statesNum):
            self.estimatedTransition[state][action][i] = (1.0 - self.updateTransitionRate) * self.estimatedTransition[state][action][i]
        
        #---- Then increase the effect of experiences association
        self.estimatedTransition[state][action][nextState] = self.estimatedTransition[state][action][nextState] + self.updateTransitionRate
                
        return

    def setOutcome(self, state,action,nextState,out):
        """Setting the outcome function of the MDP"""
        self.outcome [state][action][nextState] = out
        return 

    def getOutcome(self, state,action,nextState):
        """Obtained outcome"""
        return self.outcome[state,action,nextState]

    def updateOutcomeFunction(self, state,action,nextState,out):
        """Update the expected-outcome function"""
        self.estimatedOutcome[state][action][nextState] = (1.0 - self.updateOutcomeRate) * self.estimatedOutcome[state][action][nextState] + self.updateOutcomeRate * out
        return

    def setNonHomeostaticReward(self, state,action,nextState,rew):
        """Setting the non-homeostatic reward function of the MDP"""
        self.nonHomeostaticReward[state][action][nextState] = rew
        return 

    def getNonHomeostaticReward(self, state,action,nextState):
        """Obtained non-homeostatic reward"""
        #agent can compute two different value functions for the cocaine and noncocaine MDPs
        #t he weighted average of these two values is then used as the overall value assigned to a state-action pair
        return self.nonHomeostaticReward [state][action][nextState] 

    def updateNonHomeostaticRewardFunction(self, state,action,nextState,rew):
        """Update the expected-non-homeostatic-reward function""" 
        self.estimatedNonHomeostaticReward[state][action][nextState] = (1.0 - self.updateRewardRate) * self.estimatedNonHomeostaticReward[state][action][nextState] + self.updateRewardRate * rew
        return

    def driveReductionReward(self, inState, setpointS, outcome):
        """Homeostatically-regulated Reward"""
        d1 = np.power(np.absolute(np.power(setpointS-inState, self.n*1.0)),(1.0/self.m))
        d2 = np.power(np.absolute(np.power(setpointS-inState-outcome, self.n*1.0)),(1.0/self.m))
        return d1-d2

    def updateInState(self, inState,outcome):
        """Update internal state upon consumption"""
        interS = inState + outcome - self.cocaineDegradationRate*(inState - self.inStateLowerBound)
        if interS < self.inStateLowerBound:
            interS = self.inStateLowerBound
        return interS

    def updateSetpoint(self, optimalInState, out):
        """Update the homeostatic setpoint (Allostatic mechanism)""" 
        optInS = optimalInState + out * self.setpointShiftRate - self.setpointRecoveryRate

        if optInS < self.optimalInStateLowerBound:
            optInS = self.optimalInStateLowerBound

        if optInS > self.optimalInStateUpperBound:
            optInS = self.optimalInStateUpperBound

        return optInS
    
    def setupEnvironment(self):
        # Definition of the Markov Decison Process FR1 - Timeout 20sec
        self.cocaine = 50 # Dose of self-administered drug
        self.nonContingentCocaine = 50
        self.leverPressCost = 1 # Energy cost for pressing the lever

        self.statesNum = 5 # sequence of steps that each take 4 second to complete. 5 states x 4secs = 20secs
        self.actionsNum = 3 # number of action, e.g action 0 = Null, action 1 = Inactive Lever Press, action 2 = Active Lever Press
        self.initialExState = 0 #external state

        self.transition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
        self.outcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.nonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

    def setupAnimal(self):
        # Definition of the Animal
        #------------ Homeostatic System
        self.initialInState = 0
        self.initialSetpoint = 200
        self.inStateLowerBound = 0
        self.cocaineDegradationRate = 0.007 # Dose of cocaine that the animal loses in every time-step
        self.cocAbsorptionRatio = 0.12 # Proportion of the injected cocaine that affects the brain right after infusion 

        #------------ Allostatic (Stress) System
        self.setpointShiftRate = 0.0018
        self.setpointRecoveryRate = 0.00016
        self.optimalInStateLowerBound = 100
        self.optimalInStateUpperBound = 200

        #------------ Drive Function
        self.m = 3 # Parameter of the drive function : m-th root
        self.n = 4 # Parameter of the drive function : n-th pawer

        #------------ Goal-directed system
        self.updateOutcomeRate = 0.2  # Learning rate for updating the outcome function
        self.updateTransitionRate = 0.2  # Learning rate for updating the transition function
        self.updateRewardRate = 0.2  # Learning rate for updating the non-homeostatic reward function
        self.gamma = 1 # Discount factor
        self.beta = 0.25 # Rate of exploration
        self.searchDepth = 3 # Depth of going into the decision tree for goal-directed valuation of choices
        self.pruningThreshold = 0.1 # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree

        #this is the model the agent thinks the world will be vs what it actually becomes (e.g self.transition, self.outcome and self.nonHomeostaticReward)
        self.estimatedTransition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
        self.estimatedOutcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedNonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

        self.state = np.zeros( [4] , int) # the observation of the agent, a vector of the external state, internal state, setpoint, and trial

    def initializeAnimal(self):    
        """Creates a new animal for simulator"""   

        #this is the model the agent thinks the world will be vs what it actually becomes (e.g self.transition, self.outcome and self.nonHomeostaticReward)
        self.estimatedTransition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
        self.estimatedOutcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedNonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

        self.state = np.zeros( [4] , int) # the observation or sensors of the agent, a vector of the external state, internal state, setpoint, and trial


        self.state[0] = self.initialExState
        self.state[1] = self.initialInState
        self.state[2] = self.initialSetpoint 
        self.state[3] = 0 
        
        #initialise the transition table, outcome table and the non-homeostatic-reward table
        for i in range(0, self.statesNum):
            for j in range(0, self.actionsNum):
                for k in range(0, self.statesNum):
                    self.estimatedTransition[i][j][k] = (1.0)/( self.statesNum*1.0)
                    self.estimatedOutcome[i][j][k] = 0.0
                    self.estimatedNonHomeostaticReward[i][j][k] = 0.0
        
        #Assume animal knows its energy cost (fatigue) of pressing a lever 
        for i in range(0, self.statesNum):
            for j in range(0, self.statesNum):
                self.estimatedNonHomeostaticReward[i][1][j] = -self.leverPressCost
        return

    def isActionAvailable(self, state,action):
        """Is action a available is state s?"""
        probSum = 0 ;
        for i in range(0, self.statesNum):
            probSum = probSum + self.getTransition(state,action,i)
        if probSum == 1:
            return 1
        elif probSum == 0:
            return 0
        else:
            print("Error: There seems to be a problem in defining the transition function of the environment")      
            return

    def valueEstimation(self, state,inState,setpointS,depthLeft):
        """Goal-directed Value estimation"""
        values = np.zeros ( [self.actionsNum] , float )

        # If this is the last depth that should be searched :
        if depthLeft==1:
            for action in range(0, self.actionsNum):
                for nextState in range(0, self.statesNum):
                    homeoReward    = self.driveReductionReward(inState,setpointS, self.estimatedOutcome[state][action][nextState])
                    nonHomeoReward = self.estimatedNonHomeostaticReward[state][action][nextState]
                    transitionProb = self.estimatedTransition[state][action][nextState]
                    values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
            return values
        
        # Otherwise :
        for action in range(0, self.actionsNum):
            for nextState in range(0, self.statesNum):
                if self.estimatedTransition[state][action][nextState] < self.pruningThreshold :
                    VNextStateBest = 0
                else:    
                    VNextState = self.valueEstimation(nextState,setpointS,inState,depthLeft-1)
                    VNextStateBest = self.maxValue (VNextState)
                homeoReward = self.driveReductionReward(inState,setpointS, self.estimatedOutcome[state][action][nextState])
                nonHomeoReward = self.estimatedNonHomeostaticReward[state][action][nextState]
                transitionProb = self.estimatedTransition[state][action][nextState]
                values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + self.gamma * VNextStateBest ) 
                
        return values

    def maxValue(self, V):
        """Max ( Value[nextState,a] ) : for all a"""
        maxV = V[0]
        for action in range(0, self.actionsNum):
            if V[action]>maxV:
                maxV = V[action]    
        return maxV

    def actionSelectionSoftmax(self, state,V):
        """Action Selection : Softmax"""
        # Normalizing values, in order to be overflow due to very high values
        maxV = V[0]
        if maxV==0:
            maxV=1        
        for action in range(0, self.actionsNum):
            if maxV < V[action]:
                maxV = V[action]
        for action in range(0, self.actionsNum):
            V[action] = V[action]/maxV

        sumEV = 0
        for action in range(0, self.actionsNum):
            sumEV = sumEV + abs(cmath.exp( V[action] / self.beta ))

        index = np.random.uniform(0,sumEV)

        probSum=0
        for action in range(0, self.actionsNum):
                probSum = probSum + abs(cmath.exp( V[action] / self.beta ))
                if probSum >= index:
                    return action

        print("Error: An unexpected (strange) problem has occured in action selection...")
        return 0

    #Keramati's log and graph functions
    def loggingShA(self, trial,action,inState,setpointS,coca):
        """Logging the current information for the Short-access group"""
        if action==0: 
            self.nulDoingShA[trial] = self.nulDoingShA[trial] + 1
        elif action==1: 
            self.inactiveLeverPressShA[trial] = self.inactiveLeverPressShA[trial] + 1
        elif action==2: 
            self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial] + 1
        self.internalStateShA[trial] = self.internalStateShA[trial] + inState
        self.setpointShA[trial] = self.setpointShA[trial] + setpointS    
        if coca==self.cocaine:
            self.infusionShA[trial] = self.infusionShA[trial] + 1
        return

    def loggingLgA(self, trial,action,inState,setpointS,coca):
        """Logging the current information for the Long-access group"""
        if action==0: 
            self.nulDoingLgA[trial] = self.nulDoingLgA[trial] + 1
        elif action==1: 
            self.inactiveLeverPressLgA[trial] = self.inactiveLeverPressLgA[trial] + 1
        elif action==2:
            self.activeLeverPressLgA[trial] = self.activeLeverPressLgA[trial] + 1
        self.internalStateLgA[trial] = self.internalStateLgA[trial] + inState
        self.setpointLgA[trial] = self.setpointLgA[trial] + setpointS    
        if coca==self.cocaine:
            self.infusionLgA[trial] = self.infusionLgA[trial] + 1
        return

    def loggingFinalization(self):
        """Wrap up all the logged data"""

        for trial in range(0, self.totalTrialsNum):
            self.nulDoingShA[trial] = self.nulDoingShA[trial]/self.animalsNum
            self.inactiveLeverPressShA[trial] = self.inactiveLeverPressShA[trial]/self.animalsNum
            self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial]/self.animalsNum
            self.internalStateShA[trial] = self.internalStateShA[trial]/self.animalsNum
            self.setpointShA[trial] = self.setpointShA[trial]/self.animalsNum  
            self.infusionShA[trial] = self.infusionShA[trial]/self.animalsNum 

            self.nulDoingLgA[trial] = self.nulDoingLgA[trial]/self.animalsNum
            self.inactiveLeverPressLgA[trial] = self.inactiveLeverPressLgA[trial]/self.animalsNum
            self.activeLeverPressLgA[trial] = self.activeLeverPressLgA[trial]/self.animalsNum
            self.internalStateLgA[trial] = self.internalStateLgA[trial]/self.animalsNum
            self.setpointLgA[trial] = self.setpointLgA[trial]/self.animalsNum  
            self.infusionLgA[trial] = self.infusionLgA[trial]/self.animalsNum 
        return

    def plotInternalStateLastSession(self):
        """Plot the internal state of the last session"""
        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
        pylab.rcParams.update({'legend.fontsize': 16})
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)

        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(self.internalStateShA [self.trialsPerDay*self.sessionsNum - 50 : self.trialsPerDay*self.sessionsNum + 500 ] , linewidth = 2.5 , color='black' )
        S1 = ax1.plot(self.internalStateLgA [self.trialsPerDay*self.sessionsNum - 50 : self.trialsPerDay*self.sessionsNum + 500 ] , linewidth = 1.5 , color='black' )
    
        leg = fig1.legend((S1, S0), ('4sec time-out','20sec time-out'), loc = (0.4,0.68))
        leg.draw_frame(False)

        max = 0    
        for i in range ( self.trialsPerDay*self.sessionsNum - 50 , self.trialsPerDay*self.sessionsNum + 500 ):
            if max < self.internalStateLgA[i]:
                max = self.internalStateLgA[i]      
        pylab.ylim((-10 , max + 10))
        pylab.xlim((0,551))
    
        tick_lcs = []
        tick_lbs = []
        for i in range ( 0 , 4 ):
            tick_lcs.append( 50 + i*150 ) 
            tick_lbs.append(i*10)
        pylab.xticks(tick_lcs, tick_lbs)

        for line in ax1.get_yticklines() + ax1.get_xticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

    
        ax1.set_ylabel('Internal State')
        ax1.set_xlabel('Time (min)')
        ax1.set_title('Post-escalation')
        fig1.savefig('internalStateLastSession.png', format='png') #changed eps to png

        return

    def plotInfusionLastSession(self):
        """Plot the infusions for the last session"""
        #---------------------------------ShA rats
            
        fig1 = pylab.figure( figsize=(5,2) )
        fig1.subplots_adjust(top=0.65)
        fig1.subplots_adjust(bottom=0.3)
        fig1.subplots_adjust(left=0.16)

        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(self.infusionShA [self.trialsPerDay*self.sessionsNum - 50 : self.trialsPerDay*self.sessionsNum + 500], linewidth = 2 , color='black' )
        
        #    pylab.yticks(pylab.arange(0, 1.01, 0.2))
        pylab.ylim((0,1.5))

        pylab.xlim((0,551))
    
        tick_lcs = []
        tick_lbs = []
        pylab.yticks(tick_lcs, tick_lbs)
        for i in range ( 0 , 4 ):
            tick_lcs.append( 50 + i*150 ) 
            tick_lbs.append(i*10)
        pylab.xticks(tick_lcs, tick_lbs)

        for line in ax1.get_yticklines() + ax1.get_xticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        ax1.set_ylabel('Infusion')
        ax1.set_xlabel('Time (min)')
        ax1.set_title('20sec time-out')
        fig1.savefig('infusionShALastSession.png', format='png') #changed eps to png


    #---------------------------------LgA rats

        fig1 = pylab.figure( figsize=(5,2) )
        fig1.subplots_adjust(top=0.65)
        fig1.subplots_adjust(bottom=0.3)
        fig1.subplots_adjust(left=0.16)
        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(self.infusionLgA [self.trialsPerDay*self.sessionsNum - 50 : self.trialsPerDay*self.sessionsNum + 500], linewidth = 2 , color='black' )
        
        pylab.ylim((0,1.5))

        pylab.xlim((0,551))
    
        tick_lcs = []
        tick_lbs = []
        pylab.yticks(tick_lcs, tick_lbs)
        for i in range ( 0 , 4 ):
            tick_lcs.append( 50 + i*150 ) 
            tick_lbs.append(i*10)
        pylab.xticks(tick_lcs, tick_lbs)

        for line in ax1.get_yticklines() + ax1.get_xticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)
            
        ax1.set_ylabel('Infusions')
        ax1.set_xlabel('Time (min)')
        ax1.set_title('4sec time-out')
        fig1.savefig('infusionLgALastSession.png', format='png') #changed eps to png

        return

    def plotInfusionPer10Min(self):
        """Plot the infusions per 10 minutes for the Short-Access group"""
        infusion4sec   = np.zeros( [6] , float)
        infusion20sec  = np.zeros( [6] , float)
        x = np.arange(10, 61, 10)
        
        for i in range(0,6):
            for j in range(self.trialsPerDay + i*(self.trialsPerBlock), self.trialsPerDay + (i+1)*self.trialsPerBlock):
                infusion4sec[i] = infusion4sec[i] + self.infusionLgA[j]


        for i in range(0,6):
            for j in range(self.trialsPerDay + i*(self.trialsPerBlock), self.trialsPerDay + (i+1)*self.trialsPerBlock):
                infusion20sec[i] = infusion20sec[i] + self.infusionShA[j]

            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(bottom=0.2)
        fig1.subplots_adjust(left=0.16)
        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(x,infusion4sec  , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
        S1 = ax1.plot(x,infusion20sec , '-o', ms=8, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )

        leg=fig1.legend((S0, S1), ('4sec time-out','20sec time-out'), loc = (0.4,0.68))
        leg.draw_frame(False)

        pylab.yticks(pylab.arange(1, 11, 1))
        pylab.ylim((0,11))
        pylab.xlim((0,70))
        pylab.xticks(pylab.arange(10, 61, 10))
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        ax1.set_ylabel('Infusions / 10 min')
        ax1.set_xlabel('Time (min)')
        ax1.set_title(' Post-escalation')
        fig1.savefig('infusionPer10Min.png', format='png') #changed eps to png

        return

    def plotInterInfusionIntervals(self):
        """Plot the inter-nfusion intervals for the last session of the Long-Access group"""
        #--------------------------------------- Compute III For Long-Access
        iiiLgA  = []   # inter-infusion intervals
        xLgA = np.arange(1, 31, 1)

        for j in range(self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay), self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay)+self.seekingTrialsNumLgA):
            if self.infusionLgA[j]==1:
                previousInfTime = j
                break

        for j in range( j+1 , self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay)+self.seekingTrialsNumLgA):
            if self.infusionLgA[j]==1:
                interInf = (j - previousInfTime) * 4        # x*4 , because every trial is 4 seconds
                iiiLgA.append(interInf)
                previousInfTime = j

    #--------------------------------------- Compute III For Short-Access
        iiiShA  = []   # inter-infusion intervals
        
        for j in range(self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay),self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay)+self.seekingTrialsNumShA):
            if self.infusionShA[j]==1:
                previousInfTime = j
                break

        for j in range( j+1 , self.trialsPerDay + (self.sessionsNum-1)*(self.trialsPerDay)+self.seekingTrialsNumShA):
            if self.infusionShA[j]==1:
                interInf = (j - previousInfTime) * 4        # x*4 , because every trial is 4 seconds
                iiiShA.append(interInf)
                previousInfTime = j

        infusionsNumShA = len(iiiShA)
        xShA = np.arange(1, infusionsNumShA+1, 1)
            
        
        iiimax = 0
        for j in range( 0 , 10 ):
            if iiimax<iiiLgA[j]:
                iiimax = iiiLgA[j]
            if iiimax<iiiShA[j]:
                iiimax = iiiShA[j]

                
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(bottom=0.2)
        fig1.subplots_adjust(left=0.16)
        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(xShA,iiiShA[0:infusionsNumShA], '-o', ms=5, markeredgewidth =2, alpha=1, mfc='black',linewidth = 2 , color='black' )
        S1 = ax1.plot(xLgA,iiiLgA[0:30],              '-o', ms=5, markeredgewidth =2, alpha=1, mfc='white',linewidth = 2 , color='black' )
        
        leg=fig1.legend((S1, S0), ('4sec time-out','20sec time-out'), loc = (0.4,0.68))
        leg.draw_frame(False)
        
        pylab.ylim((-20,iiimax+20))
        pylab.xlim((0,31))
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        ax1.set_ylabel('Inter-infusion intervals (sec)')
        ax1.set_xlabel('Infusion number')
        ax1.set_title('Post-escalation')
        fig1.savefig('interInfusionIntervals.png', format='png') #changed eps to png

        return

    def plotting(self):
        """Get values then plot all Keramati's results"""
        self.loggingFinalization()

        self.plotInternalStateLastSession()
        self.plotInfusionLastSession()
        self.plotInfusionPer10Min()
        self.plotInterInfusionIntervals()
        
        pylab.show()   
        
        return

    #Keramati's 20sec vs 4 sec Experiment
    def setupKeramatiParameters(self):
        """Parameters for Keramati lever pulling experiment from original simulator"""
        #Simulation Parameters
        self.animalsNum = 1 # Number of animals
        self.animal = 0 # number of animals being tested

        self.pretrainingHours = 1
        self.sessionsNum = 1 # Number of sessions of cocain seeking, followed by rest in home-cage
        self.seekingHoursShA = 3 
        self.seekingHoursLgA = 3 
        self.extinctionHours = 0

        self.trialsPerHour = 900 # 60*60/4 = 3600/4 = 900 trials during one hour (as each trial is supposed to be 4 seconds)
        self.trialsPerDay = 24*self.trialsPerHour
        self.pretrainingTrialsNum= self.pretrainingHours* self.trialsPerHour
        self.restAfterPretrainingTrialsNum = (24 - self.pretrainingHours) * self.trialsPerHour

        self.seekingTrialsNumShA = self.seekingHoursShA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        self.restingHoursShA = 24 - self.seekingHoursShA
        self.restTrialsNumShA = self.restingHoursShA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage
        self.extinctionTrialsNum = self.extinctionHours * self.trialsPerHour      # Number of trials for each extinction session

        self.seekingTrialsNumLgA = self.seekingHoursLgA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        restingHoursLgA = 24 - self.seekingHoursLgA
        self.restTrialsNumLgA = restingHoursLgA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage
        self.extinctionTrialsNum = self.extinctionHours * self.trialsPerHour      # Number of trials for each extinction session

        self.totalTrialsNum = self.trialsPerDay + self.sessionsNum * (self.trialsPerDay)  #+ extinctionTrialsNum*2 + 1

        #Plotting Parameters
        self.trialsPerBlock = 150  # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds. ros - removed int(10*60/4)

    def initializeKeramatiEnvironment(self, sec):
        if(sec==20):
            self.statesNum = 5 # sequence of steps that each take 4 second to complete. 5 states x 4secs = 20secs
            self.actionsNum = 3 # number of action, e.g action 0 = Null, action 1 = Inactive Lever Press, action 2 = Active Lever Press
            self.initialExState = 0 #external state

            self.transition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
            self.outcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
            self.nonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

            # From state s, and by taking a, we go to state s', with probability p
            #State 0 starting state (action 0 do nothing, action 1, press inactive lever and action 2 press active lever)
            # 0 to 4 secs
            self.setTransition(0,0,0,1)
            self.setTransition(0,1,0,1)
            self.setTransition(0,2,1,1)

            # At state s, by doing action a and going to state s', we receive the outcome
            self.setOutcome(0,2,1, self.cocaine)#take a hit of cocaine 

            self.setNonHomeostaticReward(0,1,0,-self.leverPressCost)
            self.setNonHomeostaticReward(0,2,1,-self.leverPressCost)

            # 4 to 8 secs
            self.setTransition(1,0,2,1)
            self.setTransition(1,1,2,1)
            self.setTransition(1,2,2,1)

            self.setNonHomeostaticReward(1,1,2,-self.leverPressCost)
            self.setNonHomeostaticReward(1,2,2,-self.leverPressCost)

            # 8 to 12 secs
            self.setTransition(2,0,3,1)
            self.setTransition(2,1,3,1)
            self.setTransition(2,2,3,1)

            # 12 to 16 secs
            self.setTransition(3,0,4,1)
            self.setTransition(3,1,4,1)
            self.setTransition(3,2,4,1)

            self.setNonHomeostaticReward(3,1,4,-self.leverPressCost)
            self.setNonHomeostaticReward(3,2,4,-self.leverPressCost)

            # 16 to 20 secs
            self.setTransition(4,0,0,1)
            self.setTransition(4,1,0,1)
            self.setTransition(4,2,0,1)

            self.setNonHomeostaticReward(4,1,0,-self.leverPressCost)
            self.setNonHomeostaticReward(4,2,0,-self.leverPressCost)

            
        elif(sec==4):
            #------------------------------------------ ReDefining MDP
            self.statesNum  = 1                  # number of stater 
            self.actionsNum = 2                  # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
            self.initialExState = 0

            self.transition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
            self.outcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
            self.nonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

            self.setTransition(0,0,0,1) # From state s, and by taking a, we go to state s', with probability p
            self.setTransition(0,1,0,1)
            self.setOutcome(0,1,0,self.cocaine) # At state s, by doing action a and going to state s', we receive the outcome 
            self.setNonHomeostaticReward(0,1,0,-self.leverPressCost)

    def initializeKeramatiLogging(self, sec):
        if(sec==20):
            #Logging Parameters for each trial of SHA (short self administered) (on screen logging)
            self.nulDoingShA = np.zeros( [self.totalTrialsNum] , float)
            self.inactiveLeverPressShA = np.zeros( [self.totalTrialsNum] , float)
            self.activeLeverPressShA = np.zeros( [self.totalTrialsNum] , float)
            self.internalStateShA = np.zeros( [self.totalTrialsNum] , float)
            self.setpointShA = np.zeros( [self.totalTrialsNum] , float)
            self.infusionShA = np.zeros( [self.totalTrialsNum] , float)

            #Logging Parameters for each trial of LgA (long self administered)
            self.nulDoingLgA = np.zeros( [self.totalTrialsNum] , float)
            self.inactiveLeverPressLgA = np.zeros( [self.totalTrialsNum] , float)
            self.activeLeverPressLgA = np.zeros( [self.totalTrialsNum] , float)
            self.internalStateLgA = np.zeros( [self.totalTrialsNum] , float)
            self.setpointLgA = np.zeros( [self.totalTrialsNum] , float)
            self.infusionLgA = np.zeros( [self.totalTrialsNum] , float)

        elif(sec==4):
            #Logging Parameters for each trial of LgA (long self administered)
            self.nulDoingLgA = np.zeros( [self.totalTrialsNum] , float)
            self.inactiveLeverPressLgA = np.zeros( [self.totalTrialsNum] , float)
            self.activeLeverPressLgA = np.zeros( [self.totalTrialsNum] , float)
            self.internalStateLgA = np.zeros( [self.totalTrialsNum] , float)
            self.setpointLgA = np.zeros( [self.totalTrialsNum] , float)
            self.infusionLgA = np.zeros( [self.totalTrialsNum] , float)      

    def runKeramatiExperiment(self):
        """Keramati's 20sec vs 4 sec Experiment used as a test to validate simulator working correctly"""

        #simulation parameters for Keramati's 20sec vs 4 sec Experiment
        self.setupKeramatiParameters()

        # 20 second Time Out Simulation
        #construct a 20 second MDP environment
        self.initializeKeramatiEnvironment(20)
        self.initializeKeramatiLogging(20)
        self.initializeAnimal()
        
        #------------------------------------------ Simulating the 20sec time-out
        #pretrain agent
        self.pretraining('ShA')
        #put pretrained agent in a cage to rest. Homeostatic system cools off
        self.homeCage ( 0,'afterPretrainingShA' )
        #let pretrained ShA_DQN_agent seek cocaine then rest in cage
        for session in range(0, self.sessionsNum):
            self.cocaineSeeking(session, 'ShA')
            self.homeCage(session,'ShA')

        # 4 second Time Out Simulation
        self.initializeKeramatiEnvironment(4) #construct a 4 second MDP environment
        self.initializeKeramatiLogging(4)
        self.initializeAnimal()
        
        # Simulating the 4sec time-out
        #pretrain agent
        self.pretraining('LgA')
        self.homeCage(0,'afterPretrainingLgA') 
        for session in range(0, self.sessionsNum):
            #ros - let pretrained ShA_DQN_agent seek cocaine then rest in cage
            self.cocaineSeeking(session, 'LgA')
            self.homeCage(session, 'LgA') 

        #finish up and plot results
        self.plotting()
if __name__ == "__main__":
    app = HomeoRLEnv() #create so,
    #run Keramati's 20sec vs 4 sec Experiment in sim
    app.runKeramatiExperiment()