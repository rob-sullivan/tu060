# Simulating Addiction with Deep Q Learning
'''
Deep Q Learning adaption of Keramati's HomeoSim, 
by Robert S. Sullivan in 2023 using Python 3.8.10
'''

import os
from os import system, name
import random
import time
#import numpy as np #imported later in Keramati simulator
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
    """This is our RL Agent"""
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
        #softmax converts q value numbers into a probability distribution.
        #Q values are the output of the neural network
        # Temperature value = 100. closer to zero the less sure the NN will be to taking the action
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        #e.g actions[1,2,3] = prob[0.04, 0.11, 0.85] # temperature increases 0.85 value to be selected
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]#convert from pytorch tensor to action
    
    #to train our AI
    #forward propagation then backproagation
    # get our output, target, compare our output to the target to compute the loss error
    # backproagate loss error into the nn and use stochastic gradient descent we update the weights according to how much they contributed to the loss error
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
            print("saved !")
        else:
            print("no checkpoint found...")

'''
Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework By Mehdi Keramati, 
in March 2013 using Python 2.6 
'''

import scipy
import numpy
import pylab
import cmath

'''
###################################################################################################################################
###################################################################################################################################
#                                                         Functions                                                               #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the transition function of the MDP   --------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setTransition(state,action,nextState,transitionProbability):
    transition [state][action][nextState] = transitionProbability
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the outcome function of the MDP   -----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setOutcome(state,action,nextState,out):
    outcome [state][action][nextState] = out
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Setting the non-homeostatic reward function of the MDP   --------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def setNonHomeostaticReward(state,action,nextState,rew):
    nonHomeostaticReward [state][action][nextState] = rew
    return 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the probability of the transitions s-a->s'  -------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getTransition(s,a,nextS):
    return transition[s][a][nextS]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Return the next state that the animal fell into  ---------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getRealizedTransition(state,action):
           
    index = numpy.random.uniform(0,1)
    probSum = 0
    for nextS in range(0,statesNum):
        probSum = probSum + getTransition(state,action,nextS)
        if index <= probSum:
            return nextS    

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained outcome   ---------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def getOutcome(state,action,nextState):
    return outcome[state,action,nextState]

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Obtained non-homeostatic reward    ------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
def getNonHomeostaticReward(state,action,nextState):
    return nonHomeostaticReward [state][action][nextState] 

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Homeostatically-regulated Reward   ------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def driveReductionReward(inState,setpointS,outcome):
    d1 = numpy.power(numpy.absolute(numpy.power(setpointS-inState,n*1.0)),(1.0/m))
    d2 = numpy.power(numpy.absolute(numpy.power(setpointS-inState-outcome,n*1.0)),(1.0/m))
    return d1-d2

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Create a new animal   ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def initializeAnimal():    
            
    state[0] = initialExState
    state[1] = initialInState
    state[2] = initialSetpoint 
    state[3] = 0 
        
    for i in range(0,statesNum):
        for j in range(0,actionsNum):
            for k in range(0,statesNum):
                estimatedTransition[i][j][k] = (1.0)/(statesNum*1.0)
                estimatedOutcome[i][j][k] = 0.0
                estimatedNonHomeostaticReward[i][j][k] = 0.0
    
#    Assuming that the animals know the energy cost (fatigue) of pressing a lever 
    for i in range(0,statesNum):
        for j in range(0,statesNum):
            estimatedNonHomeostaticReward[i][1][j] = -leverPressCost
    
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Is action a available is state s?   ----------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def isActionAvailable(state,action):
    probSum = 0 ;
    for i in range(0,statesNum):
        probSum = probSum + getTransition(state,action,i)
    if probSum == 1:
        return 1
    elif probSum == 0:
        return 0
    else:
        print("Error: There seems to be a problem in defining the transition function of the environment")      
        return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Goal-directed Value estimation   -------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def valueEstimation(state,inState,setpointS,depthLeft):

    values = numpy.zeros ( [actionsNum] , float )

    # If this is the last depth that should be searched :
    if depthLeft==1:
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                homeoReward    = driveReductionReward(inState,setpointS,estimatedOutcome[state][action][nextState])
                nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
                transitionProb = estimatedTransition[state][action][nextState]
                values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
        return values
    
    # Otherwise :
    for action in range(0,actionsNum):
        for nextState in range(0,statesNum):
            if estimatedTransition[state][action][nextState] < pruningThreshold :
                VNextStateBest = 0
            else:    
                VNextState = valueEstimation(nextState,setpointS,inState,depthLeft-1)
                VNextStateBest = maxValue (VNextState)
            homeoReward    = driveReductionReward(inState,setpointS,estimatedOutcome[state][action][nextState])
            nonHomeoReward = estimatedNonHomeostaticReward[state][action][nextState]
            transitionProb = estimatedTransition[state][action][nextState]
            values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
            
    return values
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Max ( Value[nextState,a] ) : for all a  ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def maxValue(V):
    maxV = V[0]
    for action in range(0,actionsNum):
        if V[action]>maxV:
            maxV = V[action]    
    return maxV
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Action Selection : Softmax   ------------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def actionSelectionSoftmax(state,V):

    # Normalizing values, in order to be overflow due to very high values
    maxV = V[0]
    if maxV==0:
        maxV=1        
    for action in range(0,actionsNum):
        if maxV < V[action]:
            maxV = V[action]
    for action in range(0,actionsNum):
        V[action] = V[action]/maxV


    sumEV = 0
    for action in range(0,actionsNum):
        sumEV = sumEV + abs(cmath.exp( V[action] / beta ))

    index = numpy.random.uniform(0,sumEV)

    probSum=0
    for action in range(0,actionsNum):
            probSum = probSum + abs(cmath.exp( V[action] / beta ))
            if probSum >= index:
                return action

    print("Error: An unexpected (strange) problem has occured in action selection...")
    return 0
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update internal state upon consumption   ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateInState(inState,outcome):
    interS = inState + outcome - cocaineDegradationRate*(inState-inStateLowerBound)
    if interS<inStateLowerBound:
        interS=inStateLowerBound
    return interS

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the homeostatic setpoint (Allostatic mechanism)   -------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateSetpoint(optimalInState,out):
    
    optInS = optimalInState + out*setpointShiftRate - setpointRecoveryRate

    if optInS<optimalInStateLowerBound:
        optInS=optimalInStateLowerBound

    if optInS>optimalInStateUpperBound:
        optInS=optimalInStateUpperBound

    return optInS

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-outcome function  --------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateOutcomeFunction(state,action,nextState,out):
    estimatedOutcome[state][action][nextState] = (1.0-updateOutcomeRate)*estimatedOutcome[state][action][nextState] + updateOutcomeRate*out
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-non-homeostatic-reward function  ------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateNonHomeostaticRewardFunction(state,action,nextState,rew):
    estimatedNonHomeostaticReward[state][action][nextState] = (1.0-updateRewardRate)*estimatedNonHomeostaticReward[state][action][nextState] + updateRewardRate*rew
    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Update the expected-transition function  ------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def updateTransitionFunction(state,action,nextState):

    #---- First inhibit all associations
    for i in range(0,statesNum):
        estimatedTransition[state][action][i] = (1.0-updateTransitionRate)*estimatedTransition[state][action][i]
    
    #---- Then potentiate the experiences association
    estimatedTransition[state][action][nextState] = estimatedTransition[state][action][nextState] + updateTransitionRate
            
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Pre-training Sessions  ------------------------------------------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def pretraining  (ratType, dqn_agent):

    #ros - used to plot performance. The mean score curve (sliding window of the rewards) with respect to time.
    scores = []

    #ros - score agent wants to maximise
    reward_received = 0.0

    #ros - get current state external state, internal state, setpoint, and trial
    current_state = [state[0], state[1], state[2], state[3]]

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    trialsNum = pretrainingTrialsNum
    
    for trial in range(0,trialsNum):
        
        # ros - playing the action from the ai (dqn class)
        action = dqn_agent.update(reward_received, current_state)
        #estimatedActionValues   = valueEstimation(exState,inState,setpointS,searchDepth)
        #action                  = actionSelectionSoftmax(exState,estimatedActionValues) #ros - removed non DQL agent

        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,setpointS,out)

        if ratType=='ShA':  
            loggingShA (trial,action,inState,setpointS,out)    
            print("ShA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum))
        elif ratType=='LgA':  
            loggingLgA (trial,action,inState,setpointS,out)    
            print("LgA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum))

        updateOutcomeFunction(exState,action,nextState,out)
        updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
        updateTransitionFunction(exState,action,nextState)            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
        setpointS   = updateSetpoint(setpointS,out)

        cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

        exState   = nextState

        #ros - appending the score for DQN agent(mean of the last 100 rewards to the reward window)
        scores.append(dqn_agent.score())

        #ros - get next state for DQN agent
        current_state = [state[0], state[1], state[2], state[3]]
        
        #ros - reward or punish DQN agent
        reward_received = HomeoRew + nonHomeoRew

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

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
    pre_trained_agent = dqn_agent
    return pre_trained_agent

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Cocaine Seeking Sessions  --------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def cocaineSeeking  (sessionNum , ratType, dqn_agent):
    #ros - used to plot performance. The mean score curve (sliding window of the rewards) with respect to time.
    scores = []

    #ros - score agent wants to maximise
    reward_received = 0.0
    #ros - external state, internal state, setpoint, and trial
    current_state = [state[0], state[1], state[2], state[3]]

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    if ratType=='ShA':  
        trialsNum = seekingTrialsNumShA    
    if ratType=='LgA':  
        trialsNum = seekingTrialsNumLgA    
    
    for trial in range(trialCount,trialCount+trialsNum):

        # ros - playing the action from the ai (dqn class)
        action = dqn_agent.update(reward_received, current_state)

        #estimatedActionValues   = valueEstimation(exState,inState,setpointS,searchDepth)
        #action                  = actionSelectionSoftmax(exState,estimatedActionValues)
        nextState               = getRealizedTransition(exState,action)
        out                     = getOutcome(exState,action,nextState)
        nonHomeoRew             = getNonHomeostaticReward(exState,action,nextState)
        HomeoRew                = driveReductionReward(inState,setpointS,out)

        if ratType=='ShA':  
            loggingShA(trial,action,inState,setpointS,out)    
            print("ShA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum))
        if ratType=='LgA':  
            loggingLgA(trial,action,inState,setpointS,out)    
            print("LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum))

        updateOutcomeFunction(exState,action,nextState,out)
        updateNonHomeostaticRewardFunction(exState,action,nextState,nonHomeoRew)
        updateTransitionFunction(exState,action,nextState)            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
        setpointS   = updateSetpoint(setpointS,out)

        cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

        exState   = nextState

        #ros - appending the score for DQN agent(mean of the last 100 rewards to the reward window)
        scores.append(dqn_agent.score())

        #ros - get next state for DQN agent
        current_state = [state[0], state[1], state[2], state[3]]
        
        #ros - reward or punish DQN agent
        reward_received = HomeoRew + nonHomeoRew

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

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
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Home-cage Sessions  --------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def homeCage (sessionNum, ratType):
    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
 
    if ratType=='ShA':  
        trialsNum = restTrialsNumShA    
        print("ShA rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum))
    elif ratType=='LgA':  
        trialsNum = restTrialsNumLgA
        print("LgA rat number: %d / %d     Session Number: %d / %d                          animal rests in home cage" %(animal+1,animalsNum,sessionNum+1,sessionsNum))
    elif ratType=='afterPretrainingShA':  
        trialsNum = restAfterPretrainingTrialsNum    
        print("ShA rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum))
    elif ratType=='afterPretrainingLgA':  
        trialsNum = restAfterPretrainingTrialsNum    
        print("LgA rat number: %d / %d     After pretraining                                animal rests in home cage" %(animal+1,animalsNum))
     
    for trial in range(trialCount,trialCount+trialsNum):

        inState     = updateInState(inState,0)
        setpointS   = updateSetpoint(setpointS,0)

        if ratType=='ShA':  
            loggingShA(trial,0,inState,setpointS,0)    
        elif ratType=='LgA':  
            loggingLgA(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretrainingShA':  
            loggingShA(trial,0,inState,setpointS,0)    
        elif ratType=='afterPretrainingLgA':  
            loggingLgA(trial,0,inState,setpointS,0)    

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Extinction Sessions  -------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def extinction  ( trialsNum , ratsType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]
    cocBuffer   = 0
    
    for trial in range(trialCount,trialCount+trialsNum):
        
        estimatedActionValues   = valueEstimation               (exState,inState,setpointS,searchDepth)
        action                  = actionSelectionSoftmax        ( exState, estimatedActionValues    )
        nextState               = getRealizedTransition         ( exState, action                   )
        out                     = 0
        nonHomeoRew             = getNonHomeostaticReward       ( exState, action, nextState        )
        HomeoRew                = driveReductionReward          ( inState, setpointS, out           )
        
        if ratsType == 'ShA':
            loggingShA(trial,action,inState,setpointS,out)    
            print("ShA rat number: %d / %d     Extinction session       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,trial-trialCount+1,trialsNum))
        if ratsType == 'LgA':
            loggingLgA(trial,action,inState,setpointS,out)    
            print("LgA rat number: %d / %d     Extinction session       trial: %d / %d      Extinction of cocaine seeking" %(animal+1,animalsNum,trial-trialCount+1,trialsNum))

        updateOutcomeFunction               ( exState, action, nextState, out            )
        updateNonHomeostaticRewardFunction  ( exState, action, nextState, nonHomeoRew    )
        updateTransitionFunction            ( exState, action, nextState,                )            
        
        cocBuffer = cocBuffer + out                
        
        inState     = updateInState         ( inState, cocBuffer*cocAbsorptionRatio )
        setpointS   = updateSetpoint        ( setpointS, out )

        cocBuffer = cocBuffer * ( 1-cocAbsorptionRatio )

        exState   = nextState

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount+trialsNum

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Non-contingent cocaine infusion   ------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def noncontingentInfusion (sessionNum,ratType):

    exState     = state[0]
    inState     = state[1]
    setpointS   = state[2]
    trialCount  = state[3]

    inState     = updateInState(inState,cocaine)
    setpointS   = updateSetpoint(setpointS,cocaine)
    if ratType == 'ShA':
        loggingShA(trialCount,0,inState,setpointS,cocaine)    
        print("ShA rat number: %d / %d     Session Number: %d / %d                         animal receives non-contingent cocaine infusion" %(animal+1,animalsNum,sessionNum+1,sessionsNum))
    if ratType == 'LgA':
        loggingLgA(trialCount,0,inState,setpointS,cocaine)    
        print("LgA rat number: %d / %d     Session Number: %d / %d                         animal receives non-contingent cocaine infusion" %(animal+1,animalsNum,sessionNum+1,sessionsNum))
        

    state[0]    = exState
    state[1]    = inState
    state[2]    = setpointS
    state[3]    = trialCount + 1

    return
    
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Short-access group  ----------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingShA(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoingShA[trial]             = nulDoingShA[trial] + 1
    elif action==1: 
        inactiveLeverPressShA[trial]   = inactiveLeverPressShA[trial] + 1
    elif action==2: 
        activeLeverPressShA[trial]     = activeLeverPressShA[trial] + 1
    internalStateShA[trial]    = internalStateShA[trial] + inState
    setpointShA[trial]         = setpointShA[trial] + setpointS    
    if coca==cocaine:
        infusionShA[trial]     = infusionShA[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging the current information for the Long-access group  ------------------------------------ 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingLgA(trial,action,inState,setpointS,coca):
   
    if action==0: 
        nulDoingLgA[trial]             = nulDoingLgA[trial] + 1
    elif action==1: 
        inactiveLeverPressLgA[trial]   = inactiveLeverPressLgA[trial] + 1
    elif action==2: 
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial] + 1
    internalStateLgA[trial]    = internalStateLgA[trial] + inState
    setpointLgA[trial]         = setpointLgA[trial] + setpointS    
    if coca==cocaine:
        infusionLgA[trial]     = infusionLgA[trial] + 1
    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Wrap up all the logged data   ----------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def loggingFinalization():
    
    for trial in range(0,totalTrialsNum):
        nulDoingShA[trial]             = nulDoingShA[trial]/animalsNum
        inactiveLeverPressShA[trial]   = inactiveLeverPressShA[trial]/animalsNum
        activeLeverPressShA[trial]     = activeLeverPressShA[trial]/animalsNum
        internalStateShA[trial]        = internalStateShA[trial]/animalsNum
        setpointShA[trial]             = setpointShA[trial]/animalsNum  
        infusionShA[trial]             = infusionShA[trial]/animalsNum 

        nulDoingLgA[trial]             = nulDoingLgA[trial]/animalsNum
        inactiveLeverPressLgA[trial]   = inactiveLeverPressLgA[trial]/animalsNum
        activeLeverPressLgA[trial]     = activeLeverPressLgA[trial]/animalsNum
        internalStateLgA[trial]        = internalStateLgA[trial]/animalsNum
        setpointLgA[trial]             = setpointLgA[trial]/animalsNum  
        infusionLgA[trial]             = infusionLgA[trial]/animalsNum 

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the internal state of the last session  -------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInternalStateLastSession():

    font = {'family' : 'normal', 'size'   : 16}
    pylab.rc('font', **font)
    pylab.rcParams.update({'legend.fontsize': 16})
        
    fig1 = pylab.figure( figsize=(5,3.5) )
    fig1.subplots_adjust(left=0.16)
    fig1.subplots_adjust(bottom=0.2)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(internalStateShA [trialsPerDay*sessionsNum - 50 : trialsPerDay*sessionsNum + 500 ] , linewidth = 2.5 , color='black' )
    S1 = ax1.plot(internalStateLgA [trialsPerDay*sessionsNum - 50 : trialsPerDay*sessionsNum + 500 ] , linewidth = 1.5 , color='black' )
  
    leg = fig1.legend((S1, S0), ('4sec time-out','20sec time-out'), loc = (0.4,0.68))
    leg.draw_frame(False)

    max = 0    
    for i in range ( trialsPerDay*sessionsNum - 50 , trialsPerDay*sessionsNum + 500 ):
        if max < internalStateLgA[i]:
            max = internalStateLgA[i]      
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
    fig1.savefig('internalStateLastSession.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions for the last session ------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInfusionLastSession():

#---------------------------------ShA rats
        
    fig1 = pylab.figure( figsize=(5,2) )
    fig1.subplots_adjust(top=0.65)
    fig1.subplots_adjust(bottom=0.3)
    fig1.subplots_adjust(left=0.16)

    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(infusionShA [trialsPerDay*sessionsNum - 50 : trialsPerDay*sessionsNum + 500], linewidth = 2 , color='black' )
    
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
    fig1.savefig('infusionShALastSession.eps', format='eps')


#---------------------------------LgA rats

    fig1 = pylab.figure( figsize=(5,2) )
    fig1.subplots_adjust(top=0.65)
    fig1.subplots_adjust(bottom=0.3)
    fig1.subplots_adjust(left=0.16)
    ax1 = fig1.add_subplot(111)
    S0 = ax1.plot(infusionLgA [trialsPerDay*sessionsNum - 50 : trialsPerDay*sessionsNum + 500], linewidth = 2 , color='black' )
    
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
    fig1.savefig('infusionLgALastSession.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the infusions per 10 minutes for the Short-Access group ---------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInfusionPer10Min():
    
    infusion4sec   = numpy.zeros( [6] , float)
    infusion20sec  = numpy.zeros( [6] , float)
    x = numpy.arange(10, 61, 10)
    
    for i in range(0,6):
        for j in range(trialsPerDay + i*(trialsPerBlock), trialsPerDay + (i+1)*trialsPerBlock):
            infusion4sec[i] = infusion4sec[i] + infusionLgA[j]


    for i in range(0,6):
        for j in range(trialsPerDay + i*(trialsPerBlock), trialsPerDay + (i+1)*trialsPerBlock):
            infusion20sec[i] = infusion20sec[i] + infusionShA[j]

        
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
    fig1.savefig('infusionPer10Min.eps', format='eps')

    return


'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot the inter-nfusion intervals for the last session of the Long-Access group ---------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotInterInfusionIntervals():
 
#--------------------------------------- Compute III For Long-Access
    iiiLgA  = []   # inter-infusion intervals
    xLgA = numpy.arange(1, 31, 1)

    for j in range(trialsPerDay + (sessionsNum-1)*(trialsPerDay),trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumLgA):
        if infusionLgA[j]==1:
            previousInfTime = j
            break

    for j in range( j+1 , trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumLgA):
        if infusionLgA[j]==1:
            interInf = (j - previousInfTime) * 4        # x*4 , because every trial is 4 seconds
            iiiLgA.append(interInf)
            previousInfTime = j

#--------------------------------------- Compute III For Short-Access
    iiiShA  = []   # inter-infusion intervals
    
    for j in range(trialsPerDay + (sessionsNum-1)*(trialsPerDay),trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumShA):
        if infusionShA[j]==1:
            previousInfTime = j
            break

    for j in range( j+1 , trialsPerDay + (sessionsNum-1)*(trialsPerDay)+seekingTrialsNumShA):
        if infusionShA[j]==1:
            interInf = (j - previousInfTime) * 4        # x*4 , because every trial is 4 seconds
            iiiShA.append(interInf)
            previousInfTime = j

    infusionsNumShA = len(iiiShA)
    xShA = numpy.arange(1, infusionsNumShA+1, 1)
           
    
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
    fig1.savefig('interInfusionIntervals.eps', format='eps')

    return

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plot all the results  ------------------------------------------------------------------------- 
--------------------------------------------------------------------------------------------------------------------------------'''
def plotting():

    loggingFinalization()

    plotInternalStateLastSession()
    plotInfusionLastSession()
    plotInfusionPer10Min()
    plotInterInfusionIntervals()
    
    pylab.show()   
    
    return

'''
###################################################################################################################################
###################################################################################################################################
#                                                             Main                                                                #
###################################################################################################################################
###################################################################################################################################
'''
'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Markov Decison Process FR1 - Timeout 20sec  ---------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

cocaine         = 50             # Dose of self-administered drug
nonContingentCocaine = 50
leverPressCost  = 1              # Energy cost for pressing the lever

statesNum       = 5              # number of stater 
actionsNum      = 3              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState  = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)

# From state s, and by taking a, we go to state s', with probability p
#states when nothing done
setTransition(0,0,0,1)          
setTransition(1,0,2,1)
setTransition(2,0,3,1)
setTransition(3,0,4,1)
setTransition(4,0,0,1)

#states when inactive lever pressed
setTransition(0,1,0,1)
setTransition(1,1,2,1)
setTransition(2,1,3,1)
setTransition(3,1,4,1)
setTransition(4,1,0,1)

#states when active lever pressed
setTransition(0,2,1,1)
setTransition(1,2,2,1)
setTransition(2,2,3,1)
setTransition(3,2,4,1)
setTransition(4,2,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

# At state s, by doing action a and going to state s', we receive the outcome
setOutcome(0,2,1,cocaine)#take a hit of cocaine     

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

setNonHomeostaticReward(0,1,0,-leverPressCost)
setNonHomeostaticReward(0,2,1,-leverPressCost)
setNonHomeostaticReward(1,1,2,-leverPressCost)
setNonHomeostaticReward(1,2,2,-leverPressCost)
setNonHomeostaticReward(3,1,4,-leverPressCost)
setNonHomeostaticReward(3,2,4,-leverPressCost)
setNonHomeostaticReward(4,1,0,-leverPressCost)
setNonHomeostaticReward(4,2,0,-leverPressCost)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Definition of the Animal   --------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

#------------ Homeostatic System
initialInState          = 0
initialSetpoint         = 200
inStateLowerBound       = 0
cocaineDegradationRate  = 0.007    # Dose of cocaine that the animal loses in every time-step
cocAbsorptionRatio      = 0.12      # Proportion of the injected cocaine that affects the brain right after infusion 

#------------ Allostatic System
setpointShiftRate       = 0.0018
setpointRecoveryRate    = 0.00016
optimalInStateLowerBound= 100
optimalInStateUpperBound= 200

#------------ Drive Function
m                       = 3     # Parameter of the drive function : m-th root
n                       = 4     # Parameter of the drive function : n-th pawer

#------------ Goal-directed system
updateOutcomeRate       = 0.2  # Learning rate for updating the outcome function
updateTransitionRate    = 0.2  # Learning rate for updating the transition function
updateRewardRate        = 0.2  # Learning rate for updating the non-homeostatic reward function
gamma                   = 1     # Discount factor
beta                    = 0.25  # Rate of exploration
searchDepth             = 3     # Depth of going into the decision tree for goal-directed valuation of choices
pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

state                            = numpy.zeros ( [4] , int)     # a vector of the external state, internal state, setpoint, and trial

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation Parameters   -----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

animalsNum          = 1                                  # Number of animals

pretrainingHours    = 1
sessionsNum         = 1                                  # Number of sessions of cocain seeking, followed by rest in home-cage
seekingHoursShA     = 3            
seekingHoursLgA     = 3            
extinctionHours     = 0

trialsPerHour       = int(60*60/4)                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
trialsPerDay        = 24*trialsPerHour
pretrainingTrialsNum= pretrainingHours* trialsPerHour
restAfterPretrainingTrialsNum = (24 - pretrainingHours) *trialsPerHour

seekingTrialsNumShA = seekingHoursShA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursShA     = 24 - seekingHoursShA
restTrialsNumShA    = restingHoursShA * trialsPerHour    # Number of trials for each session of the animal being in the home cage
extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

seekingTrialsNumLgA = seekingHoursLgA * trialsPerHour    # Number of trials for each cocaine seeking session
restingHoursLgA     = 24 - seekingHoursLgA
restTrialsNumLgA    = restingHoursLgA * trialsPerHour    # Number of trials for each session of the animal being in the home cage
extinctionTrialsNum = extinctionHours*trialsPerHour      # Number of trials for each extinction session

totalTrialsNum      = trialsPerDay + sessionsNum * (trialsPerDay)  #+ extinctionTrialsNum*2 + 1

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Plotting Parameters   -------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

trialsPerBlock = int(10*60/4)            # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Logging Parameters   --------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''

nulDoingShA            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPressShA  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressShA    = numpy.zeros( [totalTrialsNum] , float)
internalStateShA       = numpy.zeros( [totalTrialsNum] , float)
setpointShA            = numpy.zeros( [totalTrialsNum] , float)
infusionShA            = numpy.zeros( [totalTrialsNum] , float)

nulDoingLgA            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPressLgA  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressLgA    = numpy.zeros( [totalTrialsNum] , float)
internalStateLgA       = numpy.zeros( [totalTrialsNum] , float)
setpointLgA            = numpy.zeros( [totalTrialsNum] , float)
infusionLgA            = numpy.zeros( [totalTrialsNum] , float)

'''--------------------------------------------------------------------------------------------------------------------------------
---------------------------------   Simulation   ----------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------'''
#ros - Create a ShA DQL Agent
#ros - get shape of numpy array sensors. i.e external state, internal state, setpoint, and trial
s = state.shape[0]
#ros - get number of actions
a = actionsNum

#ros - Agent Brain, a neural network that represents our Q-function
ShA_DQN_agent = Dqn(s,a,gamma) # e.g 5 sensors, 6 actions, gamma = 0.9

animal = 0
#------------------------------------------ Simulating the 20sec time-out
initializeAnimal          (                         )

#ros - pretrain agent
ShA_DQN_agent = pretraining('ShA',ShA_DQN_agent)
#pretraining('ShA')

#ros - put pretrain agent in a cage to rest. Homeostatic system cools off
homeCage ( 0,'afterPretrainingShA' )

for session in range(0,sessionsNum):
    #ros - let pretrained ShA_DQN_agent seek cocaine then rest in cage
    cocaineSeeking        (  session , 'ShA', ShA_DQN_agent)
    #cocaineSeeking        (  session , 'ShA'        )
    homeCage              (  session , 'ShA'        ) 
    
####################################################################
#                       4 sec Time out
####################################################################
    
#------------------------------------------ ReDefining MDP
statesNum  = 1                  # number of stater 
actionsNum = 2                  # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
initialExState = 0

transition = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
setTransition(0,1,0,1)

outcome = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setOutcome(0,1,0,cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

nonHomeostaticReward = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
setNonHomeostaticReward(0,1,0,-leverPressCost)

#------------------------------------------ ReDefining the animal
estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

#------------------------------------------ ReDefining the Logging Parameters
nulDoingLgA            = numpy.zeros( [totalTrialsNum] , float)
inactiveLeverPressLgA  = numpy.zeros( [totalTrialsNum] , float)
activeLeverPressLgA    = numpy.zeros( [totalTrialsNum] , float)
internalStateLgA       = numpy.zeros( [totalTrialsNum] , float)
setpointLgA            = numpy.zeros( [totalTrialsNum] , float)
infusionLgA            = numpy.zeros( [totalTrialsNum] , float)

#ros - Create a LgA DQL Agent
#ros - get shape of numpy array sensors. i.e external state, internal state, setpoint, and trial
s = state.shape[0]
#ros - get number of actions
a = actionsNum

#ros - Agent Brain, a neural network that represents our Q-function
LgA_DQN_agent = Dqn(s,a,gamma) # e.g 5 sensors, 6 actions, gamma = 0.9

#------------------------------------------ Simulating the 4sec time-out
initializeAnimal          (                         )
#ros - pretrain agent
LgA_DQN_agent = pretraining('LgA',LgA_DQN_agent)
#pretraining               ( 'LgA'                   )
 
homeCage                  ( 0,'afterPretrainingLgA' ) 

for session in range(0,sessionsNum):
    #ros - let pretrained ShA_DQN_agent seek cocaine then rest in cage
    cocaineSeeking        (  session , 'LgA', LgA_DQN_agent)
    #cocaineSeeking        (  session , 'ShA'        )
    homeCage              (  session , 'LgA'        ) 

plotting()


""" TEMPLATE
#ros - Create a ShA DQL Agent
# get shape of numpy array sensors. i.e external state, internal state, setpoint, and trial
s = state.shape[0]
#get number of actions
a = actionsNum

# Agent Brain - a neural network that represents our Q-function
ShA_DQN_agent = Dqn(s,a,gamma) # e.g 5 sensors, 6 actions, gamma = 0.9

#give agent actions to do
actions_available = [i for i in range(0, a)] #comprehension list to build array
agent_actions = [actions_available] # e.g 'Binge on Internet', 'Work', 'Exercise', 'Socialise', 'Drink Alcohol', 'Smoke'

# used to plot performance. The mean score curve (sliding window of the rewards) with respect to time.
scores = []

# score agent wants to maximise
reward_received = 0


def next_time_interval(self, dqn_agent):
    #external state, internal state, setpoint, and trial
    current_state = [state[0], state[1], state[2], state[3]]
    take_an_action = dqn_agent.update(reward_received, current_state) # playing the action from the ai (dqn class)
    scores.append(dqn_agent.score()) # appending the score (mean of the last 100 rewards to the reward window)
    next_state = [state[0], state[1], state[2], state[3]]
    reward_received = 0 #new reward

finish = False #trigger to end simulation
while not finish:
    finish = next_time_interval(ShA_DQN_agent)

#show results of DQN agent
plt.title("Scores: Pretraining")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.plot(scores)
plt.show()
"""