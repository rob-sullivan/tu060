# ref 2004: original experiment results; https://onlinelibrary.wiley.com/doi/10.1046/j.1471-4159.2003.01833.x
# ref 2013, Mehdi Keramati, Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework, https://github.com/mehdiKeramati
# ref: https://www.biorxiv.org/content/10.1101/029256v1.full
# Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework
# Python 3.8.10
# Robert S Sullivan
# 2023

#import libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pylab
import cmath

class Log():
    """This class will Log the current information for Long and Short access groups
    then will plot trials conducted with animals"""
    def __init__(self, totalTrialsNum): #log and plot parameters
        # Short-access (ShA) cocaine self-administration
        self.nulDoingShA = np.zeros( [totalTrialsNum] , float)
        self.activeLeverPressShA = np.zeros( [totalTrialsNum] , float)
        self.internalStateShA = np.zeros( [totalTrialsNum] , float)
        self.setpointShA = np.zeros( [totalTrialsNum] , float)
        self.infusionShA = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaShA = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaShA = np.zeros( [totalTrialsNum] , float)

        # Long-access (LgA) cocaine self-administration
        self.nulDoingLgA = np.zeros( [totalTrialsNum] , float)
        self.activeLeverPressLgA = np.zeros( [totalTrialsNum] , float)
        self.internalStateLgA = np.zeros( [totalTrialsNum] , float)
        self.setpointLgA = np.zeros( [totalTrialsNum] , float)
        self.infusionLgA = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaLgA = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaLgA = np.zeros( [totalTrialsNum] , float)

    def logTrial(self, typeOfAnimal, trial, action, inState, setpointS, cocaineInfusion, cocaine, estimatedOutcomeUnderCoca, estimatedOutcomeNoCoca):
        if typeOfAnimal == 'ShA':#'LgA'
            if action==0: 
                self.nulDoingShA[trial] = self.nulDoingShA[trial] + 1
            elif action==1: 
                self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial] + 1
            self.internalStateShA[trial] = self.internalStateShA[trial] + inState
            self.setpointShA[trial] = self.setpointShA[trial] + setpointS    
            if cocaineInfusion==cocaine:
                self.infusionShA[trial] = self.infusionShA[trial] + 1

            self.estimatedOutcomeUnderCocaShA [trial] = self.estimatedOutcomeUnderCocaShA[trial] + estimatedOutcomeUnderCoca[0][1][1]
            self.estimatedOutcomeNoCocaShA[trial] = self.estimatedOutcomeNoCocaShA[trial] + estimatedOutcomeNoCoca[0][1][1]

        elif typeOfAnimal == 'LgA':
            self.internalStateLgA[trial] = self.internalStateLgA[trial] + inState
            self.setpointLgA[trial] = self.setpointLgA[trial] + setpointS    
            if cocaineInfusion==cocaine:
                self.infusionLgA[trial] = self.infusionLgA[trial] + 1

            self.estimatedOutcomeUnderCocaLgA[trial] = self.estimatedOutcomeUnderCocaLgA[trial] + estimatedOutcomeUnderCoca[0][1][1]
            self.estimatedOutcomeNoCocaLgA[trial] = self.estimatedOutcomeNoCocaLgA[trial] + estimatedOutcomeNoCoca[0][1][1]

    def finalizeLog(self,animalsNum, totalTrialsNum):
        
        for trial in range(0,totalTrialsNum):
            # Short-access (ShA) cocaine self-administration
            self.nulDoingShA[trial] = self.nulDoingShA[trial]/animalsNum
            self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial]/animalsNum
            self.internalStateShA[trial] = self.internalStateShA[trial]/animalsNum
            self.setpointShA[trial] = self.setpointShA[trial]/animalsNum  
            self.infusionShA[trial] = self.infusionShA[trial]/animalsNum 
            self.estimatedOutcomeUnderCocaShA[trial] = self.estimatedOutcomeUnderCocaShA[trial]/animalsNum
            self.estimatedOutcomeNoCocaShA[trial] = self.estimatedOutcomeNoCocaShA[trial]/animalsNum

            # Long-access (LgA) cocaine self-administration
            self.nulDoingLgA[trial] = self.nulDoingLgA[trial]/animalsNum
            self.activeLeverPressLgA[trial] = self.activeLeverPressLgA[trial]/animalsNum
            self.internalStateLgA[trial] = self.internalStateLgA[trial]/animalsNum
            self.setpointLgA[trial] = self.setpointLgA[trial]/animalsNum  
            self.infusionLgA[trial] = self.infusionLgA[trial]/animalsNum 
            self.estimatedOutcomeUnderCocaLgA[trial] = self.estimatedOutcomeUnderCocaLgA[trial]/animalsNum
            self.estimatedOutcomeNoCocaLgA[trial] = self.estimatedOutcomeNoCocaLgA[trial]/animalsNum

    def plotInternalState45(self):
        
        font = {'family' : 'DejaVu', 'size'   : 16}#family: normal
        pylab.rc('font', **font)
        pylab.rcParams.update({'legend.fontsize': 16})
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)


        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Internal State')
        ax1.set_xlabel('Time (min)')
        ax1.set_title('')

        S0 = ax1.plot(self.internalStateLgA[0:675] , linewidth = 2 , color='black' )

        ax1.axhline(0, color='0.25',ls='--', lw=1 )
    
        pylab.yticks(pylab.arange(0, 41, 10))
        pylab.ylim((-5,47))
        pylab.xlim( ( -45 , 675 + 45 ) )
        
        tick_lcs = []
        tick_lbs = []
        for i in range ( 0 , 5 ):
            tick_lcs.append( i*15*10 ) 
            tick_lbs.append(i*10)
        pylab.xticks(tick_lcs, tick_lbs)

        #for i in range ( 0 , 5 ):
            #if i%2==0:
                #p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        fig1.savefig('internalState45min.png', format='png') #.eps format

    def plotInternalState5(self):

        font = {'family' : 'DejaVu', 'size'   : 16} #family: normal
        pylab.rc('font', **font)
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)


        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Internal State')
        ax1.set_xlabel('Time (sec)')
        ax1.set_title('')
        
        S0 = ax1.plot(self.internalStateLgA[0:60] , linewidth = 2 , color='black' )

        ax1.axhline(0, color='0.25',ls='--', lw=1 )
    
        pylab.yticks(pylab.arange(0, 41, 10))
        pylab.ylim((-5,47))
        pylab.xlim( ( -4 , 60 + 4 ) )
        
        tick_lcs = []
        tick_lbs = []
        for i in range ( 0 , 5 ):
            tick_lcs.append( i*15 ) 
            tick_lbs.append((i*240)/4)
        pylab.xticks(tick_lcs, tick_lbs)

        #for i in range ( 0 , 5 ):
            #if i%2==0:
                #p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        fig1.savefig('internalState4min.png', format='png') #eps format

class Animal():
    """This class defines an animal, human users inherit from the animal class""" 
    def __init__(self, typeOfAnimal, env): #initialize Animal
        self.animalType = typeOfAnimal
        self.states = 6
        self.actions= 2

        # Homeostatic System
        self.internalState = 0
        self.homeostaticSetpoint = 200
        self.inStateLowerBound = 0

            # Dose of cocaine that the animal loses in every time-step
        self.cocaineDegradationRate = 0.007

            # Proportion of the injected cocaine that affects the brain right after infusion
        self.cocAbsorptionRatio = 0.12 

        # Goal-directed system (two state sets total of 6 states)
            #Two sets of states; the under-cocaine state set and the not-under-cocaine state set
            ## first state set
        self.estimatedTransitionUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedOutcomeUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedNonHomeostaticRewardUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )

            ## second state set
        self.estimatedTransitionNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedOutcomeNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedNonHomeostaticRewardNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )

            #q-table state,action. with-cocaine, outcome-with-cocaine, reward-with-cocaine, without-cocaine, outcome-without-cocaine, reward-without-cocaine
        self.estimatedTransitionUnderCoca[0][0][0] = 1 #self.states , self.actions ,  self.states
        self.estimatedTransitionUnderCoca[0][1][1] = 1
        self.estimatedTransitionUnderCoca[1][0][2] = 1
        self.estimatedTransitionUnderCoca[1][1][2] = 1
        self.estimatedTransitionUnderCoca[2][0][3] = 1
        self.estimatedTransitionUnderCoca[2][1][3] = 1
        self.estimatedTransitionUnderCoca[3][0][4] = 1
        self.estimatedTransitionUnderCoca[3][1][4] = 1
        self.estimatedTransitionUnderCoca[4][0][0] = 1
        self.estimatedTransitionUnderCoca[4][1][0] = 1

        self.estimatedTransitionNoCoca[0][0][0] = 1
        self.estimatedTransitionNoCoca[0][1][1] = 1
        self.estimatedTransitionNoCoca[1][0][2] = 1
        self.estimatedTransitionNoCoca[1][1][2] = 1
        self.estimatedTransitionNoCoca[2][0][3] = 1
        self.estimatedTransitionNoCoca[2][1][3] = 1
        self.estimatedTransitionNoCoca[3][0][4] = 1
        self.estimatedTransitionNoCoca[3][1][4] = 1
        self.estimatedTransitionNoCoca[4][0][0] = 1
        self.estimatedTransitionNoCoca[4][1][0] = 1

        #Animal Sensory
            # senses for external state, internal state, homeostatic setpoint, and current trial
        self.currentState = np.zeros(4)

        self.currentState[0] = env.externalState
        self.currentState[1] = self.internalState
        self.currentState[2] = self.homeostaticSetpoint 
        self.currentState[3] = 0 #current trial

        #Rewards
        #Action Energy Cost (Assuming that the animals know the energy cost or fatigue of pressing a lever)
        for i in range(0,self.states):
            for j in range(0,self.states):
                self.estimatedNonHomeostaticRewardUnderCoca[i][1][j] = -env.leverPressCost
                self.estimatedNonHomeostaticRewardNoCoca[i][1][j] = -env.leverPressCost

        self.estimatedOutcomeUnderCoca[0][1][1] = env.cocaine
        self.estimatedOutcomeNoCoca[0][1][1] = env.cocaine

    #actions that can be taken
    def seekCocaine(self, env, log, numOfAnimal, numOfSessions, trialsNum, trialsPerDay, totalTrialsNum): #ratType #def cocaineSeeking
        for animal in range(0, numOfAnimal):
            externalState = self.currentState[0]
            internalState = self.currentState[1]
            homeostaticSetpoint = self.currentState[2]
            trialCount = self.currentState[3]

            sessionNum = 0

            for trial in range(0,trialsPerDay):
                #estimatedActionValuesUnderCoca = valueEstimationUnderCoca(exState, inState, setpointS, searchDepth )
                #estimatedActionValuesNoCoca = valueEstimationNoCoca(exState, inState, setpointS, searchDepth )        
                #underCocaineWeight = underCocaine(inState , setpointS                      )
                #estimatedActionValues = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         

                action = 0
                #nextState =0
                #nextState = 0
                #HomeoRew = 0
                if trial==0:
                    out = env.cocaine
                else:
                    out = 0
                
                if self.animalType=='LgA':  
                    log.logTrial(self.animalType, trial, action, internalState, homeostaticSetpoint, out, env.cocaine, self.estimatedOutcomeUnderCoca, self.estimatedOutcomeNoCoca)    
                    print("LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1, numOfAnimal, sessionNum+1,numOfSessions,trial-trialCount+1,trialsNum))# updateOutcomeFunction( exState , action , nextState , out ,underCocaineWeight)
                #updateNonHomeostaticRewardFunction  ( externalState , action , nextState , nonHomeoRew ,underCocaineWeight    )
                #updateTransitionFunction            ( externalState , action , nextState , underCocaineWeight                 )            
                
                env.cocaineBuffer = env.cocaineBuffer + out                
                
                internalState     = self.updateInternalState(internalState, env.cocaineBuffer * self.cocAbsorptionRatio)
                #homeostaticSetpoint   = updateSetpoint(homeostaticSetpoint,out)

                env.cocaineBuffer = env.cocaineBuffer*(1 - self.cocAbsorptionRatio)
            #externalState   = nextState

            #next state
            self.currentState[0] = externalState
            self.currentState[1] = internalState
            self.currentState[2] = homeostaticSetpoint
            self.currentState[3] = trialCount + trialsNum
        #plot the results
        log.finalizeLog(numOfAnimal, totalTrialsNum)
        log.plotInternalState45() 
        log.plotInternalState5()

    #Update internal state upon consumption
    def updateInternalState(self, internalState,outcome):
        internalS = internalState + outcome - self.cocaineDegradationRate*(internalState - self.inStateLowerBound)
        if internalS < self.inStateLowerBound:
            internalS =self.inStateLowerBound
        return internalS

class Environment():
    """This class defines the environment that an animal is in, cage, lever pressing, etc.
    For humans it could be swipping on an app."""
    def __init__(self):
        self.externalState = 0

        # Drug parameters
            # K value, Dose of self-administered drug, 
            # set K = 50 for a single infusion of 0.250mg of cocaine. 
            # #K changes proportionally for higher or lower unit doses. 
            # Repeated infusions results in the buildup of cocaine in 
            # the brain and thus, accumulation of drug influence on the internal state.
        self.cocaine = 50 
        self.cocaineBuffer = 0

        # this is energy cost or fatigue of pressing a lever
        self.leverPressCost  = 20 

class Simulator():
    def __init__(self):
        #initialise the simulation
        self.loadSimParameters()
        #create the environment
        self.env = Environment()
        #create the animal
        self.animal = Animal(self.animalType, self.env)
        #log trials
        self.log = Log(self.totalTrialsNum)

        #run sim
        self.animal.seekCocaine(self.env, self.log, self.animalsNum, self.sessionsNum, self.trialsNum, self.trialsPerDay, self.totalTrialsNum)
        
    def loadSimParameters(self):
        self.animalsNum = 1 # Number of animals
        self.animalType = 'LgA'

        # Session & Trials
        self.sessionsNum = 1 #sessions of seeking cocaine then resting in home-cage
        self.trialsPerHour = 900 # Number of trials during one hour (as each trial is supposed to be 4 seconds),  60sec*60min/4sec = 3600sec.hr/4sec = 900sec
        self.trialsPerDay = 24 * self.trialsPerHour
        self.totalTrialsNum = self.trialsPerDay

        # Pretraining
        self.pretrainingHours = 0
        self.pretrainingTrialsNum = int(self.pretrainingHours * self.trialsPerHour)
        self.restAfterPretrainingTrialsNum = int((24 - self.pretrainingHours) * self.trialsPerHour)

        # ShA
        self.seekingHoursShA = 1 
        self.seekingTrialsNumShA = self.seekingHoursShA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        self.restingHoursShA = 24 - self.seekingHoursShA
        self.restTrialsNumShA = self.restingHoursShA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage

        # LgA
        self.seekingHoursLgA = 24
        self.seekingTrialsNumLgA = self.seekingHoursLgA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        self.restingHoursLgA = 24 - self.seekingHoursLgA
        self.restTrialsNumLgA = self.restingHoursLgA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage

        if self.animalType=='ShA':  
            self.trialsNum = self.seekingTrialsNumShA   
        elif self.animalType=='LgA':  
            self.trialsNum = self.seekingTrialsNumLgA

        # Extinction
        self.extinctionHours = 0.75
        self.extinctionTrialsNum = int(self.extinctionHours * self.trialsPerHour) # Number of trials for each extinction session, we want to return an int for ranges 
        
if __name__ == "__main__":
    app = Simulator()