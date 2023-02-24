# ref 2004: original experiment results; https://onlinelibrary.wiley.com/doi/10.1046/j.1471-4159.2003.01833.x
# ref 2013, Mehdi Keramati, Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework, https://github.com/mehdiKeramati
# ref: https://www.biorxiv.org/content/10.1101/029256v1.full
# Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework
# Python 3.8.10
# Robert S Sullivan
# 2023

#import libraries 
import numpy as np
import pylab
import cmath

class Plot():
    """This class will plot trials conducted with animals"""
    def __init__(self, animalsNum, totalTrialsNum):
        self.plotParameters(totalTrialsNum)
        self.loggingFinalization(animalsNum, totalTrialsNum)
        self.plotInternalState45() 
        self.plotInternalState5()

    def plotParameters(self, totalTrialsNum):
        # Short-access (ShA) cocaine self-administration
        self.nulDoingShA = np.zeros( [totalTrialsNum] , float)
        self.activeLeverPressShA    = np.zeros( [totalTrialsNum] , float)
        self.internalStateShA       = np.zeros( [totalTrialsNum] , float)
        self.setpointShA            = np.zeros( [totalTrialsNum] , float)
        self.infusionShA            = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaShA  = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaShA     = np.zeros( [totalTrialsNum] , float)

        # Long-access (LgA) cocaine self-administration
        self.nulDoingLgA            = np.zeros( [totalTrialsNum] , float)
        self.activeLeverPressLgA    = np.zeros( [totalTrialsNum] , float)
        self.internalStateLgA       = np.zeros( [totalTrialsNum] , float)
        self.setpointLgA            = np.zeros( [totalTrialsNum] , float)
        self.infusionLgA            = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaLgA  = np.zeros( [totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaLgA     = np.zeros( [totalTrialsNum] , float)
    
    def loggingFinalization(self,animalsNum, totalTrialsNum):
        
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

        font = {'family' : 'normal', 'size'   : 16}
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

        fig1.savefig('internalState45min.eps', format='.png') #.eps format

    def plotInternalState5(self):

        font = {'family' : 'normal', 'size'   : 16}
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

        fig1.savefig('internalState4min.eps', format='png') #eps format

class Animal():
    """This class defines an animal, human users inherit from the animal class""" 
    def __init__(self, typeOfAnimal, externalState, cocaine, leverPressCost, trialsNum): #initialize Animal
        self.animalType = typeOfAnimal
        self.states = 6
        self.actions= 2

         # Homeostatic System
        self.internalState = 0
        self.homeostaticSetpoint = 200

        # Goal-directed system (6 states)
        self.estimatedTransitionUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedOutcomeUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedNonHomeostaticRewardUnderCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedTransitionNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedOutcomeNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )
        self.estimatedNonHomeostaticRewardNoCoca = np.zeros ( [ self.states , self.actions ,  self.states] , float )

        #Animal Sensory (senses for external state, internal state, homeostatic setpoint, and current trial)
        self.currentState = np.zeros(4)

        self.currentState[0] = externalState
        self.currentState[1] = self.internalState
        self.currentState[2] = self.homeostaticSetpoint 
        self.currentState[3] = 0 #current trial

        #q-table state,action. with-cocaine, outcome-with-cocaine, reward-with-cocaine, without-cocaine, outcome-without-cocaine, reward-without-cocaine
        self.estimatedTransitionUnderCoca [0][0][0] = 1 #self.states , self.actions ,  self.states
        self.estimatedTransitionUnderCoca [0][1][1] = 1
        self.estimatedTransitionUnderCoca [1][0][2] = 1
        self.estimatedTransitionUnderCoca [1][1][2] = 1
        self.estimatedTransitionUnderCoca [2][0][3] = 1
        self.estimatedTransitionUnderCoca [2][1][3] = 1
        self.estimatedTransitionUnderCoca [3][0][4] = 1
        self.estimatedTransitionUnderCoca [3][1][4] = 1
        self.estimatedTransitionUnderCoca [4][0][0] = 1
        self.estimatedTransitionUnderCoca [4][1][0] = 1

        self.estimatedTransitionNoCoca    [0][0][0] = 1
        self.estimatedTransitionNoCoca    [0][1][1] = 1
        self.estimatedTransitionNoCoca    [1][0][2] = 1
        self.estimatedTransitionNoCoca    [1][1][2] = 1
        self.estimatedTransitionNoCoca    [2][0][3] = 1
        self.estimatedTransitionNoCoca    [2][1][3] = 1
        self.estimatedTransitionNoCoca    [3][0][4] = 1
        self.estimatedTransitionNoCoca    [3][1][4] = 1
        self.estimatedTransitionNoCoca    [4][0][0] = 1
        self.estimatedTransitionNoCoca    [4][1][0] = 1

        #Action Energy Cost (Assuming that the animals know the energy cost or fatigue of pressing a lever)
        for i in range(0,self.states):
            for j in range(0,self.states):
                self.estimatedNonHomeostaticRewardUnderCoca [i][1][j] = -leverPressCost
                self.estimatedNonHomeostaticRewardNoCoca [i][1][j] = -leverPressCost

        self.estimatedOutcomeUnderCoca[0][1][1] = cocaine
        self.estimatedOutcomeNoCoca [0][1][1] = cocaine

    #actions that can be taken
    def seekCocaine(self, sessionNum): #ratType #def cocaineSeeking

        exState = self.current_state[0]
        inState = self.current_state[1]
        setpointS = self.current_state[2]
        trialCount = self.current_state[3]

        cocBuffer = 0
        
  
        
        for trial in range(0,trialsPerDay):

    #        estimatedActionValuesUnderCoca   = valueEstimationUnderCoca ( exState, inState, setpointS, searchDepth )
    #        estimatedActionValuesNoCoca      = valueEstimationNoCoca    ( exState, inState, setpointS, searchDepth )        
    #        underCocaineWeight               = underCocaine             ( inState , setpointS                      )
    #        estimatedActionValues            = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         

            action = 0
            nextState =0
            nextState = 0
            HomeoRew = 0
            if trial==0:
                out = cocaine
            else:
                out = 0
            
            if self.animalType=='LgA':  
                loggingLgA(trial,action,inState,setpointS,out)    
                print("LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,sessionNum+1,sessionsNum,trial-trialCount+1,trialsNum))

    #        updateOutcomeFunction               ( exState , action , nextState , out ,underCocaineWeight            )
    #        updateNonHomeostaticRewardFunction  ( exState , action , nextState , nonHomeoRew ,underCocaineWeight    )
    #        updateTransitionFunction            ( exState , action , nextState , underCocaineWeight                 )            
            
            cocBuffer = cocBuffer + out                
            
            inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
    #        setpointS   = updateSetpoint(setpointS,out)

            cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

    #        exState   = nextState

        self.current_state[0]    = exState
        self.current_state[1]    = inState
        self.current_state[2]    = setpointS
        self.current_state[3]    = trialCount+trialsNum

class Environment():
    """This class defines the environment that an animal is in, cage, lever pressing, etc.
    For humans it could be swipping on an app."""
    def __init__(self):
        self.externalState = 0
        self.cocaine = 50 # K value, Dose of self-administered drug, they set K = 50 for a single infusion of 0.250mg of cocaine. #K changes proportionally for higher or lower unit doses. Repeated infusions results in the buildup of cocaine in the brain and thus, accumulation of drug influence on the internal state.
        self.leverPressCost  = 20 

class Simulator():
    def __init__(self):
        #initialise the simulation
        self.simulationParameters()
        #create the environment
        self.env = Environment()
        #create the animal
        self.user = Animal(self.animalType, self.env.externalState, self.env.cocaine, self.env.leverPressCost, self.trialsNum)
        #plot the results
        self.p = Plot(self.animalsNum, self.totalTrialsNum)

    def simulationParameters(self):
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