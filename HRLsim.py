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
    #def initializeAnimal
    def __init__(self, typeOfAnimal, externalState):
        self.animalType = typeOfAnimal
        self.states = 6
        self.actions= 2

         # Homeostatic System
        self.internalState = 0
        self.homeostaticSetpoint = 200

        #sensors fed into animal     
        self.currentState = np.zeros(4) #senses for external state, internal state, homeostatic setpoint, and current

        self.currentState[0] = externalState
        self.currentState[1] = self.internalState
        self.currentState[2] = self.homeostaticSetpoint 
        self.currentState[3] = 0 #current
        


    #actions that can be taken
    def seekCocaine(self, sessionNum): #ratType #def cocaineSeeking
        cocaine = 50 # K value, Dose of self-administered drug, they set K = 50 for a single infusion of 0.250mg of cocaine. #K changes proportionally for higher or lower unit doses. Repeated infusions results in the buildup of cocaine in the brain and thus, accumulation of drug influence on the internal state.

        exState = self.current_state[0]
        inState = self.current_state[1]
        setpointS = self.current_state[2]
        trialCoun = self.current_state[3]
        cocBuffer = 0
        
        if self.animalType=='ShA':  
            trialsNum = seekingTrialsNumShA    
        if self.animalType=='LgA':  
            trialsNum = seekingTrialsNumLgA    
        
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
    def __init__(self):
        self.externalState = 0

class Simulator():
    def __init__(self):
        self.simulationParameters()
        self.env = Environment()
        self.user = Animal('LgA', self.env.externalState)
        self.p = Plot(self.animalsNum, self.totalTrialsNum)

    def simulationParameters(self):
        self.animalsNum          = 1                                  # Number of animals

        pretrainingHours    = 0
        sessionsNum         = 1                            # Number of sessions of cocain seeking, followed by rest in home-cage
        seekingHoursShA     = 1            
        seekingHoursLgA     = 24            
        extinctionHours     = 0.75

        trialsPerHour       = 900 # Number of trials during one hour (as each trial is supposed to be 4 seconds),  60sec*60min/4sec = 3600sec.hr/4sec = 900sec
        trialsPerDay        = 24*trialsPerHour
        pretrainingTrialsNum= pretrainingHours* trialsPerHour
        restAfterPretrainingTrialsNum = (24 - pretrainingHours) *trialsPerHour

        seekingTrialsNumShA = seekingHoursShA * trialsPerHour    # Number of trials for each cocaine seeking session
        restingHoursShA     = 24 - seekingHoursShA
        restTrialsNumShA    = restingHoursShA * trialsPerHour    # Number of trials for each session of the animal being in the home cage

        seekingTrialsNumLgA = seekingHoursLgA * trialsPerHour    # Number of trials for each cocaine seeking session
        restingHoursLgA     = 24 - seekingHoursLgA
        restTrialsNumLgA    = restingHoursLgA * trialsPerHour    # Number of trials for each session of the animal being in the home cage

        extinctionTrialsNum = int(extinctionHours*trialsPerHour) # Number of trials for each extinction session, we want to return an int for ranges

        self.totalTrialsNum      = trialsPerDay
        
if __name__ == "__main__":
    app = Simulator()