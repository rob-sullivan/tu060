# ref 2004: original experiment results; https://onlinelibrary.wiley.com/doi/10.1046/j.1471-4159.2003.01833.x
# ref 2013, Mehdi Keramati, Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework, https://github.com/mehdiKeramati

#import libraries 
import numpy as np
import pylab
import cmath

class Environment():
    def __init__(self):
        self.transition = None
        self.outcome = None
        self.nonHomeostaticReward = None

        # Markov Decison Process(MDK) Parameters. #Definition of the Markov Decison Process FR1 - Timeout 20sec
        self.cocaine = 50             # Dose of self-administered drug
        self.leverPressCost = 20             # Energy cost for pressing the lever

        self.statesNum = 6              # number of stater 
        self.actionsNum = 2              # number of action   action 0 = Null     action 1 = Inactive Lever Press    action 2 = Active Lever Press
        self.initialExState = 0

        self.transition = np.zeros( [self.statesNum , self.actionsNum, self.statesNum] , float)
        self.setTransition(0,0,0,1)          # From state s, and by taking a, we go to state s', with probability p
        self.setTransition(0,1,1,1)
        self.setTransition(1,0,2,1)
        self.setTransition(1,1,2,1)
        self.setTransition(2,0,3,1)
        self.setTransition(2,1,3,1)
        self.setTransition(3,0,4,1)
        self.setTransition(3,1,4,1)
        self.setTransition(4,0,0,1)
        self.setTransition(4,1,0,1)

        self.outcome = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.setOutcome(0,1,1, self.cocaine)       # At state s, by doing action a and going to state s', we receive the outcome 

        self.nonHomeostaticReward = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.setNonHomeostaticReward(0,1,1,-self.leverPressCost)
        self.setNonHomeostaticReward(1,1,2,-self.leverPressCost)
        self.setNonHomeostaticReward(2,1,3,-self.leverPressCost)
        self.setNonHomeostaticReward(3,1,4,-self.leverPressCost)
        self.setNonHomeostaticReward(4,1,0,-self.leverPressCost)

        #Simulation Parameters
        self.animalsNum = 1                                  # Number of animals

        self.pretrainingHours = 0
        self.sessionsNum = 1                            # Number of sessions of cocain seeking, followed by rest in home-cage
        self.seekingHoursShA = 1            
        self.seekingHoursLgA = 24            
        self.extinctionHours = 0.75

        self.trialsPerHour = 60 * 60 / 4                            # Number of trials during one hour (as each trial is supposed to be 4 seconds)
        self.trialsPerDay = 24 * self.trialsPerHour
        self.pretrainingTrialsNum = self.pretrainingHours * self.trialsPerHour
        self.restAfterPretrainingTrialsNum = (24 - self.pretrainingHours) * self.trialsPerHour

        self.seekingTrialsNumShA = self.seekingHoursShA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        self.restingHoursShA = 24 - self.seekingHoursShA
        self.restTrialsNumShA = self.restingHoursShA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage

        self.seekingTrialsNumLgA = self.seekingHoursLgA * self.trialsPerHour    # Number of trials for each cocaine seeking session
        self.restingHoursLgA = 24 - self.seekingHoursLgA
        self.restTrialsNumLgA = self.restingHoursLgA * self.trialsPerHour    # Number of trials for each session of the animal being in the home cage

        self.extinctionTrialsNum = self.extinctionHours * self.trialsPerHour      # Number of trials for each extinction session

        self.totalTrialsNum = self.trialsPerDay

        # Plotting Parameters
        self.trialsPerBlock = 10*60/4 # Each BLOCK is 10 minutes - Each minute 60 second - Each trial takes 4 seconds

        # Logging Parameters
        self.nulDoingShA = np.zeros( [self.totalTrialsNum] , float)
        self.activeLeverPressShA = np.zeros( [self.totalTrialsNum] , float)
        self.internalStateShA = np.zeros( [self.totalTrialsNum] , float)
        self.setpointShA = np.zeros( [self.totalTrialsNum] , float)
        self.infusionShA = np.zeros( [self.totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaShA = np.zeros( [self.totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaShA = np.zeros( [self.totalTrialsNum] , float)

        self.nulDoingLgA = np.zeros( [self.totalTrialsNum] , float)
        self.activeLeverPressLgA = np.zeros( [self.totalTrialsNum] , float)
        self.internalStateLgA = np.zeros( [self.totalTrialsNum] , float)
        self.setpointLgA = np.zeros( [self.totalTrialsNum] , float)
        self.infusionLgA = np.zeros( [self.totalTrialsNum] , float)
        self.estimatedOutcomeUnderCocaLgA = np.zeros( [self.totalTrialsNum] , float)
        self.estimatedOutcomeNoCocaLgA = np.zeros( [self.totalTrialsNum] , float)

        #run simulation
        for animal in range(0, self.animalsNum):

            self.initialSetpoint = self.optimalInStateUpperBound
            self.initializeAnimal()
            self.cocaineSeeking(0 ,'LgA')

        self.plotting()

    def setTransition(self, state,action,nextState,transitionProbability): #Setting the transition function of the MDP
        self.transition[state][action][nextState] = transitionProbability

    def setOutcome(self, state,action,nextState,out): #Setting the outcome function of the MDP
        self.outcome[state][action][nextState] = out

    def setNonHomeostaticReward(self, state,action,nextState,rew): #Setting the non-homeostatic reward function of the MDP
        self.nonHomeostaticReward [state][action][nextState] = rew
        return

    def getTransition(self, s,a,nextS): #Return the probability of the transitions s-a->s'
        return self.transition[s][a][nextS]

    def getRealizedTransition(self, state,action): #Return the next state that the animal fell into
        index = np.random.uniform(0,1)
        probSum = 0
        for nextS in range(0, self.statesNum):
            probSum = probSum + self.getTransition(state,action,nextS)
            if index <= probSum:
                return nextS

    def getOutcome(self, state,action,nextState): #Obtained outcome
        return self.outcome[state,action,nextState]

    def getNonHomeostaticReward(self, state,action,nextState): #Obtained non-homeostatic reward
        return self.nonHomeostaticReward [state][action][nextState]

    def driveReductionReward(self, inState,setpointS,outcome): #Homeostatically-regulated Reward
        d1 = np.power(np.absolute(np.power(setpointS-inState, self.n*1.0)),(1.0/self.m))
        d2 = np.power(np.absolute(np.power(setpointS-inState-outcome, self.n*1.0)),(1.0/self.m))
        return d1-d2

    def underCocaine(self, inS, setS): #To what extend the animal is under the effect of cocaine
        underCocaRate = (inS - self.inStateLowerBound) / ( setS - self.inStateLowerBound )
        if underCocaRate>1: 
            underCocaRate = 1
        return underCocaRate

    def isActionAvailable(self, state,action): #Is action a available is state s? 
        probSum = 0 ;
        for i in range(0, self.statesNum):
            probSum = probSum + self.getTransition(state,action,i)
        if probSum == 1:
            return 1
        elif probSum == 0:
            return 0
        else:
            print("Error: There seems to be a problem in defining the transition function of the environment")

    def pretraining(self, ratType): #Pre-training Sessions
        self.exState = self.state[0]
        self.inState = self.state[1]
        self.setpointS = self.state[2]
        self.trialCount = self.state[3]
        self.cocBuffer = 0
        
        self.trialsNum = self.pretrainingTrialsNum
        
        for trial in range(0, self.trialsNum):
            self.estimatedActionValuesUnderCoca = self.valueEstimationUnderCoca(self.exState, self.inState, self.setpointS, self.searchDepth)
            self.estimatedActionValuesNoCoca = self.valueEstimationNoCoca(self.exState, self.inState, self.setpointS, self.searchDepth)        
            self.underCocaineWeight = self.underCocaine(inState, setpointS)
            self.estimatedActionValues = self.estimatedActionValuesUnderCoca * self.underCocaineWeight + self.estimatedActionValuesNoCoca * ( 1 - self.underCocaineWeight)         
            
            self.action = self.actionSelectionSoftmax(self.exState, self.estimatedActionValues)
            self.nextState = self.getRealizedTransition(self.exState, self.action)
            self.out = self.getOutcome (self.exState, self.action, self.nextState)
            self.nonHomeoRew = self.getNonHomeostaticReward(self.exState, self.action, self.nextState)
            self.HomeoRew = self.driveReductionReward(self.inState, self.setpointS, self.out)

            if ratType=='ShA':  
                self.loggingShA (trial, self.action,inState,setpointS, self.out)    
                print("ShA rat number: %d / %d Pre-training session trial: %d / %d animal seeking cocaine %d   %d" %(self.animal + 1, self.animalsNum, trial + 1, self.trialsNum, self.estimatedOutcomeUnderCoca[0][1][1], self.estimatedOutcomeNoCoca[0][1][1]))
            elif ratType=='LgA':  
                self.loggingLgA (trial, self.action,inState,setpointS, self.out)    
                print("LgA rat number: %d / %d Pre-training session trial: %d / %d animal seeking cocaine" %(self.animal + 1, self.animalsNum, trial + 1, self.trialsNum))

            self.updateOutcomeFunction(self.exState, self.action, self.nextState, self.out, self.underCocaineWeight)
            self.updateNonHomeostaticRewardFunction(self.exState, self.action, self.nextState, self.nonHomeoRew, self.underCocaineWeight)
            self.updateTransitionFunction(self.exState, self.action, self.nextState, self.underCocaineWeight)            
            
            self.cocBuffer = self.cocBuffer + self.out                
            inState     = self.updateInState(inState,cocBuffer * self.cocAbsorptionRatio)
            setpointS   = self.updateSetpoint(setpointS, self.out)
            cocBuffer = cocBuffer * (1 - self.cocAbsorptionRatio)
            exState   = self.nextState

        self.state[0]    = exState
        self.state[1]    = inState
        self.state[2]    = setpointS
        self.state[3]    = self.trialCount + self.trialsNum
 
    def cocaineSeeking(self, sessionNum, ratType): #Cocaine Seeking Sessions

        self.exState = self.state[0]
        self.inState     = self.state[1]
        self.setpointS   = self.state[2]
        self.trialCount  = self.state[3]
        self.cocBuffer   = 0
        
        if ratType=='ShA':  
            trialsNum = self.seekingTrialsNumShA    
        if ratType=='LgA':  
            trialsNum = self.seekingTrialsNumLgA    
        
        for trial in range(0, self.trialsPerDay):
            # estimatedActionValuesUnderCoca = valueEstimationUnderCoca( exState, inState, setpointS, searchDepth )
            # estimatedActionValuesNoCoca = valueEstimationNoCoca( exState, inState, setpointS, searchDepth )        
            # underCocaineWeight = underCocaine( inState , setpointS)
            # estimatedActionValues = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         

            self.action = 0
            self.nextState =0
            self.nextState = 0
            self.HomeoRew = 0
            if trial==0:
                self.out = self.cocaine
            else:
                self.out = 0
            
            if ratType=='LgA':  
                self.loggingLgA(trial, self.action, self.inState, self.setpointS, self.out)    
                print("LgA rat number: %d / %d     Session Number: %d / %d     trial: %d / %d      animal seeking cocaine" %(self.animal + 1, self.animalsNum, sessionNum + 1, self.sessionsNum, trial - self.trialCount+1, self.trialsNum))

            #updateOutcomeFunction( exState , action , nextState , out ,underCocaineWeight)
            #updateNonHomeostaticRewardFunction( exState , action , nextState , nonHomeoRew ,underCocaineWeight)
            #updateTransitionFunction( exState , action , nextState , underCocaineWeight)            
            
            self.cocBuffer = self.cocBuffer + self.out                
            
            self.inState     = self.updateInState(self.inState, self.cocBuffer * self.cocAbsorptionRatio)
            #setpointS = updateSetpoint(setpointS,out)

            self.cocBuffer = self.cocBuffer*(1 - self.cocAbsorptionRatio)

            #exState = nextState

        self.state[0] = self.exState
        self.state[1] = self.inState
        self.state[2] = self.setpointS
        self.state[3] = self.trialCount + trialsNum

    def homeCage (self, sessionNum, ratType): #Home-cage Sessions
        exState = self.state[0]
        inState = self.state[1]
        setpointS = self.state[2]
        trialCount = self.state[3]
    
        if ratType=='ShA':  
            trialsNum = self.restTrialsNumShA    
            print("ShA rat number: %d / %d Session Number: %d / %d animal rests in home cage" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum))
        elif ratType=='LgA':  
            trialsNum = self.restTrialsNumLgA
            print("LgA rat number: %d / %d Session Number: %d / %d animal rests in home cage" %(self.animal+1, self.animalsNum, sessionNum+1, self.sessionsNum))
        elif ratType=='afterPretrainingShA':  
            trialsNum = self.restAfterPretrainingTrialsNum    
            print("ShA rat number: %d / %d     After pretraining animal rests in home cage" %(self.animal+1, self.animalsNum))
        elif ratType=='afterPretrainingLgA':  
            trialsNum = self.restAfterPretrainingTrialsNum    
            print("LgA rat number: %d / %d     After pretraining animal rests in home cage" %(self.animal+1, self.animalsNum))
        
        for trial in range(trialCount,trialCount+trialsNum):

            inState = self.updateInState(inState,0)
            setpointS = self.updateSetpoint(setpointS,0)

            if ratType=='ShA':  
                self.loggingShA(trial,0,inState,setpointS,0)    
            elif ratType=='LgA':  
                self.loggingLgA(trial,0,inState,setpointS,0)    
            elif ratType=='afterPretrainingShA':  
                self.loggingShA(trial,0,inState,setpointS,0)    
            elif ratType=='afterPretrainingLgA':  
                self.loggingLgA(trial,0,inState,setpointS,0)    

        self.state[0] = exState
        self.state[1] = inState
        self.state[2] = setpointS
        self.state[3] = trialCount+trialsNum

    def loggingShA(self, trial,action,inState,setpointS,coca): #Logging the current information for the Short-access group
    
        if action==0: 
            self.nulDoingShA[trial] = self.nulDoingShA[trial] + 1
        elif action==1: 
            self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial] + 1
        self.internalStateShA[trial] = self.internalStateShA[trial] + inState
        self.setpointShA[trial] = self.setpointShA[trial] + setpointS    
        if coca==self.cocaine:
            self.infusionShA[trial] = self.infusionShA[trial] + 1

        self.estimatedOutcomeUnderCocaShA[trial] = self.estimatedOutcomeUnderCocaShA [trial] + self.estimatedOutcomeUnderCoca[0][1][1]
        self.estimatedOutcomeNoCocaShA[trial] = self.estimatedOutcomeNoCocaShA [trial] + self.estimatedOutcomeNoCoca[0][1][1]

    def loggingLgA(self, trial, action, inState, setpointS, coca): #Logging the current information for the Long-access group
        self.internalStateLgA[trial] = self.internalStateLgA[trial] + inState
        self.setpointLgA[trial] = self.setpointLgA[trial] + setpointS    
        if coca == self.cocaine:
            self.infusionLgA[trial] = self.infusionLgA[trial] + 1

        self.estimatedOutcomeUnderCocaLgA [trial] = self.estimatedOutcomeUnderCocaLgA[trial] + self.estimatedOutcomeUnderCoca[0][1][1]
        self.estimatedOutcomeNoCocaLgA    [trial] = self.estimatedOutcomeNoCocaLgA[trial] + self.estimatedOutcomeNoCoca[0][1][1]

    def loggingFinalization(self): #Wrap up all the logged data
        
        for trial in range(0, self.totalTrialsNum):
            self.nulDoingShA[trial] = self.nulDoingShA[trial]/self.animalsNum
            self.activeLeverPressShA[trial] = self.activeLeverPressShA[trial]/self.animalsNum
            self.internalStateShA[trial] = self.internalStateShA[trial]/self.animalsNum
            self.setpointShA[trial] = self.setpointShA[trial]/self.animalsNum  
            self.infusionShA[trial] = self.infusionShA[trial]/self.animalsNum 
            self.estimatedOutcomeUnderCocaShA[trial] = self.estimatedOutcomeUnderCocaShA[trial]/self.animalsNum
            self.estimatedOutcomeNoCocaShA[trial] = self.estimatedOutcomeNoCocaShA[trial]/self.animalsNum

            self.nulDoingLgA[trial] = self.nulDoingLgA[trial]/self.animalsNum
            self.activeLeverPressLgA[trial] = self.activeLeverPressLgA[trial]/self.animalsNum
            self.internalStateLgA[trial] = self.internalStateLgA[trial]/self.animalsNum
            self.setpointLgA[trial] = self.setpointLgA[trial]/self.animalsNum  
            self.infusionLgA[trial] = self.infusionLgA[trial]/self.animalsNum 
            self.estimatedOutcomeUnderCocaLgA [trial] = self.estimatedOutcomeUnderCocaLgA[trial]/self.animalsNum
            self.estimatedOutcomeNoCocaLgA    [trial] = self.estimatedOutcomeNoCocaLgA[trial]/self.animalsNum

    def plotInternalState45(self): #Plot the internal state
        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
        pylab.rcParams.update({'legend.fontsize': 16})
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)



        ax1 = fig1.add_subplot(111)
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

        #    for i in range ( 0 , 5 ):
        #        if i%2==0:
        #            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        ax1.set_ylabel('Internal State')
        ax1.set_xlabel('Time (min)')
        ax1.set_title('')
        fig1.savefig('internalState45min.eps', format='eps')

    def plotInternalState5(self): #Plot the internal state

        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)


        ax1 = fig1.add_subplot(111)
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

        #    for i in range ( 0 , 5 ):
        #        if i%2==0:
        #            p = pylab.axvspan( i*extinctionTrialsNum + i, (i+1)*extinctionTrialsNum + i , facecolor='0.75',edgecolor='none', alpha=0.5)        
        
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_markeredgewidth(2)
            line.set_markersize(5)

        ax1.set_ylabel('Internal State')
        ax1.set_xlabel('Time (sec)')
        ax1.set_title('')
        fig1.savefig('internalState4min.eps', format='eps')

    def plotting(self): #Plot all the results

        self.loggingFinalization()

        self.plotInternalState45() 
        self.plotInternalState5() 

        pylab.show()

class Animal():
    def __init__(self):
        self.estimatedTransitionUnderCoca = None
        self.estimatedNonHomeostaticRewardUnderCoca = None
        self.estimatedOutcomeUnderCoca = None

        self.estimatedTransitionNoCoca = None
        self.estimatedNonHomeostaticRewardNoCoca = None
        self.estimatedOutcomeNoCoca = None

    def initializeAnimal(self): #Create a new animal, (state-action q-table)
        self.state[0] = self.initialExState
        self.state[1] = self.initialInState
        self.state[2] = self.initialSetpoint 
        self.state[3] = 0 
            
        for i in range(0, self.statesNum):
            for j in range(0, self.actionsNum):
                for k in range(0, self.statesNum):
                    self.estimatedTransitionUnderCoca            [i][j][k] = 0.0
                    self.estimatedOutcomeUnderCoca               [i][j][k] = 0.0
                    self.estimatedNonHomeostaticRewardUnderCoca  [i][j][k] = 0.0
                    self.estimatedTransitionNoCoca               [i][j][k] = 0.0
                    self.estimatedOutcomeNoCoca                  [i][j][k] = 0.0
                    self.estimatedNonHomeostaticRewardNoCoca     [i][j][k] = 0.0

        #set all q-values to 1. 
        # state-action q-table under cocaine 
        self.estimatedTransitionUnderCoca [0][0][0] = 1
        self.estimatedTransitionUnderCoca [0][1][1] = 1
        self.estimatedTransitionUnderCoca [1][0][2] = 1
        self.estimatedTransitionUnderCoca [1][1][2] = 1
        self.estimatedTransitionUnderCoca [2][0][3] = 1
        self.estimatedTransitionUnderCoca [2][1][3] = 1
        self.estimatedTransitionUnderCoca [3][0][4] = 1
        self.estimatedTransitionUnderCoca [3][1][4] = 1
        self.estimatedTransitionUnderCoca [4][0][0] = 1
        self.estimatedTransitionUnderCoca [4][1][0] = 1

        #state-action q-table without cocaine 
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

        #Assuming that the animals know the energy cost (fatigue) of pressing a lever - ros, this? https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2656.13040
        for i in range(0, self.statesNum):
            for j in range(0, self.statesNum):
                self.estimatedNonHomeostaticRewardUnderCoca[i][1][j] = -self.leverPressCost
                self.estimatedNonHomeostaticRewardNoCoca[i][1][j] = -self.leverPressCost

        self.estimatedOutcomeUnderCoca[0][1][1] = self.cocaine
        self.estimatedOutcomeNoCoca[0][1][1] = self.cocaine

    #value estimator
    def valueEstimationUnderCoca(self, state,inState,setpointS,depthLeft): #Goal-directed Value estimation, Assuming the animal is under Cocaine

        values = np.zeros ( [self.actionsNum] , float )

        # If this is the last depth that should be searched :
        if depthLeft==1:
            for action in range(0, self.actionsNum):
                for nextState in range(0, self.statesNum):
                    homeoReward    = self.driveReductionReward(inState,setpointS, self.cocaine) * self.estimatedOutcomeUnderCoca[state][action][nextState]/cocaine
                    nonHomeoReward = self.estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
                    transitionProb = self.estimatedTransitionUnderCoca[state][action][nextState]
                    values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
            return values
        
        # Otherwise :
        for action in range(0, self.actionsNum):
            for nextState in range(0, self.statesNum):
                if self.estimatedTransitionUnderCoca[state][action][nextState] < self.pruningThreshold :
                    VNextStateBest = 0
                else:    
                    VNextState = self.valueEstimationUnderCoca(nextState,setpointS,inState,depthLeft-1)
                    VNextStateBest = self.maxValue (VNextState)
                homeoReward    = self.driveReductionReward(inState,setpointS, self.cocaine) * self.estimatedOutcomeUnderCoca[state][action][nextState]/self.cocaine
                nonHomeoReward = self.estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
                transitionProb = self.estimatedTransitionUnderCoca[state][action][nextState]
                values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + self.gamma*VNextStateBest ) 
                
        return values

    def valueEstimationNoCoca(self, state,inState,setpointS,depthLeft): #Goal-directed Value estimation, Assuming the animal is not under Cocaine 

        values = np.zeros ( [self.actionsNum] , float )

        # If this is the last depth that should be searched :
        if depthLeft==1:
            for action in range(0, self.actionsNum):
                for nextState in range(0, self.statesNum):
                    homeoReward    = self.driveReductionReward(inState,setpointS, self.cocaine) * self.estimatedOutcomeNoCoca[state][action][nextState]/self.cocaine
                    nonHomeoReward = self.estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
                    transitionProb = self.estimatedTransitionNoCoca[state][action][nextState]
                    values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
            return values
        
        # Otherwise :
        for action in range(0, self.actionsNum):
            for nextState in range(0, self.statesNum):
                if self.estimatedTransitionNoCoca[state][action][nextState] < self.pruningThreshold :
                    VNextStateBest = 0
                else:    
                    VNextState = self.valueEstimationNoCoca(nextState,setpointS,inState,depthLeft-1)
                    VNextStateBest = self.maxValue (VNextState)
                homeoReward    = self.driveReductionReward(inState,setpointS, self.cocaine) * self.estimatedOutcomeNoCoca[state][action][nextState]/ self.cocaine
                nonHomeoReward = self.estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
                transitionProb = self.estimatedTransitionNoCoca[state][action][nextState]
                values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + self.gamma*VNextStateBest ) 
                
        return values
    
    def maxValue(self, V): # Max ( Value[nextState,a] ) : for all a
        maxV = V[0]
        for action in range(0, self.actionsNum):
            if V[action]>maxV:
                maxV = V[action]    
        return maxV

    def actionSelectionSoftmax(self, state,V): #Action Selection : Softmax
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

    def updateInState(self, inState,outcome): #Update internal state upon consumption
        interS = inState + outcome - self.cocaineDegradationRate*(inState- self.inStateLowerBound)
        if interS < self.inStateLowerBound:
            interS = self.inStateLowerBound
        return interS

    def updateSetpoint(self, optimalInState,out): #Update the homeostatic setpoint (Allostatic mechanism)
        optInS = optimalInState + out * self.setpointShiftRate - self.setpointRecoveryRate

        if optInS < self.optimalInStateLowerBound:
            optInS = self.optimalInStateLowerBound

        if optInS > self.optimalInStateUpperBound:
            optInS = self.optimalInStateUpperBound
        return optInS

    def updateOutcomeFunction(self, state,action,nextState,out,underCocaWeight): #Update the expected-outcome function

        learningRateUnderCoca = self.updateOutcomeRate * underCocaWeight * self.cocaineInducedLearningRateDeficiency
        learningRateNoCoca = self.updateOutcomeRate * (1 - underCocaWeight)

        self.estimatedOutcomeUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca) * self.estimatedOutcomeUnderCoca[state][action][nextState] +     learningRateUnderCoca*out
        self.estimatedOutcomeNoCoca[state][action][nextState] = (1.0-learningRateNoCoca   ) * self.estimatedOutcomeNoCoca   [state][action][nextState] +     learningRateNoCoca   *out

    def updateNonHomeostaticRewardFunction(self, state,action,nextState,rew,underCocaWeight): #Update the expected-non-homeostatic-reward function
        
        learningRateUnderCoca = self.updateOutcomeRate * underCocaWeight
        learningRateNoCoca= self.updateOutcomeRate * (1 - underCocaWeight)
        
        self.estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca) * self.estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] +     learningRateUnderCoca*rew
        self.estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] = (1.0-learningRateNoCoca   ) * self.estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] +     learningRateNoCoca   *rew

    def updateTransitionFunction(self, state,action,nextState,underCocaWeight): #Update the expected-transition function
        learningRateUnderCoca = self.updateOutcomeRate * underCocaWeight
        learningRateNoCoca    = self.updateOutcomeRate * (1 - underCocaWeight)
    
        #---- First inhibit all associations
        for i in range(0, self.statesNum):
            self.estimatedTransitionUnderCoca[state][action][i] = (1.0-learningRateUnderCoca) * self.estimatedTransitionUnderCoca[state][action][i]
            self.estimatedTransitionNoCoca   [state][action][i] = (1.0-learningRateNoCoca   ) * self.estimatedTransitionNoCoca   [state][action][i]
        
        #---- Then potentiate the experiences association
        self.estimatedTransitionUnderCoca[state][action][nextState] = self.estimatedTransitionUnderCoca[state][action][nextState] + learningRateUnderCoca
        self.estimatedTransitionNoCoca   [state][action][nextState] = self.estimatedTransitionNoCoca   [state][action][nextState] + learningRateNoCoca

    #Definition of the Animal
    def homeostaticSystem(self):
        self.initialInState          = 0
        self.initialSetpoint         = 200
        self.inStateLowerBound       = 0
        self.cocaineDegradationRate  = 0.007    # Dose of cocaine that the animal loses in every time-step
        self.cocAbsorptionRatio      = 0.12      # Proportion of the injected cocaine that affects the brain right after infusion 

    def allostaticSystem(self):
        self.setpointShiftRate       = 0.0018
        self.setpointRecoveryRate    = 0.00016
        self.optimalInStateLowerBound= 100
        self.optimalInStateUpperBound= 200

    def driveFunction(self):
        self.m = 3     # Parameter of the drive function : m-th root
        self.n = 4     # Parameter of the drive function : n-th pawer

    def goalDirectedSystem(self):
        self.updateOutcomeRate = 0.025 # Learning rate for updating the outcome function
        self.cocaineInducedLearningRateDeficiency = 0.15
        self.updateTransitionRate = 0.2   # Learning rate for updating the transition function
        self.updateRewardRate = 0.2   # Learning rate for updating the non-homeostatic reward function
        self.gamma = 1     # Discount factor
        self.beta = 0.25  # Rate of exploration
        self.searchDepth = 3     # Depth of going into the decision tree for goal-directed valuation of choices
        self.pruningThreshold = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

        self.estimatedTransitionUnderCoca = np.zeros( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedOutcomeUnderCoca = np.zeros( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedNonHomeostaticRewardUnderCoca = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedTransitionNoCoca = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedOutcomeNoCoca = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )
        self.estimatedNonHomeostaticRewardNoCoca = np.zeros ( [self.statesNum , self.actionsNum , self.statesNum] , float )

        self.state = np.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial