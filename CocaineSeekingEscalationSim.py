# ref 2004: original experiment results; https://onlinelibrary.wiley.com/doi/10.1046/j.1471-4159.2003.01833.x
# ref 2013, Mehdi Keramati, Escalation of Cocaine-Seeking in the Homeostatic Reinforcement Learning Framework, https://github.com/mehdiKeramati

#import libraries 
import numpy as np

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
        for nextS in range(0,statesNum):
            probSum = probSum + self.getTransition(state,action,nextS)
            if index <= probSum:
                return nextS

    def getOutcome(self, state,action,nextState): #Obtained outcome
        return self.outcome[state,action,nextState]

    def getNonHomeostaticReward(self, state,action,nextState): #Obtained non-homeostatic reward
        return self.nonHomeostaticReward [state][action][nextState]

    def driveReductionReward(inState,setpointS,outcome): #Homeostatically-regulated Reward
        d1 = np.power(np.absolute(np.power(setpointS-inState,n*1.0)),(1.0/m))
        d2 = np.power(np.absolute(np.power(setpointS-inState-outcome,n*1.0)),(1.0/m))
        return d1-d2

    def underCocaine(self, inS, setS): #To what extend the animal is under the effect of cocaine
        underCocaRate = (inS - inStateLowerBound) / ( setS - inStateLowerBound )
        if underCocaRate>1: 
            underCocaRate = 1
        return underCocaRate

    def isActionAvailable(self, state,action): #Is action a available is state s? 
        probSum = 0 ;
        for i in range(0,statesNum):
            probSum = probSum + self.getTransition(state,action,i)
        if probSum == 1:
            return 1
        elif probSum == 0:
            return 0
        else:
            print("Error: There seems to be a problem in defining the transition function of the environment")

    def pretraining  (ratType): #Pre-training Sessions

        exState     = state[0]
        inState     = state[1]
        setpointS   = state[2]
        trialCount  = state[3]
        cocBuffer   = 0
        
        trialsNum = pretrainingTrialsNum
        
        for trial in range(0,trialsNum):

            estimatedActionValuesUnderCoca   = valueEstimationUnderCoca ( exState, inState, setpointS, searchDepth )
            estimatedActionValuesNoCoca      = valueEstimationNoCoca    ( exState, inState, setpointS, searchDepth )        
            underCocaineWeight               = underCocaine             ( inState , setpointS                      )
            estimatedActionValues            = estimatedActionValuesUnderCoca*underCocaineWeight + estimatedActionValuesNoCoca*(1-underCocaineWeight)         
            
            action                          = actionSelectionSoftmax    ( exState , estimatedActionValues           )
            nextState                       = getRealizedTransition     ( exState , action                          )
            out                             = getOutcome                ( exState , action    , nextState           )
            nonHomeoRew                     = getNonHomeostaticReward   ( exState , action    , nextState           )
            HomeoRew                        = driveReductionReward      ( inState , setpointS , out                 )

            if ratType=='ShA':  
                loggingShA (trial,action,inState,setpointS,out)    
                print("ShA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine  %d   %d" %(animal+1,animalsNum,trial+1,trialsNum, estimatedOutcomeUnderCoca[0][1][1],estimatedOutcomeNoCoca[0][1][1]))
            elif ratType=='LgA':  
                loggingLgA (trial,action,inState,setpointS,out)    
                print("LgA rat number: %d / %d     Pre-training session     trial: %d / %d      animal seeking cocaine" %(animal+1,animalsNum,trial+1,trialsNum))

            updateOutcomeFunction               ( exState , action , nextState , out ,underCocaineWeight            )
            updateNonHomeostaticRewardFunction  ( exState , action , nextState , nonHomeoRew ,underCocaineWeight    )
            updateTransitionFunction            ( exState , action , nextState , underCocaineWeight                 )            
            
            cocBuffer = cocBuffer + out                
            
            inState     = updateInState(inState,cocBuffer*cocAbsorptionRatio)
            setpointS   = updateSetpoint(setpointS,out)

            cocBuffer = cocBuffer*(1-cocAbsorptionRatio)

            exState   = nextState

        state[0]    = exState
        state[1]    = inState
        state[2]    = setpointS
        state[3]    = trialCount+trialsNum

    def cocaineSeeking  (sessionNum , ratType): #Cocaine Seeking Sessions

        exState     = state[0]
        inState     = state[1]
        setpointS   = state[2]
        trialCount  = state[3]
        cocBuffer   = 0
        
        if ratType=='ShA':  
            trialsNum = seekingTrialsNumShA    
        if ratType=='LgA':  
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
            
            if ratType=='LgA':  
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

        state[0]    = exState
        state[1]    = inState
        state[2]    = setpointS
        state[3]    = trialCount+trialsNum

    def homeCage (sessionNum, ratType): #Home-cage Sessions

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

    def loggingShA(trial,action,inState,setpointS,coca): #Logging the current information for the Short-access group
    
        if action==0: 
            nulDoingShA[trial]             = nulDoingShA[trial] + 1
        elif action==1: 
            activeLeverPressShA[trial]     = activeLeverPressShA[trial] + 1
        internalStateShA[trial]    = internalStateShA[trial] + inState
        setpointShA[trial]         = setpointShA[trial] + setpointS    
        if coca==cocaine:
            infusionShA[trial]     = infusionShA[trial] + 1

        estimatedOutcomeUnderCocaShA [trial] = estimatedOutcomeUnderCocaShA [trial] + estimatedOutcomeUnderCoca[0][1][1]
        estimatedOutcomeNoCocaShA    [trial] = estimatedOutcomeNoCocaShA    [trial] + estimatedOutcomeNoCoca   [0][1][1]

    def loggingLgA(trial,action,inState,setpointS,coca): #Logging the current information for the Long-access group
    
        internalStateLgA[trial]    = internalStateLgA[trial] + inState
        setpointLgA[trial]         = setpointLgA[trial] + setpointS    
        if coca==cocaine:
            infusionLgA[trial]     = infusionLgA[trial] + 1

        estimatedOutcomeUnderCocaLgA [trial] = estimatedOutcomeUnderCocaLgA [trial] + estimatedOutcomeUnderCoca[0][1][1]
        estimatedOutcomeNoCocaLgA    [trial] = estimatedOutcomeNoCocaLgA    [trial] + estimatedOutcomeNoCoca   [0][1][1]

    def loggingFinalization(): #Wrap up all the logged data
        
        for trial in range(0,totalTrialsNum):
            nulDoingShA[trial]             = nulDoingShA[trial]/animalsNum
            activeLeverPressShA[trial]     = activeLeverPressShA[trial]/animalsNum
            internalStateShA[trial]        = internalStateShA[trial]/animalsNum
            setpointShA[trial]             = setpointShA[trial]/animalsNum  
            infusionShA[trial]             = infusionShA[trial]/animalsNum 
            estimatedOutcomeUnderCocaShA [trial] = estimatedOutcomeUnderCocaShA [trial]/animalsNum
            estimatedOutcomeNoCocaShA    [trial] = estimatedOutcomeNoCocaShA    [trial]/animalsNum

            nulDoingLgA[trial]             = nulDoingLgA[trial]/animalsNum
            activeLeverPressLgA[trial]     = activeLeverPressLgA[trial]/animalsNum
            internalStateLgA[trial]        = internalStateLgA[trial]/animalsNum
            setpointLgA[trial]             = setpointLgA[trial]/animalsNum  
            infusionLgA[trial]             = infusionLgA[trial]/animalsNum 
            estimatedOutcomeUnderCocaLgA [trial] = estimatedOutcomeUnderCocaLgA [trial]/animalsNum
            estimatedOutcomeNoCocaLgA    [trial] = estimatedOutcomeNoCocaLgA    [trial]/animalsNum


    def plotInternalState45(): #Plot the internal state

        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
        pylab.rcParams.update({'legend.fontsize': 16})
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)



        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(internalStateLgA[0:675] , linewidth = 2 , color='black' )

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

    def plotInternalState5(): #Plot the internal state

        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
            
        fig1 = pylab.figure( figsize=(5,3.5) )
        fig1.subplots_adjust(left=0.16)
        fig1.subplots_adjust(bottom=0.2)


        ax1 = fig1.add_subplot(111)
        S0 = ax1.plot(internalStateLgA[0:60] , linewidth = 2 , color='black' )

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

    def plotting(): #Plot all the results

        loggingFinalization()

        plotInternalState45() 
        plotInternalState5() 

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
        self.state[0] = initialExState
        self.state[1] = initialInState
        self.state[2] = initialSetpoint 
        self.state[3] = 0 
            
        for i in range(0,statesNum):
            for j in range(0,actionsNum):
                for k in range(0,statesNum):
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
        for i in range(0,statesNum):
            for j in range(0,statesNum):
                self.estimatedNonHomeostaticRewardUnderCoca      [i][1][j] = -leverPressCost
                self.estimatedNonHomeostaticRewardNoCoca         [i][1][j] = -leverPressCost

        self.estimatedOutcomeUnderCoca [0][1][1]     = cocaine
        self.estimatedOutcomeNoCoca    [0][1][1]     = cocaine

    #value estimator
    def valueEstimationUnderCoca(self, state,inState,setpointS,depthLeft): #Goal-directed Value estimation, Assuming the animal is under Cocaine

        values = numpy.zeros ( [actionsNum] , float )

        # If this is the last depth that should be searched :
        if depthLeft==1:
            for action in range(0,actionsNum):
                for nextState in range(0,statesNum):
                    homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeUnderCoca[state][action][nextState]/cocaine
                    nonHomeoReward = estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
                    transitionProb = estimatedTransitionUnderCoca[state][action][nextState]
                    values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
            return values
        
        # Otherwise :
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                if estimatedTransitionUnderCoca[state][action][nextState] < pruningThreshold :
                    VNextStateBest = 0
                else:    
                    VNextState = valueEstimationUnderCoca(nextState,setpointS,inState,depthLeft-1)
                    VNextStateBest = maxValue (VNextState)
                homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeUnderCoca[state][action][nextState]/cocaine
                nonHomeoReward = estimatedNonHomeostaticRewardUnderCoca[state][action][nextState]
                transitionProb = estimatedTransitionUnderCoca[state][action][nextState]
                values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
                
        return values

    def valueEstimationNoCoca(self, state,inState,setpointS,depthLeft): #Goal-directed Value estimation, Assuming the animal is not under Cocaine 

        values = numpy.zeros ( [actionsNum] , float )

        # If this is the last depth that should be searched :
        if depthLeft==1:
            for action in range(0,actionsNum):
                for nextState in range(0,statesNum):
                    homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeNoCoca[state][action][nextState]/cocaine
                    nonHomeoReward = estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
                    transitionProb = estimatedTransitionNoCoca[state][action][nextState]
                    values[action] = values[action] +  transitionProb * ( homeoReward + nonHomeoReward )
            return values
        
        # Otherwise :
        for action in range(0,actionsNum):
            for nextState in range(0,statesNum):
                if estimatedTransitionNoCoca[state][action][nextState] < pruningThreshold :
                    VNextStateBest = 0
                else:    
                    VNextState = valueEstimationNoCoca(nextState,setpointS,inState,depthLeft-1)
                    VNextStateBest = maxValue (VNextState)
                homeoReward    = driveReductionReward(inState,setpointS,cocaine)*estimatedOutcomeNoCoca[state][action][nextState]/cocaine
                nonHomeoReward = estimatedNonHomeostaticRewardNoCoca[state][action][nextState]
                transitionProb = estimatedTransitionNoCoca[state][action][nextState]
                values[action] = values[action] + transitionProb * ( homeoReward + nonHomeoReward + gamma*VNextStateBest ) 
                
        return values
    
    def maxValue(self, V): # Max ( Value[nextState,a] ) : for all a
        maxV = V[0]
        for action in range(0,actionsNum):
            if V[action]>maxV:
                maxV = V[action]    
        return maxV

    def actionSelectionSoftmax(state,V): #Action Selection : Softmax
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

    def updateInState(inState,outcome): #Update internal state upon consumption
        interS = inState + outcome - cocaineDegradationRate*(inState-inStateLowerBound)
        if interS<inStateLowerBound:
            interS=inStateLowerBound
        return interS

    def updateSetpoint(optimalInState,out): #Update the homeostatic setpoint (Allostatic mechanism)
        optInS = optimalInState + out*setpointShiftRate - setpointRecoveryRate

        if optInS<optimalInStateLowerBound:
            optInS=optimalInStateLowerBound

        if optInS>optimalInStateUpperBound:
            optInS=optimalInStateUpperBound
        return optInS

    def updateOutcomeFunction(state,action,nextState,out,underCocaWeight): #Update the expected-outcome function

        learningRateUnderCoca = updateOutcomeRate * underCocaWeight * cocaineInducedLearningRateDeficiency
        learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)

        estimatedOutcomeUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca)*estimatedOutcomeUnderCoca[state][action][nextState] +     learningRateUnderCoca*out
        estimatedOutcomeNoCoca   [state][action][nextState] = (1.0-learningRateNoCoca   )*estimatedOutcomeNoCoca   [state][action][nextState] +     learningRateNoCoca   *out

    def updateNonHomeostaticRewardFunction(state,action,nextState,rew,underCocaWeight): #Update the expected-non-homeostatic-reward function
        
        learningRateUnderCoca = updateOutcomeRate * underCocaWeight
        learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)
        
        estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] = (1.0-learningRateUnderCoca)*estimatedNonHomeostaticRewardUnderCoca[state][action][nextState] +     learningRateUnderCoca*rew
        estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] = (1.0-learningRateNoCoca   )*estimatedNonHomeostaticRewardNoCoca   [state][action][nextState] +     learningRateNoCoca   *rew

    def updateTransitionFunction(state,action,nextState,underCocaWeight): #Update the expected-transition function
        learningRateUnderCoca = updateOutcomeRate * underCocaWeight
        learningRateNoCoca    = updateOutcomeRate * (1 - underCocaWeight)
    
        #---- First inhibit all associations
        for i in range(0,statesNum):
            estimatedTransitionUnderCoca[state][action][i] = (1.0-learningRateUnderCoca)*estimatedTransitionUnderCoca[state][action][i]
            estimatedTransitionNoCoca   [state][action][i] = (1.0-learningRateNoCoca   )*estimatedTransitionNoCoca   [state][action][i]
        
        #---- Then potentiate the experiences association
        estimatedTransitionUnderCoca[state][action][nextState] = estimatedTransitionUnderCoca[state][action][nextState] + learningRateUnderCoca
        estimatedTransitionNoCoca   [state][action][nextState] = estimatedTransitionNoCoca   [state][action][nextState] + learningRateNoCoca

    #Definition of the Animal
    def homeostaticSystem(self):
        initialInState          = 0
        initialSetpoint         = 200
        inStateLowerBound       = 0
        cocaineDegradationRate  = 0.007    # Dose of cocaine that the animal loses in every time-step
        cocAbsorptionRatio      = 0.12      # Proportion of the injected cocaine that affects the brain right after infusion 

    def allostaticSystem(self):
        setpointShiftRate       = 0.0018
        setpointRecoveryRate    = 0.00016
        optimalInStateLowerBound= 100
        optimalInStateUpperBound= 200

    def driveFunction(self):
        self.m = 3     # Parameter of the drive function : m-th root
        self.n = 4     # Parameter of the drive function : n-th pawer

    def goalDirectedSystem(self):
        updateOutcomeRate       = 0.025 # Learning rate for updating the outcome function
        cocaineInducedLearningRateDeficiency = 0.15
        updateTransitionRate    = 0.2   # Learning rate for updating the transition function
        updateRewardRate        = 0.2   # Learning rate for updating the non-homeostatic reward function
        gamma                   = 1     # Discount factor
        beta                    = 0.25  # Rate of exploration
        searchDepth             = 3     # Depth of going into the decision tree for goal-directed valuation of choices
        pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

        estimatedTransitionUnderCoca             = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        estimatedOutcomeUnderCoca                = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        estimatedNonHomeostaticRewardUnderCoca   = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        estimatedTransitionNoCoca                = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        estimatedOutcomeNoCoca                   = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        estimatedNonHomeostaticRewardNoCoca      = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

        state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial