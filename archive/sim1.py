import numpy as np

class Environment():
    def __init__(self):
        pass

    def setupParameters(self):
        self.animalNum = 1
        self.pretrainingHours = .1 #Number of sessions of cocain seeking, followed by rest in home-cage
        self.sessionsNum = 1
        self.seekingHoursShA = 0.1 # shA = short access. Experienced seven days of one-hour sessions followed by 10 days of six-hour        
        self.seekingHoursLgA = 0.1 # lgA = long access. Experienced daily two-hour cocaine self-administration sessions until reaching 10 days of 10 or more cocaine infusions/session
        self.extinctionHours = 0
        self.trialsPerHour = 900 #900 secs or four 15 mins per hour
        self.trialsPerDay = 24*self.trialsPerHour
        self.pretrainingTrialsNum= self.pretrainingHours* self.trialsPerHour
        self.restAfterPretrainingTrialsNum = (24 - self.pretrainingHours) * self.trialsPerHour


class Animal():
    def __init__(self):
        pass

    def homeostaticSystem(self):
        self.initialInState = 0
        self.initialSetpoint = 100
        self.inStateLowerBound = 0
        self.cocaineDegradationRate = 0.007 # cocaine dose lost in every time-step
        self.cocaineAbsorptionRation = 0.12 # amount of injected cocaine that affects brain after infusion

    def allostaticSystem(self):
        self.setpointShiftRate = 0.0018
        self.setpointRecoveryRate = 0.00016
        self.optimalInStateLowerBound = 100
        self.optimalInStateUpperBound = 200

    def drive(self):
        self.m = 3 #m-th root
        self.n = 4 #n-th power

    def goalDirectedSystem(self):
        self.updateOutcomeRate       = 0.2  # Learning rate for updating the outcome function
        self.updateTransitionRate    = 0.2  # Learning rate for updating the transition function
        self.updateRewardRate        = 0.2  # Learning rate for updating the non-homeostatic reward function
        self.gamma                   = 1     # Discount factor
        self.beta                    = 0.25  # Rate of exploration
        self.searchDepth             = 3     # Depth of going into the decision tree for goal-directed valuation of choices
        self.pruningThreshold        = 0.1   # If the probability of a transition like (s,a,s') is less than "pruningThreshold", cut it from the decision tree 

        self.estimatedTransition              = numpy.zeros( [statesNum , actionsNum, statesNum] , float)
        self.estimatedOutcome                 = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )
        self.estimatedNonHomeostaticReward    = numpy.zeros ( [statesNum , actionsNum , statesNum] , float )

        self.state                            = numpy.zeros ( [4] , float )     # a vector of the external state, internal state, setpoint, and trial



