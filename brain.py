##ref habits of a happy brain
## ref sims
#  strategic life simulation video game of an agent-based artificial life program.

#A state is a daily activity of a virtual person
class State():
    def __init__(self, n, d, c):
        #meta data
        self.id = 0
        self.name = n
        self.description = d
        

        #how stressful is it
        self.cortisol = c # float, stress hormone which makes agent feel uncomfortable and wants to do something

class Action():
    def __init__(self):
        #meta data
        self.id = 0
        self.name = ""
        self.description = ""

        #type
        self.is_physical = False
        self.is_social = False
        self.is_one_up = False

        
        #e.g if physical action like jogging
        self.endorphins = 1.0 #agent gets for pushing through physical pain at different times (e.g runners high)
        #e.g if socal meeting up with friends
        self.oxytocin = 1.0  #rewarding agent for being social
        #e.g if winning at a game, feeling in control or a process or superior to peers (publich applause)
        self.serotonin = 1.0 #agent feeling of self achievement     

class Reward():
    def __init__(self, scenario, past_actions, action):
        
        #agent wants this reward for each state
        self.dopamine = 1.0 

        #but has a stressful problem to overcome in the state
        c = scenario.cortisol

        #so selects an action to overcome stress (each action has certain bonuses or weights)
        e = action.endorphins
        o = action.oxytocin
        s = action.serotonin

        #but spamming the same action all the time is not the best
        if(action in past_actions):#e.g gambling feels great the first time but not the 100th time, chasing rewards!
            #habituation will reduce the experience that makes the agent happy because the action is new
            self.habituation = 0.5

        action_reward = self.dopamine * (e + o + s)
        action_punishment = c + self.habituation

        #reward given for doing an action in a state
        state_reward = action_reward*-action_punishment

        return state_reward
        

class Environment():
    def __init__(self):
        self.past_actions = []
        self.scenarios = [] #/action_queue = []

        work = State('work', 'any type of employment', .9)
        socialise = State('socialise', 'meeting up with friends', .5)
