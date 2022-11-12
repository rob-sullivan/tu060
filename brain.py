##ref habits of a happy brain

class Environment():
    def __init__(self):
        self.past_actions = []
        self.scenarios = []


class Scenario():
    def __init__(self):
        self.id = 0
        self.name = ""
        self.description = ""
        self.cortisol = 0.0 #stress hormone which makes agent feel uncomfortable and wants to do something
        self.is_physical = False
        self.is_social = False
        self.is_one_up = False

class Reward():
    def __init__(self, scenario, past_actions, action):
        self.dopamine = 1.0 #agent gets going after a reward

        #weights or bonuses
        if(scenario.is_physical):
            self.endorphins = 1.0 #agent gets for pushing through physical pain at different times

        if(scenario.is_physical):
            self.oxytocin = 1.0  #rewarding agent for being social

        if(scenario.is_physical):
            self.serotonin = 1.0 #agent feeling of self achievement      

        if(action in past_actions):
            #habituation will reduce the experience that makes the agent happy because the action is new
            self.habituation = 1.0

        reward = (self.dopamine * (self.endorphins + self.oxytocin + self.serotonin))*(scenario.cortisol + self.habituation)

        return reward
        




