##References
# habits of a happy brain, sims, stick RPG, strategic life simulation games of agent-based artificial lives.

# I created five 'locations', a 24-hour-7-day cycle for the agent to move through each location and six actions the agent can choose from at each location. 
# If there is repetitive engagement in actions that are detrimental to the agent in a given location, we say 'agent is addicted'. 
# If we 'the user' choose actions that stop this repetitive behavior, we say 'agent not addicted'. 
# With this I'll print q values for past actions taken to demonstrate how a DQL agent seeks to maximise its dopamine reward.

#create locations and actions
#agent starts at home location
#at that location they can do an action
#action give reward but being in state for too long increases stress.
#stress causes action to change
#goal is to maximise reward while keeping stress down.

class State():
    def __init__(self, n, d, s):
        #meta data
        self.id = 0
        self.name = n
        self.description = d
        
        #stressful situations increase cortisol.
        self.stress = s # float, stress hormone which makes agent feel uncomfortable and wants to do something else

class Action():
    def __init__(self, p, s, c):
        #meta data
        self.id = 0
        self.name = ""
        self.description = ""
 
        #e.g if physical action like jogging, fighting or over exertion to release endorphins, non for social
        self.physical = p #agent gets for pushing through physical pain at different times (e.g runners high)
        #e.g if socal meeting up with friends will release oxytocin
        self.social = s  #rewarding agent for being social
        #e.g if winning at a game, feeling in control or a process or superior to peers (publich applause)
        self.in_control = c #agent feeling of self achievement

class Reward():
    def __init__(self, scenario, past_actions, action):
        
        #agent wants this dopamine reward for each state
        self.d = 1.0 

        #but has a stressful problem to overcome in the state
        c = scenario.stress

        #so selects an action to overcome stress (each action has certain bonuses or weights)
        e = action.physical
        o = action.social
        s = action.in_control

        #but spamming the same action all the time is not the best
        if(action in past_actions):#e.g gambling feels great the first time but not the 100th time, chasing rewards!
            #habituation will reduce the experience that makes the agent happy because the action is new
            self.habituation = 0.5

        action_reward = self.d * (e + o + s)
        action_punishment = c + self.habituation

        #reward given for doing an action in a state
        state_reward = action_reward - action_punishment

        return state_reward
        

class Environment():
    def __init__(self):
        t = 0 #0hrs to 23hrs
        d = 1 #1 to 7 monday to sunday
        self.past_actions = []
        self.actions = []
        self.locations = [] #/action_queue = []

        #create locations
        home = State('Home', 'a place to sleep at night', .1)
        office = State('Office', 'a place to earn an income', .9)
        college = State('College', 'a place to learn so your income potential increases', .8)
        gym = State('Gym', 'a place to work out', .5)
        pub = State('Pub', 'a place to dance, socialise, drink and smoke', .3)

        #add locations
        self.locations.append(home)
        self.locations.append(office)
        self.locations.append(college)
        self.locations.append(gym)
        self.locations.append(pub)

        #create actions
        work = Action('Work', 'completing assigned tasks', 0, 0, 1)
        socialise = Action('Socialise', 'conversing with others', 0, 1, 0)
        exercise = Action('Exercise', 'working out', 0, 1, 0)
        drink = Action('Drink', 'drinking alcohol to feel drunk and socialise', 1, 1, 0)
        smoke_cigarette = Action('Smoke', 'smoking cigarettes to feel relaxed and socialise', 1, 1, 1)
        game = Action('Game', 'playing games to relax, feel a sence of achievement and socialise', 0, 1, 1)

        #add actions
        self.actions.append(work)
        self.actions.append(socialise)
        self.actions.append(exercise)
        self.actions.append(drink)
        self.actions.append(smoke_cigarette)
        self.actions.append(game)

        #day-time system
        if t > 23:
            t = 0
        if d != 6 or d !=7: # weekday
            if t > 0 and t < 9 or t > 5 and t < 23:
                pass # at home
            elif t >= 9 and t <=5:
                pass # at work
        else: #weekend
            pass


