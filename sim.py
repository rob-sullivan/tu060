import matplotlib.pyplot as plt

class Lever():
    def __init__(self, h, active_time, inactive_time):
        self.h = h
        self.at = active_time # how many seconds h says high
        self.iat = inactive_time
        self.pressed_time = 0
        self.active = True

    #when lever is pressed give hit then deactive lever
    def press(self, t): # 1
        if(self.active):#is it active?
            self.pressed_time = t # remember when pressed
            self.active = False #deactivate now
            return self.h #give hit
        else:#next time pressed check if it should be reactivated
            dT = t - self.pressed_time # 1-0 = 1
            if(dT < self.at): #is 1 < 4?
                self.active = False
                return self.h #keep giving hit
            elif(dT >= self.at and dT <= self.iat): # is 5 less than 20
                self.active = False
                return 0 # don't give anything
            else: # is 21 greater than 20
                self.active = True
                return 0 #don't give anything but reactive

class Environment():    
    def __init__(self, t):
        self.time = t #time of simulation given in seconds
        self.episode = []
        self.H_star = 45 # a homeostatic setpoint
        self.H = 45 # an internal state. It describes an array that contains h (dopamine) along with other variables.

        #the simulation
        self.lever = Lever(45, 4, 20) #45 dopamine points for 4 seconds then disable lever for 20 seconds
        self.next_time_step()
        
        plt.figure(num='Lever Pulling Experiment')
        plt.plot(range(self.time), self.episode)
        plt.title("Homeostasis over time: H*=" + str(self.H_star))
        plt.xlabel("Time: t=" + str(self.time))
        plt.ylabel("Homeostasis [H]")
        plt.show()

    def next_time_step(self):
        for t in range(self.time):
            if(self.H < self.H_star):#are we dropping below homeostatic setpoint H*?
                h_lever = self.lever.press(t)#if so try pull lever to get a hit
                if(h_lever>0):#give hit from lever if something there.
                    self.H = h_lever
            #print("H: " + str(self.H))
            self.episode.append(self.H)
            if(self.H>0):# loose h over time
                self.H -= 1


                

    def reward(self):
        #The amount of reward (r) given for action (k) is equal to the reduced distance from the internal state H to the homeostatic setpoint H*.
        r = self.H_star - self.H
        return r

class Environment1:
  def __init__(self):
    # initial values for internal state and homeostatic setpoint
    self.Ht = []
    self.h = 0
    self.H = 0
    self.H_lower = 0
    self.H_upper = 1

    def press_lever(self):
        cocaine_dose_sec = 4

    def K_function(self, h, t):
        Ht = [cortisol, serotonin, endorphins, oxytocin, h (aka dopamine)]
        H_star = H_star_upper
        delta_Ht = Hstar – Ht
        r = delta_Ht
        self.adaptive_process(t, H_star) # trigger adaptive process slowly
        return r

    def adaptive_process(self, t, H_star):
        H_star_upper = H_star + 1
        
    def absence_process(self, t, H_star):
        if(H_star != H_star_lower):
            H_star = H_star – 1


    def expose_to_external_state(self):
        self.Ht = [0,0,0,0,0] #[cortisol, serotonin, endorphins, oxytocin, h (aka dopamine)]
        return self.Ht

    def next_time_interval(self):
        self.Ht = expose_to_external_state()
        action = softmax(reward, Ht)
        if(action=pull_lever):
            self.press_lever()
        else:
            absence_process()

            #the agent predicts the expected outcomes of possible action choices and based on that, 
            # estimates the drive-reduction rewarding values of the choices. According to the estimated 
            # values, the agent selects the action. 
            # The curved arrow represents updating outcome expectancies based 
            # on feedbacks received from the environment.



if __name__ == "__main__":
    Environment(60) # e.g 60 seconds