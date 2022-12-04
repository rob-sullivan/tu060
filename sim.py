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


if __name__ == "__main__":
    Environment(60) # e.g 60 seconds