import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

###########################################
#         Custom Environment
###########################################
class BabyRobotEnv_v1(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()

        # dimensions of the grid
        self.width = kwargs.get('width',3)
        self.height = kwargs.get('height',3)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 5 possible actions: move N,E,S,W or stay in same state
        self.action_space = Discrete(5)

        # the observation will be the coordinates of Baby Robot
        self.observation_space = MultiDiscrete([self.width, self.height])

        # Baby Robot's position in the grid
        self.x = 0
        self.y = 0

    def step(self, action):
        obs = np.array([self.x,self.y])
        reward = -1
        done = True
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset Baby Robot's position in the grid
        self.x = 0
        self.y = 0
        info = {}
        return np.array([self.x,self.y]),info

    def render(self):
        pass
###########################################
#         Stage 1 - Initialization
###########################################

# create the cartpole environment
env = gym.make('CartPole-v1', render_mode="human")

# run for 10 episodes
for episode in range(10):
  # each step or trial should be 4 secs

  # put the environment into its start state,
  env.reset()

###########################################
#            Stage 2 - Execution
###########################################

  # run until the episode completes
  terminated = False
  while not terminated:

    # show the environment
    env.render()

    # choose a random action
    action = env.action_space.sample()

    # take the action and get the information from the environment
    observation, reward, terminated, truncated, info = env.step(action)

    """
    'observation': Defines the new state of the environment. In the case of CartPole this is information about the position and velocity of the pole. In a grid-world environment it would be information about the next state, where we end up after taking the action.

    'reward': The amount of reward, if any, received as a result of taking the action.

    'terminated': A flag to indicate if we've reached the end of the episode

    'truncated': A flag to indicate if the episode has been stopped before completion.

    'info': Any additional information. In general this isn't set.
    """


###########################################
#           Stage 3 - Termination
###########################################

# terminate the environment
env.close()