from gymnasium.envs.registration import register

register(
     id="HRLSim-v0",
     entry_point="hrl_gym.envs:HRLSim"
)