from gym.envs.registration import register

register(
    id='powerworld-v0',
    entry_point='gym_powerworld.envs:VoltageControlEnv',
)
