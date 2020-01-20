from gym.envs.registration import register

register(
    id='powerworld-discrete-env-v0',
    entry_point='gym_powerworld.envs:DiscreteVoltageControlEnv',
)

register(
    id='powerworld-gridmind-env-v0',
    entry_point='gym_powerworld.envs:GridMindEnv',
)
