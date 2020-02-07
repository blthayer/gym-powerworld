from gym.envs.registration import register

register(
    id='powerworld-discrete-env-v0',
    entry_point='gym_powerworld.envs:DiscreteVoltageControlEnv',
)

register(
    id='powerworld-gridmind-env-v0',
    entry_point='gym_powerworld.envs:GridMindEnv',
)

register(
    id='powerworld-gridmind-hard-env-v0',
    entry_point='gym_powerworld.envs:GridMindHardEnv',
)

register(
    id='powerworld-gridmind-contingencies-env-v0',
    entry_point='gym_powerworld.envs:GridMindContingenciesEnv',
)

register(
    id='powerworld-discrete-env-simple-14-bus-v0',
    entry_point='gym_powerworld.envs:DiscreteVoltageControlSimple14BusEnv',
)
