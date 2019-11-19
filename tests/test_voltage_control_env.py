import unittest
from gym_powerworld.envs import voltage_control_env
import os
import numpy as np
import numpy.testing as np_test

# Get full path to this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cases are within this directory.
CASE_DIR = os.path.join(THIS_DIR, 'cases')

# IEEE 14 bus
DIR_14 = os.path.join(CASE_DIR, 'ieee_14')
PWB_14 = os.path.join(DIR_14, 'IEEE 14 bus.pwb')

# Define some constants related to the IEEE 14 bus case.
N_GENS_14 = 5
N_LOADS_14 = 11
LOAD_MW_14 = 259.0


class VoltageControlEnvInitializationTestCase(unittest.TestCase):
    """Test initializing the environment."""

    def test_something(self):
        # Define some inputs to our environment
        num_scenarios = 100
        max_load_factor = 2
        min_load_factor = 0.5
        min_load_pf = 0.8
        load_on_probability = 0.8
        num_gen_voltage_bins = 5
        gen_voltage_range = (0.9, 1.1)

        # The 14 bus case has generator minimums < 0, which is no bueno.
        # Ensure we get a warning.
        with self.assertLogs(level='WARNING') as cm:
            env = voltage_control_env.VoltageControlEnv(
                PWB_14, num_scenarios=num_scenarios,
                max_load_factor=max_load_factor,
                min_load_factor=min_load_factor,
                min_load_pf=min_load_pf,
                load_on_probability=load_on_probability,
                num_gen_voltage_bins=num_gen_voltage_bins,
                gen_voltage_range=gen_voltage_range
            )

        # TODO: Add lots of other tests in here.
        # We should get only one warning, and it should be related to
        # zeroing out generators with negative minimums.
        self.assertEqual(1, len(cm.output))
        self.assertIn('5 generators with GenMWMin < 0 have had GenMWMin',
                      cm.output[0])

        # Ensure dimensionality of loads match up.
        load_dim = (num_scenarios, N_LOADS_14)
        self.assertEqual(load_dim, env.scenario_individual_loads_mw.shape)
        self.assertEqual(load_dim, env.scenario_individual_loads_mvar.shape)

        # Ensure the individual loads match total loading.
        np_test.assert_allclose(env.scenario_individual_loads_mw.sum(axis=1),
                                env.scenario_total_loads_mw)

        # Ensure all loads are less than the maximum.
        np_test.assert_array_less(env.scenario_total_loads_mw,
                                  env.max_load_mw)

        # Ensure all loads are greater than the minimum.
        np_test.assert_array_less(env.min_load_mw,
                                  env.scenario_total_loads_mw)

        # Ensure all power factors are valid. pf = P / |S|
        s_mag = np.sqrt(np.square(env.scenario_individual_loads_mw)
                        + np.square(env.scenario_individual_loads_mvar))
        pf = env.scenario_individual_loads_mw / s_mag
        # For sake of testing, set loads with 0 power to have a
        # power factor of 1.
        pf[np.isnan(pf)] = 1
        np_test.assert_array_less(0.8, pf)


if __name__ == '__main__':
    unittest.main()
