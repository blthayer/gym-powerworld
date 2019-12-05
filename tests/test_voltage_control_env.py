import unittest
from gym_powerworld.envs import voltage_control_env
import os
import numpy as np
import numpy.testing as np_test
import logging
import warnings

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
        lead_pf_probability = 0.1
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
                lead_pf_probability=lead_pf_probability,
                load_on_probability=load_on_probability,
                num_gen_voltage_bins=num_gen_voltage_bins,
                gen_voltage_range=gen_voltage_range,
                log_level=logging.INFO,
                seed=42
            )

        # We should get a pair of warnings related to zeroing out
        # generators with negative minimums and the generator capacity.
        self.assertEqual(2, len(cm.output))
        self.assertIn('5 generators with GenMWMin < 0 have had GenMWMin',
                      cm.output[0])
        self.assertIn('The given generator capacity, ', cm.output[1])

        # Ensure our min and max loads were handled correctly.
        self.assertAlmostEqual(LOAD_MW_14 * max_load_factor,
                               env.max_load_mw, places=4)
        self.assertAlmostEqual(LOAD_MW_14 * min_load_factor,
                               env.min_load_mw, places=4)

        # TODO: Add lots of other tests in here.

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

        # Suppress numpy warnings - we'll be replacing NaNs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = env.scenario_individual_loads_mw / s_mag

        # For sake of testing, set loads with 0 power to have a
        # power factor of 1.
        pf[np.isnan(pf)] = 1
        np_test.assert_array_less(min_load_pf, pf)

        # Ensure our proportion of negative loads is appropriate.
        neg_sum = (env.scenario_individual_loads_mvar < 0).sum()
        total_elements = num_scenarios * N_LOADS_14
        self.assertLessEqual(neg_sum / total_elements, lead_pf_probability)

        # Ensure generation matches load to within the given tolerance.
        np_test.assert_array_less(
            env.scenario_gen_mw.sum(axis=1) - env.scenario_total_loads_mw,
            voltage_control_env.GEN_LOAD_DELTA_TOL
        )

        # Ensure generator outputs are within bounds.
        for gen_idx, row in enumerate(env.gen_data.itertuples()):
            gen_output = env.scenario_gen_mw[:, gen_idx]
            # noinspection PyUnresolvedReferences
            self.assertTrue((gen_output <= row.GenMWMax).all())
            # noinspection PyUnresolvedReferences
            self.assertTrue((gen_output >= row.GenMWMin).all())


if __name__ == '__main__':
    unittest.main()
