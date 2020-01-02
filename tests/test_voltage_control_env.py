import unittest
from unittest.mock import patch
from gym_powerworld.envs import voltage_control_env
from gym_powerworld.envs.voltage_control_env import LOSS
import os
import pandas as pd
import numpy as np
import numpy.testing as np_test
import logging
import warnings
from esa import SAW
from gym.spaces import Discrete

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


class VoltageControlEnv14BusTestCase(unittest.TestCase):
    """Test initializing the environment with the 14 bus model."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 2
        cls.min_load_factor = 0.5
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 9
        cls.gen_voltage_range = (0.9, 1.1)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32

        cls.env = voltage_control_env.VoltageControlEnv(
            pwb_path=PWB_14, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            dtype=cls.dtype
        )

        # For easy comparison with the original case, get a fresh SAW
        # object. Do not make any changes to this, use only "get" type
        # methods.
        cls.saw = SAW(PWB_14, early_bind=True)

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.saw.exit()
        cls.env.close()

    def test_saw_load_state(self):
        """Ensure that calling saw.LoadState() works (testing that
        saw.SaveState() has already been called).
        """
        # NOTE: This changes the state of self.env.saw, which can cause
        # issues in other tests.
        self.assertIsNone(self.env.saw.LoadState())

    def test_gen_key_fields(self):
        """Ensure the gen key fields are correct. Hard coding style."""
        self.assertListEqual(['BusNum', 'GenID'], self.env.gen_key_fields)

    def test_gen_fields(self):
        self.assertListEqual(self.env.gen_key_fields + self.env.GEN_FIELDS,
                             self.env.gen_fields)

    def test_gen_obs_fields(self):
        self.assertListEqual(self.env.gen_key_fields + self.env.GEN_OBS_FIELDS,
                             self.env.gen_obs_fields)

    def test_gen_data(self):
        self.assertIsInstance(self.env.gen_data, pd.DataFrame)
        self.assertListEqual(self.env.gen_fields,
                             self.env.gen_data.columns.tolist())

    def test_num_gens(self):
        # 15 bus case has 5 generators.
        self.assertEqual(5, self.env.num_gens)

    def test_zero_negative_gen_mw_limits(self):
        """Ensure the _zero_negative_gen_mw_limits function works as
        intended.
        """
        # First, ensure it has been called.
        self.assertTrue((self.env.gen_data['GenMWMin'] >= 0).all())

        # Now, patch gen_data and saw and call the function.
        gen_copy = self.env.gen_data.copy(deep=True)
        gen_copy['GenMWMin'] = -10
        # I wanted to use self.assertLogs, but that has trouble working
        # with nested context managers...
        with patch.object(self.env, 'gen_data', new=gen_copy):
            with patch.object(self.env, 'saw') as p:
                self.env._zero_negative_gen_mw_limits()

        # The gen_copy should have had its GenMWMin values zeroed out.
        self.assertTrue((gen_copy['GenMWMin'] == 0).all())

        # change_and_confirm_params_multiple_element should have been
        # called.
        p.change_and_confirm_params_multiple_element.assert_called_once()

        # Ensure the change was reflected in PowerWorld.
        gens = self.env.saw.GetParametersMultipleElement(
            'gen', ['BusNum', 'GenID', 'GenMWMin'])
        self.assertTrue((gens['GenMWMin'] == 0).all())

        # Finally, (this could have been done first, but oh well), make
        # sure that the case started with negative GenMWMin values.
        gens_orig = self.saw.GetParametersMultipleElement(
            'gen', ['BusNum', 'GenID', 'GenMWMin'])
        self.assertTrue((gens_orig['GenMWMin'] < 0).any())

    def test_gen_mw_capacity(self):
        # The generators are all set to a ridiculous maximum of 10 GW.
        self.assertEqual(5 * 10000.0, self.env.gen_mw_capacity)

    def test_gen_mvar_produce_capacity(self):
        self.assertEqual(50. + 40. + 24. + 24.,
                         round(self.env.gen_mvar_produce_capacity, 2))

    def test_gen_mvar_consume_capacity(self):
        self.assertEqual(-40. - 6. - 6.,
                         round(self.env.gen_mvar_consume_capacity, 2))

    def test_load_key_fields(self):
        # Hard coding!
        self.assertListEqual(self.env.load_key_fields, ['BusNum', 'LoadID'])

    def test_load_fields(self):
        self.assertListEqual(self.env.load_fields,
                             self.env.load_key_fields + self.env.LOAD_FIELDS)

    def test_load_obs_fields(self):
        self.assertListEqual(
            self.env.load_obs_fields,
            self.env.load_key_fields + self.env.LOAD_OBS_FIELDS)

    def test_load_data(self):
        self.assertIsInstance(self.env.load_data, pd.DataFrame)
        self.assertListEqual(self.env.load_data.columns.tolist(),
                             self.env.load_fields)

    def test_num_loads(self):
        self.assertEqual(11, self.env.num_loads)

    def test_zero_i_z_loads(self):
        """Patch the environment's load_data and ensure the method is
        working properly.
        """
        data = self.env.load_data.copy(deep=True)
        data[voltage_control_env.LOAD_I_Z] = 1
        with patch.object(self.env, 'load_data', new=data):
            with patch.object(self.env, 'saw') as p:
                self.env._zero_i_z_loads()

        self.assertTrue((data[voltage_control_env.LOAD_I_Z] == 0).all().all())
        p.change_and_confirm_params_multiple_element.assert_called_once()

    def test_bus_key_fields(self):
        self.assertListEqual(['BusNum'], self.env.bus_key_fields)

    def test_bus_obs_fields(self):
        self.assertListEqual(self.env.bus_key_fields + self.env.BUS_OBS_FIELDS,
                             self.env.bus_obs_fields)

    def test_bus_data(self):
        self.assertIsInstance(self.env.bus_data, pd.DataFrame)
        self.assertListEqual(self.env.bus_fields,
                             self.env.bus_data.columns.tolist())

    def test_num_buses(self):
        self.assertEqual(14, self.env.num_buses)

    def test_max_load_mw(self):
        # System loading obtained from PowerWorld's Case Summary
        # dialogue.
        self.assertEqual(round(self.env.max_load_mw, 2),
                         self.max_load_factor * LOAD_MW_14)

    def test_check_max_load_exception(self):
        """Ensure that an exception is thrown if maximum loading exceeds
        maximum generation.
        """
        with patch.object(self.env, 'max_load_mw', 10):
            with patch.object(self.env, 'gen_mw_capacity', 9.9):
                with self.assertRaisesRegex(UserWarning, 'The given max_load'):
                    self.env._check_max_load(2)

    def test_check_max_load_warning(self):
        """Ensure we get a warning if the generation is in excess of
        2x maximum load.
        """
        with self.assertLogs(logger=self.env.log, level='WARNING'):
            self.env._check_max_load(2)

    def test_min_load_mw(self):
        # System loading obtained from PowerWorld's Case Summary
        # dialogue.
        self.assertEqual(round(self.env.min_load_mw, 2),
                         self.min_load_factor * LOAD_MW_14)

    def test_check_min_load(self):
        # Get generator data.
        gens = self.env.gen_data.copy(deep=True)
        # Increase all minimum generation.
        gens['GenMWMin'] = 10
        # Patch:
        with patch.object(self.env, 'gen_data', gens):
            with patch.object(self.env, 'min_load_mw', 9.9):
                with self.assertRaisesRegex(UserWarning, 'The given min_load'):
                    self.env._check_min_load(2)

    def test_total_load_mw(self):
        # Ensure it's 1D.
        self.assertEqual(len(self.env.total_load_mw.shape), 1)
        # Check shape.
        self.assertEqual(self.env.total_load_mw.shape[0],
                         self.env.num_scenarios)
        # Ensure all loads are less than the maximum.
        np_test.assert_array_less(self.env.total_load_mw, self.env.max_load_mw)

        # Ensure all loads are greater than the minimum.
        np_test.assert_array_less(self.env.min_load_mw, self.env.total_load_mw)

    def test_loads_mw(self):
        # Check shape
        self.assertEqual(self.env.loads_mw.shape,
                         (self.num_scenarios, self.env.num_loads))
        # Ensure the individual loads match total loading.
        np_test.assert_allclose(self.env.loads_mw.sum(axis=1),
                                self.env.total_load_mw, rtol=1e-6)

    def test_loads_mvar(self):
        # Check shape.
        self.assertEqual(self.env.loads_mvar.shape,
                         (self.num_scenarios, self.env.num_loads))

        # Ensure that portion of negative var loads (leading power
        # factor) is close to the lead_pf_probability.
        neg_portion = (self.env.loads_mvar < 0).sum().sum() \
            / (self.num_scenarios * self.env.num_loads)

        # Ensure we're within 0.75 * prob and 1.25 * prob. This seems
        # reasonable.
        self.assertLessEqual(neg_portion, 1.25 * self.lead_pf_probability)
        self.assertGreaterEqual(neg_portion, 0.75 * self.lead_pf_probability)

    def test_load_power_factors(self):
        """Ensure all loads have a power factor greater than the min."""
        # Ensure all power factors are valid. pf = P / |S|
        s_mag = np.sqrt(np.square(self.env.loads_mw)
                        + np.square(self.env.loads_mvar))

        # Suppress numpy warnings - we'll be replacing NaNs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = self.env.loads_mw / s_mag

        # For sake of testing, set loads with 0 power to have a
        # power factor of 1.
        pf[np.isnan(pf)] = 1
        np_test.assert_array_less(self.min_load_pf, pf)

    def test_loads_on_match_probability(self):
        """Ensure the proportion of loads which are on matches the
        load_on_probability to a reasonable tolerance.
        """
        # First, ensure the zeros match up between loads_mw and loads_mvar.
        mw_0 = self.env.loads_mw == 0
        np.testing.assert_array_equal(mw_0, self.env.loads_mvar == 0)

        # Now, ensure the total portion of loads that are "on" is close
        # to the load_on_probability.
        # noinspection PyUnresolvedReferences
        portion = (~mw_0).sum().sum() \
            / (self.num_scenarios * self.env.num_loads)

        # Ensure we're within 0.75 * prob and 1.25 * prob. This seems
        # reasonable.
        self.assertLessEqual(portion, 1.25 * self.load_on_probability)
        self.assertGreaterEqual(portion, 0.75 * self.load_on_probability)

    def test_gen_mw(self):
        # Start with shape.
        self.assertEqual(self.env.gen_mw.shape,
                         (self.num_scenarios, self.env.num_gens))

        # Ensure total generation is close to total load plus losses.
        np_test.assert_allclose(self.env.gen_mw.sum(axis=1),
                                self.env.total_load_mw * (1 + LOSS), rtol=1e-6)

        # TODO: Since the generators in this case have ridiculously high
        #   maximums, I'm not going to bother testing that all gens are
        #   within their bounds. When we move to a more realistic case,
        #   e.g. the Texas 2000 bus case, we need to test that.
        #
        # # Ensure generator outputs are within bounds.
        # for gen_idx, row in enumerate(env.gen_data.itertuples()):
        #     gen_output = env.gen_mw[:, gen_idx]
        #     # noinspection PyUnresolvedReferences
        #     self.assertTrue((gen_output <= row.GenMWMax).all())
        #     # noinspection PyUnresolvedReferences
        # self.assertTrue((gen_output >= row.GenMWMin).all())

    def test_action_space(self):
        self.assertIsInstance(self.env.action_space, Discrete)
        self.assertEqual(self.env.action_space.n,
                         self.env.num_gens * self.num_gen_voltage_bins)

    def test_gen_bins(self):
        # Hard coding!
        np.testing.assert_allclose(
            np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1]),
            self.env.gen_bins)

    def test_action_array(self):
        self.assertEqual(self.env.action_space.n,
                         self.env.action_array.shape[0])
        self.assertEqual(2, self.env.action_array.shape[1])

        # Initialize array for comparison.
        a = np.zeros(shape=(self.env.action_space.n, 2), dtype=int)
        # Put generator indices in column 0.
        a[:, 0] = np.array(
            self.env.gen_data.index.tolist() * self.num_gen_voltage_bins)

        # Write a crappy, simple, loop to put the indices of the
        # generator voltage levels in.
        b = []
        for i in range(self.num_gen_voltage_bins):
            for _ in range(self.env.num_gens):
                b.append(i)

        a[:, 1] = np.array(b)

        np.testing.assert_array_equal(a, self.env.action_array)

    def test_num_obs(self):
        """Ensure the number of observations matches the expected number
        """
        # 14 buses + 3 * 5 gens + 3 * 11 loads
        self.assertEqual(14 + 3 * 5 + 3 * 11, self.env.num_obs)

    def test_observation_space(self):
        """Ensure the observation space has the appropriate properties.
        """
        # Test shape.
        self.assertEqual(self.env.observation_space.shape, (self.env.num_obs,))

        # Test bounds. At present, we're using the range [0, 1] for all
        # bounds.
        self.assertTrue((self.env.observation_space.high == 1.).all())
        self.assertTrue((self.env.observation_space.low == 0.).all())

    def test_observation_attributes(self):
        """After initialization, several observation related attributes
        should be initialized to None.
        """
        self.assertIsNone(self.env.gen_obs)
        self.assertIsNone(self.env.load_obs)
        self.assertIsNone(self.env.bus_obs)

        self.assertIsNone(self.env.gen_obs_prev)
        self.assertIsNone(self.env.load_obs_prev)
        self.assertIsNone(self.env.bus_obs_prev)

    def test_action_count(self):
        """After initialization, the action count should be 0."""
        self.assertEqual(0, self.env.action_count)


if __name__ == '__main__':
    unittest.main()
