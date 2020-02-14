import unittest
from unittest.mock import patch
from gym_powerworld.envs import voltage_control_env
# noinspection PyProtectedMember
from gym_powerworld.envs.voltage_control_env import LOSS, \
    MinLoadBelowMinGenError, MaxLoadAboveMaxGenError, OutOfScenariosError, \
    MIN_V, MAX_V, MIN_V_SCALED, MAX_V_SCALED, _scale_voltages
import os
import pandas as pd
import numpy as np
import numpy.testing as np_test
import logging
import warnings
from esa import SAW, PowerWorldError
from gym.spaces import Discrete
import shutil

# Get full path to this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cases are within this directory.
CASE_DIR = os.path.join(THIS_DIR, 'cases')

# IEEE 14 bus
DIR_14 = os.path.join(CASE_DIR, 'ieee_14')
PWB_14 = os.path.join(DIR_14, 'IEEE 14 bus.pwb')
AXD_14 = os.path.join(DIR_14, 'IEEE 14 bus.axd')
CONTOUR = os.path.join(DIR_14, 'contour.axd')

# Case with 3 gens modeled as condensers:
PWB_14_CONDENSERS = os.path.join(DIR_14, 'IEEE 14 bus condensers.pwb')

# Case with min and max MW limits on all 5 generators.
PWB_14_LIMITS = os.path.join(DIR_14, 'IEEE 14 bus limits.pwb')

# IL 200
PWB_200 = os.path.join(CASE_DIR, 'il_200', 'ACTIVSg200.pwb')

# TX 2000
PWB_2000 = os.path.join(CASE_DIR, 'tx_2000',
                        'ACTIVSg2000_AUG-09-2018_Ride_mod.PWB')

# Define some constants related to the IEEE 14 bus case.
N_GENS_14 = 5
N_LOADS_14 = 11
LOAD_MW_14 = 259.0


# noinspection DuplicatedCode
class DiscreteVoltageControlEnv14BusTestCase(unittest.TestCase):
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
        cls.log_buffer = 100

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
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
            dtype=cls.dtype,
            log_buffer=cls.log_buffer
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

    def test_branches_to_open(self):
        """Ensure branches_to_open is the right shape and is in the
        appropriate range.
        """
        self.assertIsNotNone(self.env.branches_to_open)
        self.assertEqual((self.num_scenarios,),
                         self.env.branches_to_open.shape)
        self.assertTrue(self.env.branches_to_open.min() >= 0)
        self.assertTrue(
            self.env.branches_to_open.max()
            < self.env.branch_init_data.shape[0])

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

    def test_gen_init_fields(self):
        self.assertListEqual(
            self.env.gen_key_fields + self.env.GEN_INIT_FIELDS,
            self.env.gen_init_fields)

    def test_gen_obs_fields(self):
        self.assertListEqual(self.env.gen_key_fields + self.env.GEN_OBS_FIELDS,
                             self.env.gen_obs_fields)

    def test_gen_init_data(self):
        self.assertIsInstance(self.env.gen_init_data, pd.DataFrame)
        self.assertListEqual(self.env.gen_init_fields,
                             self.env.gen_init_data.columns.tolist())

    def test_num_gens(self):
        # 15 bus case has 5 generators.
        self.assertEqual(5, self.env.num_gens)

    def test_zero_negative_gen_mw_limits(self):
        """Ensure the _zero_negative_gen_mw_limits function works as
        intended.
        """
        # First, ensure it has been called.
        self.assertTrue((self.env.gen_init_data['GenMWMin'] >= 0).all())

        # Now, patch gen_init_data and saw and call the function.
        gen_copy = self.env.gen_init_data.copy(deep=True)
        gen_copy['GenMWMin'] = -10
        # I wanted to use self.assertLogs, but that has trouble working
        # with nested context managers...
        with patch.object(self.env, '_gen_init_data', new=gen_copy):
            with patch.object(self.env, 'saw') as p:
                self.env._zero_negative_gen_mw_limits()

        # The gen_copy should have had its GenMWMin values zeroed out.
        self.assertTrue((gen_copy['GenMWMin'] == 0).all())

        # change_parameters_multiple_element_df should have been
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

    def test_load_init_fields(self):
        self.assertListEqual(self.env.load_init_fields,
                             self.env.load_key_fields
                             + self.env.LOAD_INIT_FIELDS)

    def test_load_obs_fields(self):
        self.assertListEqual(
            self.env.load_obs_fields,
            self.env.load_key_fields + self.env.LOAD_OBS_FIELDS)

    def test_load_init_data(self):
        self.assertIsInstance(self.env.load_init_data, pd.DataFrame)
        self.assertListEqual(self.env.load_init_data.columns.tolist(),
                             self.env.load_init_fields)

    def test_num_loads(self):
        self.assertEqual(11, self.env.num_loads)

    def test_zero_i_z_loads(self):
        """Patch the environment's load_init_data and ensure the method is
        working properly.
        """
        data = self.env.load_init_data.copy(deep=True)
        data[voltage_control_env.LOAD_I_Z] = 1
        with patch.object(self.env, '_load_init_data', new=data):
            with patch.object(self.env, 'saw') as p:
                self.env._zero_i_z_loads()

        self.assertTrue((data[voltage_control_env.LOAD_I_Z] == 0).all().all())
        p.change_and_confirm_params_multiple_element.assert_called_once()

    def test_bus_key_fields(self):
        self.assertListEqual(['BusNum'], self.env.bus_key_fields)

    def test_bus_obs_fields(self):
        self.assertListEqual(self.env.bus_key_fields + self.env.BUS_OBS_FIELDS,
                             self.env.bus_obs_fields)

    def test_bus_init_data(self):
        self.assertIsInstance(self.env.bus_init_data, pd.DataFrame)
        self.assertListEqual(self.env.bus_init_fields,
                             self.env.bus_init_data.columns.tolist())

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
                with self.assertRaisesRegex(MaxLoadAboveMaxGenError,
                                            'The given max_load'):
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
        gens = self.env.gen_init_data.copy(deep=True)
        # Increase all minimum generation.
        gens['GenMWMin'] = 10
        # Patch:
        with patch.object(self.env, '_gen_init_data', gens):
            with patch.object(self.env, 'min_load_mw', 9.9):
                with self.assertRaisesRegex(MinLoadBelowMinGenError,
                                            'The given min_load'):
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
        # for gen_idx, row in enumerate(env.gen_init_data.itertuples()):
        #     gen_output = env.gen_mw[:, gen_idx]
        #     # noinspection PyUnresolvedReferences
        #     self.assertTrue((gen_output <= row.GenMWMax).all())
        #     # noinspection PyUnresolvedReferences
        # self.assertTrue((gen_output >= row.GenMWMin).all())

    def test_gen_v(self):
        # Shape.
        self.assertEqual(self.env.gen_v.shape,
                         (self.env.num_scenarios, self.env.num_gens))

        # Values.
        self.assertTrue(
            ((self.env.gen_v >= self.gen_voltage_range[0]).all()
             and
             (self.env.gen_v <= self.gen_voltage_range[1]).all()
             )
        )

    def test_action_space(self):
        self.assertIsInstance(self.env.action_space, Discrete)
        # Plus 1 because no-op action
        self.assertEqual(self.env.action_space.n,
                         self.env.num_gens * self.num_gen_voltage_bins + 1)

    def test_gen_bins(self):
        # Hard coding!
        np.testing.assert_allclose(
            np.array([0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1]),
            self.env.gen_bins)

    def test_gen_action_array(self):
        # Minus 1 because no-op action.
        self.assertEqual(self.env.action_space.n - 1,
                         self.env.gen_action_array.shape[0])
        self.assertEqual(2, self.env.gen_action_array.shape[1])

        # Initialize array for comparison. Again, -1 due to no-op.
        a = np.zeros(shape=(self.env.action_space.n - 1, 2), dtype=int)
        # Put generator bus numbers in column 0. No need to worry about
        # multiple generators at the same bus for this case.
        a[:, 0] = np.array(
            self.env.gen_init_data['BusNum'].tolist()
            * self.num_gen_voltage_bins)

        # Write a crappy, simple, loop to put the indices of the
        # generator voltage levels in.
        b = []
        for i in range(self.num_gen_voltage_bins):
            for _ in range(self.env.num_gens):
                b.append(i)

        a[:, 1] = np.array(b)

        np.testing.assert_array_equal(a, self.env.gen_action_array)

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

        # Test bounds. Bus voltages should have a high of 2, and the
        # rest should have a high of 1.
        self.assertTrue((self.env.observation_space.high[
                         0:self.env.num_buses] == 2.).all())
        self.assertTrue((self.env.observation_space.high[
                         self.env.num_buses:] == 1.).all())
        self.assertTrue((self.env.observation_space.low == 0.).all())

    def test_observation_attributes(self):
        """After initialization, several observation related attributes
        should be initialized to None.
        """
        self.assertIsNone(self.env.gen_obs_data)
        self.assertIsNone(self.env.load_obs_data)
        self.assertIsNone(self.env.bus_obs_data)

        self.assertIsNone(self.env.gen_obs_data_prev)
        self.assertIsNone(self.env.load_obs_data_prev)
        self.assertIsNone(self.env.bus_obs_data_prev)

    def test_action_count(self):
        """After initialization, the action count should be 0."""
        self.assertEqual(0, self.env.action_count)

    def test_reward_matches(self):
        """For this simple initialization, the rewards should be the
        same as the class constant.
        """
        self.assertDictEqual(self.env.rewards, self.env.REWARDS)

    def test_override_reward(self):
        """Ensure overriding a portion of the rewards behaves as
        expected.
        """
        # Create a new env, but use new rewards.
        env = voltage_control_env.DiscreteVoltageControlEnv(
            pwb_path=PWB_14, num_scenarios=10,
            max_load_factor=self.max_load_factor,
            min_load_factor=self.min_load_factor,
            num_gen_voltage_bins=5,
            rewards={'v_delta': 1000})

        # Loop and assert.
        for key, value in env.rewards.items():
            if key == 'v_delta':
                self.assertNotEqual(env.REWARDS[key], value)
            else:
                self.assertEqual(env.REWARDS[key], value)

        # Ensure the keys are the same.
        self.assertListEqual(list(env.rewards.keys()),
                             list(env.REWARDS.keys()))

    def test_bad_reward_key(self):
        """Ensure an exception is raised if a bad reward key is given.
        """
        with self.assertRaisesRegex(KeyError, 'The given rewards key, v_detl'):
            _ = voltage_control_env.DiscreteVoltageControlEnv(
                pwb_path=PWB_14, num_scenarios=10,
                max_load_factor=self.max_load_factor,
                min_load_factor=self.min_load_factor,
                rewards={'v_detla': 1000})

    def test_log_columns(self):
        """Ensure the log columns are as they should be."""
        self.assertListEqual(
            ['episode', 'action_taken', 'reward']
            + [f'bus_{x+1}_v' for x in range(14)]
            + [f'gen_{x}_{y}' for x, y in zip([1, 2, 3, 6, 8], [1] * 5)],
            self.env.log_columns
        )

    def test_log_array(self):
        self.assertEqual(self.env.log_array.shape,
                         # 14 + 3 --> num buses plus ep, action, reward,
                         # and num gens.
                         (self.log_buffer, 14 + 3 + 5))

    def test_no_op_action(self):
        # Cover several data types because the action comes directly
        # from a neural network, which could have different data types.
        self.assertEqual(0, self.env.no_op_action)
        self.assertEqual(0.0, self.env.no_op_action)
        self.assertEqual(np.float64(0), self.env.no_op_action)
        self.assertEqual(np.float32(0), self.env.no_op_action)
        self.assertEqual(np.int(0.0), self.env.no_op_action)

    def test_last_action(self):
        self.assertIsNone(self.env.last_action)


# noinspection DuplicatedCode
class DiscreteVoltageControlEnv14BusLimitsTestCase(unittest.TestCase):
    """Test initializing the environment with the 14 bus model with
    limits added.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        # Create a ton of scenarios so the generator dispatch is
        # thoroughly exercised.
        cls.num_scenarios = 100000
        cls.max_load_factor = 1.44
        cls.min_load_factor = 0.5
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 9
        cls.gen_voltage_range = (0.9, 1.1)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 100

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
            pwb_path=PWB_14_LIMITS, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            dtype=cls.dtype,
            log_buffer=cls.log_buffer
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    # noinspection PyUnresolvedReferences
    def test_gens_in_bounds(self):
        self.assertTrue(
            (self.env.gen_mw
             <= self.env.gen_init_data['GenMWMax'].to_numpy()).all()
        )

        self.assertTrue(
            (self.env.gen_mw
             >= self.env.gen_init_data['GenMWMin'].to_numpy()).all()
        )

    def test_gen_meets_load(self):
        np.testing.assert_allclose(self.env.total_load_mw * (1 + LOSS),
                                   self.env.gen_mw.sum(axis=1))


# noinspection DuplicatedCode
class DiscreteVoltageControlEnv14BusResetTestCase(unittest.TestCase):
    """Test the reset method of the environment."""
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

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
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

        # Extract generator data needed for testing the reset method.
        cls.gens = cls.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=cls.env.gen_key_fields + cls.env.GEN_RESET_FIELDS)

        # Extract generator data needed for testing the reset method.
        cls.loads = cls.saw.GetParametersMultipleElement(
            ObjectType='load',
            ParamList=cls.env.load_key_fields + cls.env.LOAD_RESET_FIELDS
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.saw.exit()
        cls.env.close()

    def setUp(self) -> None:
        """Reset the scenario index for each run, and restore the
        case.
        """
        self.env.scenario_idx = 0
        self.env.saw.LoadState()

    def test_scenario_idx_increments(self):
        """Ensure subsequent calls to reset update the scenario index.
        """
        # Patch the changing of parameters so that we'll get a
        # a consistent incrementing of the index (no failed power flow).
        with patch.object(self.env.saw,
                          'change_parameters_multiple_element_df'):
            self.env.reset()
            self.assertEqual(1, self.env.scenario_idx)
            self.env.reset()
            self.assertEqual(2, self.env.scenario_idx)
            self.env.reset()
            self.assertEqual(3, self.env.scenario_idx)

    def test_action_count_reset(self):
        """Ensure subsequent calls to reset reset the action_count."""
        # Patch the changing of parameters so that we'll get a
        # a consistent incrementing of the index (no failed power flow).
        with patch.object(self.env.saw,
                          'change_parameters_multiple_element_df'):
            self.env.action_count = 10
            self.env.reset()
            self.assertEqual(0, self.env.action_count)
            self.env.action_count = 17
            self.env.reset()
            self.assertEqual(0, self.env.action_count)
            self.env.action_count = 1
            self.env.reset()
            self.assertEqual(0, self.env.action_count)

    def test_load_state_called(self):
        """Ensure the SAW object's LoadState method is called in reset.
        """
        # Patch the changing of parameters so that we'll get a
        # a consistent incrementing of the index (no failed power flow).
        with patch.object(self.env.saw,
                          'change_parameters_multiple_element_df'):
            with patch.object(
                    self.env.saw, 'LoadState',
                    side_effect=self.env.saw.LoadState) as p:
                self.env.reset()

        p.assert_called_once()

    def test_gens_and_loads_set_correctly(self):
        """Ensure that the appropriate generators get opened and closed,
        and that the power levels get set correctly in the case for both
        generators and loads.
        """
        # There are 5 generators in the 14 bus case. In the base case,
        # only gens at buses 1 and 2 are providing active power, but
        # the others are "Closed" and thus regulating their voltage.
        # We'll patch the environment's gen_mw to have all gens on
        # and sharing the load evenly except the generator at bus 2.
        # We'll also patch all gens to be regulating to 1.05 per unit.
        p = LOAD_MW_14 / 4
        gen_mw_row = np.array([p, 0, p, p, p])
        gen_mw = self.env.gen_mw.copy()
        gen_mw[0, :] = gen_mw_row

        gen_v_row = np.array([1.05] * 5)
        gen_v = self.env.gen_v.copy()
        gen_v[0, :] = gen_v_row

        # Extract the original loading, but we'll bump one load by 1 MW
        # and 1 MVAR and decrement another by 1 MW and 1 MVAR.
        loads_mw_row = self.loads['LoadSMW'].to_numpy()
        loads_mw_row[3] += 1
        loads_mw_row[4] -= 1
        loads_mw = self.env.loads_mw.copy()
        loads_mw[0, :] = loads_mw_row

        loads_mvar_row = self.loads['LoadSMVR'].to_numpy()
        loads_mvar_row[3] += 1
        loads_mvar_row[4] -= 1
        loads_mvar = self.env.loads_mvar.copy()
        loads_mvar[0, :] = loads_mvar_row

        # Patch the scenario index, generator output, and loading. Then
        # reset the environment.
        with patch.object(self.env, 'gen_mw', new=gen_mw):
            with patch.object(self.env, 'gen_v', new=gen_v):
                with patch.object(self.env, 'loads_mw', new=loads_mw):
                    with patch.object(self.env, 'loads_mvar', new=loads_mvar):
                        # Patch branches_to_open so a line does not get
                        # opened.
                        with patch.object(self.env, 'branches_to_open',
                                          new=None):
                            self.env.reset()

        # Pull the generator data from PowerWorld and ensure that both
        # the status and output match up.
        gen_reset_data = self.env.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=self.env.gen_key_fields + self.env.GEN_RESET_FIELDS)

        # All gens except for the 2nd should be closed.
        status = ['Closed'] * 5
        status[1] = 'Open'
        self.assertListEqual(status, gen_reset_data['GenStatus'].tolist())

        # Excluding the slack, generator MW output should exactly match
        # what was commanded.
        np.testing.assert_allclose(
            gen_mw_row[1:], gen_reset_data['GenMW'].to_numpy()[1:])

        # The slack should be equal to within our assumed line losses.
        np.testing.assert_allclose(
            gen_mw_row[0], gen_reset_data['GenMW'].to_numpy()[0],
            rtol=LOSS, atol=0
        )

        # Generator voltage setpoints should match.
        np.testing.assert_allclose(
            gen_v_row, gen_reset_data['GenVoltSet'].to_numpy()
        )

        # Pull the load data from PowerWorld and ensure that both the
        # MW and MVAR outputs match up.
        load_init_data = self.env.saw.GetParametersMultipleElement(
            ObjectType='load',
            ParamList=self.env.load_key_fields + self.env.LOAD_RESET_FIELDS
        )

        np.testing.assert_allclose(
            loads_mw_row, load_init_data['LoadSMW'].to_numpy()
        )

        np.testing.assert_allclose(
            loads_mvar_row, load_init_data['LoadSMVR'].to_numpy()
        )

    def test_failed_power_flow(self):
        """Ensure that if the power flow fails to solve, we move on
        to the next scenario.
        """
        # Patch SolvePowerFlow so that the second call fails, while
        # the first, third, and fourth succeed.
        with patch.object(
                self.env.saw, 'SolvePowerFlow',
                side_effect=[None, PowerWorldError('failure'), None,
                             None]):
            self.env.reset()

        # Our first attempt should fail, and the second should succeed.
        # The index is always bumped at the end of each iteration, so
        # it should end up at 2 (starts at 0, bumped to 1 after first
        # failed iteration, bumped to 2 after second successful
        # iteration).
        self.assertEqual(2, self.env.scenario_idx)

    def test_hit_max_iterations(self):
        """Exception should be raised once all scenarios are exhausted.
        """
        # We want every other power flow solve to fail.
        side_effect = [None, PowerWorldError('failure')] * 10
        with patch.object(self.env.saw, 'SolvePowerFlow',
                          side_effect=side_effect):
            with patch.object(self.env, 'num_scenarios', new=5):
                with self.assertRaisesRegex(
                        OutOfScenariosError,
                        'We have gone through all scenarios'):
                    self.env.reset()

    def test_reset_returns_proper_observation(self):
        """Ensure a single call to reset calls _get_observation and
        returns the observation.
        """
        with patch.object(self.env, '_get_observation',
                          side_effect=self.env._get_observation) as p:
            obs = self.env.reset()

        # _get_observation should be called once only. Note if we get
        # into a bad state where the voltages are two low, it may
        # be called more than once. Bad test design due to the fact
        # we can't just spin up new ESA instances for each test.
        p.assert_called_once()

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_extra_reset_actions_called(self):
        with patch.object(self.env, '_set_gens_for_scenario') as p:
            self.env.reset()

        p.assert_called_once()

    def test_set_gens_for_scenario_called(self):
        with patch.object(self.env, '_set_gens_for_scenario') as p:
            self.env.reset()

        p.assert_called_once()

    def test_set_loads_for_scenario_called(self):
        with patch.object(self.env, '_set_loads_for_scenario') as p:
            with patch.object(self.env, '_solve_and_observe'):
                self.env.reset()

        p.assert_called_once()

    def test_solve_and_observe_called(self):
        with patch.object(self.env, '_solve_and_observe') as p:
            self.env.reset()

        p.assert_called_once()

    def test_current_reward_cleared(self):
        self.env.current_reward = 10
        self.env.reset()
        self.assertTrue(np.isnan(self.env.current_reward))

    def test_last_action_cleared(self):
        self.env.last_action = 7
        self.env.reset()
        self.assertIsNone(self.env.last_action)

    def test_branch_opened(self):
        """Ensure branch is opened when reset is called."""
        # Ensure the branch starts closed.
        # Rely on the fact that our setup method sets the scenario index
        # to 0.
        branch_idx = self.env.branches_to_open[0]
        branch = self.env.branch_init_data.iloc[branch_idx]
        kf = self.env.branch_key_fields
        branch_state = self.env.saw.GetParametersSingleElement(
            ObjectType='branch',
            ParamList=kf + ['LineStatus'],
            Values=branch[kf].tolist() + [0]
        )
        self.assertEqual('Closed', branch_state['LineStatus'])

        # Now call reset and ensure the branch is open.
        self.env.reset()

        branch_state = self.env.saw.GetParametersSingleElement(
            ObjectType='branch',
            ParamList=kf + ['LineStatus'],
            Values=branch[kf].tolist() + [0]
        )
        self.assertEqual('Open', branch_state['LineStatus'])

    def test_scenario_init_success_array_properly_updated(self):
        """Ensure the scenario_init_success array gets updated as it
        should.
        """
        # Patch _solve_and_observe to sometimes succeed, sometimes fail.
        se = [None, PowerWorldError('stuff'), PowerWorldError('stuff'), None]
        with patch.object(self.env, '_solve_and_observe', side_effect=se):
            with patch.object(self.env, '_add_to_log'):
                # Run reset twice, as that's all that is required since
                # it'll keep trying on failures.
                for _ in range(2):
                    self.env.reset()

        # Test.
        expected = np.array([True if x is None else False for x in se])
        np.testing.assert_array_equal(
            expected, self.env.scenario_init_success[0:4]
        )


# noinspection DuplicatedCode
class DiscreteVoltageControlEnv14BusStepTestCase(unittest.TestCase):
    """Test the step method of the environment."""
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

        cls.rewards = {'no_op': 50}

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
            pwb_path=PWB_14, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            rewards=cls.rewards,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            dtype=cls.dtype
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        """Reset the scenario index and call reset for each run.
        """
        self.env.scenario_idx = 0
        self.env.reset()

    def action_helper(self, action, gen_bus, v_set):
        """Helper for testing that actions work correctly."""
        # We call reset() in setUp, so that last_action should get
        # reset.
        self.assertIsNone(self.env.last_action)

        # Perform the step.
        self.env.step(action)

        # Ensure the "last_action" is being performed properly.
        self.assertEqual(action, self.env.last_action)

        # Hard-code access to the 0th generator. It's at bus 1.
        gen_init_data = self.env.saw.GetParametersSingleElement(
            ObjectType='gen', ParamList=['BusNum', 'GenID', 'GenVoltSet'],
            Values=[gen_bus, '1', 0]
        )

        self.assertAlmostEqual(v_set, gen_init_data['GenVoltSet'], places=3)

    def test_action_0(self):
        """Action 0 is the no-op action."""
        # Call _solve_and_observe to rotate all the observation data.
        self.env._solve_and_observe()

        # Take the 0th action.
        _, reward, _, _ = self.env.step(0)

        # The observation DataFrames should be identical, indicating
        # that no action was taken.
        pd.testing.assert_frame_equal(
            self.env.bus_obs_data_prev, self.env.bus_obs_data)
        pd.testing.assert_frame_equal(
            self.env.gen_obs_data_prev, self.env.gen_obs_data
        )
        pd.testing.assert_frame_equal(
            self.env.load_obs_data_prev, self.env.load_obs_data
        )

        # The absolute value should be equal to the no-op reward.
        self.assertEqual(self.rewards['no_op'], abs(reward))

    def test_action_1(self):
        """Action 1 should set the 0th generator to the minimum."""
        # The 0th generator is at bus 1.
        self.action_helper(1, 1, self.gen_voltage_range[0])

    def test_action_last(self):
        """The last action should put the last generator to its maximum.
        """
        # The last generator is at bus 8.
        self.action_helper(self.env.action_space.n - 1, 8,
                           self.gen_voltage_range[1])

    def test_action_middle(self):
        """Test an action not on the book ends and ensure the generator
        set point is updated correctly.
        """
        # Action 18 should put the 3rd generator at the 4th voltage
        # level. The 3rd generator is at bus 3. Hard code the fact that
        # the bins are in 0.025pu increments.
        self.action_helper(18, 3, self.gen_voltage_range[0] + 3 * 0.025)

    def test_action_count_increments(self):
        """Ensure each subsequent call to step bumps the action_count.
        """
        self.assertEqual(0, self.env.action_count)
        self.env.step(4)
        self.assertEqual(1, self.env.action_count)
        self.env.step(10)
        self.assertEqual(2, self.env.action_count)
        self.env.step(13)
        self.assertEqual(3, self.env.action_count)

    def test_failed_power_flow(self):
        """If a PowerWorldError is raised while calling SolvePowerFlow,
        the observation should come back with zeros in the voltage
        positions, and the reward should be negative.
        """
        with patch.object(self.env.saw, 'SolvePowerFlow',
                          side_effect=PowerWorldError('failure')):
            obs, reward, done, info = self.env.step(12)

        # Ensure there are zeroes in the appropriate slots.
        self.assertTrue((obs[0:self.env.num_buses] == 0.0).all())

        # Ensure the observation is of the expected size.
        self.assertEqual(obs.shape, (self.env.num_obs,))

        # TODO: This fails because we actually can have numbers less
        #   than 0. So, also need to fix the observation space
        #   definition.
        self.assertTrue((obs[self.env.num_buses:] >= 0.0).all())

        # Make sure the reward is as expected.
        self.assertEqual(
            reward, self.env.rewards['action'] + self.env.rewards['fail'])

    def test_compute_end_of_episode_reward_called_correctly(self):
        with patch.object(self.env, '_solve_and_observe'):
            with patch.object(self.env, '_compute_reward'):

                # Have _check_done return True.
                with patch.object(self.env, '_check_done',
                                  return_value=True):
                    with patch.object(self.env,
                                      '_compute_end_of_episode_reward') as p1:
                        self.env.step(1)

                # Now, have _check_done return False
                with patch.object(self.env, '_check_done',
                                  return_value=False):
                    with patch.object(self.env,
                                      '_compute_end_of_episode_reward') as p2:
                        self.env.step(1)

        p1.assert_called_once()
        self.assertEqual(p2.call_count, 0)


# noinspection DuplicatedCode
class DiscreteVoltageControlEnv14BusComputeRewardTestCase(unittest.TestCase):
    """Test the _compute_reward method of the environment."""
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

        cls.rewards = {
            "action": -10,
            "v_delta": 1,
            "v_in_bounds": 10,
            "v_out_bounds": -10,
            "gen_var_delta": 1,
            "fail": -1000,
            'no_op': 12
        }

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
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
            rewards=cls.rewards,
            dtype=cls.dtype
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        """Override the relevant observation DataFrames.
        """
        # 6 buses with unity per unit voltage.
        v_df = pd.DataFrame(
            [[1., 'a'], [1., 'b'], [1., 'c'], [1., 'd'], [1., 'e'], [1., 'f']],
            columns=['BusPUVolt', 'junk'])

        self.env.bus_obs_data_prev = v_df.copy()
        self.env.bus_obs_data = v_df.copy()

        # 6 gens at 80% var loading.
        var_df = pd.DataFrame(
            [[.8, 'a'], [.8, 'b'], [.8, 'c'], [.8, 'd'], [.8, 'e'], [.8, 'f']],
            columns=['GenMVRPercent', 'junk'])

        self.env.gen_obs_data_prev = var_df.copy()
        self.env.gen_obs_data = var_df.copy()

    def get_reward_no_action(self):
        """Helper to compute the reward but temporarily zero out the
        action penalty.
        """
        with patch.dict(self.env.rewards, {'action': 0}):
            reward = self.env._compute_reward()

        return reward

    def test_action_only(self):
        """No values change, should only get the action penalty."""
        self.assertEqual(self.env._compute_reward(), self.rewards['action'])

    def test_low_voltage_moved_up(self):
        """Test a single low bus voltage moving up, but not in bounds.
        """
        self.env.bus_obs_data_prev.loc[2, 'BusPUVolt'] = 0.8
        self.env.bus_obs_data.loc[2, 'BusPUVolt'] = 0.85

        # The bus voltage moved up 5 1/100ths per unit.
        self.assertAlmostEqual(self.get_reward_no_action(),
                               5 * self.rewards['v_delta'])

    def test_high_voltage_moved_down(self):
        """Test a single high bus voltage moving down, but not in bounds.
        """
        self.env.bus_obs_data_prev.loc[0, 'BusPUVolt'] = 1.1
        self.env.bus_obs_data.loc[0, 'BusPUVolt'] = 1.07

        # The bus voltage moved down 3 1/100ths per unit.
        self.assertAlmostEqual(self.get_reward_no_action(),
                               3 * self.rewards['v_delta'])

    def test_low_voltage_moved_up_past_nominal(self):
        """Test a single low bus voltage moving up and exceeding nominal
        voltage.
        """
        self.env.bus_obs_data_prev.loc[2, 'BusPUVolt'] = 0.93
        self.env.bus_obs_data.loc[2, 'BusPUVolt'] = 1.02

        # The bus voltage should get credit for reducing its distance to
        # nominal, as well as a bonus for moving into the good band.
        self.assertAlmostEqual(
            self.get_reward_no_action(),
            # ((1.02 - 1) - (1 - 0.93)) * 100 = 5
            5 * self.rewards['v_delta'] + self.rewards['v_in_bounds'])

    def test_high_voltage_moved_down_past_nominal(self):
        """Test a single high bus voltage moving down and going below
        nominal voltage.
        """
        self.env.bus_obs_data_prev.loc[5, 'BusPUVolt'] = 1.1
        self.env.bus_obs_data.loc[5, 'BusPUVolt'] = 0.98

        # The bus voltage should get credit for moving to nominal, and
        # also get a bonus for moving into the good band.
        self.assertAlmostEqual(
            self.get_reward_no_action(),
            # ((1.1 - 1) - (1 - 0.98)) * 100 = 8
            8 * self.rewards['v_delta'] + self.rewards['v_in_bounds'])

    def test_low_voltage_moved_in_range(self):
        """Should also get a bonus for moving a voltage in bounds."""
        self.env.bus_obs_data_prev.loc[1, 'BusPUVolt'] = 0.91
        self.env.bus_obs_data.loc[1, 'BusPUVolt'] = 0.95

        # The bus voltage moved up 4 1/100ths per unit, and also moved
        # into the "good" range.
        self.assertAlmostEqual(
            self.get_reward_no_action(),
            4 * self.rewards['v_delta'] + self.rewards['v_in_bounds'])

    def test_high_voltage_moved_in_range(self):
        """Should also get a bonus for moving a voltage in bounds."""
        self.env.bus_obs_data_prev.loc[3, 'BusPUVolt'] = 1.2
        self.env.bus_obs_data.loc[3, 'BusPUVolt'] = 1.05

        # The bus voltage moved by 15 1/100ths per unit, and also
        # moved into the "good" range.
        self.assertAlmostEqual(
            self.get_reward_no_action(),
            15 * self.rewards['v_delta'] + self.rewards['v_in_bounds'])

    def test_high_and_low_moved_in_range(self):
        """Test multiple buses moving opposite directions, but in bounds
        """
        self.env.bus_obs_data_prev.loc[3, 'BusPUVolt'] = 1.07
        self.env.bus_obs_data.loc[3, 'BusPUVolt'] = 1.05

        self.env.bus_obs_data_prev.loc[0, 'BusPUVolt'] = 0.91
        self.env.bus_obs_data.loc[0, 'BusPUVolt'] = 1.05

        self.assertAlmostEqual(
            self.get_reward_no_action(),
            # high moved down
            2 * self.rewards['v_delta']
            # low moved up, but overshot
            + ((1 - 0.91) - (1.05 - 1)) * 100 * self.rewards['v_delta']
            # bonus for moving in band
            + 2 * self.rewards['v_in_bounds'])

    def test_changes_but_all_in_bounds(self):
        """If voltages change, but all stay in bounds, there should be
        no reward, only the penalty for taking an action.
        """
        self.env.bus_obs_data_prev.loc[0, 'BusPUVolt'] = 0.95
        self.env.bus_obs_data.loc[0, 'BusPUVolt'] = 0.96

        self.env.bus_obs_data_prev.loc[1, 'BusPUVolt'] = 1.0
        self.env.bus_obs_data.loc[1, 'BusPUVolt'] = 1.01

        self.env.bus_obs_data_prev.loc[2, 'BusPUVolt'] = 1.05
        self.env.bus_obs_data.loc[2, 'BusPUVolt'] = 1.01

        self.env.bus_obs_data_prev.loc[3, 'BusPUVolt'] = 0.8
        self.env.bus_obs_data.loc[3, 'BusPUVolt'] = 0.8

        self.assertAlmostEqual(self.env._compute_reward(),
                               self.rewards['action'])

    def test_low_v_gets_lower(self):
        """Should get a penalty for moving a low voltage lower."""
        self.env.bus_obs_data_prev.loc[2, 'BusPUVolt'] = 0.93
        self.env.bus_obs_data.loc[2, 'BusPUVolt'] = 0.91

        self.assertAlmostEqual(
            self.get_reward_no_action(), -2 * self.rewards['v_delta'])

    def test_high_v_gets_higher(self):
        """Should get a penalty for moving a high voltage higher."""
        self.env.bus_obs_data_prev.loc[3, 'BusPUVolt'] = 1.06
        self.env.bus_obs_data.loc[3, 'BusPUVolt'] = 1.09

        self.assertAlmostEqual(
            self.get_reward_no_action(), -3 * self.rewards['v_delta'])

    def test_in_bounds_moves_low(self):
        """Should get penalty for voltage that was in bounds moving
        out of bounds.
        """
        self.env.bus_obs_data_prev.loc[3, 'BusPUVolt'] = 1.05
        self.env.bus_obs_data.loc[3, 'BusPUVolt'] = 0.9

        self.assertAlmostEqual(
            self.get_reward_no_action(),
            # Moved 0.05 pu away from lower boundary, also gets extra
            # penalty for leaving bounds.
            -5 * self.rewards['v_delta'] + self.rewards['v_out_bounds'])

    def test_in_bounds_moves_high(self):
        """Should get penalty for voltage that was in bounds moving
        out of bounds.
        """
        self.env.bus_obs_data_prev.loc[0, 'BusPUVolt'] = 0.96
        self.env.bus_obs_data.loc[0, 'BusPUVolt'] = 0.94

        self.assertAlmostEqual(
            self.get_reward_no_action(),
            # Moved 0.01 pu away from lower boundary, also gets extra
            # penalty for leaving bounds.
            -1 * self.rewards['v_delta'] + self.rewards['v_out_bounds'])

    def test_no_op_reward(self):
        """With no violations, should receive the no_op reward if the
        no_op action is the last_action and buses are within bounds.
        """
        with patch.object(self.env, 'last_action', 0):
            r = self.env._compute_reward()

        self.assertEqual(self.rewards['no_op'], r)

    def test_no_op_penalty(self):
        """With any violations, should receive the no_op penalty if the
        no_op action is the last_action and any buses are out of bounds.
        """
        # Put one voltage below.
        self.env.bus_obs_data.loc[0, 'BusPUVolt'] = 0.94

        with patch.object(self.env, 'last_action', 0):
            r = self.env._compute_reward()

        self.assertEqual(self.rewards['no_op'], -r)


# noinspection DuplicatedCode
class GridMindControlEnv14BusInitTestCase(unittest.TestCase):
    """Test the initialization of the environment."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindEnv(
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
            rewards=cls.rewards,
            dtype=cls.dtype
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_branches_to_open(self):
        """By default, branches_to_open should be None."""
        self.assertIsNone(self.env.branches_to_open)

    def test_set_branches_for_scenario_does_not_call_power_world(self):
        """By default, branches_to_open should be None, and thus
        calling _set_branches_for_scenario should not use the saw
        object.
        """
        # Bad practice patching private method from external class.
        with patch.object(self.env.saw, '_call_simauto') as p:
            out = self.env._set_branches_for_scenario()

        self.assertIsNone(out)
        self.assertFalse(p.called)

    def test_loading(self):
        """Ensure all load values are set and are in bounds."""
        # Compare MW.
        original_mw = self.env.load_init_data['LoadSMW'].to_numpy()

        # I feel like there has to be a better way to do this, but
        # I failed to find it.
        #
        # Ensure all loads are at or above the minimum.
        # noinspection PyUnresolvedReferences
        self.assertTrue(
            ((np.tile(original_mw, (self.num_scenarios, 1))
             * self.min_load_factor)
             <= self.env.loads_mw).all()
        )

        # Ensure all loads are at or below the maximum.
        # noinspection PyUnresolvedReferences
        self.assertTrue(
            ((np.tile(original_mw, (self.num_scenarios, 1))
              * self.max_load_factor)
             >= self.env.loads_mw).all()
        )

        # Ensure total loading matches.
        np.testing.assert_array_equal(
            self.env.total_load_mw, self.env.loads_mw.sum(axis=1))

        # Ensure shapes are correct.
        self.assertEqual(self.env.total_load_mw.shape, (self.num_scenarios,))
        self.assertEqual(self.env.loads_mw.shape,
                         (self.num_scenarios, self.env.num_loads))
        self.assertEqual(self.env.loads_mvar.shape,
                         (self.num_scenarios, self.env.num_loads))

    def test_generation(self):
        """Change loading in the case, solve the power flow, and ensure
        all gens pick up the difference.
        """
        try:
            load_copy = self.env.load_init_data.copy(deep=True)

            # Increase loading.
            load_copy['LoadSMW'] = load_copy['LoadSMW'] * 1.2
            self.env.saw.change_and_confirm_params_multiple_element(
                ObjectType='load', command_df=load_copy)

            # Solve the power flow.
            self.env.saw.SolvePowerFlow()

            # Now get generator information.
            gen_data = self.env.saw.GetParametersMultipleElement(
                ObjectType='gen', ParamList=self.env.gen_init_fields
            )

            # Take the difference.
            delta = (gen_data['GenMW']
                     - self.env.gen_init_data['GenMW']).to_numpy()

            # All generators should take on some load.
            np.testing.assert_array_less(0, delta)

            # All generator increases should be nearly the same. The
            # slack will have some differences - we'll allow for 0.5%
            # relative tolerance.
            np.testing.assert_allclose(actual=delta, desired=delta[-1],
                                       rtol=0.005)
        finally:
            self.env.saw.LoadState()

    def test_gen_action_array(self):
        """Ensure the action array is of the correct dimension."""
        # Check the shape.
        self.assertEqual(self.env.gen_action_array.shape[0],
                         self.env.action_space.n)
        self.assertEqual(self.env.gen_action_array.shape[1], self.env.num_gens)

        # Spot check
        np.testing.assert_array_equal(self.env.gen_action_array[0, :],
                                      np.array([self.gen_voltage_range[0]] * 5)
                                      )

        np.testing.assert_array_equal(self.env.gen_action_array[-1, :],
                                      np.array([self.gen_voltage_range[1]] * 5)
                                      )

        a = np.array([self.gen_voltage_range[0]] * 5)
        a[-1] = self.env.gen_bins[1]
        np.testing.assert_array_equal(self.env.gen_action_array[1, :], a)

        b = np.array([self.gen_voltage_range[-1]] * 5)
        b[-1] = self.env.gen_bins[-2]
        np.testing.assert_array_equal(self.env.gen_action_array[-2, :], b)

        c = np.array([self.gen_voltage_range[0]] * 5)
        c[-2] = self.env.gen_bins[1]
        np.testing.assert_array_equal(self.env.gen_action_array[5], c)


# noinspection DuplicatedCode
class GridMindControlEnv14BusRewardTestCase(unittest.TestCase):
    """Test the _compute_reward method."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindEnv(
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
            rewards=cls.rewards,
            dtype=cls.dtype
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        """Override the relevant observation DataFrames, clear the
        cumulative reward.
        """
        # Call reset and decrement the scenario index for consistency.
        self.env.reset()
        self.env.scenario_idx = 0

        # Overwrite bus observations.
        # 6 buses with unity per unit voltage.
        v_df = pd.DataFrame(
            [[1., 'a'], [1., 'b'], [1., 'c'], [1., 'd'], [1., 'e'], [1., 'f']],
            columns=['BusPUVolt', 'junk'])

        self.env.bus_obs_data_prev = v_df.copy()
        self.env.bus_obs_data = v_df.copy()

        # Clear cumulative reward.
        self.env.cumulative_reward = 0

    def test_all_normal(self):
        """All buses in normal zone."""
        self.assertEqual(0, self.env.cumulative_reward)
        reward = self.env._compute_reward()
        self.assertEqual(reward, self.rewards['normal'])

    def test_all_diverged(self):
        """All buses in diverged zone."""
        self.assertEqual(0, self.env.cumulative_reward)
        self.env.bus_obs_data['BusPUVolt'] = \
            np.array([0.0, 1.25, 200, 0.8, 0.5, 1.26])

        reward = self.env._compute_reward()
        self.assertEqual(reward, self.rewards['diverged'])

    def test_all_violation(self):
        """All buses in violation zone."""
        self.assertEqual(0, self.env.cumulative_reward)
        self.env.bus_obs_data['BusPUVolt'] = \
            np.array([0.81, 1.06, 1.249, 0.949, 0.9, 1.1])

        reward = self.env._compute_reward()
        self.assertEqual(reward, self.rewards['violation'])

    def test_mixed(self):
        """Test a mixture of bus zones."""
        self.assertEqual(0, self.env.cumulative_reward)
        self.env.bus_obs_data['BusPUVolt'] = \
            np.array([0.81, 0.79, 1., 1.02, 1.06, 1.04])

        reward = self.env._compute_reward()
        # The presence of any diverged buses means we should get the
        # "diverged" reward.
        self.assertEqual(reward, self.rewards['diverged'])

    def test_cumulative_reward_correct_under_failed_pf(self):
        """Ensure the cumulative reward is correctly computed under
        a failed power flow.
        """
        # Ensure the cumulative reward is 0.
        self.assertEqual(0, self.env.cumulative_reward)

        # Ensure the current reward is NaN.
        self.assertTrue(np.isnan(self.env.current_reward))

        # Patch solve and observe to throw an exception. Also patch
        # _take_action to do nothing. Need to patch _add_to_log so it
        # doesn't get upset about bad sized dataframe.
        with patch.object(self.env, '_solve_and_observe',
                          side_effect=PowerWorldError('bleh')):
            with patch.object(self.env, '_take_action'):
                with patch.object(self.env, '_add_to_log'):
                    # Take a step.
                    obs, rew, d, i = self.env.step(3)

        # Current and cumulative rewards should be equal.
        self.assertEqual(self.env.current_reward, self.env.cumulative_reward)

        # Penalty should be equal to 2* the diverged reward.
        self.assertEqual(self.rewards['diverged'] * 2,
                         self.env.cumulative_reward)


# noinspection DuplicatedCode
class GridMindControlEnv14BusMiscTestCase(unittest.TestCase):
    """Test a few miscellaneous aspects."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindEnv(
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
            rewards=cls.rewards,
            dtype=cls.dtype
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_reset_clears_cumulative_reward(self):
        self.env.cumulative_reward = 10
        self.env.reset()
        self.assertEqual(self.env.cumulative_reward, 0)

    def test_compute_failed_pf_reward(self):
        self.assertEqual(self.env._compute_failed_pf_reward(), -100)

    def test_get_observation(self):
        df = pd.DataFrame([[1., 'a'], [2., 'b']],
                          columns=['BusPUVolt', 'bleh'])
        with patch.object(self.env, 'bus_obs_data', df):
            obs = self.env._get_observation()

        np.testing.assert_array_equal(obs, np.array([1., 2.]))

    def test_take_action_0(self):
        """Action 0 should put all gens at the minimum voltage.
        """
        self.env._take_action(0)
        gens = self.env.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=(self.env.gen_key_fields + ['GenVoltSet']))

        np.testing.assert_allclose(gens['GenVoltSet'].to_numpy(),
                                   self.gen_voltage_range[0])

    def test_take_last_action(self):
        """The last action should put all gens at the maximum voltage.
        """
        self.env._take_action(self.env.action_space.n - 1)
        gens = self.env.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=(self.env.gen_key_fields + ['GenVoltSet']))

        np.testing.assert_allclose(gens['GenVoltSet'].to_numpy(),
                                   self.gen_voltage_range[1])

    def test_failed_pf_obs_zero(self):
        obs = self.env._get_observation_failed_pf()
        self.assertTrue((obs == 0.0).all())
        self.assertEqual(obs.shape, (self.env.num_buses,))


# noinspection DuplicatedCode
class GridMindControlEnv14BusCondensersTestCase(unittest.TestCase):
    """Test the case with condensers and make sure behavior is
    expected.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindEnv(
            pwb_path=PWB_14_CONDENSERS, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            rewards=cls.rewards,
            dtype=cls.dtype
        )

    def test_generation(self):
        """Change loading in the case, solve the power flow, and ensure
        only two gens pick up the difference.
        """
        try:
            load_copy = self.env.load_init_data.copy(deep=True)

            # Increase loading.
            load_copy['LoadSMW'] = load_copy['LoadSMW'] * 1.2
            self.env.saw.change_and_confirm_params_multiple_element(
                ObjectType='load', command_df=load_copy)

            # Solve the power flow.
            self.env.saw.SolvePowerFlow()

            # Now get generator information.
            gen_data = self.env.saw.GetParametersMultipleElement(
                ObjectType='gen', ParamList=self.env.gen_init_fields
            )

            # Take the difference.
            delta = (gen_data['GenMW']
                     - self.env.gen_init_data['GenMW']).to_numpy()

            # The generators at buses 3, 6, and 8 should a) have 0 MW
            # and b) have 0 change in MW.
            gen_3_6_8 = gen_data['BusNum'].isin([3, 6, 8]).to_numpy()

            np.testing.assert_array_equal(
                gen_data.loc[gen_3_6_8, 'GenMW'].to_numpy(), 0.0)

            np.testing.assert_array_equal(delta[gen_3_6_8], 0.0)

            # The remaining generators should take on load.
            np.testing.assert_array_less(0, delta[~gen_3_6_8])

            # All generator increases should be nearly the same. The
            # slack will have some differences - we'll allow for 0.5%
            # relative tolerance.
            np.testing.assert_allclose(actual=delta[~gen_3_6_8],
                                       desired=delta[1], rtol=0.005)
        finally:
            self.env.saw.LoadState()


# noinspection DuplicatedCode
class GridMindControlEnv14BusRenderTestCase(unittest.TestCase):
    """Test rendering."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.oneline_axd = AXD_14
        cls.contour_axd = CONTOUR
        cls.image_dir = os.path.join(THIS_DIR, 'render_dir')
        cls.render_interval = 0.1

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        # noinspection PyTypeChecker
        cls.env = voltage_control_env.GridMindEnv(
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
            rewards=cls.rewards,
            dtype=cls.dtype,
            oneline_axd=cls.oneline_axd, contour_axd=cls.contour_axd,
            image_dir=cls.image_dir, render_interval=cls.render_interval
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()
        shutil.rmtree(cls.image_dir)

    def _get_files_in_image_dir(self):
        # https://stackoverflow.com/a/3207973/11052174
        return [f for f in os.listdir(self.env.image_dir)
                if os.path.isfile(os.path.join(self.env.image_dir, f))]

    def test_rendering(self):
        # Before render has been called, several attributes should be
        # None.
        self.assertIsNone(self.env.image_path)
        self.assertIsNone(self.env.image)
        self.assertIsNone(self.env.image_axis)
        self.assertIsNone(self.env.fig)
        self.assertIsNone(self.env.ax)

        # The render flag should be False.
        self.assertFalse(self.env._render_flag)

        # Reset should be called before render.
        self.env.reset()

        # Render flag should still be False.
        self.assertFalse(self.env._render_flag)

        # Calling render should initialize all sorts of stuff.
        self.env.render()
        self.assertIsNotNone(self.env.image_path)
        self.assertIsNotNone(self.env.image)
        self.assertIsNotNone(self.env.image_axis)
        self.assertIsNotNone(self.env.fig)
        self.assertIsNotNone(self.env.ax)

        # We should have one file in our image directory.
        files = self._get_files_in_image_dir()
        self.assertEqual(len(files), 1)

        # Take a couple steps and render each time.
        for i in range(2):
            self.env.step(self.env.action_space.sample())
            self.env.render()

            files = self._get_files_in_image_dir()
            self.assertEqual(len(files), i+2)


# noinspection DuplicatedCode
class GridMindControlEnv14BusLoggingTestCase(unittest.TestCase):
    """Test that the logging is working as it should.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindEnv(
            pwb_path=PWB_14_CONDENSERS, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            rewards=cls.rewards,
            dtype=cls.dtype,
            log_buffer=cls.log_buffer,
            csv_logfile=cls.csv_logfile
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_log(self):
        """Step through some training-like steps and ensure the logging
        works as expected.
        """
        # Ensure the log array starts empty.
        zeros = np.zeros((self.log_buffer, 14 + 3 + 5))
        np.testing.assert_array_equal(zeros, self.env.log_array)

        # Calling reset should create a log entry.
        self.env.reset()

        entry_1 = self.env.log_array[0, :]
        # Episode:
        self.assertEqual(entry_1[0], 0)
        # Action:
        self.assertTrue(np.isnan(entry_1[1]))
        # Reward:
        self.assertTrue(np.isnan(entry_1[2]))
        np.testing.assert_array_equal(
            zeros[1:, :], self.env.log_array[1:, :])

        # We haven't hit the "buffer" limit yet.
        self.assertEqual(0, self.env.log_flush_count)
        self.assertFalse(os.path.isfile(self.env.csv_logfile))

        # If we run 9 actions, we should hit the buffer.
        actions = [500 + x for x in range(9)]
        for a in actions:
            self.env.step(a)

        # The log should have been flushed.
        self.assertEqual(1, self.env.log_flush_count)
        self.assertTrue(os.path.isfile(self.env.csv_logfile))

        # The log index should have been reset.
        self.assertEqual(0, self.env.log_idx)

        # Read the log file.
        log_data = pd.read_csv(self.env.csv_logfile, index_col=None)

        # Columns should line up.
        self.assertListEqual(log_data.columns.tolist(), self.env.log_columns)

        # There should be the same number of entries as our "buffer"
        # size.
        self.assertEqual(log_data.shape[0], self.env.log_buffer)

        # Ensure the episode number is 0 for all rows.
        self.assertTrue((log_data['episode'] == 0).all())

        # First action should be NaN, while the rest should line up
        # with our action list.
        self.assertTrue(np.isnan(log_data['action_taken'].to_numpy()[0]))
        np.testing.assert_array_equal(
            np.array(actions), log_data['action_taken'].to_numpy()[1:])

        # First reward should be NaN, while the rest should not.
        self.assertTrue(np.isnan(log_data['reward'].to_numpy()[0]))
        self.assertFalse(np.isnan(log_data['reward'].to_numpy()[1:]).any())

        # Bus voltages and generator setpoints should be greater than 0.
        bus_cols = log_data.columns.to_numpy()[
            log_data.columns.str.startswith('bus_') |
            log_data.columns.str.startswith('gen_')]
        self.assertEqual((14+5,), bus_cols.shape)
        self.assertTrue((log_data[bus_cols].to_numpy() > 0).all())

        # Reset the environment and take another set of actions that
        # will cause the buffer to flush.
        self.env.reset()

        # If we run 9 actions, we should hit the buffer.
        actions = [600 + x for x in range(9)]
        for a in actions:
            self.env.step(a)

        # The log should have been flushed for the 2nd time.
        self.assertEqual(2, self.env.log_flush_count)
        self.assertTrue(os.path.isfile(self.env.csv_logfile))

        # The log index should have been reset.
        self.assertEqual(0, self.env.log_idx)

        # Read the log file.
        log_data = pd.read_csv(self.env.csv_logfile, index_col=None)

        # Columns should line up.
        self.assertListEqual(log_data.columns.tolist(), self.env.log_columns)

        # Should now have 2x buffer size entries.
        self.assertEqual(log_data.shape[0], 2 * self.env.log_buffer)

        # Perform a reset and run two actions.
        self.env.reset()
        self.env.step(1502)
        self.env.step(1242)

        # Manually flush the log.
        self.env._flush_log()

        # Now we should get three more rows.
        log_data = pd.read_csv(self.env.csv_logfile, index_col=None)
        self.assertEqual(log_data.shape[0], 2 * self.env.log_buffer + 3)
        # If the last row is 0's then the indexing is bad.
        self.assertFalse(np.array_equal(
            np.zeros(log_data.shape[1]), log_data.to_numpy()[-1, :]))

        # Finally, ensure the "reset_log" method works as intended.
        with patch.object(self.env, '_flush_log') as p:
            self.env.reset_log(new_file='mynewlog.csv')

        # Ensure _flush_log gets called, and that the appropriate
        # variables get reset.
        p.assert_called_once()
        self.assertEqual(self.env.log_idx, 0)
        self.assertEqual(self.env.log_flush_count, 0)
        self.assertEqual(self.env.csv_logfile, 'mynewlog.csv')


# noinspection DuplicatedCode
class GridMindContingenciesEnv14BusLineOpenTestCase(unittest.TestCase):
    """Test that line opening is happening as it should.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.rewards = {
            "normal": 100,
            "violation": -50,
            "diverged": -100
        }

        cls.env = voltage_control_env.GridMindContingenciesEnv(
            pwb_path=PWB_14_CONDENSERS, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            rewards=cls.rewards,
            dtype=cls.dtype,
            log_buffer=cls.log_buffer,
            csv_logfile=cls.csv_logfile
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self):
        # Load the state between runs.
        self.env.saw.LoadState()
        self.env.saw.SolvePowerFlow()

    def _all_closed(self):
        line_data = self.env.saw.GetParametersMultipleElement(
            'branch', self.env.branch_key_fields+['LineStatus'])

        self.assertTrue((line_data['LineStatus'] == 'Closed').all())

    def _one_open(self):
        line_data = self.env.saw.GetParametersMultipleElement(
            'branch', self.env.branch_key_fields+['LineStatus'])

        # Ensure we have a single open line.
        closed = line_data['LineStatus'] == 'Closed'
        self.assertFalse(closed.all())
        self.assertEqual(1, line_data[~closed].shape[0])

    def test_set_branches_for_scenario(self):
        # Ensure the lines are actually all closed right now.
        self._all_closed()

        # Run the method.
        self.env._set_branches_for_scenario()

        # Ensure a single line is open.
        self._one_open()

    def test_reset_opens_branch(self):
        """Ensure a branch is opened after calling reset."""
        # Ensure all closed now.
        self._all_closed()

        # Run reset.
        self.env.reset()

        # One line should be open.
        self._one_open()


# noinspection DuplicatedCode
class DiscreteVoltageControlGenState14BusEnvTestCase(unittest.TestCase):
    """Quick testing of DiscreteVoltageControlGenState14BusEnv."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.env = voltage_control_env.DiscreteVoltageControlGenState14BusEnv(
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
            dtype=cls.dtype,
            log_buffer=cls.log_buffer,
            csv_logfile=cls.csv_logfile
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_branches_to_open(self):
        """By default, branches_to_open should not be None."""
        self.assertIsNotNone(self.env.branches_to_open)
        self.assertEqual((self.env.num_scenarios,),
                         self.env.branches_to_open.shape)
        self.assertTrue(self.env.branches_to_open.min() >= 0)
        self.assertTrue(self.env.branches_to_open.max()
                        < len(self.env.LINES_TO_OPEN))

    def test_set_branches_for_scenario_does_not_call_power_world(self):
        """By default, branches_to_open should be None, and thus
        calling _set_branches_for_scenario should not use the saw
        object.
        """
        # Bad practice patching private method from external class.
        with patch.object(self.env.saw, '_call_simauto') as p:
            out = self.env._set_branches_for_scenario()

        self.assertIsNone(out)
        self.assertTrue(p.called)

    def test_observation_shape(self):
        # 14 buses, 5 generators.
        self.assertEqual((14+5,), self.env.observation_space.shape)

    def test_observation_bounds(self):
        # All lower bounds should be 0.
        np.testing.assert_array_equal(0, self.env.observation_space.low)
        # Upper bounds corresponding to buses should be 2.
        np.testing.assert_array_equal(2, self.env.observation_space.high[0:14])
        # Remaining upper bounds correspond to generator states and
        # should be at 1.
        np.testing.assert_array_equal(1, self.env.observation_space.high[14:])

    def test_num_obs(self):
        # 14 buses, 5 generators.
        self.assertEqual(14+5, self.env.num_obs)

    def test_get_observation(self):
        """Patch the observation data frames and ensure we get back
        what we expect.
        """
        v_arr = np.ones(14, dtype=self.env.dtype)
        v_arr += 0.1
        v = pd.DataFrame({'BusPUVolt': v_arr, 'BusNum': np.arange(1, 15)})

        g_list = ['Closed'] * 5
        g_list[1] = 'Open'
        g_arr = np.ones(5, dtype=self.env.dtype)
        g_arr[1] = 0

        g = pd.DataFrame({'GenStatus': g_list, 'Bleh': [1] * 5})

        with patch.object(self.env, 'bus_obs_data', v):
            with patch.object(self.env, 'gen_obs_data', g):
                obs = self.env._get_observation()

        np.testing.assert_array_equal(v_arr, obs[0:14])

        np.testing.assert_array_equal(g_arr, obs[14:])

    def test_get_observation_failed_pf(self):
        v_arr = np.zeros(14, dtype=self.env.dtype)

        g_list = ['Closed'] * 5
        g_list[3] = 'Open'
        g_list[4] = 'Open'
        g_arr = np.ones(5, dtype=self.env.dtype)
        g_arr[3] = 0
        g_arr[4] = 0

        g = pd.DataFrame({'GenStatus': g_list, 'Stuff': [2] * 5})
        with patch.object(self.env, 'gen_obs_data', g):
            obs = self.env._get_observation_failed_pf()

        np.testing.assert_array_equal(v_arr, obs[0:14])
        np.testing.assert_array_equal(g_arr, obs[14:])


# noinspection DuplicatedCode
class DiscreteVoltageControlBranchState14BusEnvTestCase(unittest.TestCase):
    """Quick testing of DiscreteVoltageControlBranchState14BusEnv."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.env = \
            voltage_control_env.DiscreteVoltageControlBranchState14BusEnv(
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
                dtype=cls.dtype,
                log_buffer=cls.log_buffer,
                csv_logfile=cls.csv_logfile
            )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.env.scenario_idx = 0
        self.env.reset()

    def test_branches_to_open(self):
        """By default, branches_to_open should have an entry for each
        scenario."""
        self.assertIsNotNone(self.env.branches_to_open)
        self.assertEqual((self.env.num_scenarios,),
                         self.env.branches_to_open.shape)
        self.assertTrue(self.env.branches_to_open.min() >= 0)
        self.assertTrue(self.env.branches_to_open.max()
                        < len(self.env.LINES_TO_OPEN))

    def test_set_branches_for_scenario_does_not_call_power_world(self):
        """Since branches_to_open should not be None,
        _set_branches_for_scenario should call PowerWorld.
        """
        # Bad practice patching private method from external class.
        with patch.object(self.env.saw, '_call_simauto') as p:
            out = self.env._set_branches_for_scenario()

        self.assertIsNone(out)
        self.assertTrue(p.called)

    def test_observation_shape(self):
        # 14 buses, 4 "openable" lines.
        self.assertEqual((14+4,), self.env.observation_space.shape)

    def test_observation_bounds(self):
        # All lower bounds should be 0.
        np.testing.assert_array_equal(0, self.env.observation_space.low)
        # Upper bounds corresponding to buses should be 2.
        np.testing.assert_array_equal(2, self.env.observation_space.high[0:14])
        # Remaining upper bounds correspond to line states and
        # should be at 1.
        np.testing.assert_array_equal(1, self.env.observation_space.high[14:])

    def test_num_obs(self):
        # 14 buses, 4 lines to open.
        self.assertEqual(14+4, self.env.num_obs)

    def test_get_observation_and_get_observation_failed_pf(self):
        # Get a copy of the line observation DataFrame.
        branch_obs = self.env.branch_obs_data.copy(deep=True)
        # Convert to multi-index to make selecting buses a bit
        # simpler.
        branch_obs.set_index(['BusNum', 'BusNum:1'], inplace=True)

        # Open all the eligible lines in this DataFrame.
        for l_t in self.env.LINES_TO_OPEN:
            branch_obs.loc[(l_t[0], l_t[1]), 'LineStatus'] = 'Open'

        # Patch the existing DataFrame, call _get_observation.
        with patch.object(self.env, 'branch_obs_data',
                          branch_obs.reset_index()):
            obs = self.env._get_observation()
            obs_failed = self.env._get_observation_failed_pf()

        # The first 14 values correspond to bus voltages.
        np.testing.assert_array_less(0, obs[0:14])
        np.testing.assert_array_equal(0, obs_failed[0:14])

        # The remaining correspond to line states, and should all be 0
        # since we opened all the lines.
        np.testing.assert_array_equal(0, obs[14:])
        np.testing.assert_array_equal(0, obs_failed[14:])

        # Now close all the lines.
        for l_t in self.env.LINES_TO_OPEN:
            branch_obs.loc[(l_t[0], l_t[1]), 'LineStatus'] = 'Closed'

        # Patch the existing DataFrame, call _get_observation.
        with patch.object(self.env, 'branch_obs_data',
                          branch_obs.reset_index()):
            obs = self.env._get_observation()
            obs_failed = self.env._get_observation_failed_pf()

        # The first 14 values correspond to bus voltages.
        np.testing.assert_array_less(0, obs[0:14])
        np.testing.assert_array_equal(0, obs_failed[0:14])

        # The remaining correspond to line states, and should all be 1
        # since we closed all the lines.
        np.testing.assert_array_equal(1, obs[14:])
        np.testing.assert_array_equal(1, obs_failed[14:])

        # Open the second eligible single line.
        l_t = self.env.LINES_TO_OPEN[1]
        branch_obs.loc[(l_t[0], l_t[1]), 'LineStatus'] = 'Open'

        # Patch the existing DataFrame, call _get_observation.
        with patch.object(self.env, 'branch_obs_data',
                          branch_obs.reset_index()):
            obs = self.env._get_observation()
            obs_failed = self.env._get_observation_failed_pf()

        # The first 14 values correspond to bus voltages.
        np.testing.assert_array_less(0, obs[0:14])
        np.testing.assert_array_equal(0, obs_failed[0:14])

        # The remaining correspond to line states. All but the second
        # entry should be closed.
        expected = np.ones(len(self.env.LINES_TO_OPEN), dtype=self.env.dtype)
        expected[1] = 0
        np.testing.assert_array_equal(expected, obs[14:])
        np.testing.assert_array_equal(expected, obs_failed[14:])


# noinspection DuplicatedCode
class DiscreteVoltageControlBranchAndGenState14BusEnvTestCase(
        unittest.TestCase):
    """Quick testing of DiscreteVoltageControlBranchAndGenState14BusEnv.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.env = \
            voltage_control_env.DiscreteVoltageControlBranchAndGenState14BusEnv(
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
                dtype=cls.dtype,
                log_buffer=cls.log_buffer,
                csv_logfile=cls.csv_logfile
            )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.env.scenario_idx = 0
        self.env.reset()

    def test_observation_shape(self):
        # 14 buses, 4 "openable" lines, 5 generators.
        self.assertEqual((14+4+5,), self.env.observation_space.shape)

    def test_observation_bounds(self):
        # All lower bounds should be 0.
        np.testing.assert_array_equal(0, self.env.observation_space.low)
        # Upper bounds corresponding to buses should be 2.
        np.testing.assert_array_equal(2, self.env.observation_space.high[0:14])
        # Remaining upper bounds correspond to line states and
        # generator states should be at 1.
        np.testing.assert_array_equal(1, self.env.observation_space.high[14:])

    def test_num_obs(self):
        # 14 buses, 4 lines to open, 5 generators.
        self.assertEqual(14+4+5, self.env.num_obs)

    def test_get_observation_and_get_observation_failed_pf(self):
        # Get a copy of the line observation DataFrame.
        branch_obs = self.env.branch_obs_data.copy(deep=True)
        # Convert to multi-index to make selecting buses a bit
        # simpler.
        branch_obs.set_index(['BusNum', 'BusNum:1'], inplace=True)

        # Close all lines.
        branch_obs['LineStatus'] = 'Closed'

        # Open the first and last eligible lines.
        for idx in [0, len(self.env.LINES_TO_OPEN) - 1]:
            l_t = self.env.LINES_TO_OPEN[idx]
            branch_obs.loc[(l_t[0], l_t[1]), 'LineStatus'] = 'Open'

        # Get a copy of the generator observation DataFrame.
        gen_obs = self.env.gen_obs_data.copy(deep=True)

        # Close all generators.
        gen_obs['GenStatus'] = 'Closed'

        # Open the first and last generators.
        gen_obs.loc[0, 'GenStatus'] = 'Open'
        gen_obs.loc[gen_obs.shape[0] - 1, 'GenStatus'] = 'Open'

        # Patch the existing DataFrames, call _get_observation.
        with patch.object(self.env, 'branch_obs_data',
                          branch_obs.reset_index()):
            with patch.object(self.env, 'gen_obs_data', gen_obs):
                obs = self.env._get_observation()
                obs_failed = self.env._get_observation_failed_pf()

        # Ensure we get back the correct number of observations.
        self.assertEqual(14+4+5, obs.shape[0])
        self.assertEqual(14 + 4 + 5, obs_failed.shape[0])

        # The first 14 values correspond to bus voltages.
        np.testing.assert_array_less(0, obs[0:14])
        np.testing.assert_array_equal(0, obs_failed[0:14])

        # The remaining correspond to generator and line states.
        line_arr = obs[14:18]
        line_arr_failed = obs[14:18]
        gen_arr = obs[18:]
        gen_arr_failed = obs[18:]

        # Gen states and line states should be the same for the normal
        # and failed cases.
        np.testing.assert_array_equal(line_arr, line_arr_failed)
        np.testing.assert_array_equal(gen_arr, gen_arr_failed)

        # Create array for expected line states.
        line_expected = np.ones(len(self.env.LINES_TO_OPEN),
                                dtype=self.env.dtype)
        line_expected[0] = 0
        line_expected[-1] = 0

        # Create array for expected gen states.
        gen_expected = np.ones(self.env.num_gens, dtype=self.env.dtype)
        gen_expected[0] = 0
        gen_expected[-1] = 0

        # Test.
        np.testing.assert_array_equal(line_expected, line_arr)
        np.testing.assert_array_equal(gen_expected, gen_arr)


# noinspection DuplicatedCode
class IL200BusShuntsTestCase(unittest.TestCase):
    """Test case for shunts with the 200 bus case and the
    DiscreteVoltageControlGenAndShuntNoContingenciesEnv.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 1000
        cls.max_load_factor = 1.4
        cls.min_load_factor = 0.6
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.9
        cls.shunt_closed_probability = 0.5
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        cls.env = \
            voltage_control_env.DiscreteVoltageControlGenAndShuntNoContingenciesEnv(
                pwb_path=PWB_200, num_scenarios=cls.num_scenarios,
                max_load_factor=cls.max_load_factor,
                min_load_factor=cls.min_load_factor,
                min_load_pf=cls.min_load_pf,
                lead_pf_probability=cls.lead_pf_probability,
                load_on_probability=cls.load_on_probability,
                shunt_closed_probability=cls.shunt_closed_probability,
                num_gen_voltage_bins=cls.num_gen_voltage_bins,
                gen_voltage_range=cls.gen_voltage_range,
                seed=cls.seed,
                log_level=logging.INFO,
                dtype=cls.dtype,
                log_buffer=cls.log_buffer,
                csv_logfile=cls.csv_logfile
            )

    def setUp(self) -> None:
        self.env.scenario_idx = 0
        self.env.reset()

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_branches_to_open_none(self):
        self.assertIsNone(self.env.branches_to_open)

    def _shunt_action_helper(self, shunt_idx, start_state, finish_state):
        """The last action should toggle the last shunt. Ensure that
        shunt is open first. shunt_idx should be negative only.
        """
        # Grab the last shunt.
        last_shunt = self.env.shunt_init_data.iloc[shunt_idx].copy()
        # Open it.
        last_shunt['SSStatus'] = start_state
        self.env.saw.ChangeParametersSingleElement(
            ObjectType='shunt',
            ParamList=last_shunt.index.tolist(),
            Values=last_shunt.tolist()
        )
        # Update observations.
        self.env._rotate_and_get_observation_frames()

        # Helper to pull this shunt.
        def get_shunt():
            s = self.env.saw.GetParametersSingleElement(
                ObjectType='shunt',
                ParamList=['BusNum', 'ShuntID', 'SSStatus'],
                Values=[last_shunt['BusNum'], last_shunt['ShuntID'], 0])
            return s

        # Confirm shunt changed state in PowerWorld.
        shunt_out = get_shunt()
        self.assertEqual(start_state, shunt_out['SSStatus'])

        # Now, take the last action, which should toggle this shunt.
        self.env._take_action(self.env.action_space.n + shunt_idx)

        # Confirm it's closed now.
        shunt_out = get_shunt()
        self.assertEqual(finish_state, shunt_out['SSStatus'])

    def test_take_action_last_shunt_open_to_closed(self):
        """Toggle the last shunt from open to closed."""
        self._shunt_action_helper(
            shunt_idx=-1, start_state='Open', finish_state='Closed')

    def test_take_action_first_shunt_closed_to_open(self):
        """Toggle the first shunt from closed to open."""
        # Hard-code the fact that there are 4 shunts in this case.
        self._shunt_action_helper(
            shunt_idx=-4, start_state='Closed', finish_state='Open'
        )

    def test_observation_space(self):
        """Ensure the observation space is the correct size via
        hard-coding.
        """
        # 200 buses, 49 generators, 4 shunts.
        n = 200 + 49 + 4

        self.assertEqual(
            (n,), self.env.observation_space.shape
        )

        # Lower bound should be 0.
        np.testing.assert_array_equal(
            np.zeros(n, dtype=self.env.dtype), self.env.observation_space.low
        )

        # Voltage cap at 2.
        np.testing.assert_array_equal(
            np.ones(200, dtype=self.dtype) + 1,
            self.env.observation_space.high[0:200])

        # All else at 1 (gen and shunt states)
        np.testing.assert_array_equal(
            np.ones(49+4, dtype=self.dtype),
            self.env.observation_space.high[200:]
        )

    def test_action_1_puts_gen_at_min(self):
        """Ensure action 1 puts the first generator at the lowest
        set point.
        """
        # Get the set point.
        gen_data = self.env.gen_obs_data.iloc[0]
        initial_v = gen_data['GenVoltSet']

        # Ensure we dont' start at the minimum.
        self.assertNotAlmostEqual(initial_v, self.gen_voltage_range[0])

        # Take action 1 (0 is no-op).
        self.env._take_action(1)

        # Pull voltage for this generator.
        gen = self.env.saw.GetParametersSingleElement(
            ObjectType='gen',
            ParamList=self.env.gen_key_fields + ['GenVoltSet'],
            Values=gen_data[self.env.gen_key_fields].tolist() + [0]
        )

        # Ensure it's voltage set point is at the minimum.
        self.assertAlmostEqual(self.gen_voltage_range[0], gen['GenVoltSet'])

    def test_last_gen_action_puts_gen_at_max(self):
        """Ensure the last possible gen action puts the last generator
        at the highest set point.
        """
        # Get the starting set point.
        gen_data = self.env.gen_obs_data.iloc[-1]
        initial_v = gen_data['GenVoltSet']

        # Ensure we don't start at the maximum.
        self.assertNotAlmostEqual(initial_v, self.gen_voltage_range[1])

        # Take the last generator action. No need for -1 because action
        # 0 is no-op.
        self.env._take_action(self.env.num_gens * self.num_gen_voltage_bins)

        # Pull voltage for this generator.
        gen = self.env.saw.GetParametersSingleElement(
            ObjectType='gen',
            ParamList=self.env.gen_key_fields + ['GenVoltSet'],
            Values=gen_data[self.env.gen_key_fields].tolist() + [0]
        )

        # Ensure it's voltage set point is at the maximum.
        self.assertAlmostEqual(self.gen_voltage_range[1], gen['GenVoltSet'],
                               6)


# noinspection DuplicatedCode
class TX2000BusShuntsTapsGensTestCase(unittest.TestCase):
    """Test case for shunts, taps, and generators in the Texas 2000 bus
    case.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 10
        cls.max_load_factor = 1.2
        cls.min_load_factor = 0.8
        cls.min_load_pf = 0.8
        cls.lead_pf_probability = 0.1
        cls.load_on_probability = 0.8
        cls.shunt_closed_probability = 0.6
        cls.num_gen_voltage_bins = 5
        cls.gen_voltage_range = (0.95, 1.05)
        cls.seed = 18
        cls.log_level = logging.INFO
        cls.dtype = np.float32
        cls.log_buffer = 10
        cls.csv_logfile = 'log.csv'

        # Expected number of shunts.
        cls.expected_shunts = 264

        # Ensure we remove the logfile if it was created by other
        # test cases.
        try:
            os.remove(cls.csv_logfile)
        except FileNotFoundError:
            pass

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
            pwb_path=PWB_2000, num_scenarios=cls.num_scenarios,
            max_load_factor=cls.max_load_factor,
            min_load_factor=cls.min_load_factor,
            min_load_pf=cls.min_load_pf,
            lead_pf_probability=cls.lead_pf_probability,
            load_on_probability=cls.load_on_probability,
            shunt_closed_probability=cls.shunt_closed_probability,
            num_gen_voltage_bins=cls.num_gen_voltage_bins,
            gen_voltage_range=cls.gen_voltage_range,
            seed=cls.seed,
            log_level=logging.INFO,
            dtype=cls.dtype,
            log_buffer=cls.log_buffer,
            csv_logfile=cls.csv_logfile
        )

    def setUp(self) -> None:
        self.env.saw.LoadState()
        self.env.scenario_idx = 0

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_gens_at_same_bus_have_same_voltage_set_point_in_gen_v(self):
        # Extract boolean array indicating generators that regulate the
        # same bus.
        dup_arr = self.env.gen_dup_reg.to_numpy()
        # Shift that boolean array backwards one slot so we can test.
        dup_shifted = np.roll(dup_arr, -1)

        # Ensure voltage set points are the same for generators on the
        # same buses.
        np.testing.assert_array_equal(
            self.env.gen_v[:, dup_arr],
            self.env.gen_v[:, dup_shifted]
        )

    def test_take_action_multiple_gens_same_bus(self):
        """Ensure that commanding a voltage set point for generators
        that share a bus works properly.
        """
        # Bus 4192 has a bunch of generators.
        bus = 4192

        # Pull the initial voltage. It should be at 1.02.
        gen_volt_before = self.env.gen_init_data[
            self.env.gen_init_data['BusNum'] == bus
        ]['GenVoltSet']

        np.testing.assert_allclose(gen_volt_before.to_numpy(), 1.02)

        # Get the action which will put these generators at their max.
        # Add 1 due to the no-op action.
        action = np.argmax(
            (self.env.gen_action_array[:, 0] == bus)
            & (self.env.gen_action_array[:, 1]
               == self.env.gen_bins.shape[0] - 1)) + 1

        # Take the action.
        self.env._take_action(action)

        # Pull gens from PowerWorld.
        gens = self.env.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=['BusNum', 'GenID', 'GenVoltSet']
        )

        # All generators at our bus should have the maximum voltage
        # set point.
        np.testing.assert_allclose(
            gens[gens['BusNum'] == bus]['GenVoltSet'].to_numpy(),
            self.env.gen_bins[-1]
        )

    def test_shunt_init_data(self):
        """Ensure the right number of shunts have been picked up."""
        self.assertEqual(self.env.shunt_init_data.shape[0],
                         self.expected_shunts)

    def test_shunt_shunt_states(self):
        """Ensure the shunt_states attribute is as expected."""
        # Check shape.
        self.assertEqual(self.env.shunt_states.shape,
                         (self.num_scenarios, self.expected_shunts))

        # Ensure the "on" percentage is fairly close (say, within 5%).
        on_pct = self.env.shunt_states.sum().sum() \
            / (self.num_scenarios * self.expected_shunts)

        self.assertGreaterEqual(on_pct, self.shunt_closed_probability - 0.05)
        self.assertLessEqual(on_pct, self.shunt_closed_probability + 0.05)

    def _shunt_helper(self, shunt_patch, state):
        with patch.object(self.env, 'shunt_states', new=shunt_patch):
            self.env._set_shunts_for_scenario()

        # Retrieve.
        df = self.env.saw.GetParametersMultipleElement(
            'shunt', self.env.shunt_key_fields + ['SSStatus'])

        # noinspection PyUnresolvedReferences
        self.assertTrue((df['SSStatus'] == state).all())

    def test_set_shunts_for_scenario_closed(self):
        """Ensure the shunt setting works properly."""
        # Close all shunts.
        shunt_patch = np.ones((self.num_scenarios, self.expected_shunts),
                              dtype=bool)

        self._shunt_helper(shunt_patch, 'Closed')

    def test_set_shunts_for_scenario_open(self):
        # Open all shunts.
        shunt_patch = np.zeros((self.num_scenarios, self.expected_shunts),
                               dtype=bool)

        self._shunt_helper(shunt_patch, 'Open')

    def test_auto_control_overridden_in_init(self):
        """Ensure that all shunts have their AutoControl property
        turned off after initialization.
        """
        # Ensure that some shunts did indeed start in auto mode.
        self.assertTrue(
            (self.env.shunt_init_data['AutoControl'] == 'YES').any())

        # Fetch the current settings.
        shunts = self.env.saw.GetParametersMultipleElement(
            ObjectType='shunt',
            ParamList=self.env.shunt_key_fields + ['AutoControl'])

        # Now all shunts should not have auto control.
        self.assertTrue((shunts['AutoControl'] == 'NO').all())

    def test_ltc_data(self):
        """Ensure we're getting the expected LTC data."""
        # There should be 35 LTCs in the case.
        self.assertEqual(35, self.env.ltc_init_data.shape[0])

        # Fetch the integer (well, they're floats for whatever silly
        # reason) min and max tap positions. They should come back as
        # -16 to 16.
        ltc = self.env.saw.GetParametersMultipleElement(
            ObjectType='branch',
            ParamList=(self.env.branch_key_fields
                       + ['XFTapPos:1', 'XFTapPos:2']),
            FilterName=self.env.ltc_filter
        )

        self.assertTrue((ltc['XFTapPos:1'] == -16.0).all())
        self.assertTrue((ltc['XFTapPos:2'] == 16.0).all())

    def test_change_taps(self):
        """Test that changing taps creates the expected change in
        voltage.
        """
        # We'll be hard-coding bus numbers for an LTC.
        # b_f = 7402  # bus from
        # b_t = 7401  # bus to
        # b_f = 7423
        # b_t = 7422
        b_f = 8140
        b_t = 8139
        c = '1'     # circuit ID

        # print(f'From bus: {b_f}')
        # print(f'To bus: {b_t}')

        # List of parameters for commanding.
        c_list = ['BusNum', 'BusNum:1', 'LineCircuit', 'XFTapPos']

        # Set the tap to 0 and solve the power flow.
        self.env.saw.ChangeParametersSingleElement(
            ObjectType='branch', ParamList=c_list, Values=[b_f, b_t, c, 0])

        # Solve the power flow.
        self.env.saw.SolvePowerFlow()

        # Ensure the tap is in the right position and that the min and
        # max are as expected.
        # Ensure the tap changed.
        xf = self.env.saw.GetParametersSingleElement(
            ObjectType='branch',
            ParamList=c_list + ['XFTapPos:1', 'XFTapPos:2', 'BusPUVolt',
                                'BusPUVolt:1', 'LineAmp', 'LineR', 'LineX',
                                'XFTapMin', 'XFTapMax'],
            Values=([b_f, b_t, c] + [0] * 10))

        self.assertEqual(xf['XFTapPos:1'], -16)
        self.assertEqual(xf['XFTapPos:2'], 16)
        self.assertEqual(xf['XFTapPos'], 0)
        self.assertEqual(xf['XFTapMin'], 0.9)
        self.assertEqual(xf['XFTapMax'], 1.1)

        # Voltages should be almost equal.
        # self.assertAlmostEqual(xf['BusPUVolt'], xf['BusPUVolt:1'], 2)
        # print('With tap at 0:')
        # print(xf['BusPUVolt'] / xf['BusPUVolt:1'])
        # print(f"V1: {xf['BusPUVolt']:.3f}")
        # print(f"V2: {xf['BusPUVolt:1']:.3f}")

        # Put the tap at the maximum
        self.env.saw.ChangeParametersSingleElement(
            ObjectType='branch', ParamList=c_list, Values=[b_f, b_t, c, 16])

        # Solve the power flow.
        self.env.saw.SolvePowerFlow()

        # Ensure the tap changed.
        xf = self.env.saw.GetParametersSingleElement(
            ObjectType='branch',
            ParamList=c_list + ['BusPUVolt', 'BusPUVolt:1', 'XFTapPercent'],
            Values=[b_f, b_t, c, 0, 0, 0, 0])

        self.assertEqual(xf['XFTapPos'], 16)
        self.assertEqual(xf['XFTapPercent'], 100.0)

        # Ensure the first voltage is higher than the second.
        self.assertTrue(xf['BusPUVolt'] > xf['BusPUVolt:1'])
        # print('With tap at 16:')
        # print(xf['BusPUVolt'] / xf['BusPUVolt:1'])
        # print(f"V1: {xf['BusPUVolt']:.3f}")
        # print(f"V2: {xf['BusPUVolt:1']:.3f}")

        # The voltage out should be ~1.1 * the voltage in.
        # self.assertAlmostEqual(xf['BusPUVolt'] * 1.1, xf['BusPUVolt:1'])

        # Can't put the tap at the minimum without causing the power
        # flow to diverge. Do -8 instead of -16.

        # Put the tap low.
        self.env.saw.ChangeParametersSingleElement(
            ObjectType='branch', ParamList=c_list, Values=[b_f, b_t, c, -16])

        self.env.saw.SolvePowerFlow()

        # Ensure the tap changed.
        xf = self.env.saw.GetParametersSingleElement(
            ObjectType='branch',
            ParamList=c_list + ['BusPUVolt', 'BusPUVolt:1', 'XFTapPercent'],
            Values=[b_f, b_t, c, 0, 0, 0, 0])

        self.assertEqual(xf['XFTapPos'], -16)
        self.assertEqual(xf['XFTapPercent'], -100)

        # Ensure the second voltage is higher than the first.
        self.assertTrue(xf['BusPUVolt'] < xf['BusPUVolt:1'])
        # print('With tap at -16:')
        # print(xf['BusPUVolt'] / xf['BusPUVolt:1'])
        # print(f"V1: {xf['BusPUVolt']:.3f}")
        # print(f"V2: {xf['BusPUVolt:1']:.3f}")

    def test_action_space(self):
        """Ensure the action space is the correct size accounting for
        gens, ltcs, and shunts.
        """
        self.assertTrue(False)

    def test_action_ltc(self):
        """Ensure an action corresponding to an LTC is handled
        correctly.
        """
        self.assertTrue(False)

    def test_action_shunt(self):
        """Ensure an action corresponding to a shunt is handled
        correctly.
        """
        self.assertTrue(False)

    # def test_solve(self):
    #     while self.env.scenario_idx < self.env.num_scenarios:
    #         self.env.reset()
    #
    #     ratio = self.env.reset_successes / self.env.num_scenarios
    #     self.assertGreaterEqual(ratio, 0.9)
    #     print(f'Success ratio: {ratio:.3f}')


# noinspection DuplicatedCode
class DiscreteVoltageControlEnvFilterScenariosTestCase(unittest.TestCase):
    """Test the filter_scenarios method of the environment."""
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 10
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

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
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

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        """Reset the scenario index for each run, and restore the
        case.
        """
        self.env.scenario_idx = 0
        self.env.saw.LoadState()

    def test_filter(self):
        """Ensure the filtering is successful."""
        # Create a filter in which every other scenario is feasible.
        num_success = int(self.num_scenarios / 2)
        mask = np.array([True, False] * num_success)

        # Perform the filtering.
        self.env.filter_scenarios(mask)

        # Check shapes.
        all_none = True
        for attr in self.env.SCENARIO_INIT_ATTRIBUTES:
            arr = getattr(self.env, attr)
            if arr is None:
                continue
            all_none = False
            self.assertEqual(num_success, arr.shape[0])

        self.assertFalse(all_none)

        # Run reset until we exhaust the scenarios. Patch SolvePowerFlow
        # so we always have a success.
        with patch.object(self.env.saw, 'SolvePowerFlow'):
            i = 0
            while i < self.num_scenarios:
                try:
                    self.env.reset()
                except OutOfScenariosError:
                    break
                i += 1

        # Now, the scenario index should equal the number of successes.
        self.assertEqual(num_success, self.env.scenario_idx)


class ScaleVoltagesTestCase(unittest.TestCase):
    """Test _scale_voltages."""

    # noinspection PyMethodMayBeStatic
    def test_works(self):
        # Assuming MIN_V is 0.7 and MAX_V is 1.2, this array should be
        # [0.7, 0.95, 1.2]
        a = np.array([MIN_V, MIN_V + (MAX_V - MIN_V)/2, MAX_V])
        # Assuming MIN_V_SCALED is 0 and MAX_V_SCALED is 1, expected
        # should be [0, 0.5, 1]
        expected = np.array([MIN_V_SCALED, (MAX_V_SCALED - MIN_V_SCALED)/2,
                             MAX_V_SCALED])
        np.testing.assert_allclose(expected, _scale_voltages(a))


class DiscreteVoltageControlEnvVoltBoundsTestCase(unittest.TestCase):
    """Test using the truncate_voltages parameter to cause
    power flows to count as failed if the voltages are out of range.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Initialize the environment. Then, we'll use individual test
        # methods to test various attributes, methods, etc.

        # Define inputs to the constructor.
        cls.num_scenarios = 10
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
        cls.truncate_voltages = True

        cls.env = voltage_control_env.DiscreteVoltageControlEnv(
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
            dtype=cls.dtype,
            truncate_voltages=cls.truncate_voltages
        )

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def patch_helper(self, df, exc=True):
        with patch.object(self.env.saw, 'SolvePowerFlow') as p:
            with patch.object(self.env, 'bus_obs_data', new=df):
                with patch.object(self.env,
                                  '_rotate_and_get_observation_frames'):
                    with patch.object(self.env, '_get_observation'):
                        if exc:
                            with self.assertRaisesRegex(PowerWorldError,
                                                        'Scenario rejected'):
                                self.env._solve_and_observe()
                        else:
                            self.env._solve_and_observe()

    def test_solve_and_observe_low(self):
        """Ensure _solve_and_observe throws a PowerWorld error for
        voltages below the threshold.
        """
        # Patch the bus_obs_data DataFrame.
        df = pd.DataFrame({'BusPUVolt': np.ones(self.env.num_buses)})
        df.iloc[3]['BusPUVolt'] = MIN_V - 0.01

        self.patch_helper(df, True)

    def test_solve_and_observe_high(self):
        """Ensure _solve_and_observe throws a PowerWorld error for
        voltages below the threshold.
        """
        # Patch the bus_obs_data DataFrame.
        df = pd.DataFrame({'BusPUVolt': np.ones(self.env.num_buses)})
        df.iloc[7]['BusPUVolt'] = MAX_V + 0.01

        self.patch_helper(df, True)

    def test_solve_and_observe_in_bounds(self):
        """Ensure _solve_and_observe throws a PowerWorld error for
        voltages below the threshold.
        """
        # Patch the bus_obs_data DataFrame.
        df = pd.DataFrame({'BusPUVolt': np.ones(self.env.num_buses)})
        df.iloc[11]['BusPUVolt'] = MAX_V
        df.iloc[1]['BusPUVolt'] = MIN_V

        self.patch_helper(df, False)

#
# # noinspection DuplicatedCode
# class GridMindHardSolveTestCase(unittest.TestCase):
#     """Ensure a certain percentage of hard cases are solvable."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         # Initialize the environment. Then, we'll use individual test
#         # methods to test various attributes, methods, etc.
#
#         # Define inputs to the constructor.
#         cls.num_scenarios = 1000
#         # 50% to 150% loading.
#         cls.max_load_factor = 1.5
#         cls.min_load_factor = 0.5
#         cls.min_load_pf = 0.8
#         cls.lead_pf_probability = 0.1
#         cls.load_on_probability = 0.8
#         cls.num_gen_voltage_bins = 5
#         cls.gen_voltage_range = (0.95, 1.05)
#         cls.seed = 18
#         cls.log_level = logging.INFO
#         cls.dtype = np.float32
#         cls.log_buffer = 10
#         cls.csv_logfile = 'log.csv'
#
#         # Ensure we remove the logfile if it was created by other
#         # test cases.
#         try:
#             os.remove(cls.csv_logfile)
#         except FileNotFoundError:
#             pass
#
#         cls.rewards = {
#             "normal": 100,
#             "violation": -50,
#             "diverged": -100
#         }
#
#         cls.env = voltage_control_env.GridMindHardEnv(
#             pwb_path=PWB_14_CONDENSERS, num_scenarios=cls.num_scenarios,
#             max_load_factor=cls.max_load_factor,
#             min_load_factor=cls.min_load_factor,
#             min_load_pf=cls.min_load_pf,
#             lead_pf_probability=cls.lead_pf_probability,
#             load_on_probability=cls.load_on_probability,
#             num_gen_voltage_bins=cls.num_gen_voltage_bins,
#             gen_voltage_range=cls.gen_voltage_range,
#             seed=cls.seed,
#             log_level=logging.INFO,
#             rewards=cls.rewards,
#             dtype=cls.dtype,
#             log_buffer=cls.log_buffer,
#             csv_logfile=cls.csv_logfile
#         )
#
#     # noinspection PyUnresolvedReferences
#     @classmethod
#     def tearDownClass(cls) -> None:
#         cls.env.close()
#
#     def test_solve(self):
#         while self.env.scenario_idx < self.env.num_scenarios:
#             self.env.reset()
#
#         ratio = self.env.reset_successes / self.env.num_scenarios
#         self.assertGreaterEqual(ratio, 0.9)
#         print(f'Success ratio: {ratio:.3f}')


if __name__ == '__main__':
    unittest.main()
