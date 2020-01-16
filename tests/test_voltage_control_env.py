import unittest
from unittest.mock import patch
from gym_powerworld.envs import voltage_control_env
from gym_powerworld.envs.voltage_control_env import LOSS, MIN_V
import os
import pandas as pd
import numpy as np
import numpy.testing as np_test
import logging
import warnings
from esa import SAW, PowerWorldError
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


# noinspection DuplicatedCode
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
        env = voltage_control_env.VoltageControlEnv(
            pwb_path=PWB_14, num_scenarios=10,
            max_load_factor=self.max_load_factor,
            min_load_factor=self.min_load_factor,
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
            env = voltage_control_env.VoltageControlEnv(
                pwb_path=PWB_14, num_scenarios=10,
                max_load_factor=self.max_load_factor,
                min_load_factor=self.min_load_factor,
                rewards={'v_detla': 1000})

# noinspection DuplicatedCode
class VoltageControlEnv14BusResetTestCase(unittest.TestCase):
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
        """Reset the scenario index for each run."""
        self.env.scenario_idx = 0

    def test_scenario_idx_increments(self):
        """Ensure subsequent calls to reset update the scenario index.
        """
        # Patch the changing of parameters so that we'll get a
        # a consistent incrementing of the index (no failed power flow).
        with patch.object(self.env.saw,
                          'change_and_confirm_params_multiple_element'):
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
                          'change_and_confirm_params_multiple_element'):
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
                          'change_and_confirm_params_multiple_element'):
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
        p = LOAD_MW_14 / 4
        gen_mw_row = np.array([p, 0, p, p, p])
        gen_mw = self.env.gen_mw.copy()
        gen_mw[0, :] = gen_mw_row

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
            with patch.object(self.env, 'loads_mw', new=loads_mw):
                with patch.object(self.env, 'loads_mvar', new=loads_mvar):
                    self.env.reset()

        # Pull the generator data from PowerWorld and ensure that both
        # the status and output match up.
        gen_data = self.env.saw.GetParametersMultipleElement(
            ObjectType='gen',
            ParamList=self.env.gen_key_fields + self.env.GEN_RESET_FIELDS)

        # All gens except for the 2nd should be closed.
        status = ['Closed'] * 5
        status[1] = 'Open'
        self.assertListEqual(status, gen_data['GenStatus'].tolist())

        # Excluding the slack, generator MW output should exactly match
        # what was commanded.
        np.testing.assert_allclose(
            gen_mw_row[1:], gen_data['GenMW'].to_numpy()[1:])

        # The slack should be equal to within our assumed line losses.
        np.testing.assert_allclose(
            gen_mw_row[0], gen_data['GenMW'].to_numpy()[0],
            rtol=LOSS, atol=0
        )

        # Pull the load data from PowerWorld and ensure that both the
        # MW and MVAR outputs match up.
        load_data = self.env.saw.GetParametersMultipleElement(
            ObjectType='load',
            ParamList=self.env.load_key_fields + self.env.LOAD_RESET_FIELDS
        )

        np.testing.assert_allclose(
            loads_mw_row, load_data['LoadSMW'].to_numpy()
        )

        np.testing.assert_allclose(
            loads_mvar_row, load_data['LoadSMVR'].to_numpy()
        )

    def test_failed_power_flow(self):
        """Ensure that if the power flow fails to solve, we move on
        to the next scenario.
        """
        # Patch SolvePowerFlow so the first call raises a
        # PowerWorldError but the second simply returns None (indicating
        # success).
        with patch.object(
                self.env.saw, 'SolvePowerFlow',
                side_effect=[PowerWorldError('failure'), None]):
            self.env.reset()

        # Our first attempt should fail, and the second should succeed.
        # The index is always bumped at the end of each iteration, so
        # it should end up at 2 (starts at 0, bumped to 1 after first
        # failed iteration, bumped to 2 after second successful
        # iteration).
        self.assertEqual(2, self.env.scenario_idx)

    def test_low_voltage_skipped(self):
        """Ensure that if a bus voltage comes back lower than allowed,
        the case is skipped.
        """
        self.assertEqual(0, self.env.scenario_idx)

        # Having trouble getting some good patching going in order to
        # get _get_observation to set the bus_obs differently for each
        # run. So, I'm going to do this the "bad" way and set up the
        # first scenario such that all generators are off (the power
        # flow will solve, but all buses have 0 pu voltage),
        # and set up the second scenario to ensure the power flow will
        # converge with all buses above the minimum threshold.
        gen_mw = np.array([[0] * N_GENS_14,
                           self.gens['GenMW'].tolist()])

        load_mw = np.array([[100] * N_LOADS_14,
                            self.loads['LoadSMW'].tolist()])

        load_mvar = np.array([[0] * N_LOADS_14,
                             self.loads['LoadSMVR'].tolist()])

        # Patch the environment's generator and load scenario data.
        with patch.object(self.env, 'gen_mw', new=gen_mw):
            with patch.object(self.env, 'loads_mw', new=load_mw):
                with patch.object(self.env, 'loads_mvar', new=load_mvar):
                    self.env.reset()

        # The scenario index should now be at 2.
        self.assertEqual(2, self.env.scenario_idx)

    def test_hit_max_iterations(self):
        """Exception should be raised once all scenarios are exhausted.
        """
        with patch.object(self.env.saw, 'SolvePowerFlow',
                          side_effect=PowerWorldError('failure')):
            with patch.object(self.env, 'num_scenarios', new=5):
                with self.assertRaisesRegex(
                        UserWarning, 'We have gone through all scenarios'):
                    self.env.reset()


if __name__ == '__main__':
    unittest.main()
