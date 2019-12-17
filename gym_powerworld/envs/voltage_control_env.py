import gym
from gym import spaces
import numpy as np
import logging
from esa import SAW
from typing import Union, Tuple

# When generating scenarios, we're drawing random generation to meet
# the load. There will be some rounding error, so set a reasonable
# tolerance. Note this is in MW. In the power flow, the slack bus will
# pick up this rounding error - that's okay.
GEN_LOAD_DELTA_TOL = 0.001

# For safety we'll have a maximum number of loop iteration.
ITERATION_MAX = 100

# Constants related to PowerWorld (for convenience and cleanliness):
# Constant power portion of PowerWorld loads.
LOAD_P = ['LoadSMW', 'LoadSMVR']

# Constant current and constant impedance portions of PowerWorld
# loads.
LOAD_I_Z = ['LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']


class VoltageControlEnv(gym.Env):
    """Environment for performing voltage control with the PowerWorld
    Simulator.
    """

    # All PowerWorld generator fields that will be used during
    # environment initialization, sans key fields.
    GEN_FIELDS = ['BusCat', 'GenMW', 'GenMVR', 'GenVoltSet', 'GenMWMax',
                  'GenMWMin', 'GenMVRMax', 'GenMVRMin', 'GenStatus']

    # PowerWorld generator fields which will be used when generating
    # an observation during a time step.
    GEN_OBS_FIELDS = ['GenMW', 'GenMVA', 'GenVoltSet', 'GenMVRPercent',
                      'GenStatus']

    # All PowerWorld load fields that will be used during environment
    # initialization, sans key fields.
    LOAD_FIELDS = LOAD_P + LOAD_I_Z

    # PowerWorld load fields that will be used when generating an
    # observation during a time step. Voltage will be handled at the
    # bus level rather than the load level.
    LOAD_OBS_FIELDS = LOAD_P

    # Bus fields for environment initialization will simply be the
    # key fields, which are not listed here. During a time step, we'll
    # want the per unit voltage at the bus.
    BUS_OBS_FIELDS = ['BusPUVolt']

    def __init__(self, pwb_path: str, num_scenarios: int,
                 max_load_factor: Union[str, float] = None,
                 min_load_factor: Union[str, float] = None,
                 min_load_pf: float = 0.8,
                 lead_pf_probability: float = 0.1,
                 load_on_probability: float = 0.8,
                 num_gen_voltage_bins: int = 5,
                 gen_voltage_range: Tuple[float, float] = (0.9, 1.1),
                 seed: Union[str, float] = None,
                 log_level=logging.INFO,
                 dtype=np.float32):
        """

        :param pwb_path: Full path to a PowerWorld .pwb file
            representing the grid which the agent will control.
        :param num_scenarios: Number of different case initialization
            scenarios to create. Note it is not guaranteed that the
            power flow will solve successfully for all cases, so the
            number of actual usable cases will be less than the
            num_scenarios.
        :param max_load_factor: Factor which when multiplied by the
            current total system load represents the maximum allowable
            system load for training states. E.g., if the current sum
            of all the loads is 100MVA and the max_load_factor is
            2, then the maximum possible loading in a training state
            will be 200MVA. If the max_load_factor is None, the maximum
            system loading will be computed by the sum of generator
            maximum MW outputs.
        :param min_load_factor: Similar to the max_load_factor, this
            number is multiplied with the current total system load to
            determine the minimum loading for a training state. If None,
            the minimum system loading will be computed by the sum of
            generator minimum MW outputs.
        :param min_load_pf: Minimum load power factor to be used when
            generating loading scenarios. Should be a positive number
            in the interval (0, 1].
        :param lead_pf_probability: Probability on a load by load
            basis that the power factor will be leading. Should be
            a positive number in the interval [0, 1].
        :param load_on_probability: For each scenario, probability
            to determine on a load by load basis which are "on" (i.e.
            have power consumption > 0). Should be a positive number
            on the interval (0, 1].
        :param num_gen_voltage_bins: Number of intervals/bins to split
            generator voltage set points into. I.e.,
            if gen_voltage_range=(0.95, 1.05) and gen_bins is 5, the
            generator set points will be discretized into the set
            {0.95, 0.975, 1.0, 1.025, 1.05}.
        :param gen_voltage_range: Minimum and maximum allowed generator
            voltage regulation set points (in per unit).
        :param seed: Seed for random number.
        :param log_level: Log level for the environment's logger. Pass
            a constant from the logging module, e.g. logging.DEBUG or
            logging.INFO.
        :param dtype: Numpy datatype for observations.
        """
        ################################################################
        # Logging, seeding, SAW initialization, simple attributes
        ################################################################
        # Set up log.
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)

        # Handle random seeding up front.
        self.rng = np.random.default_rng(seed)

        # Initialize a SimAuto wrapper. Use early binding since it's
        # faster.
        self.saw = SAW(pwb_path, early_bind=True)
        self.log.debug('PowerWorld case loaded.')

        # Track data type.
        self.dtype = dtype

        self.num_scenarios = num_scenarios
        ################################################################
        # Generator fields and data
        ################################################################
        # Get generator data.
        # Start by getting a listing of the key fields.
        self.gen_key_fields = self.saw.get_key_field_list('gen')
        # Define all the parameters we'll need for setup.
        self.gen_fields = self.gen_key_fields + self.GEN_FIELDS

        # The following fields will be used for observations during
        # learning.
        self.gen_obs_fields = self.gen_key_fields + self.GEN_OBS_FIELDS

        # Use the SimAuto Wrapper to get generator data from PowerWorld.
        self.gen_data = self.saw.GetParametersMultipleElement(
            ObjectType='gen', ParamList=self.gen_fields)

        # For convenience, get the number of generators.
        self.num_gens = self.gen_data.shape[0]

        # Zero out negative minimum generation.
        self._zero_negative_gen_mw_limits()

        # For convenience, compute the maximum generation capacity. This
        # will also represent maximum loading.
        # TODO: Confirm that the produce/consume convention is correct.
        self.gen_mw_capacity = self.gen_data['GenMWMax'].sum()
        self.gen_mvar_produce_capacity = self.gen_data['GenMVRMax'].sum()
        self.gen_mvar_consume_capacity = self.gen_data['GenMVRMin'].sum()

        ################################################################
        # Load fields and data
        ################################################################
        # Get load data.
        # TODO: Somehow need to ensure that the only active load models
        #   are ZIP.
        # Get key fields for loads.
        self.load_key_fields = self.saw.get_key_field_list('load')
        self.load_fields = self.load_key_fields + self.LOAD_FIELDS
        # Query PowerWorld for the load data.
        self.load_data = self.saw.GetParametersMultipleElement(
            ObjectType='load', ParamList=self.load_fields)

        # Number of loads for convenience.
        self.num_loads = self.load_data.shape[0]

        # The following fields will be used for observations.
        self.load_obs_fields = self.load_key_fields + self.LOAD_OBS_FIELDS

        # Zero out constant current and constant impedance portions so
        # we simply have constant power.
        # TODO: in the future, different loads should be considered.
        self._zero_i_z_loads()

        ################################################################
        # Bus fields and data
        ################################################################
        # Get bus data.
        self.bus_key_fields = self.saw.get_key_field_list('bus')

        # The following fields will be used for observations.
        self.bus_obs_fields = self.bus_key_fields + ['BusPUVolt']
        # Get bus data from PowerWorld.
        self.bus_data = self.saw.GetParametersMultipleElement(
            ObjectType='bus', ParamList=self.bus_key_fields)
        # For convenience, track number of buses.
        self.num_buses = self.bus_data.shape[0]

        ################################################################
        # Minimum and maximum system loading
        ################################################################
        # Compute maximum system loading.
        if max_load_factor is not None:
            # If given a max load factor, multiply it by the current
            # system load.
            self.max_load_mw = \
                self.load_data['LoadSMW'].sum() * max_load_factor
        else:
            # If not given a max load factor, compute the maximum load
            # as the sum of the generator maximums.
            self.max_load_mw = self.gen_mw_capacity

        # Compute minimum system loading.
        if min_load_factor is not None:
            self.min_load_mw = \
                self.load_data['LoadSMW'].sum() * min_load_factor
        else:
            self.min_load_mw = self.gen_data['GenMWMin'].sum()

        # Warn if our generation capacity is more than double the max
        # load - this could mean generator maxes aren't realistic.
        gen_factor = self.gen_mw_capacity / self.max_load_mw
        if gen_factor >= 2:
            self.log.warning(
                f'The given generator capacity, {self.gen_mw_capacity:.2f} MW,'
                f' is {gen_factor:.2f} times larger than the maximum load, '
                f'{self.max_load_mw:.2f} MW. This could indicate that '
                'the case does not have proper generator limits set up.')

        ################################################################
        # Scenario/episode initialization: loads
        ################################################################
        # Time to generate scenarios.
        # Initialize list to hold all information pertaining to all
        # scenarios.
        self.scenarios = []

        self.total_load_mw, self.loads_mw, self.loads_mvar = \
            self._compute_loading(load_on_probability=load_on_probability,
                                  min_load_pf=min_load_pf,
                                  lead_pf_probability=lead_pf_probability)

        ################################################################
        # Scenario/episode initialization: generation
        ################################################################
        # Now, we'll take a similar procedure to set up generation
        # levels. Unfortunately, this is a tad more complex since we
        # have upper and lower limits.
        #
        self.gen_mw = self._compute_generation()

        ################################################################
        # Action space definition
        ################################################################
        # Create action space by discretizing generator set points.
        self.action_space = spaces.Discrete(self.num_gens
                                            * num_gen_voltage_bins)

        # Create the generator bins.
        self.gen_bins = np.linspace(gen_voltage_range[0], gen_voltage_range[1],
                                    num_gen_voltage_bins)

        # Columns we'll use for voltage control with SAW's
        # ChangeParametersSingleElement.
        self.gen_voltage_control_fields = self.gen_key_fields + ['GenVoltSet']

        # Now, map each action to a generator set-point.
        self.action_map = dict()

        i = 0
        # Loop over the index.
        for gen_data_idx in self.gen_data.index:
            # Extract the identifying information for this generator.
            gen_key_values = \
                self.gen_data.loc[gen_data_idx, self.gen_key_fields].tolist()

            # Create a list compatible with
            # SAW.ChangeParametersSingleElement for each voltage level.
            for v in self.gen_bins:
                self.action_map[i] = gen_key_values + [v]
                i += 1

        ################################################################
        # Observation space definition
        ################################################################

        # Time for the observation space. This will include:
        #   - bus voltages
        #   - generator voltage set points
        #   - generator power factor
        #   - generator portion of maximum reactive power
        #   - generator status
        # TODO: How best to handle low/high? Could use independent
        #   bounds for each observation type.
        self.num_obs = self.num_buses + 4 * self.num_gens
        self.observation_space = spaces.Box(
            low=0.9, high=1.2, shape=(self.num_obs,), dtype=self.dtype)

        # Create indices for the various components of the observations.
        self.bus_v_indices = (0, self.num_buses)
        self.gen_v_indices = (self.num_buses, self.num_buses + self.num_gens)
        self.gen_pf_indices = (self.num_buses + self.num_gens,
                               self.num_buses + 2 * self.num_gens)
        self.gen_var_indices = (self.num_buses + 2 * self.num_gens,
                                self.num_buses + 3 * self.num_gens)
        self.gen_status_indices = (self.num_buses + 3 * self.num_gens,
                                   self.num_buses + 4 * self.num_gens)

        # TODO: Remove this stuff.
        # Open generators which have 0 real power set.
        zero_mw = self.gen_data['GenMW'] == 0
        subset = self.gen_data.loc[:, self.gen_key_fields
                                      + ['GenMW', 'GenStatus']]
        subset.loc[zero_mw, 'GenStatus'] = 'Open'
        self.saw.change_and_confirm_params_multiple_element(
            ObjectType='gen', command_df=subset)
        self.saw.SolvePowerFlow()
        obs = self._get_observation()

        print('stuff')
        # TODO: regulators
        # TODO: shunts

    # def seed(self, seed=None):
    #     """Borrowed from Gym.
    #     https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
    #     """
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def _zero_negative_gen_mw_limits(self):
        """Helper to zero out generator MW limits which are < 0."""
        gen_less_0 = self.gen_data['GenMWMin'] < 0
        if (self.gen_data['GenMWMin'] < 0).any():
            self.gen_data.loc[gen_less_0, 'GenMWMin'] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'gen', self.gen_data.loc[:, self.gen_key_fields
                                            + ['GenMWMin']])
            self.log.warning(f'{gen_less_0.sum()} generators with '
                             'GenMWMin < 0 have had GenMWMin set to 0.')

    def _zero_i_z_loads(self):
        """Helper to zero out constant current and constant impedance
        portions of loads.
        """
        # Warn if we have constant current or constant impedance load
        # values. Then, zero out the constant current and constant
        # impedance portions.
        if (self.load_data[LOAD_I_Z] != 0.0).any().any():
            self.log.warning('The given PowerWorld case has loads with '
                             'non-zero constant current and constant impedance'
                             ' portions. These will be zeroed out.')
            self.load_data.loc[:, LOAD_I_Z] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'Load', self.load_data.loc[:, self.load_key_fields + LOAD_I_Z])

    def _compute_loading(self, load_on_probability,
                         min_load_pf, lead_pf_probability):
        # Define load array shape for convenience.
        shape = (self.num_scenarios, self.num_loads)

        # Draw an active power loading condition for each case.
        scenario_total_loads_mw = \
            np.zeros(self.num_scenarios, dtype=self.dtype)
        scenario_total_loads_mw[:] = self.rng.uniform(
            self.min_load_mw, self.max_load_mw, self.num_scenarios
        )

        # Draw to determine which loads will be "on" for each scenario.
        scenario_loads_off = \
            self.rng.random(shape, dtype=self.dtype) > load_on_probability

        # Draw initial loading levels. Loads which are "off" will be
        # removed, and then each row will be scaled.
        scenario_individual_loads_mw = self.rng.random(shape, dtype=self.dtype)

        # Zero out loads which are off.
        scenario_individual_loads_mw[scenario_loads_off] = 0.0

        # Scale each row to meet the appropriate scenario total loading.
        # First, get our vector of scaling factors (factor per row).
        scale_factor_vector = scenario_total_loads_mw \
            / scenario_individual_loads_mw.sum(axis=1)

        # Then, multiply each element in each row by its respective
        # scaling factor. The indexing with None creates an additional
        # dimension to our vector to allow for that element-wise
        # scaling.
        scenario_individual_loads_mw = (
                scenario_individual_loads_mw * scale_factor_vector[:, None]
        )

        # Now, come up with reactive power levels for each load based
        # on the minimum power factor.
        # Start by randomly generating power factors for each load for
        # each scenario.
        pf = np.zeros(shape, dtype=self.dtype)
        pf[:] = self.rng.uniform(min_load_pf, 1, shape)

        # Use the power factor and MW to get Mvars.
        # Q = P * tan(arccos(pf))
        scenario_individual_loads_mvar = (
                scenario_individual_loads_mw * np.tan(np.arccos(pf)))

        # Possibly flip the sign of the reactive power for some loads
        # in order to make their power factor leading.
        lead = self.rng.random(shape, dtype=self.dtype) < lead_pf_probability
        scenario_individual_loads_mvar[lead] *= -1

        return scenario_total_loads_mw, scenario_individual_loads_mw, \
            scenario_individual_loads_mvar

    def _compute_generation(self):
        # Initialize the generator power levels to 0.
        scenario_gen_mw = np.zeros((self.num_scenarios, self.num_gens))

        # Initialize indices that we'll be shuffling.
        gen_indices = np.arange(0, self.num_gens)

        # Loop over each scenario.
        # TODO: this should be vectorized.
        # TODO: should we instead draw which generators are on like
        #   what's done with the load? It'll have similar issues.
        for scenario_idx in range(self.num_scenarios):
            # Draw random indices for generators. In this way, we'll
            # start with a different generator each time.
            self.rng.shuffle(gen_indices)

            # Get our total load for this scenario.
            load = self.total_load_mw[scenario_idx]

            # Randomly draw generation until we meet the load.
            # The while loop is here in case we "under draw" generation
            # such that generation < load.
            i = 0
            while (abs(scenario_gen_mw[scenario_idx, :].sum() - load)
                   > GEN_LOAD_DELTA_TOL) and (i < ITERATION_MAX):

                # Ensure generation is zeroed out from the last
                # last iteration of the loop.
                scenario_gen_mw[scenario_idx, :] = 0.0

                # Initialize the generation total to 0.
                gen_total = 0

                # For each generator, draw a power level between its
                # minimum and maximum.
                for gen_idx in gen_indices:
                    # Draw generation between this generator's minimum
                    # and the minimum of the generator's maximum and
                    # load.
                    gen_mw = self.rng.uniform(
                        self.gen_data.iloc[gen_idx]['GenMWMin'],
                        min(self.gen_data.iloc[gen_idx]['GenMWMax'], load))

                    # Place the generation in the appropriate spot.
                    if (gen_mw + gen_total) > load:
                        # Generation cannot exceed load. Set this
                        # generator power output to the remaining load
                        # and break out of the loop. This will keep the
                        # remaining generators at 0.
                        scenario_gen_mw[scenario_idx, gen_idx] = \
                            load - gen_total
                        break
                    else:
                        # Use the randomly drawn gen_mw.
                        scenario_gen_mw[scenario_idx, gen_idx] = gen_mw

                    # Increment the generation total.
                    gen_total += gen_mw

                i += 1

            if i >= ITERATION_MAX:
                # TODO: better exception.
                raise UserWarning(f'Iterations exceeded {ITERATION_MAX}')

            self.log.debug(f'It took {i} iterations to create generation for '
                           f'scenario {scenario_idx}')

        return scenario_gen_mw

    def render(self, mode='human'):
        """Not planning to implement this for now. However, it could
        make for some nice graphics/presentations related to learning.

        Also, PowerWorld does have the option to show the one line, so
        this could be interesting down the line.
        """
        raise NotImplementedError

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        """Tear down SimAuto wrapper."""
        self.saw.exit()

    def _get_observation(self) -> np.ndarray:
        """Helper to return an observation. For the given simulation,
        the power flow should already have been solved.
        """
        # Start by getting bus voltages.
        bus_voltage = self.saw.GetParametersMultipleElement(
            ObjectType='bus', ParamList=self.bus_obs_fields)

        # Now, get all relevant generator data.
        gen_data = self.saw.GetParametersMultipleElement(
            ObjectType='gen', ParamList=self.gen_obs_fields)

        # Initialize return.
        out = np.ones(self.num_obs, dtype=self.dtype)

        # Assign the load voltage data.
        out[self.bus_v_indices[0]:self.bus_v_indices[1]] = \
            bus_voltage['BusPUVolt'].to_numpy(dtype=self.dtype)

        # Set generator voltage set points.
        out[self.gen_v_indices[0]:self.gen_v_indices[1]] = \
            gen_data['GenVoltSet'].to_numpy(dtype=self.dtype)

        # Compute and assign power factor for the generators. Set the
        # power factor equal to 1 where generators are off.
        # pf = P / |S|
        out[self.gen_pf_indices[0]:self.gen_pf_indices[1]] = \
            (gen_data['GenMW'] / gen_data['GenMVA']).fillna(1).to_numpy(
            dtype=self.dtype)

        # Assign percentage MVR loading for generators.
        out[self.gen_var_indices[0]:self.gen_var_indices[1]] = \
            gen_data['GenMVRPercent'].to_numpy(dtype=self.dtype) / 100

        # Assign status.
        out[self.gen_status_indices[0]:self.gen_status_indices[1]] = \
            gen_data['GenStatus'].map(gen_numeric_status_map).to_numpy(
                dtype=self.dtype)

        # All done.
        return out


def gen_numeric_status_map(x: str):
    """Map for generator states: closed --> 1, open --> 0

    :param x: Generator status, either 'Open' or 'Closed' (case
        insensitive).
    :returns: 1 if open, 0 if closed.
    """
    if x.lower() == 'closed':
        return 1
    else:
        return 0
