from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging
from esa import SAW, PowerWorldError
from typing import Union, Tuple, List
from copy import deepcopy

# When generating scenarios, we're drawing random generation to meet
# the load. There will be some rounding error, so set a reasonable
# tolerance. Note this is in MW. In the power flow, the slack bus will
# pick up this rounding error - that's okay.
GEN_LOAD_DELTA_TOL = 0.001

# For safety we'll have a maximum number of loop iterations.
ITERATION_MAX = 100

# Constants related to PowerWorld (for convenience and cleanliness):
# Constant power portion of PowerWorld loads.
LOAD_P = ['LoadSMW', 'LoadSMVR']

# Constant current and constant impedance portions of PowerWorld
# loads.
LOAD_I_Z = ['LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']

# Assumed transmission system losses as a fraction of energy delivered.
LOSS = 0.03

# Minimum allowed bus voltage (per unit) for a case. After solving the
# power flow, voltages below this threshold will signify the case is
# "bad." This will avoid weird behavior like PowerWorld converting loads
# to constant current/impedance, etc.
MIN_V = 0.75

# Specify bus voltage bounds.
# TODO: May want to consider 0.9 to 1.1.
LOW_V = 0.95
HIGH_V = 1.05
NOMINAL_V = 1.0

# Instead of writing code to manage English rules, just hard code plural
# mappings.
PLURAL_MAP = {
    'gen': 'gens',
    'load': 'loads',
    'bus': 'buses',
    'branch': 'branches',
    'shunt': 'shunts'
}


# noinspection PyPep8Naming
class DiscreteVoltageControlEnvBase(ABC, gym.Env):
    """Base class for discrete voltage control environments.

    Subclasses must set the following attributes in __init__:
    - action_space
    - num_obs
    - observation_space

    Subclasses must implement the following methods:
    - _compute_loading
    - _compute_generation
    - _take_action

    Note that the initialization method of this class solves the power
    flow and calls the SaveState method, so subclasses may need to
    repeat that if they make case modifications that they want reloaded
    for each call to "reset."
    """

    ####################################################################
    # Abstract class properties
    ####################################################################

    @property
    @abstractmethod
    def GEN_INIT_FIELDS(self) -> List[str]:
        """All PowerWorld generator fields to be used during environment
        initialization, sans key fields. Set to an empty list in
        subclasses which do no use generator data for environment
        initialization.
        """

    @property
    @abstractmethod
    def GEN_OBS_FIELDS(self) -> List[str]:
        """All PowerWorld generator fields to be used when generating
        an observation during a time step, sans key fields. Set to an
        empty list in subclasses which do no use generator data for
        observations.
        """

    @property
    @abstractmethod
    def GEN_RESET_FIELDS(self) -> List[str]:
        """All PowerWorld generator fields to be used in the "reset"
        method, sans key fields. Set to an empty list in subclasses
        which do not change generator parameters during calls to
        "reset."
        """

    @property
    @abstractmethod
    def LOAD_INIT_FIELDS(self) -> List[str]:
        """All PowerWorld load fields to be used during environment
        initialization, sans key fields. Set to an empty list in
        subclasses which do no use load data for environment
        initialization.
        """

    @property
    @abstractmethod
    def LOAD_OBS_FIELDS(self) -> List[str]:
        """All PowerWorld load fields to be used when generating
        an observation during a time step, sans key fields. Set to an
        empty list in subclasses which do no use load data for
        observations.
        """

    @property
    @abstractmethod
    def LOAD_RESET_FIELDS(self) -> List[str]:
        """All PowerWorld load fields to be used in the "reset"
        method, sans key fields. Set to an empty list in subclasses
        which do not change load parameters during calls to
        "reset."
        """

    @property
    @abstractmethod
    def BUS_INIT_FIELDS(self) -> List[str]:
        """All PowerWorld bus fields to be used during environment
        initialization, sans key fields. Set to an empty list in
        subclasses which do no use bus data for environment
        initialization.
        """

    @property
    @abstractmethod
    def BUS_OBS_FIELDS(self) -> List[str]:
        """All PowerWorld bus fields to be used when generating
        an observation during a time step, sans key fields. Set to an
        empty list in subclasses which do no use bus data for
        observations.
        """

    @property
    @abstractmethod
    def BUS_RESET_FIELDS(self) -> List[str]:
        """All PowerWorld load fields to be used in the "reset"
        method, sans key fields. Set to an empty list in subclasses
        which do not change load parameters during calls to
        "reset."
        """

    @property
    @abstractmethod
    def BRANCH_INIT_FIELDS(self) -> List[str]:
        """All PowerWorld branch fields to be used during environment
        initialization, sans key fields. Set to an empty list in
        subclasses which do no use branch data for environment
        initialization.
        """

    @property
    @abstractmethod
    def BRANCH_OBS_FIELDS(self) -> List[str]:
        """All PowerWorld branch fields to be used when generating
        an observation during a time step, sans key fields. Set to an
        empty list in subclasses which do no use branch data for
        observations.
        """

    @property
    @abstractmethod
    def BRANCH_RESET_FIELDS(self) -> List[str]:
        """All PowerWorld branch fields to be used in the "reset"
        method, sans key fields. Set to an empty list in subclasses
        which do not change branch parameters during calls to
        "reset."
        """

    @property
    @abstractmethod
    def SHUNT_INIT_FIELDS(self) -> List[str]:
        """All PowerWorld shunt fields to be used during environment
        initialization, sans key fields. Set to an empty list in
        subclasses which do no use shunt data for environment
        initialization.
        """

    @property
    @abstractmethod
    def SHUNT_OBS_FIELDS(self) -> List[str]:
        """All PowerWorld shunt fields to be used when generating
        an observation during a time step, sans key fields. Set to an
        empty list in subclasses which do no use shunt data for
        observations.
        """

    @property
    @abstractmethod
    def SHUNT_RESET_FIELDS(self) -> List[str]:
        """All PowerWorld shunt fields to be used in the "reset"
        method, sans key fields. Set to an empty list in subclasses
        which do not change shunt parameters during calls to
        "reset."
        """

    @property
    @abstractmethod
    def REWARDS(self) -> dict:
        """Subclasses should implement a dictionary with default
        rewards. Rewards should be given a positive sign, and penalties
        should be given a negative sign.
        """

    ####################################################################
    # Initialization
    ####################################################################

    def __init__(self, pwb_path: str, num_scenarios: int,
                 max_load_factor: float = None,
                 min_load_factor: float = None,
                 min_load_pf: float = 0.8,
                 lead_pf_probability: float = 0.1,
                 load_on_probability: float = 0.8,
                 num_gen_voltage_bins: int = 5,
                 gen_voltage_range: Tuple[float, float] = (0.9, 1.1),
                 seed: float = None,
                 log_level=logging.INFO,
                 rewards: Union[dict, None] = None,
                 dtype=np.float32):
        """Initialize the environment. Pull data needed up front,
        create gen/loading cases, perform case checks, etc.

        :param pwb_path: Full path to a PowerWorld .pwb file
            representing the grid which the agent will control.
        :param num_scenarios: Number of different case initialization
            scenarios to create. Note it is not guaranteed that the
            power flow will solve successfully for all cases, so the
            number of actual usable cases will be less than the
            num_scenarios.
        :param max_load_factor: Factor which when multiplied by the
            current total system load represents the maximum allowable
            system loading for training episodes in MW. E.g., if the
            current sum of all the active power components of the loads
            is 100 MW and the max_load_factor is 2, then the maximum
            possible active power loading for any given episode will be
            200 MW. If the max_load_factor is None, the maximum
            system loading will be computed by the sum of generator
            maximum MW outputs.
        :param min_load_factor: Similar to the max_load_factor, this
            number is multiplied with the current total system load to
            determine the minimum active power loading for an episode.
            If None, the minimum system loading will be computed by the
            sum of generator minimum MW outputs.
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
            if gen_voltage_range=(0.95, 1.05) and num_gen_voltage_bins
            is 5, the generator set points will be discretized into the
            set {0.95, 0.975, 1.0, 1.025, 1.05}.
        :param gen_voltage_range: Minimum and maximum allowed generator
            voltage regulation set points (in per unit).
        :param seed: Seed for random number.
        :param log_level: Log level for the environment's logger. Pass
            a constant from the logging module, e.g. logging.DEBUG or
            logging.INFO.
        :param rewards: Dictionary of rewards/penalties. For available
            fields and their descriptions, see the class constant
            REWARDS. This dictionary can be partial, i.e. include only
            a subset of fields contained in REWARDS. To use the default
            rewards, pass in None.
        :param dtype: Numpy datatype to be used for most numbers. It's
            common to use np.float32 for machine learning tasks to
            reduce memory consumption.
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

        # Number of scenarios/episodes to generate. Note that not all
        # will be usable.
        self.num_scenarios = num_scenarios

        # Scenario index starts at one, and is incremented for each
        # episode.
        self.scenario_idx = 0

        ################################################################
        # Initialize PowerWorld related attributes, get data from case
        ################################################################
        # Following the principle of least astonishment, initialize all
        # attributes that will hold data. The attributes will be
        # overridden in more compact/generic helper methods.
        #
        # Start with key fields.
        self.gen_key_fields: Union[list, None] = None
        self.load_key_fields: Union[list, None] = None
        self.bus_key_fields: Union[list, None] = None
        self.branch_key_fields: Union[list, None] = None
        self.shunt_key_fields: Union[list, None] = None

        # Fields used for environment initialization. These will include
        # key fields.
        self.gen_init_fields: Union[list, None] = None
        self.load_init_fields: Union[list, None] = None
        self.bus_init_fields: Union[list, None] = None
        self.branch_init_fields: Union[list, None] = None
        self.shunt_init_fields: Union[list, None] = None

        # Fields used for observations. These will include key fields.
        self.gen_obs_fields: Union[list, None] = None
        self.load_obs_fields: Union[list, None] = None
        self.bus_obs_fields: Union[list, None] = None
        self.branch_obs_fields: Union[list, None] = None
        self.shunt_obs_fields: Union[list, None] = None

        # Fields used during "reset." These will include key fields.
        self.gen_reset_fields: Union[list, None] = None
        self.load_reset_fields: Union[list, None] = None
        self.bus_reset_fields: Union[list, None] = None
        self.branch_reset_fields: Union[list, None] = None
        self.shunt_reset_fields: Union[list, None] = None

        # Now data which will be used during initialization.
        self.gen_init_data: Union[pd.DataFrame, None] = None
        self.load_init_data: Union[pd.DataFrame, None] = None
        self.bus_init_data: Union[pd.DataFrame, None] = None
        self.branch_init_data: Union[pd.DataFrame, None] = None
        self.shunt_init_data: Union[pd.DataFrame, None] = None

        # Data which will be used in observations.
        self.gen_obs_data: Union[pd.DataFrame, None] = None
        self.load_obs_data: Union[pd.DataFrame, None] = None
        self.bus_obs_data: Union[pd.DataFrame, None] = None
        self.branch_obs_data: Union[pd.DataFrame, None] = None
        self.shunt_obs_data: Union[pd.DataFrame, None] = None

        # Some environments may want to keep data on the previous
        # observations.
        self.gen_obs_data_prev: Union[pd.DataFrame, None] = None
        self.load_obs_data_prev: Union[pd.DataFrame, None] = None
        self.bus_obs_data_prev: Union[pd.DataFrame, None] = None
        self.branch_obs_data_prev: Union[pd.DataFrame, None] = None
        self.shunt_obs_data_prev: Union[pd.DataFrame, None] = None

        # Data which will be used/set in the "reset" method.
        self.gen_reset_data: Union[pd.DataFrame, None] = None
        self.load_reset_data: Union[pd.DataFrame, None] = None
        self.bus_reset_data: Union[pd.DataFrame, None] = None
        self.branch_reset_data: Union[pd.DataFrame, None] = None
        self.shunt_reset_data: Union[pd.DataFrame, None] = None

        # To avoid constant calls to .shape[0], we'll track the number
        # of elements of each type in the case. None will indicate
        # we didn't even look.
        self.num_gens: Union[int, None] = None
        self.num_loads: Union[int, None] = None
        self.num_buses: Union[int, None] = None
        self.num_branches: Union[int, None] = None
        self.num_shunts: Union[int, None] = None

        # Call helper to fill in most of the attributes we just listed
        # above. All "fields" attributes will be filled, unless the
        # corresponding class constant is the empty list. All
        # "init_data" attributes will be filled by querying PowerWorld.
        self._fill_init_attributes()

        ################################################################
        # Generator fields and data
        ################################################################
        # Zero out negative minimum generation limits. A warning will
        # be emitted if generators have negative limits.
        self._zero_negative_gen_mw_limits()

        # For convenience, compute the maximum generation capacity.
        # Depending on max_load_factor, gen_mw_capacity could also
        # represent maximum loading.
        # TODO: Confirm that the produce/consume convention is correct.
        self.gen_mw_capacity = self.gen_init_data['GenMWMax'].sum()
        self.gen_mvar_produce_capacity = self.gen_init_data['GenMVRMax'].sum()
        # TODO: Should we use abs here?
        self.gen_mvar_consume_capacity = self.gen_init_data['GenMVRMin'].sum()

        ################################################################
        # Load fields and data
        ################################################################
        # Zero out constant current and constant impedance portions so
        # we simply have constant power. A warning will be emitted if
        # there are loads with non-zero constant current or impedance
        # portions.
        # TODO: in the future, we should allow for loads beyond constant
        #  power.
        self._zero_i_z_loads()

        ################################################################
        # Minimum and maximum system loading
        ################################################################
        # Compute maximum system loading.
        if max_load_factor is not None:
            # If given a max load factor, multiply it by the current
            # system load.
            self.max_load_mw = \
                self.load_init_data['LoadSMW'].sum() * max_load_factor

            # Ensure the maximum loading is <= generation capacity.
            self._check_max_load(max_load_factor)
        else:
            # If not given a max load factor, the maximum loading will
            # simply be generation capacity.
            self.max_load_mw = self.gen_mw_capacity

        # Compute minimum system loading.
        if min_load_factor is not None:
            self.min_load_mw = \
                self.load_init_data['LoadSMW'].sum() * min_load_factor

            # Ensure the minimum loading is feasible based on the
            # generation.
            self._check_min_load(min_load_factor)
        else:
            # If not given a min load factor, minimum loading is simply
            # the minimum generation of minimum generation.
            self.min_load_mw = self.gen_init_data['GenMWMin'].min()

        ################################################################
        # Scenario/episode initialization: loads
        ################################################################
        # Time to generate scenarios.
        # Initialize list to hold all information pertaining to all
        # scenarios.
        self.scenarios = []

        # Compute total loading and loading for each individual load
        # for all scenarios.
        self.total_load_mw, self.loads_mw, self.loads_mvar = \
            self._compute_loading(load_on_probability=load_on_probability,
                                  min_load_pf=min_load_pf,
                                  lead_pf_probability=lead_pf_probability)

        ################################################################
        # Scenario/episode initialization: generation
        ################################################################
        # Compute each individual generator's active power contribution
        # for each loading scenario.
        self.gen_mw = self._compute_generation()

        ################################################################
        # Action space
        ################################################################
        # Start by creating the generator bins.
        self.gen_bins = np.linspace(gen_voltage_range[0], gen_voltage_range[1],
                                    num_gen_voltage_bins)

        # Subclasses should set the action_space based on the gen_bins
        # and/or other factors.

        ################################################################
        # Observation space definition
        ################################################################
        # Subclasses should set num_obs and observation_space
        # attributes.

        # Initialize attribute for checking if all voltages are in
        # range. This will be used to check if an episode is done, and
        # will be reset in the "reset" method.
        self.all_v_in_range = False

        # We'll track how many actions the agent has taken in an episode
        # as part of the stopping criteria.
        self.action_count = 0

        ################################################################
        # Set rewards.
        ################################################################
        self.rewards = deepcopy(self.REWARDS)
        self._overwrite_rewards(rewards)

        ################################################################
        # Solve power flow, save state.
        ################################################################
        # Solve the initial power flow, and save the state for later
        # loading. This is done so that PowerWorld doesn't get stuck
        # in a low voltage solution when moving from a bad case to a
        # feasible case.
        #
        # Ensure this is the absolute LAST thing done
        # in __init__ so that changes we've made to the case don't get
        # overridden.
        self.saw.SolvePowerFlow()
        self.saw.SaveState()

        # All done.

    ####################################################################
    # Public methods
    ####################################################################

    def render(self, mode='human'):
        """Not planning to implement this for now. However, it could
        make for some nice graphics/presentations related to learning.

        Also, PowerWorld does have the option to show the one line, so
        this could be interesting down the line.
        """
        raise NotImplementedError

    def reset(self):
        """Advance to the next episode/scenario. To do so, generators
        will be turned on/off, generator MW set points will be set,
        and load P/Q will be set. Finally, the power flow will be
        solved and an initial observation will be returned.
        """
        # Reset the action counter.
        self.action_count = 0

        # Reset all_v_in_range.
        self.all_v_in_range = False

        done = False

        while (not done) & (self.scenario_idx < self.num_scenarios):
            # Load the initial state of the system to avoid getting
            # stuck in a low voltage solution from a previous solve.
            self.saw.LoadState()

            # Get generators and loads set up for this scenario.
            self._set_gens_for_scenario()
            self._set_loads_for_scenario()

            # Solve the power flow.
            try:
                obs = self._solve_and_observe()
            except (PowerWorldError, LowVoltageError) as exc:
                # This scenario is bad. Move on.
                self.log.debug(
                    f'Scenario {self.scenario_idx} failed. Error message: '
                    f'{exc.args[0]}')
            else:
                # Success! The power flow solved, and no voltages went
                # below the minimum. Signify we're done looping.
                done = True
            finally:
                # Always increment the scenario index.
                self.scenario_idx += 1

        # Raise exception if we've gone through all the scenarios.
        # TODO: better exception.
        if self.scenario_idx >= self.num_scenarios:
            raise UserWarning('We have gone through all scenarios.')

        # Return the observation.
        # noinspection PyUnboundLocalVariable
        return obs

    def step(self, action):
        """Change generator set point, solve power flow, compute reward.
        """
        # Bump the action counter.
        self.action_count += 1

        # Take the action.
        self._take_action(action)

        # Solve the power flow and get an observation.
        try:
            obs = self._solve_and_observe()
        except (PowerWorldError, LowVoltageError):
            # The power flow failed to solve or bus voltages went below
            # the minimum. This episode is complete.
            # TODO: Should our observation be None?
            obs = None
            done = True
            # An action was taken, so include both the action and
            # failure penalties.
            reward = self.rewards['fail'] + self.rewards['action']
        else:
            # The power flow successfully solved. Compute the reward
            # and check to see if this episode is done.
            reward = self._compute_reward()
            done = self._check_done()

        # TODO: update the fourth return (info) to, you know, actually
        #   give info.
        # That's it.
        return obs, reward, done, dict()

    def close(self):
        """Tear down SimAuto wrapper."""
        self.saw.exit()

    ####################################################################
    # Private methods
    ####################################################################

    def _fill_init_attributes(self):
        """Helper to loop and set attributes for gens, loads, buses,
        branches, and shunts. This method will set the following:

        <obj>_key_fields
        <obj>_init_fields
        <obj>_obs_fields
        <obj>_reset_fields
        <obj>_init_data
        num_<obj_plural>
        """
        for obj in ('gen', 'load', 'bus', 'branch', 'shunt'):
            # Get the key fields.
            kf = self.saw.get_key_field_list(obj)

            # Set attribute.
            setattr(self, obj + '_key_fields', kf)

            # Add key fields to init, obs, and reset fields.
            for attr in ('init', 'obs', 'reset'):
                # Get class constant (cc) list.
                cc = getattr(self, f'{obj.upper()}_{attr.upper()}_FIELDS')

                # Only set the attribute if the class constant has a
                # length greater than 0. Otherwise, leave it as the
                # default, None. We always want it set for init,
                # however. This is because we'll pull data from
                # PowerWorld to get the number of elements, for
                # instance.
                if (attr == 'init') or (len(cc) > 0):
                    # Set instance attribute by combining key fields and
                    # class constant.
                    setattr(self, f'{obj}_{attr}_fields', kf + cc)

            # Pull and set initialization data, as well as computing
            # the number of elements present.
            data = self.saw.GetParametersMultipleElement(
                ObjectType=obj,
                ParamList=getattr(self, f'{obj}_init_fields'))

            plural = PLURAL_MAP[obj]

            # ESA will return None if the objects are not present.
            if data is not None:
                setattr(self, f'{obj}_init_data', data)
                setattr(self, f'num_{plural}', data.shape[0])
            else:
                setattr(self, f'num_{plural}', 0)

    def _zero_negative_gen_mw_limits(self):
        """Helper to zero out generator MW limits which are < 0."""
        gen_less_0 = self.gen_init_data['GenMWMin'] < 0
        if (self.gen_init_data['GenMWMin'] < 0).any():
            self.gen_init_data.loc[gen_less_0, 'GenMWMin'] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'gen', self.gen_init_data.loc[:, self.gen_key_fields
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
        if (self.load_init_data[LOAD_I_Z] != 0.0).any().any():
            self.log.warning('The given PowerWorld case has loads with '
                             'non-zero constant current and constant impedance'
                             ' portions. These will be zeroed out.')
            self.load_init_data.loc[:, LOAD_I_Z] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'Load', self.load_init_data.loc[:, self.load_key_fields
                                                + LOAD_I_Z])

    def _check_max_load(self, max_load_factor):
        """Ensure maximum loading is less than generation capacity. Also
        warn if generation capacity is >= 2 * maximum loading.
        """
        if self.max_load_mw > self.gen_mw_capacity:
            # TODO: Better exception.
            raise UserWarning(
                f'The given max_load_factor, {max_load_factor:.3f} '
                f'resulted in maximum loading of {self.max_load_mw:.3f} MW'
                ', but the generator active power capacity is only '
                f'{self.gen_mw_capacity:.3f} MW. Reduce the '
                'max_load_factor and try again.')

        # Warn if our generation capacity is more than double the max
        # load - this could mean generator maxes aren't realistic.
        gen_factor = self.gen_mw_capacity / self.max_load_mw
        if gen_factor >= 1.5:
            self.log.warning(
                f'The given generator capacity, {self.gen_mw_capacity:.2f} MW,'
                f' is {gen_factor:.2f} times larger than the maximum load, '
                f'{self.max_load_mw:.2f} MW. This could indicate that '
                'the case does not have proper generator limits set up.')

    def _check_min_load(self, min_load_factor):
        """Ensure minimum loading is greater than the minimum generator
        minimum generation.
        """
        min_gen = self.gen_init_data['GenMWMin'].min()
        if self.min_load_mw < min_gen:
            # TODO: better exception.
            raise UserWarning(
                f'The given min_load_factor, {min_load_factor:.3f}, '
                'results in a minimum system loading of '
                f'{self.min_load_mw:3f} MW, but the lowest generation '
                f'possible is {min_gen:.3f} MW. Increase the '
                'min_load_factor and try again.')

    def _overwrite_rewards(self, rewards):
        """Simple helper to overwrite default rewards with user provided
         rewards.
         """
        if rewards is None:
            # If not given rewards, do nothing.
            return
        else:
            # If given rewards, loop over the dictionary and set fields.
            for key, value in rewards.items():
                # Raise exception if key is invalid.
                if key not in self.REWARDS:
                    raise KeyError(
                        f'The given rewards key, {key}, is invalid. Please '
                        'only use keys in the class constant, REWARDS.')

                # If we're here, the key is valid. Set it.
                self.rewards[key] = value

    def _solve_and_observe(self):
        """Helper to solve the power flow and get an observation.

        :raises LowVoltageError: If any bus voltage is below MIN_V after
            solving the power flow.
        :raises PowerWorldError: If PowerWorld fails to solve the power
            flow.
        """
        # Start by solving the power flow. This will raise a
        # PowerWorldError if it fails to solve.
        self.saw.SolvePowerFlow()

        # Get new observations, rotate old ones.
        self._rotate_and_get_observation_frames()

        # Check if all voltages are in range.
        self.all_v_in_range = (
                ((self.bus_obs_data['BusPUVolt'] < LOW_V)
                 & (self.bus_obs_data['BusPUVolt'] > HIGH_V)).sum()
                == 0
        )

        # If any voltages are too low, raise exception.
        if (self.bus_obs_data['BusPUVolt'] < MIN_V).any():
            num_low = (self.bus_obs_data['BusPUVolt'] < MIN_V).sum()
            raise LowVoltageError(
                f'{num_low} buses were below {MIN_V:.2f} p.u.'
            )

        # Get and return a properly arranged observation.
        return self._get_observation()

    def _rotate_and_get_observation_frames(self):
        """Simple helper to get new observation DataFrames, and rotate
        the previous frames into the correct attributes.
        """
        # Rotate.
        self.bus_obs_data_prev = self.bus_obs_data
        self.gen_obs_data_prev = self.gen_obs_data
        self.load_obs_data_prev = self.load_obs_data

        # Get new data.
        # Buses:
        if self.bus_obs_fields is not None:
            self.bus_obs_data = self.saw.GetParametersMultipleElement(
                ObjectType='bus', ParamList=self.bus_obs_fields)
        # Generators:
        if self.gen_obs_fields is not None:
            self.gen_obs_data = self.saw.GetParametersMultipleElement(
                ObjectType='gen', ParamList=self.gen_obs_fields)
        # Loads:
        if self.load_obs_fields is not None:
            self.load_obs_data = self.saw.GetParametersMultipleElement(
                ObjectType='load', ParamList=self.load_obs_fields
            )

        # That's it.
        return None

    def _set_gens_for_scenario(self):
        """Helper to set up generators in the case for this
        episode/scenario. This method should only be called by reset.
        """
        # Extract a subset of the generator data.
        gens = self.gen_init_data.loc[:, self.gen_key_fields
                                      + self.GEN_RESET_FIELDS]

        # Turn generators on/off and set their MW set points.
        gens.loc[:, 'GenMW'] = self.gen_mw[self.scenario_idx, :]
        gen_g_0 = self.gen_mw[self.scenario_idx, :] > 0
        gens.loc[gen_g_0, 'GenStatus'] = 'Closed'
        gens.loc[~gen_g_0, 'GenStatus'] = 'Open'
        # TODO: may want to use a faster command like
        #   ChangeParametersMultipleElement. NOTE: Will need to
        #   update patching in tests if this route is taken.
        self.saw.change_and_confirm_params_multiple_element('gen', gens)

    def _set_loads_for_scenario(self):
        """Helper to set up loads in the case for this episode/scenario.
        This method should only be called by reset.
        """
        # Extract a subset of the load data.
        loads = self.load_init_data.loc[:, self.load_key_fields
                                        + self.LOAD_RESET_FIELDS]

        # Set P and Q.
        loads.loc[:, 'LoadSMW'] = self.loads_mw[self.scenario_idx, :]
        loads.loc[:, 'LoadSMVR'] = self.loads_mvar[self.scenario_idx, :]
        # TODO: may want to use a faster command like
        #   ChangeParametersMultipleElement. NOTE: Will need to
        #   update patching in tests if this route is taken.
        self.saw.change_and_confirm_params_multiple_element('load', loads)

    def _check_done(self):
        """Check whether (True) or not (False) and episode is done. Call
        this after calling _solve_and_observe.
        """
        # If the number of actions taken in this episode has exceeded
        # a threshold, we're done.
        # TODO: Stop hard-coding number of actions
        if self.action_count > (self.num_gens * 2):
            return True

        # If all voltages are in range, we're done.
        if self.all_v_in_range:
            return True

        # Otherwise, we're not done.
        return False

    ####################################################################
    # Abstract methods
    ####################################################################

    @abstractmethod
    def _compute_loading(self, load_on_probability, min_load_pf,
                         lead_pf_probability):
        """Subclasses should implement a _compute_loading method which
        computes loading for each scenario. The method should return
        three arrays: one of dimension (num_scenarios,) representing
        total load for each scenario and two arrays of dimension
        (num_scenarios, num_loads) representing individual load MW and
        Mvar, respectively.

        See initialization inputs for descriptions of parameters.
        """
        pass

    @abstractmethod
    def _compute_generation(self):
        """Subclasses should implement a _compute_generation method
        which allocates generator output for each loading scenario. The
        return array should be of dimension (num_scenarios, num_gens).
        """
        pass

    @abstractmethod
    def _take_action(self, action):
        """Subclasses should implement a _take_action method which takes
        a given action, looks up what it does, and takes the action in
        the PowerWorld simulator. The _take_action method does not need
        to solve the power flow.
        """
        pass

    @abstractmethod
    def _get_observation(self):
        """Subclasses should implement a _get_observation method which
        returns an observation.
        """

    @abstractmethod
    def _compute_reward(self):
        """Subclasses should implement a _compute_reward method which
        computes a reward for a given action.
        """


class DiscreteVoltageControlEnv(DiscreteVoltageControlEnvBase):
    """Environment for performing voltage control with the PowerWorld
    Simulator.
    """
    # Gen fields. See base class for comments.
    GEN_INIT_FIELDS = ['BusCat', 'GenMW', 'GenMVR', 'GenVoltSet', 'GenMWMax',
                       'GenMWMin', 'GenMVRMax', 'GenMVRMin', 'GenStatus']
    GEN_OBS_FIELDS = ['GenMW', 'GenMWMax', 'GenMVA', 'GenMVRPercent',
                      'GenStatus']
    # TODO: May need to add voltage set point here.
    GEN_RESET_FIELDS = ['GenMW', 'GenStatus']

    # Load fields.
    LOAD_INIT_FIELDS = LOAD_P + LOAD_I_Z
    LOAD_OBS_FIELDS = LOAD_P + ['PowerFactor', ]
    LOAD_RESET_FIELDS = LOAD_P

    # Bus fields.
    BUS_INIT_FIELDS = []
    BUS_OBS_FIELDS = ['BusPUVolt', ]
    BUS_RESET_FIELDS = []

    # Branch fields.
    BRANCH_INIT_FIELDS = []
    BRANCH_OBS_FIELDS = []
    BRANCH_RESET_FIELDS = []

    # Shunt fields. TODO
    SHUNT_INIT_FIELDS = []
    SHUNT_OBS_FIELDS = []
    SHUNT_RESET_FIELDS = []

    # Specify default rewards.
    # NOTE: When computing rewards, all reward components will be
    #   treated with a positive sign. Thus, specify penalties in this
    #   dictionary with a negative sign.
    REWARDS = {
        # Negative reward (penalty) given for taking an action.
        # TODO: May want different penalties for different types of
        #   actions, e.g. changing gen set point vs. switching cap.
        "action": -10,
        # Reward per 0.01 pu voltage movement in the right direction
        # (i.e. a voltage below the lower bound moving upward).
        "v_delta": 1,
        # Bonus reward for moving a voltage that was not in-bounds
        # in-bounds. This is set to be equal to the action penalty so
        # that moving a single bus in-bounds makes an action worth it.
        "v_in_bounds": 10,
        # Penalty for moving a voltage that was in-bounds out-of-bounds.
        "v_out_bounds": -10,
        # Reward per 1% increase in generator var reserves (or penalty
        # per 1% decrease).
        # TODO: Really, this should be on a per Mvar basis. By basing
        #   this on the generator's maximum, we lose information about
        #   generator sizing. This percentage change scheme will equally
        #   reward large and small generators, but those large and small
        #   generators have different sized roles in voltage support.
        "gen_var_delta": 1,
        # Penalty for taking an action which causes the power flow to
        # fail to converge (or voltages get below the MIN_V threshold).
        "fail": -1000
    }

    def __init__(self, pwb_path: str, num_scenarios: int,
                 max_load_factor: float = None,
                 min_load_factor: float = None,
                 min_load_pf: float = 0.8,
                 lead_pf_probability: float = 0.1,
                 load_on_probability: float = 0.8,
                 num_gen_voltage_bins: int = 5,
                 gen_voltage_range: Tuple[float, float] = (0.9, 1.1),
                 seed: float = None,
                 log_level=logging.INFO,
                 rewards: Union[dict, None] = None,
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
            system loading for training episodes in MW. E.g., if the
            current sum of all the active power components of the loads
            is 100 MW and the max_load_factor is 2, then the maximum
            possible active power loading for any given episode will be
            200 MW. If the max_load_factor is None, the maximum
            system loading will be computed by the sum of generator
            maximum MW outputs.
        :param min_load_factor: Similar to the max_load_factor, this
            number is multiplied with the current total system load to
            determine the minimum active power loading for an episode.
            If None, the minimum system loading will be computed by the
            sum of generator minimum MW outputs.
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
            if gen_voltage_range=(0.95, 1.05) and num_gen_voltage_bins
            is 5, the generator set points will be discretized into the
            set {0.95, 0.975, 1.0, 1.025, 1.05}.
        :param gen_voltage_range: Minimum and maximum allowed generator
            voltage regulation set points (in per unit).
        :param seed: Seed for random number.
        :param log_level: Log level for the environment's logger. Pass
            a constant from the logging module, e.g. logging.DEBUG or
            logging.INFO.
        :param rewards: Dictionary of rewards/penalties. For available
            fields and their descriptions, see the class constant
            REWARDS. This dictionary can be partial, i.e. include only
            a subset of fields contained in REWARDS. To use the default
            rewards, pass in None.
        :param dtype: Numpy datatype to be used for most numbers. It's
            common to use np.float32 for machine learning tasks to
            reduce memory consumption.
        """
        # Start by calling super constructor.
        super().__init__(
            pwb_path=pwb_path, num_scenarios=num_scenarios,
            max_load_factor=max_load_factor, min_load_factor=min_load_factor,
            min_load_pf=min_load_pf,
            lead_pf_probability=lead_pf_probability,
            load_on_probability=load_on_probability,
            num_gen_voltage_bins=num_gen_voltage_bins,
            gen_voltage_range=gen_voltage_range,
            seed=seed, log_level=log_level, rewards=rewards,
            dtype=dtype)

        ################################################################
        # Action space definition
        ################################################################
        # Create action space by discretizing generator set points.
        self.action_space = spaces.Discrete(self.num_gens
                                            * num_gen_voltage_bins)

        # The action array is a simple mapping array which corresponds
        # 1:1 with the action_space. Each row represents an action,
        # and within each row is an index into self.gen_data and into
        # self.gen_bins, respectively. Note: if we need to really trim
        # memory use, the mapping can be computed on the fly rather
        # than stored in an array. For now, let's stick with a simple
        # approach.
        self.action_array = np.zeros(shape=(self.action_space.n, 2),
                                     dtype=int)

        # Repeat the generator indices in the first column of the
        # action_array.
        self.action_array[:, 0] = np.tile(self.gen_init_data.index.to_numpy(),
                                          num_gen_voltage_bins)

        # Create indices into self.gen_bins in the second column.
        # It feels like this could be better vectorized, but this should
        # be close enough.
        for i in range(num_gen_voltage_bins):
            self.action_array[
                i * self.num_gens:(i + 1) * self.num_gens, 1] = i

        ################################################################
        # Observation space definition
        ################################################################

        # Time for the observation space. This will include:
        #   - bus voltages
        #   - generator active power output divided by maximum active
        #       power output
        #   - generator power factor
        #   - generator portion of maximum reactive power
        #   - load level (MW) divided by maximum MW loading
        #   - load power factor
        #   - flag (0/1) for if load power factor is lagging/leading
        #
        # We'll leave out a flag for generator lead/lag, because they
        # will almost always be producing rather than consuming vars.
        # May want to change this in the future.
        # TODO: How best to handle low/high? Could use independent
        #   bounds for each observation type.
        self.num_obs = self.num_buses + 3 * self.num_gens + 3 * self.num_loads
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=self.dtype)

        # All done.

    def render(self, mode='human'):
        """Putting this here strictly to get PyCharm to quit
        complaining about unimplemented abstract methods. I don't want
        to completely suppress the warning, though."""
        raise NotImplementedError

    def _compute_loading(self, load_on_probability,
                         min_load_pf, lead_pf_probability):
        # Define load array shape for convenience.
        shape = (self.num_scenarios, self.num_loads)

        # Randomly draw an active power loading condition for each
        # episode from the uniform distribution between min load and
        # max load.
        # Initialize then fill the array to get the proper data type.
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

            # Get our total load for this scenario. Multiply by losses
            # so the slack doesn't have to make up too much.
            load = self.total_load_mw[scenario_idx] * (1 + LOSS)

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
                        self.gen_init_data.iloc[gen_idx]['GenMWMin'],
                        min(self.gen_init_data.iloc[gen_idx]['GenMWMax'],
                            load))

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

    def _take_action(self, action):
        """Helper to make the appropriate updates in PowerWorld for a
        given action.
        """
        # Look up action and send to PowerWorld.
        gen_idx = self.action_array[action, 0]
        # TODO: it might be worth short-circuiting everything if the
        #   action won't do anything (e.g. the generator is off).
        voltage = self.gen_bins[self.action_array[action, 1]]
        self.saw.ChangeParametersSingleElement(
            ObjectType='gen', ParamList=self.gen_key_fields + ['GenVoltSet'],
            Values=self.gen_init_data.loc[gen_idx,
                                          self.gen_key_fields].tolist()
            + [voltage]
        )

    def _get_observation(self) -> np.ndarray:
        """Helper to return an observation. For the given simulation,
        the power flow should already have been solved.
        """
        # Add a column to load_data for power factor lead/lag
        self.load_obs_data['lead'] = \
            (self.load_obs_data['LoadSMVR'] < 0).astype(self.dtype)

        # Create observation by concatenating the relevant data. No
        # need to scale per unit data.
        return np.concatenate([
            # Bus voltages.
            self.bus_obs_data['BusPUVolt'].to_numpy(dtype=self.dtype),
            # TODO: Bus voltage angles?
            # Generator active power divide by maximum active power.
            (self.gen_obs_data['GenMW']
             / self.gen_obs_data['GenMWMax']).to_numpy(dtype=self.dtype),
            # Generator power factor.
            (self.gen_obs_data['GenMW'] / self.gen_obs_data['GenMVA']).fillna(
                1).to_numpy(dtype=self.dtype),
            # Generator var loading.
            self.gen_obs_data['GenMVRPercent'].to_numpy(
                dtype=self.dtype) / 100,
            # Load MW consumption divided by maximum MW loading.
            (self.load_obs_data['LoadSMW'] / self.max_load_mw).to_numpy(
                dtype=self.dtype),
            # Load power factor.
            self.load_obs_data['PowerFactor'].to_numpy(dtype=self.dtype),
            # Flag for leading power factors.
            self.load_obs_data['lead'].to_numpy(dtype=self.dtype)
        ])

    def _compute_reward(self):
        """Helper to compute a reward after an action. This method
        should only be called after _rotate_and_get_observation_frames,
        as that method updates the observation DataFrame attributes.
        """
        # First of all, any action gets us a negative reward. We'd like
        # to avoid changing set points if possible.
        reward = self.rewards['action']

        # Compute the difference in the distance to nominal voltage for
        # all buses before and after the action. Multiply by 100 so that
        # we reward change per 0.01 pu. A positive value indicates
        # reducing the distance to nominal, while a negative value
        # indicates increasing the distance to nominal.
        nom_delta_diff = \
            ((self.bus_obs_data_prev['BusPUVolt'] - NOMINAL_V).abs()
             - (self.bus_obs_data['BusPUVolt'] - NOMINAL_V).abs()) * 100

        # Get masks for bus voltages which are too high or too low for
        # both the previous (pre-action) data frame and the current
        # (post-action) data frame.
        low_v_prev = self.bus_obs_data_prev['BusPUVolt'] < LOW_V
        high_v_prev = self.bus_obs_data_prev['BusPUVolt'] > HIGH_V
        low_v_now = self.bus_obs_data['BusPUVolt'] < LOW_V
        high_v_now = self.bus_obs_data['BusPUVolt'] > HIGH_V

        # Get masks for voltages.
        in_prev = ~low_v_prev & ~high_v_prev  # in bounds before
        out_prev = low_v_prev | high_v_prev   # out of bounds before
        in_now = ~low_v_now & ~high_v_now     # in bounds now
        out_now = low_v_now | high_v_now      # out of bounds now

        # Now, get more "composite" masks
        in_out = in_prev & out_now          # in before, out now
        out_in = out_prev & in_now          # out before, in now
        in_out_low = in_prev & low_v_now    # in before, low now
        in_out_high = in_prev & high_v_now  # in before, high now
        # Out of bounds before, but moved in the right direction.
        out_right_d = out_prev & nom_delta_diff[out_prev] > 0

        # Give reward for voltages that were out of bounds, but moved in
        # the right direction, based on the change in distance from
        # nominal voltage.
        reward += (nom_delta_diff[out_right_d] * self.rewards['v_delta']).sum()

        # Give penalty for voltages that were in bounds, but moved out
        # of bounds. Penalty should be based on how far away from the
        # boundary (upper or lower) that they moved.
        reward += ((self.bus_obs_data['BusPUVolt'][in_out_low] - LOW_V) / 0.01
                   * self.rewards['v_delta']).sum()
        reward += ((HIGH_V - self.bus_obs_data['BusPUVolt'][in_out_high])
                   / 0.01 * self.rewards['v_delta']).sum()

        # Give an extra penalty for moving buses out of bounds.
        reward += in_out.sum() * self.rewards['v_out_bounds']

        # Give an extra reward for moving buses in bounds.
        reward += out_in.sum() * self.rewards['v_in_bounds']

        # Check if all voltages are in range.
        if (low_v_now & high_v_now).sum() == 0:
            self.all_v_in_range = True
        else:
            self.all_v_in_range = False

        # Give a positive reward for lessening generator var loading,
        # and a negative reward for increasing it.
        # TODO: This really should account for actual vars not just
        #   percent loading.
        var_delta = (self.gen_obs_data_prev['GenMVRPercent']
                     - self.gen_obs_data['GenMVRPercent'])

        reward += (var_delta * self.rewards['gen_var_delta']).sum()

        # All done.
        return reward


# noinspection PyAbstractClass
class GridMindEnv(DiscreteVoltageControlEnvBase):
    """Environment for attempting to replicate the work done by the
    State Grid Corporation of China, described in the following paper:

    https://www.researchgate.net/publication/332630883_Autonomous_Voltage_Control_for_Grid_Operation_Using_Deep_Reinforcement_Learning

    There should be a version on IEEEXplore soon.

    Here's the summary:

    States: Power flow solution, i.e. bus voltages and magnitudes and
        line flows (P and Q).
    Rewards: Each action receives the following rewards:
        -> +100 for each bus between 0.95 and 1.05 p.u.
        -> -50 for buses in the range [0.8, 0.95) or (1.05, 1.25]
        -> -100 for buses < 0.8 or > 1.25.
        Additionally, a final reward is given at the end of the episode,
        which is the sum of each action reward divided by the number of
        actions taken.
    Control: In this paper, a single action is considered setting the
        voltage set point of all generators at once. Generator voltage
        set points are discretized to be in [0.95, 0.975, 1.0, 1.025,
        1.05].
    """


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class LowVoltageError(Error):
    """Raised if any bus voltages go below MIN_V."""
    pass
