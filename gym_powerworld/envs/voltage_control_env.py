from abc import ABC, abstractmethod
import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging
from esa import SAW, PowerWorldError
from typing import Union, Tuple, List
from copy import deepcopy
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from PIL import Image
import pickle

# Get full path to this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Specify bus voltage bounds.
LOW_V = 0.95
HIGH_V = 1.05
NOMINAL_V = 1.0
V_TOL = 0.0001

# Instead of writing code to manage English rules, just hard code plural
# mappings.
PLURAL_MAP = {
    'gen': 'gens',
    'load': 'loads',
    'bus': 'buses',
    'branch': 'branches',
    'shunt': 'shunts'
}

# Lines which are allowed to be opened in the 14 bus case for some
# environments.
LINES_TO_OPEN_14 = ((1, 5, '1'), (2, 3, '1'), (4, 5, '1'), (7, 9, '1'))

# Map for open/closed states.
STATE_MAP = {
    1: 'Closed',
    0: 'Open',
    True: 'Closed',
    False: 'Open',
    1.0: 'Closed',
    0.0: 'Open'
}

# Some environments may reject scenarios with a certain voltage range.
MIN_V = 0.7
MAX_V = 1.2

# Some environments may min/max scale voltages.
MIN_V_SCALED = 0
MAX_V_SCALED = 1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
V_SCALER = (MAX_V_SCALED - MIN_V_SCALED) / (MAX_V - MIN_V)
V_ADD_TERM = MIN_V_SCALED - MIN_V * V_SCALER


def save_env(env, file):
    """Pickle the given env and save to the given file. The saw object
    will be set to None for saving, and restored after saving is
    complete.
    """
    saw = env.saw
    env.saw = None
    try:
        with open(file, 'wb') as f:
            pickle.dump(env, f)
    finally:
        env.saw = saw

    return None


def load_env(file, pwb_path=None):
    """Load an environment from file. If given a pwb_path, initialize
    the environment's SAW object from the given path.
    """
    with open(file, 'rb') as f:
        env = pickle.load(f)

    if pwb_path is not None:
        p = pwb_path
    else:
        p = env.pwb_path

    env.saw = SAW(FileName=p, early_bind=True)

    # Prep the PowerWorld case.
    env.prep_case()

    return env


def _scale_voltages(arr_in: np.ndarray) -> np.ndarray:
    """Scale voltages which are assumed to already be on interval
    [MIN_V, MAX_V] to the interval [MIN_V_SCALED, MAX_V_SCALED]
    """
    return V_SCALER * arr_in + V_ADD_TERM


def _set_gens_for_scenario_gen_mw_and_v_set_point(self) -> None:
    """Set generator Open/Closed states and set Gen MW setpoints based
    on self's "gen_mw" attribute.

    :param self: An initialized child class of
        DiscreteVoltageControlEnvBase.
    """
    # Extract a subset of the generator data.
    gens = self.gen_com_data.loc[:, self.gen_key_fields
                                 + self.GEN_RESET_FIELDS]

    # Turn generators on/off and set their MW set points.
    gens.loc[:, 'GenMW'] = self.gen_mw[self.scenario_idx, :]
    gen_g_0 = self.gen_mw[self.scenario_idx, :] > 0
    gens.loc[gen_g_0, 'GenStatus'] = 'Closed'
    gens.loc[~gen_g_0, 'GenStatus'] = 'Open'

    # Change voltage set points.
    gens.loc[:, 'GenVoltSet'] = self.gen_v[self.scenario_idx, :]

    self.saw.change_parameters_multiple_element_df('gen', gens)

    # Nothing to return.
    return None


def write_ltc_filter_aux_file(file_name) -> str:
    """Helper to save an aux file which creates a branch filter to get
    on load tap changing (LTC) transformers.

    :param file_name: Path to save the .aux file.
    :returns: Name of filter ("ltc_filter").
    """
    # String defining the filter.
    s = \
        """
DATA (Filter, [ObjectType,FilterName,FilterLogic,Number,FilterPre,Enabled,DataMaintainerAssign])
{
"Branch" "ltc_filter" "AND" 1 "NO " "YES" ""
   <SUBDATA Condition>
     LineXFType = "LTC"
   </SUBDATA>
}"""
    with open(file_name, 'w') as f:
        f.write(s)

    return "ltc_filter"


# noinspection PyPep8Naming
class DiscreteVoltageControlEnvBase(ABC, gym.Env):
    """Base class for discrete voltage control environments.

    Subclasses must set the following attributes in __init__:
    - action_space

    Subclasses must implement the following methods:
    - _compute_loading
    - _compute_generation
    - _compute_gen_v_set_points
    - _take_action
    - _get_observation
    - _compute_reward
    - _extra_reset_actions
    - _compute_end_of_episode_reward
    - _compute_reward_failed_pf
    - _get_num_obs_and_space

    Note that the initialization method of this class solves the power
    flow and calls the SaveState method, so subclasses may need to
    repeat that if they make case modifications that they want reloaded
    for each call to "reset."
    """

    ####################################################################
    # Concrete class properties
    ####################################################################
    # The following parameters are all related to scenario
    # initialization.
    # TODO: Update as needed. For instance, add LTC params when we get
    #   there.
    SCENARIO_INIT_ATTRIBUTES = ['total_load_mw', 'loads_mw', 'loads_mvar',
                                'shunt_states', 'gen_mw', 'gen_v',
                                'branches_to_open']

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

    @property
    @abstractmethod
    def action_cap(self) -> int:
        """Subclasses should define an action_cap."""
        pass

    @property
    @abstractmethod
    def LINES_TO_OPEN(self) -> Union[Tuple[Tuple[int, int, str]], None]:
        """List of lists denoting lines that are allowed to be opened
        during scenario initialization. Set to None or an empty list to
        disable line opening. Each sub-list should have three elements:
        from bus, to bus, and line circuit ID. These correspond to the
        PowerWorld legacy variables BusNum, BusNum:1, LineCircuit
        """
        pass

    @property
    @abstractmethod
    def CONTINGENCIES(self) -> bool:
        """Subclasses should define a CONTINGENCIES flag indicating
        whether or not contingencies are included in scenario
        initialization.
        """
        pass

    ####################################################################
    # Initialization
    ####################################################################

    def __init__(self, pwb_path: str, num_scenarios: int,
                 max_load_factor: float = None,
                 min_load_factor: float = None,
                 min_load_pf: float = 0.8,
                 lead_pf_probability: float = 0.1,
                 load_on_probability: float = 0.8,
                 shunt_closed_probability: float = 0.6,
                 num_gen_voltage_bins: int = 5,
                 gen_voltage_range: Tuple[float, float] = (0.9, 1.1),
                 seed: float = None,
                 log_level=logging.INFO,
                 rewards: Union[dict, None] = None,
                 dtype=np.float32,
                 low_v: float = LOW_V,
                 high_v: float = HIGH_V,
                 oneline_axd: str = None,
                 contour_axd: str = None,
                 image_dir: str = None,
                 render_interval: float = 1.0,
                 log_buffer: int = 10000,
                 csv_logfile: str = 'log.csv',
                 truncate_voltages=False,
                 scale_voltage_obs=False,
                 clipped_reward=False,
                 vtol=V_TOL):
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
        :param shunt_closed_probability: For each scenario, probability
            to determine if shunts are closed. Should be on the
            interval (0, 1].
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
        :param low_v: Low end of voltage range that is considered
            acceptable (inclusive). Defaults to LOW_V constant, which at
            the time of writing was 0.95 p.u.
        :param high_v: high end of voltage range that is considered
            acceptable (inclusive). Defaults to HIGH_V constant, which
            at the time of writing was 1.05 p.u.
        :param oneline_axd: Full path to PowerWorld oneline file (.axd).
            This is required if using the "render" function. This should
            be created AFTER contours have been added to the oneline.
            Simply click "File" -> "Save Oneline As" and then save as
            the .axd type.
        :param contour_axd: Full path to PowerWorld .axd file used for
            adding voltage contours. This is required if using the
            "render" function. Create this file by going to "Onelines",
            clicking on "Contouring," and then clicking on "Contouring"
            in the drop down. Configure the contour, and then click
            "Save to AXD..."
        :param image_dir: Full path to directory to store images for
            rendering. If not provided (None) and the "render" method is
            called, a directory called "render_images" will be created
            in the current directory.
        :param render_interval: Interval in seconds to pause while
            displaying the oneline and voltage contours after a call
            to reset() or step(). Must be > 0 in order for rendering
            to work.
        :param log_buffer: How many log entries to hold before flushing
            to file.
        :param csv_logfile: Path to .csv file which will be used to
            log states/actions.
        :param truncate_voltages: Whether (True) or not (False) to
            consider a power flow solution with voltages outside of
            [MIN_V, MAX_V] to be failed.
        :param scale_voltage_obs: Whether (True) or not (False) to
            scale bus voltage observations. Cannot be True if
            truncate_voltages is False.
        :param clipped_reward: Whether (True) or not (False) to use
            the clipped reward scheme.
        :param vtol: Tolerance for comparing with low_v and high_v.
            This helps avoid rounding error. The class instance's
            low_v attribute will be set to low_v - vtol, and the
            high_v attribute will be set to high_v + vtol. Then,
            when checking v > low_v or v < high_v, we have some extra
            tolerance built in.
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
        self.pwb_path = pwb_path
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

        # Track low and high v.
        self.low_v = low_v - vtol
        self.high_v = high_v + vtol

        # Logging.
        self.log_buffer = log_buffer
        self.csv_logfile = csv_logfile

        # Load factors.
        self.min_load_factor = min_load_factor
        self.max_load_factor = max_load_factor

        # Shunt probability.
        self.shunt_closed_probability = shunt_closed_probability

        # Track reset successes and failures.
        self.reset_successes = 0
        self.reset_failures = 0

        # No-op action. Subclasses should override if they wish to have
        # a no-op action.
        self.no_op_action = None

        # Track the last action taken.
        self.last_action = None
        ################################################################
        # Rendering related stuff
        ################################################################
        # Track pwd and axd files.
        self.oneline_axd = oneline_axd
        self.contour_axd = contour_axd

        # For rendering, we'll simply call the oneline "my oneline"
        self.oneline_name = 'my oneline'

        # Image directory:
        if image_dir is None:
            self.image_dir = 'render_images'
        else:
            self.image_dir = image_dir

        # Current image to render.
        self.image_path: Union[str, None] = None
        self.image: Union[Image.Image, None] = None
        self.image_axis: Union[AxesImage, None] = None

        # Matplotlib stuff.
        self.fig: Union[plt.Figure, None] = None
        self.ax: Union[plt.axes.Axes, None] = None

        self.render_interval = render_interval

        # Initialize the _render_flag property to False. It'll be
        # flipped when "render" is called.
        self._render_flag = False

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

        # Now data which will be used during initialization. Note there
        # are formal properties defined for the non-underscored versions
        # of these. This is to protect them from being overridden.
        self._gen_init_data: Union[pd.DataFrame, None] = None
        self._load_init_data: Union[pd.DataFrame, None] = None
        self._bus_init_data: Union[pd.DataFrame, None] = None
        self._branch_init_data: Union[pd.DataFrame, None] = None
        self._shunt_init_data: Union[pd.DataFrame, None] = None

        # Data for creating commands. This will be initialized to a
        # copy of the init data, and then can safely be modified
        # without overwriting data from the original case.
        self.gen_com_data: Union[pd.DataFrame, None] = None
        self.load_com_data: Union[pd.DataFrame, None] = None
        self.bus_com_data: Union[pd.DataFrame, None] = None
        self.branch_com_data: Union[pd.DataFrame, None] = None
        self.shunt_com_data: Union[pd.DataFrame, None] = None

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
        self.gen_mw_capacity = self.gen_init_data['GenMWMax'].sum()
        self.gen_mvar_produce_capacity = self.gen_init_data['GenMVRMax'].sum()
        self.gen_mvar_consume_capacity = self.gen_init_data['GenMVRMin'].sum()

        # Get mask indicating which generators are regulating the same
        # buses.
        self.gen_dup_reg = self.gen_init_data.duplicated('GenRegNum', 'first')

        # Track the number of generator regulated buses.
        self.num_gen_reg_buses = (~self.gen_dup_reg).sum()

        # Create a multi-indexed version of the generators to ease
        # sending voltage set point commands to generators at the same
        # bus. "mi" for multi-index.
        self.gen_init_data_mi = self.gen_init_data.set_index(
            ['BusNum', 'GenID'])

        # Keep things simple by ensuring all generators simply regulate
        # their own bus.
        # TODO: This isn't necessary if the logic for ensuring voltage
        #   set points for generators at the same bus are the same is
        #   improved.
        # noinspection PyUnresolvedReferences
        if not (self.gen_init_data['BusNum']
                == self.gen_init_data['GenRegNum']).all():
            raise UserWarning(
                'Not currently supporting generators that regulate buses '
                'other than their own. This can be fixed without a horrendous '
                'amount of effort.'
            )
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
        # Shunt fields and data
        ################################################################
        # Turn off automatic control for all shunts.
        if self.num_shunts > 0:
            # We'll tweak both the com_data and init_data as we're
            # essentially tweaking the base case.
            self.shunt_com_data['AutoControl'] = 'NO'
            self.shunt_init_data['AutoControl'] = 'NO'
            self.saw.change_parameters_multiple_element_df(
                'shunt', self.shunt_com_data)

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
        # Scenario/episode initialization: shunts
        ################################################################
        self.shunt_states = self._compute_shunts()

        ################################################################
        # Action space
        ################################################################
        # Start by creating the generator bins.
        self.gen_bins = np.linspace(gen_voltage_range[0], gen_voltage_range[1],
                                    num_gen_voltage_bins)

        # Subclasses should set the action_space based on the gen_bins
        # and/or other factors.

        ################################################################
        # Scenario/episode initialization: generation
        ################################################################
        # Compute each individual generator's active power contribution
        # for each loading scenario.
        self.gen_mw = self._compute_generation()
        self.gen_v = self._compute_gen_v_set_points()

        ################################################################
        # Scenario/episode initialization: branches
        ################################################################
        self.branches_to_open = self._compute_branches()

        ################################################################
        # Manage voltage truncation and scaling.
        ################################################################
        if scale_voltage_obs and (not truncate_voltages):
            raise ValueError('If scale_voltage_obs is True, truncate_voltages '
                             'must be True.')

        # Simply set scale_voltage_obs attribute.
        self.scale_voltage_obs = scale_voltage_obs

        # Determine which _solve_and_observe method to use.
        if truncate_voltages:
            self._solve_and_observe = self._solve_and_observe_truncate
        else:
            self._solve_and_observe = self._solve_and_observe_default

        ################################################################
        # Observation space definition
        ################################################################
        self.num_obs, self.observation_space = self._get_num_obs_and_space()

        # We'll track how many actions the agent has taken in an episode
        # as part of the stopping criteria.
        self.action_count = 0

        ################################################################
        # Set rewards and reward methods.
        ################################################################
        self.rewards = deepcopy(self.REWARDS)
        self._overwrite_rewards(rewards)
        self.current_reward = np.nan
        # Initialize attribute for tracking episode cumulative rewards.
        # It'll be reset in reset.
        self.cumulative_reward = 0

        # Set _compute_reward and _compute_reward_failed_pf attributes
        # depending on whether or not a clipped reward is desired. Note
        # these reward schemes are markedly different, and will lead to
        # different learning behavior.
        if clipped_reward:
            self._compute_reward = self._compute_reward_volt_change_clipped
            self._compute_reward_failed_pf = \
                self._compute_reward_failed_power_flow_clipped
        else:
            self._compute_reward = self._compute_reward_volt_change
            self._compute_reward_failed_pf =\
                self._compute_reward_failed_pf_volt_change
        ################################################################
        # Action/State logging
        ################################################################
        # Initialize logging columns.
        # Start by retrieving a list of bus numbers.
        buses = self.saw.GetParametersMultipleElement(
            'bus', self.bus_key_fields)
        # Log episode #, action taken, all bus voltages, all generator
        # set points.
        self.log_columns = \
            ['episode', 'action_taken', 'reward'] \
            + [f'bus_{n}_v' for n in buses['BusNum'].tolist()] \
            + [f'gen_{bus}_{i}' for bus, i in
               zip(self.gen_init_data['BusNum'].to_numpy(),
                   self.gen_init_data['GenID'].to_numpy())]

        # Initialize the logging array.
        self.log_array = np.zeros((self.log_buffer, len(self.log_columns)))

        # Keep track of the index of the logging array.
        self.log_idx = 0

        # For purposes of appending/writing headers, track how many
        # times we've flushed.
        self.log_flush_count = 0

        ################################################################
        # Scenario success tracking
        ################################################################
        # Keep an array that tells us if the scenario was successfully
        # initialized.
        self.scenario_init_success = np.zeros(self.num_scenarios, dtype=bool)

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
    # Initialization data - make sure they have getters but not setters.
    ####################################################################
    @property
    def gen_init_data(self):
        return self._gen_init_data

    @property
    def load_init_data(self):
        return self._load_init_data

    @property
    def bus_init_data(self):
        return self._bus_init_data

    @property
    def branch_init_data(self):
        return self._branch_init_data

    @property
    def shunt_init_data(self):
        return self._shunt_init_data

    ####################################################################
    # Misc properties
    ####################################################################
    @property
    def all_v_in_range(self):
        """True if all voltages are on interval
        [self.low_v, self.high_v], False otherwise."""
        return self.bus_obs_data['BusPUVolt'].between(
            self.low_v, self.high_v, inclusive=True).all()

    @property
    def bus_pu_volt_arr(self) -> np.ndarray:
        """Bus per unit voltages as a numpy array."""
        return self.bus_obs_data['BusPUVolt'].to_numpy(dtype=self.dtype)

    @property
    def bus_pu_volt_arr_scaled(self) -> np.ndarray:
        """(Possibly) scaled per unit bus voltages as a numpy array."""
        if self.scale_voltage_obs:
            return _scale_voltages(self.bus_pu_volt_arr)
        else:
            return self.bus_pu_volt_arr

    @property
    def bus_pu_volt_arr_prev(self) -> np.ndarray:
        """Previous per unit voltages as a numpy array."""
        return self.bus_obs_data_prev['BusPUVolt'].to_numpy(dtype=self.dtype)

    @property
    def gen_volt_set_arr(self) -> np.ndarray:
        """Generator voltage setpoints as an array."""
        return self.gen_obs_data['GenVoltSet'].to_numpy(dtype=self.dtype)

    @property
    def gen_status_arr(self) -> np.ndarray:
        """Get the generator states (Open/Closed) as a numeric vector.
        Truthy values correspond to Closed, falsey values correspond to
        Open.
        """
        return (self.gen_obs_data['GenStatus'] == 'Closed').to_numpy(
            dtype=self.dtype)

    @property
    def branch_status_arr(self) -> np.ndarray:
        """Get the branch (line) states (Open/Closed) as a numeric
        vector. Truthy values correspond to Closed, falsey values
        correspond to Open.
        """
        return (self.branch_obs_data['LineStatus'] == 'Closed').to_numpy(
            dtype=self.dtype)

    @property
    def shunt_status_arr(self) -> np.ndarray:
        """Get shunt states (Open/Closed) as a numeric vector. Truthy
        values correspond to Closed, falsey values correspond to Open.
        """
        return (self.shunt_obs_data['SSStatus'] == 'Closed').to_numpy(
            dtype=self.dtype)

    ####################################################################
    # Public methods
    ####################################################################
    def prep_case(self):
        """Perform updates to the PowerWorld case as part of
        initialization. This method really only ever needs called during
        initialization and after loading a pickled environment.
        """
        raise NotImplementedError()

    def render(self, mode='human'):
        """The rendering here is quite primitive due to limitations in
        interacting with PowerWorld's oneline diagrams via SimAuto.
        The general flow looks like this:
        - Open the oneline .axd file (once only).
        - Open the contour .axd file (each time).
        - Export oneline to image file (.bmp in this case).
        - Use pillow/PIL to load the image.
        - Use matplotlib to display the image.
        """
        # Configure if necessary.
        if not self._render_flag:
            self._render_config()

        # Toggle the render flag to True.
        self._render_flag = True

        # Render.
        self._render()

    def reset(self):
        """Advance to the next episode/scenario. To do so, generators
        will be turned on/off, generator MW set points will be set,
        and load P/Q will be set. Finally, the power flow will be
        solved and an initial observation will be returned.
        """
        # Take extra reset actions as defined by the subclass.
        self._extra_reset_actions()

        # Reset the action counter.
        self.action_count = 0

        # Clear the current reward.
        self.current_reward = np.nan

        # Clear the cumulative reward.
        self.cumulative_reward = 0

        # Clear the last action.
        self.last_action = None

        done = False
        obs = None
        while (not done) & (self.scenario_idx < self.num_scenarios):
            # Load the initial state of the system to avoid getting
            # stuck in a low voltage solution from a previous solve.
            self.saw.LoadState()
            self.saw.SolvePowerFlow()

            # Get generators, loads, and lines set up for this scenario.
            # noinspection PyArgumentList
            self._set_gens_for_scenario()
            self._set_loads_for_scenario()
            self._set_branches_for_scenario()
            self._set_shunts_for_scenario()

            # Solve the power flow.
            try:
                obs = self._solve_and_observe()
            except PowerWorldError as exc:
                # This scenario is bad. Move on.
                self.reset_failures += 1
                self.scenario_init_success[self.scenario_idx] = False
                self.log.warning(
                    f'Scenario {self.scenario_idx} failed. Error message: '
                    f'{exc.args[0]}')
                obs = None
            else:
                # Success! The power flow solved. Signify we're done
                # looping.
                self.reset_successes += 1
                self.scenario_init_success[self.scenario_idx] = True
                done = True
            finally:
                # Always increment the scenario index.
                self.scenario_idx += 1

        # Log. Since no action was taken, pass NaN.
        self._add_to_log(action=np.nan)

        # Raise exception if we've gone through all the scenarios.
        if (self.scenario_idx >= self.num_scenarios) and (obs is None):
            raise OutOfScenariosError('We have gone through all scenarios.')

        # Return the observation.
        # noinspection PyUnboundLocalVariable
        return obs

    def step(self, action):
        """Change generator set point, solve power flow, compute reward.
        """
        # Bump the action counter.
        self.action_count += 1

        # Track the action.
        self.last_action = action

        # Take the action.
        self._take_action(action)

        # Initialize info dict.
        info = dict()

        # Solve the power flow and get an observation.
        try:
            obs = self._solve_and_observe()
        except PowerWorldError:
            # The power flow failed to solve or bus voltages went below
            # the minimum. This episode is complete.
            #
            # Get an observation for the special case where the power
            # flow failed (or we have really low voltages).
            obs = self._get_observation_failed_pf()
            done = True
            # An action was taken, so include both the action and
            # failure penalties.
            reward = self._compute_reward_failed_pf()
            info['is_success'] = False
        else:
            # The power flow successfully solved. Compute the reward
            # and check to see if this episode is done.
            reward = self._compute_reward()
            done = self._check_done()

            if done and self.all_v_in_range:
                info['is_success'] = True
            elif done:
                info['is_success'] = False

        # Update the cumulative reward for this episode.
        self.cumulative_reward += reward

        # Some subclasses may wish to add an end of episode reward.
        # Ensure this is done after updating the cumulative reward in
        # case this reward depends on the cumulative reward.
        if done:
            eor = self._compute_end_of_episode_reward()
            if eor is not None:
                reward += eor
                self.cumulative_reward += eor

        # Update current reward.
        self.current_reward = reward

        # Always log.
        self._add_to_log(action=action)

        # That's it.
        return obs, reward, done, info

    def close(self):
        """Tear down SimAuto wrapper, flush the log."""
        self.saw.exit()
        self._flush_log()

    def reset_log(self, new_file) -> None:
        """Reset the log. This is useful if you, for example, have
        stopped training and want to create a new log for testing the
        trained agent.

        :param new_file: Full path to the new csv log file to be
            created.
        """
        # Start by ensuring the log has been completely flushed.
        self._flush_log()

        # Overwrite the csv_logfile.
        self.csv_logfile = new_file

        # Reset the log index and flush count.
        self.log_idx = 0
        self.log_flush_count = 0

        # All done.
        return None

    def filter_scenarios(self, mask):
        """Using a boolean mask of shape (self.num_scenarios,), filter
        all the scenario related data. This is useful if you want to
        pre-screen scenarios which are invalid.
        """
        # Simply loop and filter.
        for attr in self.SCENARIO_INIT_ATTRIBUTES:

            arr = getattr(self, attr)

            # Nothing to do if the array is actually None.
            if arr is None:
                continue

            # index 1-d arrays differently than 2-d arrays.
            s = len(arr.shape)
            if s == 1:
                filtered_arr = arr[mask]
            elif s == 2:
                filtered_arr = arr[mask, :]
            else:
                raise UserWarning('Something odd is afoot.')

            # Overwrite.
            setattr(self, attr, filtered_arr)

        # Adjust the number of scenarios.
        self.num_scenarios = mask.sum()

    ####################################################################
    # Private methods
    ####################################################################
    def _render_config(self):
        """Helper to perform rendering configuration."""
        # If either pwd or axd are None, we can't render.
        if (self.oneline_axd is None) or (self.contour_axd is None):
            self.log.error('Cannot render without providing the "oneline_axd" '
                           'and "contour_axd" parameters during environment '
                           'initialization. Rendering will not occur.')
            return

        # Do some matplotlib configuration. The ion() call will put it
        # in interactive mode, which will allow us to show figures
        # without blocking, and show() will kick of the "showing."
        plt.ion()
        plt.show()
        plt.axis('off')

        # Initialize figure and axes.
        self.fig, self.ax = plt.subplots(frameon=False)

        # Make the figure full screen.
        self.fig.canvas.manager.full_screen_toggle()

        # Turn off axis.
        self.ax.set_axis_off()

        # Open the oneline.
        self.saw.RunScriptCommand(
            fr'LoadAXD("{self.oneline_axd}", "{self.oneline_name}");')

        # Create the image directory.
        try:
            os.mkdir(self.image_dir)
        except FileExistsError:
            self.log.warning(f'The directory {self.image_dir} already exists. '
                             'Existing images will be overwritten.')

    def _render(self):
        """Do the work of rendering."""
        # Load the contour AXD file to update the contours.
        self.saw.RunScriptCommand(
            fr'LoadAXD("{self.contour_axd}", "{self.oneline_name}");')

        # Create the file name.
        # Note that we're using .bmp files. This is because in some
        # testing, I found that PowerWorld is actually fastest at
        # writing out BMP files, despite their larger file size. While
        # it does take Pillow ~1.5x longer to read BMP than JPG, the
        # writing times are much longer than the reading times, so the
        # trade off is worth it. Also note that Pillow is waaaaay faster
        # at loading both BMPs and JPGs than cv2 (provided by
        # the opencv-python package).
        self.image_path = os.path.join(
            self.image_dir,
            f'episode_{self.scenario_idx}_action_{self.action_count}.bmp')

        # Save to file. Arguments:
        # "filename", "OnelineName", ImageType, "view", FullScreen,
        # ShowFull
        self.saw.RunScriptCommand(
            fr'ExportOneline("{self.image_path}", "{self.oneline_name}", BMP, '
            + r'"", YES, YES)'
        )

        # Load up the image.
        self.image = Image.open(self.image_path)

        # Display the image.
        if self.image_axis is None:
            self.image_axis = self.ax.imshow(self.image)
        else:
            self.image_axis.set_data(self.image)

        # Add text.
        txt = (f'Episode: {self.scenario_idx}, Action Count: '
               f'{self.action_count}, Current Reward: {self.current_reward}')

        # Using the axis title seems to be best (as opposed to the
        # figure supertitle or adding a text box).
        self.ax.set_title(txt, fontsize=32, fontweight='bold')
        # self.fig.suptitle(txt, fontsize=32, fontweight='bold')
        # self.ax.text(
        #     0.1, 0.9, txt, color='black',
        #     bbox=dict(facecolor='white', edgecolor='black'))

        # Ensure the title is nice and tight to the figure.
        plt.tight_layout()

        # Redraw and pause.
        # Do we want/need to tighten the layout?
        # plt.tight_layout()
        plt.draw()
        plt.pause(self.render_interval)

    def _fill_init_attributes(self):
        """Helper to loop and set attributes for gens, loads, buses,
        branches, and shunts. This method will set the following:

        <obj>_key_fields
        <obj>_init_fields
        <obj>_obs_fields
        <obj>_reset_fields
        <obj>_init_data
        <obj>_com_data
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

            # Pull and set initialization and command data, as well as
            # compute the number of elements present.
            data = self.saw.GetParametersMultipleElement(
                ObjectType=obj,
                ParamList=getattr(self, f'{obj}_init_fields'))

            plural = PLURAL_MAP[obj]

            # ESA will return None if the objects are not present.
            if data is not None:
                setattr(self, f'_{obj}_init_data', data)
                setattr(self, f'{obj}_com_data', data.copy(deep=True))
                setattr(self, f'num_{plural}', data.shape[0])
            else:
                setattr(self, f'num_{plural}', 0)

    def _zero_negative_gen_mw_limits(self):
        """Helper to zero out generator MW limits which are < 0."""
        gen_less_0 = self.gen_init_data['GenMWMin'] < 0
        if (self.gen_init_data['GenMWMin'] < 0).any():
            # In this case we'll zero out the initialization data since
            # we're essentially modifying the base case.
            self.gen_init_data.loc[gen_less_0, 'GenMWMin'] = 0.0
            self.gen_com_data.loc[gen_less_0, 'GenMWMin'] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'gen', self.gen_com_data.loc[:, self.gen_key_fields
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
            # In this case we'll zero out the actual initialization data
            # since we're effectively modifying the "base" case.
            self.load_com_data.loc[:, LOAD_I_Z] = 0.0
            self.load_init_data.loc[:, LOAD_I_Z] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'Load', self.load_com_data.loc[:, self.load_key_fields
                                               + LOAD_I_Z])

    def _check_max_load(self, max_load_factor):
        """Ensure maximum loading is less than generation capacity. Also
        warn if generation capacity is >= 2 * maximum loading.
        """
        max_with_loss = self.max_load_mw * (1 + LOSS)
        if max_with_loss >= self.gen_mw_capacity:
            raise MaxLoadAboveMaxGenError(
                f'The given max_load_factor, {max_load_factor:.3f} '
                f'resulted in maximum loading of {max_with_loss:.3f} MW '
                '(loss estimation included), but the generator active '
                'power capacity is only '
                f'{self.gen_mw_capacity:.3f} MW. Reduce the '
                'max_load_factor and try again.')

        load_gen_ratio = max_with_loss / self.gen_mw_capacity
        if load_gen_ratio >= 0.98:
            self.log.warning(
                'The ratio of maximum loading (with a loss factor included) '
                f'to the maximum generation is {load_gen_ratio:.2f}. Having '
                'maximum load so near to maximum generation could result in '
                'very slow scenario generation.'
            )

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
            raise MinLoadBelowMinGenError(
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

    def _solve_and_observe_default(self):
        """Helper to solve the power flow and get an observation.

        :raises PowerWorldError: If PowerWorld fails to solve the power
            flow.
        """
        # Start by solving the power flow. This will raise a
        # PowerWorldError if it fails to solve.
        self.saw.SolvePowerFlow()

        # Get new observations, rotate old ones.
        self._rotate_and_get_observation_frames()

        # Get and return a properly arranged observation.
        return self._get_observation()

    def _solve_and_observe_truncate(self):
        """Helper to solve the power flow and get an observation.
        The case will be considered to "fail" if voltages are outside
        the [MIN_V, MAX_V] range.

        :raises PowerWorldError: If PowerWorld fails to solve the power
            flow or voltages are outside of [MIN_V, MAX_V].
        """
        # Start by solving the power flow. This will raise a
        # PowerWorldError if it fails to solve.
        self.saw.SolvePowerFlow()

        # Get new observations, rotate old ones.
        self._rotate_and_get_observation_frames()

        # Reject this power flow solution if voltages are below the min
        # or above the max.
        # noinspection PyArgumentList
        if ((self.bus_pu_volt_arr.min() < self.dtype(MIN_V))
                or (self.bus_pu_volt_arr.max() > self.dtype(MAX_V))):
            raise PowerWorldError(
                'Scenario rejected as there was at least one bus voltage '
                f'less than {MIN_V:.2f} or greater than {MAX_V:.2f}')

        # Get and return a properly arranged observation.
        return self._get_observation()

    def _rotate_and_get_observation_frames(self):
        """Simple helper to get new observation DataFrames, and rotate
        the previous frames into the correct attributes.
        """
        # Loop over the various object types.
        for obj in ['bus', 'gen', 'load', 'branch', 'shunt']:
            # Rotate <obj>_obs_data into <obj>_obs_data_prev.
            setattr(self, obj + '_obs_data_prev',
                    getattr(self, obj + '_obs_data'))

            # Get new data if applicable.
            fields = getattr(self, obj + '_obs_fields')
            if fields is not None:
                setattr(
                    self, obj + '_obs_data',
                    self.saw.GetParametersMultipleElement(
                        ObjectType=obj, ParamList=fields))

        # That's it.
        return None

    _set_gens_for_scenario = _set_gens_for_scenario_gen_mw_and_v_set_point

    def _set_loads_for_scenario(self):
        """Helper to set up loads in the case for this episode/scenario.
        This method should only be called by reset.
        """
        # Extract a subset of the load data.
        loads = self.load_com_data.loc[:, self.load_key_fields
                                       + self.LOAD_RESET_FIELDS]

        # Set P and Q.
        loads.loc[:, 'LoadSMW'] = self.loads_mw[self.scenario_idx, :]
        loads.loc[:, 'LoadSMVR'] = self.loads_mvar[self.scenario_idx, :]
        self.saw.change_parameters_multiple_element_df('load', loads)

    def _set_shunts_for_scenario(self):
        """Helper to set up shunts. Some subclasses will do nothing."""
        pass

    def _check_done(self):
        """Check whether (True) or not (False) and episode is done. Call
        this after calling _solve_and_observe.
        """
        # If the number of actions taken in this episode has exceeded
        # a threshold, we're done.
        # TODO: Stop hard-coding number of actions
        if self.action_count >= self.action_cap:
            return True

        # If all voltages are in range, we're done.
        if self.all_v_in_range:
            return True

        # Otherwise, we're not done.
        return False

    def _flush_log(self):
        """Helper to flush the log to file."""
        # Create DataFrame with logging data. Need to handle ending
        # indices differently for if we're flushing the whole array
        # or a subset of it.
        if self.log_idx == self.log_buffer:
            idx = self.log_idx + 1
        else:
            idx = self.log_idx

        df = pd.DataFrame(self.log_array[0:idx],
                          columns=self.log_columns)

        # If this is our first flush, we need to use writing mode and
        # include the headers.
        if self.log_flush_count == 0:
            df.to_csv(self.csv_logfile, mode='w', index=False, header=True)
        else:
            # Append.
            df.to_csv(self.csv_logfile, mode='a', index=False, header=False)

        # Increase the flush count.
        self.log_flush_count += 1

        # Clear the logging array.
        self.log_array[:, :] = 0.0

        # Reset the log index.
        self.log_idx = 0

    def _add_to_log(self, action):
        """Helper to add data to the log and flush if necessary."""
        # Logging occurs after the scenario index is bumped, so
        # subtract 1.
        self.log_array[self.log_idx, 0] = self.scenario_idx - 1
        # Action.
        self.log_array[self.log_idx, 1] = action
        # Reward.
        self.log_array[self.log_idx, 2] = self.current_reward
        # Data.
        self.log_array[self.log_idx, 3:] = \
            np.concatenate((self.bus_pu_volt_arr, self.gen_volt_set_arr))

        # Increment the log index.
        self.log_idx += 1

        # Flush if necessary.
        if self.log_idx == self.log_buffer:
            self._flush_log()

    def _compute_reward_volt_change(self) -> float:
        """Compute reward based on voltage movement.
        """
        # If the action taken was the no-op action, give a simple reward
        # if all voltages are in bounds, and a simple penalty otherwise.
        if self.last_action == self.no_op_action:
            if self.all_v_in_range:
                return self.rewards['no_op']
            else:
                return -self.rewards['no_op']

        # First of all, any action gets us a negative reward. We'd like
        # to avoid changing set points if possible.
        reward = self.rewards['action']

        # Get aliases for the voltages to simplify this gross method.
        v_prev = self.bus_obs_data_prev['BusPUVolt']
        v_now = self.bus_obs_data['BusPUVolt']

        # Compute the difference in the distance to nominal voltage for
        # all buses before and after the action. Multiply by 100 so that
        # we reward change per 0.01 pu. Take the absolute value.
        nom_delta_diff = \
            ((v_prev - NOMINAL_V).abs() - (
                        v_now - NOMINAL_V).abs()).abs() * 100

        # Use helper function to get dictionary of masks.
        d = _get_voltage_masks(
            v_prev=v_prev.to_numpy(dtype=self.dtype),
            v_now=v_now.to_numpy(dtype=self.dtype),
            low_v=self.low_v,
            high_v=self.high_v)

        # Give reward for voltages that were out of bounds, but moved in
        # the right direction, based on the change in distance from
        # nominal voltage.
        reward += (nom_delta_diff[d['out_right_d']]
                   * self.rewards['v_delta']).sum()

        # Similarly, give a penalty for voltages that were out of bounds,
        # and moved in the wrong direction, based on the change in distance
        # from nominal voltage.
        reward -= (nom_delta_diff[d['out_wrong_d']]
                   * self.rewards['v_delta']).sum()

        # Give penalty for voltages that were in bounds, but moved out
        # of bounds. Penalty should be based on how far away from the
        # boundary (upper or lower) that they moved.
        reward += ((v_now[d['in_out_low']] - LOW_V) / 0.01
                   * self.rewards['v_delta']).sum()
        reward += ((HIGH_V - v_now[d['in_out_high']])
                   / 0.01 * self.rewards['v_delta']).sum()

        # Give an extra penalty for moving buses out of bounds.
        reward += d['in_out'].sum() * self.rewards['v_out_bounds']

        # Give an extra reward for moving buses in bounds.
        reward += d['out_in'].sum() * self.rewards['v_in_bounds']

        # All done.
        return reward

    def _compute_reward_failed_pf_volt_change(self) -> float:
        """Simply combine the fail and action rewards."""
        return self.rewards['fail'] + self.rewards['action']

    def _compute_reward_volt_change_clipped(self) -> float:
        """Simplified voltage movement reward with a minimum at -1 and a
        maximum at 1. The -1 reward won't be realized here, but rather in
        the corresponding _compute_reward_failed_pf-like function.
        """
        # If the action taken was the no-op action, there's no reward or
        # penalty.
        if self.last_action == self.no_op_action:
            return 0.0

        # If all voltages are now in bounds, give the full reward of 1.
        if self.all_v_in_range:
            return 1.0

        # Get aliases for the voltages to simplify. Round voltages to
        # get rid of annoying floating point errors.
        v_prev = self.bus_pu_volt_arr_prev
        v_now = self.bus_pu_volt_arr

        # Get dictionary of masks.
        d = _get_voltage_masks(v_prev=v_prev, v_now=v_now, low_v=self.low_v,
                               high_v=self.high_v)

        # Compute number of buses that moved from out of the band to in
        # the band, and vice versa.
        out_in_sum = d['out_in'].sum()
        in_out_sum = d['in_out'].sum()

        # If more voltages moved in bounds, give a reward.
        if (out_in_sum > in_out_sum) and (out_in_sum > 0):
            net = out_in_sum - in_out_sum

            if net > 1:
                # 2nd highest possible reward.
                return 0.75
            else:
                # 3rd highest possible reward.
                return 0.5

        # If more voltages moved out of bounds, give a penalty.
        if (in_out_sum > out_in_sum) and (in_out_sum > 0):
            net = in_out_sum - out_in_sum

            if net > 1:
                # Lowest possible reward aside from failed power flow.
                return -0.75
            else:
                # Less severe penalty for just moving one bus.
                return -0.5

        # If we're here, no voltages moved in or out of bounds. In that
        # case, we'll give lesser rewards/penalties depending on movement
        # in the right/wrong direction.
        right_d_sum = d['out_right_d'].sum()
        wrong_d_sum = d['out_wrong_d'].sum()

        # If more voltages moved in the right direction, give a reward.
        if (right_d_sum > wrong_d_sum) and (right_d_sum > 0):
            return 0.25

        # If more voltages moved in the wrong direction, give a penalty.
        if (wrong_d_sum > right_d_sum) and (wrong_d_sum > 0):
            return -0.25

        # TODO: If there were overshoots or undershoots, give a penalty if
        #  we're now further from the band or a reward if we're now closer
        #  to the band.

        # If we're here, the given action was not very helpful or harmful.
        # Give a small penalty for taking a "useless" action.
        return -0.1

    # noinspection PyMethodMayBeStatic
    def _compute_reward_failed_power_flow_clipped(self) -> float:
        """Clipped reward goes -1 to 1. So, causing a failure is the
        biggest penalty.
        """
        return -1.0

    def _compute_reward_gen_var_change(self) -> float:
        """Compute a reward based on how generator var loading changes."""
        # Give a positive reward for lessening generator var loading,
        # and a negative reward for increasing it.
        # TODO: This really should account for actual vars not just
        #   percent loading.
        var_delta = (self.gen_obs_data_prev['GenMVRPercent']
                     - self.gen_obs_data['GenMVRPercent'])

        return (var_delta * self.rewards['gen_var_delta']).sum()

    def _compute_reward_volt_and_var_change(self) -> float:
        """Composite reward that accounts for both bus voltage movement and
        generator var loading movement.
        """
        reward = 0
        reward += self._compute_reward_volt_change()
        reward += self._compute_reward_gen_var_change()
        return reward

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
    def _compute_gen_v_set_points(self):
        """Subclasses should implement this method to create an array
        of voltage set points for generators for each scenario.
        """
        pass

    @abstractmethod
    def _compute_shunts(self):
        """Subclasses should implement this method to create an array
        of shunt states.
        """

    @abstractmethod
    def _compute_branches(self):
        """Subclasses should implement this method to create an array
        of data related to branch states. In most cases, this will
        likely just be an array of shape (num_scenarios,) with a branch
        index in each spot.
        """

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
    def _get_observation_failed_pf(self):
        """Subclasses should implement a _get_observation method which
        returns an observation ONLY in the case of a failed power flow
        or LowVoltageError.
        """

    @abstractmethod
    def _extra_reset_actions(self):
        """Subclasses should implement this method, which is called at
        the beginning of "reset".
        """

    @abstractmethod
    def _compute_end_of_episode_reward(self):
        """Subclasses should implement this method, which will, well,
        compute an end of episode reward (if desired). If not desired,
        just return 0.
        """

    @abstractmethod
    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Space]:
        """Return the number of observations, and an observation space.
        """

    @abstractmethod
    def _set_branches_for_scenario(self):
        """Helper to set up lines and/or taps using
        self.branches_to_open, which is created by calling
        self._compute_branches.
        """


def _compute_loading_robust(self: DiscreteVoltageControlEnvBase,
                            load_on_probability, min_load_pf,
                            lead_pf_probability) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    "Robust" computation of loading, can be used as a drop-in for the
    _compute_loading method of child classes of
    DiscreteVoltageControlEnvBase.

    1. Randomly draw total loading between min and max

    2. Randomly draw which loads are on or off based on the
    load_on_probability

    3. Draw load levels for "on" loads from the uniform distribution,
    scale each scenario to match the appropriate total loading

    4. Randomly draw power factors for each load, and then compute
    var loading accordingly

    5. Randomly flip the sign of some var loads in order to make some
    loads leading.

    :param self: An initialized child class of
        DiscreteVoltageControlEnvBase
    :param load_on_probability: Probability a given load is "on." Should
        be pretty close to 1.
    :param min_load_pf: Minimum allowed power factor.
    :param lead_pf_probability: Probability an individual load will
        have a leading power factor. Should be closer to 0 than 1 in
        most cases.
    :returns: scenario_total_loads_mw, scenario_individual_loads_mw,
        scenario_individual_loads_mvar.

    """
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


# noinspection PyUnusedLocal
def _compute_loading_gridmind(self: DiscreteVoltageControlEnvBase, *args,
                              **kwargs) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As far as I can tell, the GridMind loading is dead simple -
    they scale each load between 80% and 120% of original.

    Here's the text from the paper:
    "Random  load  changes  are  applied across the entire system,
    and each load fluctuates within 80%-120% of its original value."

    :param self: An initialized child class of
        DiscreteVoltageControlEnvBase
    """
    # Solve the power flow for the initial case and get loading.
    # We can't assume that the case has already been solved.
    self.saw.SolvePowerFlow()
    loads = self.saw.GetParametersMultipleElement(
        ObjectType='load',
        ParamList=(self.load_key_fields + LOAD_P)
    )

    # Define load array shape for convenience.
    shape = (self.num_scenarios, self.num_loads)

    # Randomly draw an array between the minimum load factor and
    # maximum load factor.
    load_factor = self.rng.uniform(
        self.min_load_factor, self.max_load_factor, shape)

    # Create arrays to hold individual load MW and Mvar loading.
    scenario_individual_loads_mw = (
        np.tile(loads['LoadSMW'].to_numpy(), (self.num_scenarios, 1))
        * load_factor
    )
    scenario_individual_loads_mvar = (
        np.tile(loads['LoadSMVR'].to_numpy(), (self.num_scenarios, 1))
        * load_factor
    )

    # Return.
    return scenario_individual_loads_mw.sum(axis=1),\
        scenario_individual_loads_mw, scenario_individual_loads_mvar


def _compute_generation_and_dispatch(self: DiscreteVoltageControlEnvBase) -> \
        np.ndarray:
    """
    Drop in replacement for _compute_generation in child classes of
    DiscreteVoltageControlEnvBase. This method simultaneously computes
    generator commitment and dispatch in a vectorized manner. The
    process goes something like:

    1. Draw MW output for all generators between their minimum and
    maximum for all scenarios.

    2. Shuffle the generator order for each scenario.

    3. Compute a cumulative generation sum for each scenario. Once this
    cumulative sum exceeds load, zero out remaining generator outputs
    (effectively turning them off).

    4. For each scenario, proportionally reduce the MW output of all
    generators such that generation meets load.

    5. If this MW reduction resulted in some generators going below
    their minimum, set these generators to their minimum. Go back to
    the previous step, but only consider generators that are NOT at
    their minimum for the proportional reduction.

    :param self: An initialized child class of
        DiscreteVoltageControlEnvBase.
    :returns: scenario_gen_mw
    """
    # Extract generator minimums and maximums, expand them such that
    # we have a row for each scenario. There may be a more memory
    # efficient way to do this, but I haven't run out of memory yet :)
    gen_min = np.tile(
        self.gen_init_data['GenMWMin'].to_numpy(), (self.num_scenarios, 1))
    gen_max = np.tile(
        self.gen_init_data['GenMWMax'].to_numpy(), (self.num_scenarios, 1))

    # Get the loading for each scenario, multiply it by the loss factor.
    load = self.total_load_mw * (1 + LOSS)

    # Get the elementwise minimum between the generator maximums and the
    # loading for each condition.
    gen_max = np.minimum(gen_max, load[:, None])

    # Randomly draw generation between the minimum and maximum.
    scenario_gen_mw = self.rng.uniform(gen_min, gen_max)

    # Get indices for shuffling the array row-wise.
    # Sources:
    # https://stackoverflow.com/a/45438143/11052174
    # https://stackoverflow.com/a/55317373/11052174
    shuffle_idx = self.rng.random(scenario_gen_mw.shape).argsort(-1)

    # Shuffle the generation array, as well as the mins and maxes.
    gen_shuffled = np.take_along_axis(scenario_gen_mw, shuffle_idx, 1)
    gen_min_shuffled = np.take_along_axis(gen_min, shuffle_idx, 1)
    gen_max_shuffled = np.take_along_axis(gen_max, shuffle_idx, 1)

    # Get the cumulative sum of the shuffled generation rows.
    gen_cumsum = np.cumsum(gen_shuffled, axis=1)

    # Compare with total load, g_l --> greater than load.
    g_l = gen_cumsum >= load[:, None]

    # We need to loop until all scenarios have loading met.
    it = 0
    while it < 1000:
        it += 1
        # Check to see if the rows are capable of meeting generation.
        # noinspection PyUnresolvedReferences
        g_l_any = g_l.any(axis=1)

        # If all rows meet generation, we're done.
        if g_l_any.all():
            break

        # Bump up the minimums for the offending rows.
        gen_min_shuffled[~g_l_any, :] = gen_shuffled[~g_l_any, :]

        # Re-draw for the offending rows.
        gen_shuffled[~g_l_any, :] = self.rng.uniform(
            gen_min_shuffled[~g_l_any, :], gen_max_shuffled[~g_l_any, :]
        )

        # Re-compute the cumulative sum.
        gen_cumsum = np.cumsum(gen_shuffled, axis=1)

        # Compare with total load, g_l --> greater than load.
        g_l = gen_cumsum >= load[:, None]

    if it >= 1000:
        raise UserWarning(f'Hit 1000 iterations.')

    # Now, we need to zero out excess generation, and ensure each
    # scenario exactly matches the given load (with losses).
    #
    # Start by getting the indices of the first column where generation
    # exceeds load.
    # noinspection PyUnresolvedReferences
    gen_ex_load_idx = g_l.argmax(axis=1)

    # Zero out excess generation.
    gen_shuffled[
        np.tile(np.arange(self.num_gens), (self.num_scenarios, 1))
        > gen_ex_load_idx[:, None]] = 0.0

    # Get mask of active generators.
    active_mask = gen_shuffled > 0.0

    # Fix up the minimum array.
    gen_min_shuffled = np.take_along_axis(gen_min, shuffle_idx, 1)

    # Now, reduce generation and ensure minimums are respected.
    it = 0
    while it < 1000:
        it += 1
        # Apportion that difference among all active generators that
        # are not at their minimum.
        viable = (gen_shuffled != gen_min_shuffled) & active_mask
        num_viable = viable.sum(axis=1)
        # Get the difference between generation and load. Repeat it so
        # it has the same number of elements as are viable generators.
        # TODO: This reduces the MW output for ALL viable generators by
        #   the same amount. It may be more stable to allot more
        #   reduction to generators which are further away from their
        #   minimum. However, would this skew the randomness more?
        diff = np.repeat((gen_shuffled.sum(axis=1) - load) / num_viable,
                         num_viable)

        gen_shuffled[viable] -= diff

        # See if any generators violated their minimums after the
        # reapportionment.
        viol_min = gen_shuffled < gen_min_shuffled

        # If there are no violations, we're done.
        # noinspection PyUnresolvedReferences
        if not viol_min.any():
            break

        # Set violating generators at their minimum.
        gen_shuffled[viol_min] = gen_min_shuffled[viol_min]

    if it >= 1000:
        raise UserWarning(f'Hit 1000 iterations.')

    # Place the shuffled generation back in place.
    np.put_along_axis(scenario_gen_mw, shuffle_idx, gen_shuffled, 1)

    # All done.
    return scenario_gen_mw


def _compute_generation_gridmind(self: DiscreteVoltageControlEnvBase) -> None:
    """Here's the relevant text from the paper:

    "When loads change, generators are re-dispatched based on a
    participation factor list to maintain system power balance."

    I have to assume that the participation factor list is fixed,
    but maybe it isn't?

    The simplest approach here is to simply use PowerWorld to
    enforce participation factors. Several steps are required:
    - Turn on participation factor AGC for the area
    - Set AGC tolerance to be low (say 0.01MW) for the area
    - Turn on AGC for all generators
    - Set participation factor for all generators based on their
        MW rating.

    :param self: An initialized child class of
        DiscreteVoltageControlEnvBase.
    """
    # Start by getting data for areas.
    area_kf = self.saw.get_key_field_list(ObjectType='area')
    areas = self.saw.GetParametersMultipleElement(
        ObjectType='area',
        ParamList=(area_kf + ['ConvergenceTol', 'BGAGC']))

    # Put the a areas in participation factor AGC with a small
    # tolerance.
    areas['ConvergenceTol'] = 0.01
    areas['BGAGC'] = 'Part. AGC'
    self.saw.change_and_confirm_params_multiple_element(
        ObjectType='area', command_df=areas)

    # Get data for generators.
    gens = self.saw.GetParametersMultipleElement(
        ObjectType='gen',
        ParamList=(self.gen_key_fields
                   + ['GenAGCAble', 'GenEnforceMWLimits'])
    )

    # Make all generators AGCAble and ensure MW limits are followed.
    # TODO: Need exceptions for wind and other non-dispatchable
    #  resources
    gens['GenAGCAble'] = 'YES'
    gens['GenEnforceMWLimits'] = 'YES'
    self.saw.change_and_confirm_params_multiple_element(
        ObjectType='gen', command_df=gens
    )

    # Set participation factors for all generators in the system.
    self.saw.RunScriptCommand(
        'SetParticipationFactors(MAXMWRAT, 0, SYSTEM);')

    # Nothing to return.
    return None


def _compute_gen_v_set_points_draw(self: DiscreteVoltageControlEnvBase) -> \
        np.ndarray:
    """Compute generator voltage set points by drawing from the
    environment's gen_bins attribute. We also need to ensure that
    generators that regulate the same bus are set to the same set point.
    """
    # Start by simply randomly drawing for all generators.
    df = pd.DataFrame(self.rng.choice(
        self.gen_bins, size=(self.num_scenarios, self.num_gens), replace=True)
    )

    if self.gen_dup_reg.any():
        # Now, set gens which regulate the same bus as another to NaN,
        # except for the first one. E.g., if generators at buses 3 and 4
        # both regulate bus 5, the generator at bus 4 will have it's
        # set point reset to NaN.
        df.loc[:, self.gen_dup_reg.to_numpy()] = np.nan

        # ASSUMPTION NOTE: Here, we assume generators are ALWAYS
        # regulating their own bus, and not another. This could be
        # improved.
        # TODO: Allow for generators regulating other buses.
        # Back fill to ensure generators which regulate the same
        # bus have the same voltage set point. This counts on the fact
        # that our gen_dup_reg Series marks duplicates as True except
        # for the first occurrence.
        df.fillna(method='pad', inplace=True, limit=None, axis='columns')

    return df.to_numpy()


def _compute_branches_from_lines_to_open(self):
    """Draw a random line index from LINES_TO_OPEN for each
    scenario.
    """
    if (self.LINES_TO_OPEN is None) or (len(self.LINES_TO_OPEN) == 0):
        # Nothing to do.
        return None

    # Simply draw an index for each line.
    return self.rng.integers(low=0, high=len(self.LINES_TO_OPEN),
                             size=self.num_scenarios)


def _get_voltage_masks(v_prev, v_now, low_v, high_v):
    # Get masks for bus voltages which are too high or too low for
    # both the previous (pre-action) data frame and the current
    # (post-action) data frame.
    low_v_prev = v_prev < low_v
    high_v_prev = v_prev > high_v
    low_v_now = v_now < low_v
    high_v_now = v_now > high_v

    # Get masks for voltages.
    in_prev = (~low_v_prev) & (~high_v_prev)    # in bounds before
    out_prev = low_v_prev | high_v_prev         # out of bounds before
    in_now = (~low_v_now) & (~high_v_now)       # in bounds now
    out_now = low_v_now | high_v_now            # out of bounds now

    # Movement masks.
    v_diff = v_prev - v_now
    moved_up = v_diff < 0
    moved_down = v_diff > 0

    # Now, get more "composite" masks
    in_out = in_prev & out_now              # in before, out now
    out_in = out_prev & in_now              # out before, in now
    in_out_low = in_prev & low_v_now        # in before, low now
    in_out_high = in_prev & high_v_now      # in before, high now
    # Out of bounds before, but moved in the right direction.
    out_right_d = (out_prev
                   & ((high_v_prev & moved_down) | (low_v_prev & moved_up)))
    # Out of bounds before, but moved in the wrong direction.
    out_wrong_d = (out_prev
                   & ((high_v_prev & moved_up) | (low_v_prev & moved_down)))

    # TODO: May want to consider the edge case of over/under shooting.
    #   With the current experiments, the allowable generator set points
    #   make this extremely unlikely to happen. Besides, the agent is
    #   trying to maximize it's reward, and should thus prefer the
    #   larger reward of moving buses in bounds over some sort of
    #   marginal reward for overshooting but decreasing distance to
    #   the band.
    # Out of bounds before, moved in the right direction, out of bounds
    # now. AKA, "overshot" or "undershot" the band.
    # over_under_shoot = out_prev & out_now & out_right_d

    out = {
        # 'low_v_prev': low_v_prev, 'high_v_prev': high_v_prev,
        # 'low_v_now': low_v_now, 'high_v_now': high_v_now,
        # 'in_prev': in_prev, out_prev: 'out_prev', 'in_now': in_now,
        # 'out_now': out_now, 'moved_up': moved_up, 'moved_down': moved_down,
        'in_out': in_out, 'out_in': out_in, 'in_out_low': in_out_low,
        'in_out_high': in_out_high, 'out_right_d': out_right_d,
        'out_wrong_d': out_wrong_d,   # 'over_under_shoot': over_under_shoot
    }

    return out


def _set_branches_for_scenario_from_lines_to_open(
        self: DiscreteVoltageControlEnvBase):
    """Open a line from LINES_TO_OPEN.
    """
    # Do nothing if we have no lines to open.
    if self.branches_to_open is None:
        # Do nothing.
        return

    # Extract the line from LINES_TO_OPEN.
    line = self.LINES_TO_OPEN[self.branches_to_open[self.scenario_idx]]

    # Open the line.
    self.saw.ChangeParametersSingleElement(
        ObjectType='branch',
        ParamList=['BusNum', 'BusNum:1', 'LineCircuit', 'LineStatus'],
        Values=[*line, 'Open']
    )


def _get_observation_bus_pu_only(
        self: DiscreteVoltageControlEnvBase) -> np.ndarray:
    # Note the voltage will only be scaled if self.scale_voltage_obs
    # is True. Otherwise, it'll be raw voltages.
    return self.bus_pu_volt_arr_scaled


def _get_num_obs_and_space_v_only(
        self: DiscreteVoltageControlEnvBase) -> Tuple[int, spaces.Box]:
    """Number of observations is simply the number of buses,
    observation space is a box for voltages.
    """
    # Set voltage cap at 2 if we're not scaling, set to 1 otherwise.
    if self.scale_voltage_obs:
        high = 1
    else:
        high = 2

    return self.num_buses, spaces.Box(
        low=0, high=high, shape=(self.num_buses,), dtype=self.dtype)


def _get_observation_failed_pf_volt_only(self: DiscreteVoltageControlEnvBase)\
        -> np.ndarray:
    """If the power flow fails to solve, return an observation of 0
    voltages.
    """
    return np.zeros(self.num_buses, dtype=self.dtype)


class DiscreteVoltageControlEnv(DiscreteVoltageControlEnvBase):
    """Environment for performing voltage control with the PowerWorld
    Simulator.
    """
    # Gen fields. See base class for comments.
    GEN_INIT_FIELDS = ['BusCat', 'GenMW', 'GenMVR', 'GenVoltSet', 'GenMWMax',
                       'GenMWMin', 'GenMVRMax', 'GenMVRMin', 'GenStatus',
                       'GenRegNum']
    GEN_OBS_FIELDS = ['GenMW', 'GenMWMax', 'GenMVA', 'GenMVRPercent',
                      'GenStatus', 'GenVoltSet']
    GEN_RESET_FIELDS = ['GenMW', 'GenStatus', 'GenVoltSet']

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
    CONTINGENCIES = False

    # By default, do not open any lines.
    LINES_TO_OPEN = None

    # Shunt fields.
    # Really only grabbing the AutoControl for testing purposes.
    SHUNT_INIT_FIELDS = ['AutoControl', 'SSStatus']
    # We'll be using the status for generating an observation.
    SHUNT_OBS_FIELDS = ['SSStatus']
    SHUNT_RESET_FIELDS = []

    # LTC fields.
    LTC_INIT_FIELDS = ['XFAuto', 'XFRegMin', 'XFRegMax', 'XFTapMin',
                       'XFTapMax', 'XFStep', 'XFTapPos', 'XFTapPos:1',
                       'XFTapPos:2']
    LTC_OBS_FIELDS = ['XFTapPos']

    # Specify default rewards.
    # NOTE: When computing rewards, all reward components will be
    #   treated with a positive sign. Thus, specify penalties in this
    #   dictionary with a negative sign.
    REWARDS = {
        # Negative reward (penalty) given for taking an action.
        # TODO: May want different penalties for different types of
        #   actions, e.g. changing gen set point vs. switching cap.
        "action": -10,
        # If no action is taken, give a reward if all voltages are in
        # bounds, otherwise penalize.
        "no_op": 50,
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

    def __init__(self, **kwargs):
        """See parent class for parameter definitions.
        """

        # Start by calling super constructor.
        super().__init__(**kwargs)

        # Extract keyword arguments we need.
        num_gen_voltage_bins = kwargs['num_gen_voltage_bins']
        ################################################################
        # Get/set load tap changing (LTC) transformers.
        ################################################################
        # Write LTC filter to file.
        ltc_file = os.path.join(THIS_DIR, 'ltc_filter.aux')
        self.ltc_filter = write_ltc_filter_aux_file(ltc_file)

        # Process the aux file.
        self.saw.ProcessAuxFile(ltc_file)

        # Get LTC data.
        self.ltc_init_data = self.saw.GetParametersMultipleElement(
            ObjectType='branch',
            ParamList=self.branch_key_fields + self.LTC_INIT_FIELDS,
            FilterName=self.ltc_filter
        )

        # For now, we'll only support regulators with taps from
        # -16 to 16 and a regulation range from 0.9 to 1.1.
        # TODO: Support more regulator configurations.
        if self.ltc_init_data is not None:
            self.ltc_com_data = self.ltc_init_data.copy(deep=True)
            self.num_ltc = self.ltc_init_data.shape[0]
            assert (self.ltc_init_data['XFTapPos:1'] == -16.0).all()
            assert (self.ltc_init_data['XFTapPos:2'] == 16.0).all()
            assert (self.ltc_init_data['XFRegMin'] == 0.9).all()
            assert (self.ltc_init_data['XFTapMin'] == 0.9).all()
            assert (self.ltc_init_data['XFRegMax'] == 1.1).all()
            assert (self.ltc_init_data['XFTapMax'] == 1.1).all()
        else:
            self.num_ltc = 0
            self.ltc_com_data = None

        ################################################################
        # Action space definition
        ################################################################
        # TODO: Add regulators.
        # TODO: support more regulator configurations.

        # Start by getting a listing of regulators that are in parallel.
        if self.num_ltc > 0:
            self.parallel_ltc = self.ltc_init_data.duplicated(
                ['BusNum', 'BusNum:1'])

        # Generator and shunt action space. Don't include more than one
        # generator per bus. Shunts are toggle, so only need one action
        # per shunt. +1 for no-op action.
        self.action_space = spaces.Discrete(
            int(self.num_gen_reg_buses * num_gen_voltage_bins
                + self.num_shunts + 1))

        # The gen action array is a mapping where the first row entry
        # is a generator bus number which can be used with
        # self.gen_init_data_mi and the second entry in each row is an
        # index into self.gen_bins. Note: if we need to really trim
        # memory use, the mapping can be computed on the fly rather than
        # stored in an array. For now, stick with a simple approach.
        self.gen_action_array = np.zeros(
            shape=(self.num_gen_reg_buses * num_gen_voltage_bins, 2),
            dtype=int)

        # Repeat the unique generator bus numbers in the first column of
        # the gen_action_array.
        self.gen_action_array[:, 0] = \
            np.tile(
                self.gen_init_data.loc[~self.gen_dup_reg, 'BusNum'].to_numpy(),
                num_gen_voltage_bins)

        # Create indices into self.gen_bins in the second column.
        # It feels like this could be better vectorized, but this should
        # be close enough.
        for i in range(num_gen_voltage_bins):
            # Get starting and ending indices into the array.
            s_idx = i * self.num_gen_reg_buses
            e_idx = (i + 1) * self.num_gen_reg_buses
            # Place index into the correct spot.
            self.gen_action_array[s_idx:e_idx, 1] = i

        ################################################################
        # Misc.
        ################################################################
        # Set the action cap to be double the number of generators.
        # TODO: Include other controllable assets, e.g. taps
        self._action_cap = 2 * self.num_gens + 2 * self.num_shunts

        # No-op action will be 0.
        self.no_op_action = 0

        # All done.

    @property
    def action_cap(self) -> int:
        return self._action_cap

    # Use helper functions to simplify the class definition.
    _compute_loading = _compute_loading_robust
    _compute_generation = _compute_generation_and_dispatch
    _compute_gen_v_set_points = _compute_gen_v_set_points_draw

    def _compute_branches(self):
        """Draw from all available branches."""
        if self.branch_init_data is None:
            # Nothing to do.
            return None

        # Simply draw a line index for each scenario.
        return self.rng.integers(low=0, high=self.branch_init_data.shape[0],
                                 size=self.num_scenarios)

    def _set_branches_for_scenario(self):
        """Open the line with the appropriate index."""
        # Do nothing if we have no lines to open.
        if self.branches_to_open is None:
            # Do nothing.
            return

        # Extract the line the initialization data.
        line = self.branch_init_data.iloc[
            self.branches_to_open[self.scenario_idx]][self.branch_key_fields]

        # Open the line.
        self.saw.ChangeParametersSingleElement(
            ObjectType='branch',
            ParamList=self.branch_key_fields + ['LineStatus'],
            Values=line.tolist() + ['Open']
        )

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Space]:
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
        num_obs = self.num_buses + 3 * self.num_gens + 3 * self.num_loads
        # TODO: This is wrong. Several of these values can actually go
        #   below zero.
        low = np.zeros(num_obs, dtype=self.dtype)
        # Put a cap of 2 p.u. voltage on observations - I don't see how
        # bus voltages could ever get that high.
        bus_high = np.ones(self.num_buses, dtype=self.dtype)
        if not self.scale_voltage_obs:
            bus_high += 1

        # The rest will have a maximum of 1.
        rest_high = np.ones(3 * self.num_gens + 3 * self.num_loads,
                            dtype=self.dtype)

        return num_obs, spaces.Box(
            low=low, high=np.concatenate((bus_high, rest_high)),
            dtype=self.dtype)

    def _take_action(self, action):
        """Routing method which adjusts the action and sends it to the
         correct helper method (_take_action_gens, _take_action_shunts,
         or _take_action_ltcs).
         """
        # The 0th action is a "do nothing" action.
        if action == self.no_op_action:
            return

        # Subtract one from the action to keep things simple.
        action -= 1

        if action < self.gen_action_array.shape[0]:
            # Pass the action into the gens method.
            self._take_action_gens(action)
            return

        # Adjust the action for shunts.
        action -= self.gen_action_array.shape[0]

        if action < self.num_shunts:
            # Pass the action into the shunts method.
            self._take_action_shunts(action)
            return

        # Adjust the action for LTCs.
        action -= self.num_shunts
        self._take_action_ltcs(action)
        return

    def _take_action_gens(self: DiscreteVoltageControlEnvBase, action):
        """Helper to make the appropriate updates in PowerWorld for
        generator voltage set point updates.
        """
        # Extract generator bus number and voltage bin index.
        # noinspection PyUnresolvedReferences
        gen_bus = self.gen_action_array[action, 0]
        # noinspection PyUnresolvedReferences
        voltage = self.gen_bins[self.gen_action_array[action, 1]]

        # Get generators at this bus.
        gens = self.gen_init_data_mi.loc[(gen_bus,), :]

        # Send command in to PowerWorld.
        if gens.shape[0] == 1:
            # If there's just one generator at the bus, keep it simple.
            self.saw.ChangeParametersSingleElement(
                ObjectType='gen',
                ParamList=['BusNum', 'GenID', 'GenVoltSet'],
                # Since we pulled from gen_init_data_mi only using the
                # first level of the multi-index (bus number), the
                # remaining part of the index is the generator ID.
                Values=[gen_bus, gens.index[0], voltage]
            )
        else:
            # We have multiple generators that need commanded. Use
            # ChangeParametersMultipleElement.
            self.saw.ChangeParametersMultipleElement(
                ObjectType='gen',
                ParamList=['BusNum', 'GenID', 'GenVoltSet'],
                ValueList=[[gen_bus, gen_id, voltage] for gen_id in
                           gens.index]
                )

    def _take_action_shunts(self, action):
        """Helper to make the appropriate update in PowerWorld for
        toggling switched shunts.
        """
        # Grab the current state for this shunt.
        s = self.shunt_status_arr[action]

        # Look up the new state.
        new = STATE_MAP[not s]

        # Grab the key fields for the shunt in question.
        kf = self.shunt_init_data.iloc[action][self.shunt_key_fields].tolist()

        # Send in the command.
        self.saw.ChangeParametersSingleElement(
            ObjectType='shunt',
            ParamList=self.shunt_key_fields + ['SSStatus'],
            Values=kf + [new]
        )

    # noinspection PyMethodMayBeStatic
    def _take_action_ltcs(self, action):
        return NotImplementedError()

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
            _get_observation_bus_pu_only(self),
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

    def _get_observation_failed_pf(self):
        """Use the regular _get_observation call, but zero out the
        voltages.
        """
        # Use the normal method of getting an observation.
        obs = self._get_observation()
        # Zero out the voltages.
        obs[0:self.num_buses] = 0.0
        # Done.
        return obs

    def _extra_reset_actions(self):
        """No extra reset actions needed here."""
        pass

    def _compute_end_of_episode_reward(self):
        """For now, no end of episode reward.
        """
        return 0

    def _compute_shunts(self):
        """Set up shunts."""
        if self.shunt_init_data is None:
            # If there are no shunts, there's not work to do.
            return None

        # There are shunts. Draw from the uniform distribution, and
        # consider any shunts < the prob. to be closed. So, True maps
        # to closed, False maps to open.
        return self.rng.uniform(0, 1, (self.num_scenarios, self.num_shunts)) \
            < self.shunt_closed_probability

    def _set_shunts_for_scenario(self):
        """"""
        # Don't do anything if there aren't any shunts.
        if self.shunt_init_data is None:
            return None

        # Extract a subset of the shunt data.
        shunts = self.shunt_com_data.loc[:, self.shunt_key_fields
                                         + ['SSStatus']]

        # Map the shunts.
        s = pd.Series(self.shunt_states[self.scenario_idx, :],
                      index=shunts.index)
        shunts.loc[:, 'SSStatus'] = s.map({True: 'Closed', False: 'Open'})

        # Send command to PowerWorld.
        self.saw.change_parameters_multiple_element_df(
            ObjectType='shunt', command_df=shunts)


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
    # Gen fields. See base class for comments.
    GEN_INIT_FIELDS = ['GenMW', 'GenMVR', 'GenVoltSet', 'GenMWMax',
                       'GenMWMin', 'GenMVRMax', 'GenMVRMin', 'GenStatus',
                       'GenRegNum']
    GEN_OBS_FIELDS = ['GenMW', 'GenMWMax', 'GenMVA', 'GenMVRPercent',
                      'GenStatus', 'GenVoltSet']
    GEN_RESET_FIELDS = ['GenMW', 'GenStatus']

    # Load fields.
    LOAD_INIT_FIELDS = LOAD_P + LOAD_I_Z
    LOAD_OBS_FIELDS = []
    LOAD_RESET_FIELDS = LOAD_P

    # Bus fields.
    BUS_INIT_FIELDS = []
    BUS_OBS_FIELDS = ['BusPUVolt', 'BusAngle']
    BUS_RESET_FIELDS = []

    # Branch fields.
    BRANCH_INIT_FIELDS = []
    BRANCH_OBS_FIELDS = ['LineMW', 'LineMVR']
    BRANCH_RESET_FIELDS = []
    CONTINGENCIES = False

    # By default, do not open any lines.
    LINES_TO_OPEN = None

    # Shunt fields. TODO
    SHUNT_INIT_FIELDS = []
    SHUNT_OBS_FIELDS = []
    SHUNT_RESET_FIELDS = []

    # Rewards. We'll use similar terminology to that given in the
    # paper.
    REWARDS = {
        # [0.95, 1.05] p.u.
        'normal': 100,
        # [0.8, 0.95) p.u. and (1.05, 1.25] p.u.
        'violation': -50,
        # [0.0, 0.8) p.u. and (1.25, inf) p.u.
        'diverged': -100
    }

    def __init__(self, **kwargs):
        """See parent class for parameter descriptions.
        """
        # Start by calling super constructor.
        super().__init__(**kwargs)

        ################################################################
        # Action space definition
        ################################################################
        # The GridMind action space is all possible combinations of the
        # generator voltage bins.
        self.action_space = spaces.Discrete(
            kwargs['num_gen_voltage_bins'] ** self.num_gens)

        # Being lazy, just create an action array.
        # TODO: It's silly to store this giant array in memory when you
        #   could compute the necessary permutation given an index on
        #   the fly.
        self.gen_action_array = \
            np.array(list(itertools.product(
                *[self.gen_bins for _ in range(self.num_gens)])
            ))

        ################################################################
        # Override reward methods.
        ################################################################
        self._compute_reward = self._compute_reward_gridmind
        self._compute_reward_failed_pf = \
            self._compute_reward_failed_pf_gridmind

        ################################################################
        # Misc.
        ################################################################

        # Cap the actions per episode at 15 (co-author said 10-20 would
        # be fine, so split the difference).
        self._action_cap = 15

        # All done.

    @property
    def action_cap(self) -> int:
        return self._action_cap

    _get_num_obs_and_space = _get_num_obs_and_space_v_only
    _compute_loading = _compute_loading_gridmind
    _compute_generation = _compute_generation_gridmind
    _compute_branches = _compute_branches_from_lines_to_open
    _set_branches_for_scenario = _set_branches_for_scenario_from_lines_to_open

    # After consulting with a co-author on the GridMind paper
    # (Jiajun Duan) I've confirmed that for this voltage control problem
    # the only input states are bus per unit voltage magnitudes,
    # contrary to what is listed in the paper (line flows (P and Q),
    # bus voltages (angle and magnitude)).
    _get_observation = _get_observation_bus_pu_only

    def _set_gens_for_scenario(self):
        """Since we're using PowerWorld's participation factor AGC to
        change generators, no need to actually set generation. So,
        we'll simply override this method so it does nothing.
        """
        pass

    def _compute_gen_v_set_points(self):
        """The GridMind implementation does not change generator voltage
        set points.
        """
        return None

    def _compute_reward_gridmind(self):
        """The reward structure for GridMind is pretty primitive -
        rewards for getting all buses in the normal zone, penalties if
        any buses are outside that zone. It would seem the notation
        they used in the paper is incorrect/misleading - I followed up
        with an author and found that there use of the "forall" and
        "exists" symbols really should be interpreted as "if all" and
        "if any."
        """
        # Get voltage data.
        v = self.bus_obs_data['BusPUVolt']

        # Reward if all buses are in bounds.
        if v.between(self.low_v, self.high_v, inclusive=True).all():
            reward = self.rewards['normal']
        # Penalize heavily if any voltages are in the "diverged" zone.
        elif (v <= 0.8).any() or (v >= 1.25).any():
            reward = self.rewards['diverged']
        # Otherwise, penalize for buses being in the "violation" zone.
        else:
            reward = self.rewards['violation']

        return reward

    _get_observation_failed_pf = _get_observation_failed_pf_volt_only

    def _take_action(self, action: int):
        """Send the generator set points into PowerWorld.
        """
        # Update the command df.
        self.gen_com_data.loc[:, 'GenVoltSet'] = \
            self.gen_action_array[action, :]
        self.saw.change_parameters_multiple_element_df(
            ObjectType='gen',
            command_df=self.gen_com_data.loc[:, self.gen_key_fields
                                             + ['GenVoltSet']])

    def _extra_reset_actions(self):
        """No extra reset actions needed."""
        pass

    def _compute_end_of_episode_reward(self):
        """Simply cumulative reward divided by number of actions.
        """
        return self.cumulative_reward / self.action_count

    def _compute_reward_failed_pf_gridmind(self):
        """After consulting with a co-author on the paper (Jiajun Duan)
        I've confirmed that if the power flow fails to converge, they
        simply give a single instance of the "diverged" penalty.
        """
        return self.rewards['diverged']

    def _compute_shunts(self):
        """GridMind does not deal with shunts."""
        pass


class GridMindContingenciesEnv(GridMindEnv):
    """GridMind environment, but with hard-coded contingencies. This
    is hard-coded to work only with the IEEE 14 bus case."""
    # Get line status.
    BRANCH_INIT_FIELDS = ['LineStatus']
    CONTINGENCIES = True

    # In the paper, the allowed lines are 1-5, 2-3, 4-5, and 7-9.
    LINES_TO_OPEN = LINES_TO_OPEN_14


class GridMindHardEnv(GridMindContingenciesEnv):
    """Modified GridMind environment that uses the more difficult
    loading and generation scenario generation. It also opens a random
    line - see GridMindContingenciesEnv case.
    """
    _compute_loading = _compute_loading_robust
    _compute_generation = _compute_generation_and_dispatch
    _compute_gen_v_set_points = _compute_gen_v_set_points_draw
    _set_gens_for_scenario = _set_gens_for_scenario_gen_mw_and_v_set_point


class DiscreteVoltageControlGenAndShuntNoContingenciesEnv(
        DiscreteVoltageControlEnv):
    """
    Actions: Generator set point, toggle shunt
    Observations: Bus voltage, generator state, shunt state
    Reward: Voltage movement only
    """
    # No contingencies.
    CONTINGENCIES = False

    def _compute_branches(self):
        """No contingencies, do nothing."""
        return None

    def _set_branches_for_scenario(self):
        """No contingencies, do nothing."""
        return None

    def _get_observation(self) -> np.ndarray:
        """Concatenate bus voltages, generator states, and shunt states.
        """
        # Note the voltage will only be scaled if self.scale_voltage_obs
        # is True. Otherwise, it'll be raw voltages.
        return np.concatenate(
            (self.bus_pu_volt_arr_scaled, self.gen_status_arr,
             self.shunt_status_arr)
        )

    def _get_observation_failed_pf(self):
        """Concatenate bus voltages (0 due to failed power flow),
        generator states, and shunt states.
        """
        return np.concatenate(
            (_get_observation_failed_pf_volt_only(self),
             self.gen_status_arr, self.shunt_status_arr)
        )

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Box]:
        """Number of observations is number of buses plus number of shunts,
        observation space is box for voltages and shunt states.
        """
        # Observations consist of bus per unit voltage, generator
        # states and shunt states.
        n = self.num_buses + self.num_gens + self.num_shunts
        # Low is 0 for all quantities.
        low = np.zeros(n, dtype=self.dtype)
        # High will be 2 for bus voltages, 1 for everything else.
        high = np.ones(n, dtype=self.dtype)
        if not self.scale_voltage_obs:
            high[0:self.num_buses] = 2
        return n, spaces.Box(low=low, high=high, dtype=self.dtype)


class DiscreteVoltageControlGensBranchesShuntsEnv(DiscreteVoltageControlEnv):
    """Environment with single line contingencies.

    Actions: Generator set point, toggle shunt
    Observations: Bus voltages, generator states, shunt states, and
        branch states
    Reward: Voltage movement only
    """
    # Get line states during observation.
    BRANCH_OBS_FIELDS = ['LineStatus']
    CONTINGENCIES = True

    def _get_observation(self) -> np.ndarray:
        """Concatenate bus voltages, generator states, shunt states, and
        branch states.
        """
        # Note the voltage will only be scaled if self.scale_voltage_obs
        # is True. Otherwise, it'll be raw voltages.
        return np.concatenate(
            (self.bus_pu_volt_arr_scaled, self.gen_status_arr,
             self.shunt_status_arr, self.branch_status_arr)
        )

    def _get_observation_failed_pf(self):
        """Concatenate bus voltages (0 due to failed power flow),
        generator states, shunt states, and branch states.
        """
        return np.concatenate(
            (_get_observation_failed_pf_volt_only(self),
             self.gen_status_arr, self.shunt_status_arr,
             self.branch_status_arr)
        )

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Box]:
        """Number of observations is total of number of buses, gens,
        shunts, and branches.
        """
        # Observations consist of bus per unit voltage, generator
        # states and shunt states.
        n = self.num_buses + self.num_gens + self.num_shunts \
            + self.num_branches
        # Low is 0 for all quantities.
        low = np.zeros(n, dtype=self.dtype)
        # High will be 2 for bus voltages, 1 for everything else.
        high = np.ones(n, dtype=self.dtype)
        if not self.scale_voltage_obs:
            high[0:self.num_buses] = 2
        return n, spaces.Box(low=low, high=high, dtype=self.dtype)


class DiscreteVoltageControlSimpleEnv(DiscreteVoltageControlEnv):
    """Simplified version of the DiscreteVoltageControlEnv will use
    only bus magnitudes for observations, and will only consider voltage
    movement in the reward (no generator var reserves). Additionally,
    the only lines considered for opening will come from LINES_TO_OPEN.
    """
    # Use only bus per unit voltages in the observations.
    _get_observation = _get_observation_bus_pu_only
    _get_observation_failed_pf = _get_observation_failed_pf_volt_only
    _get_num_obs_and_space = _get_num_obs_and_space_v_only

    # Draw branches to open from LINES_TO_OPEN, rather than all
    # available branches.
    _compute_branches = _compute_branches_from_lines_to_open
    _set_branches_for_scenario = _set_branches_for_scenario_from_lines_to_open


class DiscreteVoltageControlSimple14BusEnv(DiscreteVoltageControlSimpleEnv):
    """Include line contingencies to be consistent with the
    GridMindHardEnv.
    """
    # Get line status.
    BRANCH_INIT_FIELDS = ['LineStatus']

    # In the paper, the allowed lines are 1-5, 2-3, 4-5, and 7-9.
    LINES_TO_OPEN = LINES_TO_OPEN_14
    CONTINGENCIES = True


class DiscreteVoltageControlGenState14BusEnv(
        DiscreteVoltageControlSimple14BusEnv):
    """Identical to parent class, except generator states are included
    in observations.
    """
    def _get_observation(self) -> np.ndarray:
        """Observation is both bus per unit voltage and generator states.
        """
        # Initialize.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Put bus voltages in the first slots.
        # Note the voltage will only be scaled if self.scale_voltage_obs
        # is True. Otherwise, it'll be raw voltages.
        out[0:self.num_buses] = self.bus_pu_volt_arr_scaled
        # Put gen states in the remaining slots.
        out[self.num_buses:] = self.gen_status_arr
        return out

    def _get_observation_failed_pf(self) -> np.ndarray:
        """If the power flow fails to solve, return an observation of 0
        voltages, as well as the generator states.
        """
        # Initialize. No need to fill voltage slots since we'll put
        # them all at 0 for failure.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Fill appropriate slots with generator states.
        out[self.num_buses:] = self.gen_status_arr
        return out

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Box]:
        """Number of observations is the number of buses plus the number
        of generators. Observation space is a box for voltages and
        generator states.
        """
        # Voltages should never get above 2 p.u, while gen states will be
        # either 0 or 1.
        n = self.num_buses + self.num_gens
        low = np.zeros(n)
        high = np.ones(n)
        if not self.scale_voltage_obs:
            high[0:self.num_buses] = 2
        return n, spaces.Box(low=low, high=high, dtype=self.dtype)


class DiscreteVoltageControlBranchState14BusEnv(
        DiscreteVoltageControlSimple14BusEnv):
    """Identical to parent class, except branch states are included in
    observations.
    """
    # Get line states during observation.
    BRANCH_OBS_FIELDS = ['LineStatus']

    def _get_branch_state_14(self) -> np.ndarray:
        """Get line states as a vector, but hard-coded to only extract the
        lines which are opened in GridMind for the 14 bus case.
        """
        # Hard-code select the lines. Indices correspond to lines
        # 1-5, 2-3, 4-5, and 7-9. We can count on the array coming back in
        # the same order every time.
        return self.branch_status_arr[[1, 2, 6, 14]]

    def _get_observation(self) -> np.ndarray:
        # Initialize.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Fill first part with voltage.
        # Note the voltage will only be scaled if self.scale_voltage_obs
        # is True. Otherwise, it'll be raw voltages.
        out[0:self.num_buses] = self.bus_pu_volt_arr_scaled
        # Fill remaining with line states.
        out[self.num_buses:] = self._get_branch_state_14()
        # Done.
        return out

    def _get_observation_failed_pf(self) -> np.ndarray:
        # Initialize output.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Fill the branch state part, leaving the bus voltages at 0.
        out[self.num_buses:] = self._get_branch_state_14()
        return out

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Box]:
        """This function/method is specific to the 14 bus case, where
        only four lines are being opened.
        """
        # Number of buses plus the lines that could be opened.
        n = self.num_buses + len(self.LINES_TO_OPEN)
        low = np.zeros(n)
        high = np.ones(n)
        # Voltages can go above one.
        if not self.scale_voltage_obs:
            high[0:self.num_buses] = 2
        return n, spaces.Box(low=low, high=high, dtype=self.dtype)


class DiscreteVoltageControlBranchAndGenState14BusEnv(
        DiscreteVoltageControlBranchState14BusEnv):
    """Include both generator and branch states in observations."""
    def _get_observation(self) -> np.ndarray:
        # Initialize.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Fill first part with voltage.
        # Note the voltage will only be scaled if self.scale_voltage_obs
        # is True. Otherwise, it'll be raw voltages.
        out[0:self.num_buses] = self.bus_pu_volt_arr_scaled
        # Fill the line and generator states.
        return self._fill_gen_and_line_states(out)

    def _get_observation_failed_pf(self) -> np.ndarray:
        # Initialize output.
        out = np.zeros(self.observation_space.shape, dtype=self.dtype)
        # Fill line and generator portions.
        return self._fill_gen_and_line_states(out)

    def _fill_gen_and_line_states(self, arr_in):
        end_idx = self.num_buses + len(self.LINES_TO_OPEN)
        # Fill the branch state part, leaving the bus voltages at 0.
        arr_in[self.num_buses:end_idx] = self._get_branch_state_14()
        # Fill the generator state part.
        arr_in[end_idx:] = self.gen_status_arr
        return arr_in

    def _get_num_obs_and_space(self) -> Tuple[int, spaces.Box]:
        """This function/method is specific to the 14 bus case, where
        only four lines are being opened.
        """
        # Number of buses plus the lines that could be opened.
        n = self.num_buses + len(self.LINES_TO_OPEN) + self.num_gens
        low = np.zeros(n)
        high = np.ones(n)
        # Voltages can go above one.
        if not self.scale_voltage_obs:
            high[0:self.num_buses] = 2
        return n, spaces.Box(low=low, high=high, dtype=self.dtype)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class MinLoadBelowMinGenError(Error):
    """Raised if an environment's minimum possible load is below the
    minimum possible generation.
    """
    pass


class MaxLoadAboveMaxGenError(Error):
    """Raised if an environment's maximum possible load is below the
    maximum possible generation.
    """


class OutOfScenariosError(Error):
    """Raised when an environment's reset() method is called to move to
    the next episode, but there are none remaining.
    """


class ComputeGenMaxIterationsError(Error):
    """Raised when generation for a given scenario/episode cannot be
    computed within the given iteration limit.
    """