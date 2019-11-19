import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import logging
from esa import SAW
from typing import Union, Tuple


class VoltageControlEnv(gym.Env):
    """Environment for performing voltage control with the PowerWorld
    Simulator.
    """
    def __init__(self, pwb_path: str, num_scenarios: int,
                 max_load_factor: Union[str, float] = None,
                 min_load_factor: Union[str, float] = None,
                 min_load_pf: float = 0.8,
                 load_on_probability: float = 0.8,
                 num_gen_voltage_bins: int = 5,
                 gen_voltage_range: Tuple[float, float] = (0.9, 1.1),
                 seed: Union[str, float] = None):
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
        """

        # Set up log.
        self.log = logging.getLogger(self.__class__.__name__)

        # Handle random seeding up front.
        self.np_random = None  # Type: np.random.RandomState
        # Note that calling the seed function will override np_random.
        self.seed(seed)

        # Initialize a SimAuto wrapper.
        self.saw = SAW(pwb_path, early_bind=True)
        self.log.debug('PowerWorld case loaded.')

        # Get generator data.
        gen_key_field_df = self.saw.get_key_fields_for_object_type('gen')
        gen_key_fields = gen_key_field_df['internal_field_name'].tolist()
        bus_type = ['BusCat']
        gen_p_q = ['GenMW', 'GenMVR']
        volt_set = ['GenVoltSet']
        gen_limits = ['GenMWMax', 'GenMWMin', 'GenMVRMax', 'GenMVRMin']
        gen_params = (gen_key_fields + bus_type + gen_p_q + volt_set
                      + gen_limits)

        # Get the per unit voltage magnitude the generators regulate to.
        self.gen_data = self.saw.GetParametersMultipleElement(
            ObjectType='gen', ParamList=gen_params)

        # If we have negative minimum generation levels, zero them
        # out.
        gen_leq_0 = self.gen_data['GenMWMin'] < 0
        if (self.gen_data['GenMWMin'] < 0).any():
            self.gen_data.loc[gen_leq_0, 'GenMWMin'] = 0.0
            cols = gen_key_fields + ['GenMWMin']
            self.saw.change_and_confirm_params_multiple_element(
                'gen', self.gen_data.loc[:, cols])
            self.log.warning(f'{gen_leq_0.sum()} generators with '
                             'GenMWMin < 0 have had GenMWMin set to 0.')
            pass

        # Get load data.
        # TODO: Somehow need to ensure that the only active load models
        #   are ZIP.
        load_key_field_df = self.saw.get_key_fields_for_object_type('load')
        load_key_fields = load_key_field_df['internal_field_name'].tolist()
        load_p = ['LoadSMW', 'LoadSMVR']
        load_i_z = ['LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']
        load_params = load_key_fields + load_p + load_i_z
        self.load_data = self.saw.GetParametersMultipleElement(
            ObjectType='load', ParamList=load_params
        )

        # Warn if we have constant current or constant impedance
        # values. Then, zero out the constant current and constant
        # impedance portions.
        if (self.load_data[load_i_z] != 0.0).any().any():
            self.log.warning('The given PowerWorld case has loads with '
                             'non-zero constant current and constant impedance'
                             ' portions. These will be zeroed out.')
            self.load_data.loc[:, load_i_z] = 0.0
            self.saw.change_and_confirm_params_multiple_element(
                'Load', self.load_data.loc[:, load_key_fields + load_i_z])

        # Compute maximum system loading.
        if max_load_factor is not None:
            # If given a max load factor, multiply it by the current
            # system load.
            self.max_load_mw = \
                self.load_data['LoadSMW'].sum() * max_load_factor
        else:
            # If not given a max load factor, compute the maximum load
            # as the sum of the generator maximums.
            self.max_load_mw = self.gen_data['GenMWMax'].sum()

        # Compute minimum system loading.
        if min_load_factor is not None:
            self.min_load_mw = \
                self.load_data['LoadSMW'].sum() * min_load_factor
        else:
            self.min_load_mw = self.gen_data['GenMWMin'].sum()

        # Time to generate scenarios.
        # Initialize list to hold all information pertaining to all
        # scenarios.
        self.scenarios = []

        # Draw an active power loading condition for each case.
        self.scenario_total_loads_mw = self.np_random.uniform(
            self.min_load_mw, self.max_load_mw, num_scenarios
        )

        # Draw to determine which loads will be "on" for each scenario.
        self.num_loads = self.load_data.shape[0]
        loads_on_draw = self.np_random.rand(num_scenarios, self.num_loads)
        self.scenario_loads_off = loads_on_draw > load_on_probability

        # Draw initial loading levels. Loads which are "off" will be
        # removed, and then each row will be scaled.
        self.scenario_individual_loads_mw = \
            self.np_random.rand(num_scenarios, self.num_loads)

        # Zero out loads which are off.
        self.scenario_individual_loads_mw[self.scenario_loads_off] = 0.0

        # Scale each row to meet the appropriate scenario total loading.
        # First, get our vector of scaling factors (factor per row).
        scale_factor_vector = (
            self.scenario_total_loads_mw
            / self.scenario_individual_loads_mw.sum(axis=1)
        )

        # Then, multiply each element in each row by its respective
        # scaling factor. The indexing with None creates an additional
        # dimension to our vector to allow for that element-wise
        # scaling.
        self.scenario_individual_loads_mw = (
                self.scenario_individual_loads_mw
                * scale_factor_vector[:, None]
        )

        # Now, come up with reactive power levels for each load based
        # on the minimum power factor.
        pf = self.np_random.uniform(min_load_pf, 1,
                                    (num_scenarios, self.num_loads))

        # Q = P * tan(arccos(pf))
        self.scenario_individual_loads_mvar = (
            self.scenario_individual_loads_mw
            * np.tan(np.arccos(pf))
        )

        # Now, we'll take a similar procedure to set up generation
        # levels. Unfortunately, this is a tad more complex since we
        # have upper and lower limits.

        # TODO: regulators
        # TODO: shunts

        # Start by solving the power flow.
        # TODO: Handle exceptions.
        self.saw.SolvePowerFlow()

    def seed(self, seed=None):
        """Borrowed from Gym.
        https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
