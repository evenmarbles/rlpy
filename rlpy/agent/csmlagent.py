from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
from random import shuffle
from copy import deepcopy

from rlglued.agent.agent import Agent
from rlglued.utils.taskspecvrlglue3 import TaskSpecParser, TaskSpecError
from rlglued.types import Action

from .mdp.mdp import MDP
from .mdp.action import MDPAction
from .mdp.state import MDPState
from .funcapprox.csml import CsmlApproximator
from .funcapprox.kernel import KernelApproximator
from .funcapprox.interpolation import InterpolationApproximator
from .learner.learner_abc import Learner
from .planner.prioritizedsweeping import PrioritizedSweeping


class CSMLAgent(Agent):
    _config = {
        'model_tau': 0.02,
        'model_sigma': 0.004,
        'n_components': 2,
        'n_iter': 10,
        'value_tau': 0.03,
        # The amount of data required before considering a state-action explored, including data
        # generalized (and weighted) from nearby states.
        'explorationthreshold': 1.,
        # Factor multiplied to expected future rewards, known as gamma in the RL literature.
        'discountfactor': 1.0,
        # Terminal Bellman residual threshold for value iteration.
        # Planning terminates after each time step when the largest value change is smaller than
        # this threshold.
        'epsilon': 0.01,
        # Degree of generalization used to estimate the model for each action. A Gaussian kernel with
        # this standard deviation (in the scaled state space) weights the contribution of nearby states
        # to the model for a given action at a given state.
        'modelbreadth': 1.0 / 16.0,
        # Threshold for the weight of a transition used to approximate an action's model, as a fraction of
        # the maximum possible weight of a transition. Combined with modelbreadth, determines the maximum
        # distance over which generalization can occur. Setting this number to 1.0 would remove all
        # generalization.
        'minweight': 0.01,
        # Threshold for the weight of a transition used to approximate an action's model, as a fraction of
        # the cumulative weight of higher-weighted transitions.
        'minfraction': 0.01,
        # Controls the resolution of the evenly spaced grid used to approximate the value function.
        # For each unit distance in the scaled state space, the grid will have math:`2^(resolutionfactor)`
        # points.
        'resolutionfactor': 4,
    }

    def __init__(self, config=None):
        self._config.update(config if config is not None else {})

        self._laststate = None
        self._lastaction = None

        self._learner = None
        """Learner"""

    def __setstate__(self, d):
        self.__dict__.update(d)

    def init(self, taskspec):
        ts = TaskSpecParser(taskspec)
        if not ts.valid:
            raise TaskSpecError('TaskSpec Error: Invalid task spec version')

        _, maxval = ts.get_reward_range()

        extra = ts.get_extra()
        v = ['OBSDESCR', 'ACTDESCR', 'COPYRIGHT']
        pos = []
        for i, id_ in enumerate(list(v)):
            try:
                pos.append(extra.index(id_))
            except ValueError:
                v.remove(id_)
        sorted_v = sorted(zip(pos, v))

        act_desc = {}
        for i, (_, id_) in enumerate(sorted_v):
            val = ts.get_value(i, extra, v)
            if id_ == 'OBSDESCR':
                pass
            elif id_ == 'ACTDESCR':
                act_desc = eval(val)

        obs = ts.get_double_obs()
        dimensions = [1.0] * ts.get_num_int_obs()
        dimensions += (1.0 / np.asarray(zip(*obs)[1] - np.asarray(zip(*obs)[0]))).tolist()

        MDPState.set_feature_limits(obs)

        act_limits = ts.get_int_act()
        act_limits += ts.get_double_act()

        discrete_dim = ts.get_num_int_act()
        assert (discrete_dim > 0)
        continuous_dim = ts.get_num_double_act()
        assert (continuous_dim == 0)

        if discrete_dim > 1:
            min_ = list(zip(*act_limits)[0])
            max_ = (np.asarray(list(zip(*act_limits)[1])) + 1).tolist()
            actions = [range(*a) for a in zip(min_, max_)]

            import itertools
            act = list(itertools.product(*actions))
        else:
            act = act_limits[0][:]
            act[1] += 1

        bb = self._config['modelbreadth'] * self._config['modelbreadth']
        maxd = np.sqrt(-bb * np.log(self._config['minweight']))
        kernelfn = lambda x: np.exp(-x * x / bb)

        def model_inclusion(approximator, state, delta):
            try:
                approx = approximator._queries[state]
            except KeyError:
                approx = approximator.Approximation(approximator, state, approximator._kernelfn)

            do_add = True
            for _, s, d in approx._neighbors:
                delta_error = np.linalg.norm(d - delta)
                if delta_error <= self._config['model_sigma']:
                    # At least one of the cases in the case base correctly estimated the query case,
                    # the query case does not add any new information, do not add.
                    do_add = False
                    break

            do_add = do_add or approx._neighbors[0][0] > self._config['model_tau']
            return do_add

        actions = []
        for i in range(*act):
            model_approximator = CsmlApproximator(self._config['minfraction'], maxd, dimensions, kernelfn,
                                                  self._config['n_components'], self._config['n_iter'], model_inclusion)
            actions.append(
                MDPAction.create(i, self._config['explorationthreshold'], maxval, model_approximator,
                                 name=act_desc[i] if i in act_desc else None,
                                 feature_limits=act_limits))

        # shuffle(actions)
        actions = [actions[0], actions[1], actions[2]]

        # def value_inclusion(approximator, state):
        #     try:
        #         approx = approximator._queries[state]
        #     except KeyError:
        #         approx = approximator.Approximation(approximator, state, approximator._kernelfn)
        #
        #     return len(approx._weights) <= 0 or approx._neighbors[0][0] > self._config['value_tau']
        #
        # value_approximator = KernelApproximator(0.01, maxd, dimensions, lambda x: np.exp(-x ** 2), value_inclusion)
        value_approximator = InterpolationApproximator(self._config['resolutionfactor'], dimensions)
        mdp = MDP(actions, value_approximator)
        planner = PrioritizedSweeping(mdp, lambda x: False, lambda x: 0, self._config['discountfactor'],
                                      self._config['epsilon'])

        self._learner = Learner(planner)

    def start(self, observation):
        self._lastaction = None
        self._laststate = MDPState.create(list(observation.intArray) + list(observation.doubleArray))
        self._learner.initialize()
        return self.choose_action()

    def step(self, reward, observation):
        succ = MDPState.create(list(observation.intArray) + list(observation.doubleArray))

        self._learner.update(self._laststate, self._lastaction, succ)
        self._lastaction.update(self._laststate, reward, succ)
        self._learner.learn()
        self._laststate = deepcopy(succ)

        return self.choose_action()

    def end(self, reward):
        self._lastaction.update(self._laststate, reward)
        self._learner.learn()
        not_add_bases = 0
        not_add_count = 0
        num_bases = 0
        for a in self._learner._planner.mdp.actions:
            not_add_bases += a.approximator._not_add_bases
            not_add_count += a.approximator._not_add_count
            num_bases += a.approximator._num_bases
        print("not add bases: %i, not added: %i, num cases: %i" % (not_add_bases, not_add_count, num_bases))

    def cleanup(self):
        pass

    def message(self, msg):
        pass

    def choose_action(self):
        pi_s = self._learner.policy(self._laststate)

        lastaction = pi_s.policy_action
        pi_s = lastaction.policy(self._laststate)

        self._lastaction = lastaction
        assert self._lastaction
        # print(lastaction)

        return_action = Action()
        return_action.intArray = self._lastaction.tolist()
        return return_action
