from __future__ import division, print_function, absolute_import

import weakref
import numpy as np
from collections import OrderedDict

from ...framework.observer import Listener, Observable
from .primitive import MDPPrimitive


class StateDistribution(object):
    """Probability Distribution.

    This class handles evaluation of empirically derived states and
    calculates the probability distribution from them.

    """
    def __init__(self):
        self._states = OrderedDict()
        """:type: dict[MDPState, tuple[float]]"""

    # def __getstate__(self):
    #     return {
    #         "_states": self._states,
    #         "_proba_calc_method": {
    #             "module": self._proba_calc_method.__class__.__module__,
    #             "name": self._proba_calc_method.__class__.__name__
    #         }
    #     }
    #
    # def __setstate__(self, d):
    #     for name, value in d.iteritems():
    #         if name == "_proba_calc_method":
    #             module = __import__(value["module"])
    #             try:
    #                 value = getattr(module, value["name"])()
    #             except:
    #                 path = value["module"].split(".")
    #                 mod = "module"
    #                 for i, ele in enumerate(path):
    #                     if i != 0:
    #                         mod += '.'
    #                         mod += ele
    #                 value = getattr(eval(mod), value["name"])()
    #
    #         setattr(self, name, value)
    #
    #     self._dirty = False

    def __repr__(self):
        return repr(self._states)

    def __len__(self):
        return len(self._states)

    def __getitem__(self, state):
        return self._states.get(state, 0.)

    def __setitem__(self, state, proba):
        self._states[state] = proba

    def __delitem__(self, state):
        del self._states[state]

    def clear(self):
        """Clear the probability distribution."""
        self._states.clear()
        return self._states

    def pop(self, k, d=None):
        return self._states.pop(k, d)

    def keys(self):
        return self._states.keys()

    def values(self):
        return self._states.values()

    def items(self):
        """Retrieve the probability distribution.

        Returns
        -------
        dict[MDPState, float] :
            A list of probabilities for all possible transitions.
        """
        return self._states.items()

    def iterkeys(self):
        return self._states.iterkeys()

    def itervalues(self):
        return self._states.itervalues()

    def iteritems(self):
        return self._states.iteritems()

    def __iter__(self):
        return iter(self._states)

    def __contains__(self, item):
        return item in self._states

    def sample(self):
        """Returns a successor state according to the probability distribution.

        Returns
        -------
        MDPState :
            The next state sampled from the probability distribution.
        """
        keys = self._states.keys()

        if not keys:
            raise UserWarning("No initial states defined.")

        idx = np.random.choice(range(len(keys)), p=[v for v in self._states.values()])
        return keys[idx]


class MDPState(MDPPrimitive):
    """

    """
    class StateAction(Listener):
        """The models interface.

        Contains all relevant information predicted by a model for a
        given state-action pair. This includes the (predicted) reward and
        transition probabilities to possible next states.

        Attributes
        ----------

        """
        _instance = object()

        @property
        def reward(self):
            """float: The reward"""
            return self._model.reward

        @property
        def successor_probabilities(self):
            """StateDistribution: The state probability distribution"""
            return self._successor_proba

        def __init__(self, token, parent, action, model):
            if token is not self._instance:
                raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))

            super(MDPState.StateAction, self).__init__()

            self._parent = parent
            """:type: StateData"""
            self._action = action
            """:type: MDPAction"""

            self._model = model
            """:type: StateActionModel"""
            self._approx = {}
            """:type: dict[Primitive, Approximation]"""

            self._successor_proba = StateDistribution()

        def __getstate__(self):
            return self.__dict__.copy()

        def __setstate__(self, d):
            self.__dict__.update(d)

        @classmethod
        def create(cls, parent, action, model):
            result = cls(cls._instance, parent, action, model)
            parent._state_action_map[action] = result
            result._model.subscribe(result, 'model_change')
            result.compute_successors()
            return result

        def notify(self, event):
            if event.name == 'model_change' or event.name == 'average_change':
                self._parent._mdp._inbox.add(self)

        def compute_successors(self):
            state = self._parent.state

            self._successor_proba.clear()

            effects = self._model.effects_map

            # remove erstwhile approximations
            for delta_s, approx in self._approx.items():
                if delta_s not in effects:
                    if approx is not None:
                        approx.unsubscribe(self, 'average_change')
                    del self._approx[delta_s]

            for delta_s, model_weight in effects.iteritems():
                # add a new approximation
                if delta_s not in self._approx:
                    successor = MDPState.create(delta_s + state)
                    succ_approx = None
                    if successor != state:
                        # not a self transition, so approximate
                        succ_approx = self._parent._mdp._approximator.approximate(successor)
                        succ_approx.subscribe(self, 'average_change')
                    self._approx[delta_s] = succ_approx

                # Add the translated successors
                approx = self._approx.get(delta_s, None)
                if approx is None:
                    # effect was a self transition, use parent's state
                    self._successor_proba[self._parent.state] += model_weight
                else:
                    for s, (_, value_weight) in approx.basis_weights.iteritems():
                        self._successor_proba[s] += value_weight * model_weight

            self._parent.dispatch('mdp_change', action=self._action)

    class StateData(Observable):
        """State information interface.

        Information about the state can be accessed here.

        Parameters
        ----------

        Attributes
        ----------

        """
        _instance = object()

        @property
        def state(self):
            return self._approx.state

        @property
        def state_actions(self):
            return self._state_action_map

        def __init__(self, token, mdp, approx):
            if token is not self._instance:
                raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))

            super(MDPState.StateData, self).__init__()

            self._mdp = mdp
            """:type: MDP"""

            self._approx = approx
            """:type: Approximation"""
            self._state_action_map = OrderedDict()
            """:type: dict[MDPAction, StateAction]"""

        def __getstate__(self):
            return self.__dict__.copy()

        def __setstate__(self, d):
            self.__dict__.update(d)

        @classmethod
        def create(cls, mdp, approx):
            result = cls(cls._instance, mdp, approx)
            state = approx.state
            mdp._state_map[state] = result
            for act in mdp.actions:
                if act.available(state):
                    sa = MDPState.StateAction.create(weakref.proxy(result), act, act.model(state))
                    assert act in result._state_action_map
                    assert sa == result._state_action_map[act]
            return result

    # -----------------------------
    # MDPState
    # -----------------------------
    def __init__(self, token, features, name=None):
        super(MDPState, self).__init__(token, features, name)

    @classmethod
    def create(cls, features, name=None, feature_limits=None):
        features = cls._process_parameters(features, feature_limits)
        return cls(cls._instance, features, name)

    def encode(self):
        # noinspection PyUnresolvedReferences,PyUnusedLocal
        """Encodes the state into a human readable representation.

        Returns
        -------
        ndarray :
            The encoded state.

        Notes
        -----
        Optionally this method can be overwritten at runtime.

        Examples
        --------
        >>> def my_encode(self)
        ...     pass
        ...
        >>> MDPState.encode = my_encode

        """
        return self._features
