import weakref
import numpy as np

from ...framework.observer import Observable, Listener
from .primitive import Primitive, MDPPrimitive

__all__ = ['MDPAction']


class MDPAction(MDPPrimitive):
    """

    """
    class StateActionData(object):
        """

        """
        def __init__(self):
            self.count = 0
            self.cumulative_reward = 0
            self.effect_counts = {}
            """:type: dict[Primitive, int]"""
            self.observers = weakref.WeakSet()
            """:type: set[StateActionModel]"""

    class StateActionModel(Observable, Listener):
        """

        """
        _instance = object()

        @property
        def state(self):
            return self._approx.state

        @property
        def reward(self):
            return self._parent._maxval if self._sum < self._parent._threshold else self._reward

        @property
        def effects_map(self):
            """

            Returns
            -------
            dict[Primitive, double]

            """
            return {} if self._sum < self._parent._threshold else self._effects_map

        def __init__(self, token, parent, approx):
            if token is not self._instance:
                raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))

            super(MDPAction.StateActionModel, self).__init__()

            self._parent = parent
            """:type: MDPAction"""
            self._approx = approx
            """:type: Approximation"""

            self._sum = 0.0
            self._reward = 0.0

            self._translation = {}
            """:type: dict[MDPState, StateActionData]"""
            self._effects_map = {}
            """:type: dict[Primitive, double]"""

        def __repr__(self):
            return self._mid

        @classmethod
        def create(cls, parent, approx):
            result = cls(cls._instance, parent, approx)
            parent._models[approx.state] = result
            result._approx.subscribe(result, 'average_change')
            result.compute_model()
            return result

        def notify(self, event):
            if event.name == 'average_change':
                self._parent._inbox.add(self)

        def compute_model(self):
            self._sum = 0.0
            self._reward = 0.0
            self._effects_map = {}

            instances = self._approx.basis_weights
            sum_ = 0.0

            # remove erstwhile instances
            for s, sad in self._translation.items():
                if s not in instances.keys():
                    sad.observers.discard(self)
                    del self._translation[s]

            for s, (w, proba) in instances.iteritems():
                try:
                    data = self._translation[s]
                except KeyError:
                    data = self._parent._get_data(s)
                    data.observers.add(self)
                    self._translation[s] = data

                self._sum += w * data.count

                sum_ += proba * data.count
                self._reward += proba * data.cumulative_reward

                for delta_s, cnt in data.effect_counts.iteritems():
                    self._effects_map[delta_s] = self._effects_map.get(delta_s, 0.0) + (proba * cnt)

            # normalize
            try:
                self._reward /= sum_
            except ZeroDivisionError:
                self._reward = np.NaN
            for delta_s in self._effects_map.iterkeys():
                self._effects_map[delta_s] /= sum_

            if self._sum > self._parent._threshold:
                self.dispatch('model_change')

    # -----------------------------
    # MDPAction
    # -----------------------------
    @property
    def approximator(self):
        return self._approximator

    def __init__(self, token, features, threshold, maxval, approximator, name=None):
        super(MDPAction, self).__init__(token, features, name)

        self._threshold = threshold
        self._maxval = maxval

        self._approximator = approximator
        """:type: Approximator"""
        self._precondition = None

        self._data = {}
        """:type: dict[MDPState, StateActionData]"""
        self._models = weakref.WeakValueDictionary()
        """:type: dict[MDPState, StateActionModel]"""

        self._inbox = weakref.WeakSet()
        """:type: set[StateActionModel]"""

    # noinspection PyMethodOverriding
    @classmethod
    def create(cls, features, threshold, maxval, modelapproximator, name=None, feature_limits=None):
        features = cls._process_parameters(features, feature_limits)
        return cls(cls._instance, features, threshold, maxval, modelapproximator, name)

    def available(self, state):
        return True

    def terminal(self, state):
        return True

    def policy(self, state):
        return None

    def model(self, state):
        """

        Parameters
        ----------
        state

        Returns
        -------
        StateActionModel

        """
        approx = self._approximator.approximate(state, self)

        try:
            m = self._models[approx.state]
        except KeyError:
            m = MDPAction.StateActionModel.create(self, approx)
            assert state in self._models
            assert m == self._models[state]
        return m

    def propagate_changes(self):
        for i in self._inbox:
            i.compute_model()
        self._inbox.clear()

    def initialize(self):
        self._approximator.initialize()

    def update(self, state, reward, succ=None):
        """Incorporate an experience to update the model

        Parameters
        ----------
        state : MDPState
            The state at which the MDP action was executed
        reward : float
            The immediate reward obtained
        succ : MDPState, optional
            The successor state observed

        """
        dat = self._get_data(self._approximator.add_basis(state, self, succ))
        if succ is not None:
            effect = Primitive(succ - state)
            dat.effect_counts[effect] = dat.effect_counts.get(effect, 0) + 1

        dat.count += 1
        dat.cumulative_reward += reward

        for o in dat.observers:
            self._inbox.add(o)

    def _get_data(self, state):
        """Fetches or creates a data structure for recording the observed
        data for a given state (with this MDP action).

        Parameters
        ----------
        state : MDPState
            A state whose data to fetch.

        Returns
        -------
        StateActionData :
            A StateActionData that contains all known data at the given
            state for this MDP action.

        """
        return self._data.setdefault(state, MDPAction.StateActionData())
