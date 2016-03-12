import weakref

from .state import MDPState


class MDP(object):
    """

    """
    @property
    def actions(self):
        return self._actions

    def __init__(self, actions, approximator):
        self._actions = actions
        """:type: list[MDPAction]"""
        self._approximator = approximator
        """:type: Approximator"""

        self._state_map = weakref.WeakValueDictionary()
        """:type: dict[MDPState, StateData]"""

        self._inbox = weakref.WeakSet()
        """:type: set[StateAction]"""

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)

    def initialize(self):
        for act in self._actions:
            act.approximator.initialize()
        self._approximator.initialize()

    def update(self, state, act, succ):
        self._approximator.add_basis(state, act, succ)

    def state_data(self, state):
        approx = self._approximator.approximate(state)
        try:
            state_data = self._state_map[approx.state]
        except KeyError:
            state_data = MDPState.StateData.create(self, approx)
            assert state in self._state_map
            assert state_data == self._state_map[state]
        return state_data

    def propagate_changes(self):
        for i in self._inbox:
            i.compute_successors()
        self._inbox.clear()
