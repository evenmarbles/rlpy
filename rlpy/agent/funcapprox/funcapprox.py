from ...framework.modules import UniqueModule
from ...framework.observer import Observable
from ..mdp.state import StateDistribution


class FunctionApproximator(UniqueModule):
    """Interface for objects that construct Approximation objects."""

    class Approximation(Observable):
        """Interface for objects that approximate a given state as a weighted
        approximation of other states. Such approximations are used to approximate
        the model of a given state using data approximations from visited states,
        as well as to approximate the value of a given state using the values of
        a fixed finite set of states.

        Parameters
        ----------
        state : MDPState
            The state to approximate.

        """
        @property
        def state(self):
            """
            Returns
            -------
            MDPState :
                The state that this Approximation approximates.

            """
            return self._state

        @property
        def basis_weights(self):
            """
            Returns
            -------
            dict :
                A mapping of states to UNNORMALIZED weights. Each state
                reference in the mapping is non-None.

            """
            return self._weights

        def __init__(self, state):
            super(FunctionApproximator.Approximation, self).__init__()

            self._state = state
            self._weights = StateDistribution()

    # -----------------------------
    # Approximator
    # -----------------------------
    def __init__(self):
        super(FunctionApproximator, self).__init__()

    def initialize(self):
        """Prepare for a new episode."""
        pass

    def add_basis(self, state, act, succ):
        """Adds a state to the set of bases used to approximate query
        states.

        Parameters
        ----------
        state : MDPState
            The state to add
        act : MDPAction
            The action performed in that state
        succ : MDPState:
            The successor state.

        Returns
        -------
        MDPState :
            The approximated state.

        """
        pass

    def approximate(self, state, act):
        """Approximates a given state using an Approximation.

        Parameters
        ----------
        state : MDPState
            The state to approximate.
        act : MDPAction
            The action performed in that state

        Returns
        -------
        Approximation :
            The Approximation approximating state.

        """
        raise NotImplementedError
