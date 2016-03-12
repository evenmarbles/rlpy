from ...framework.observer import Listener
from ...auxiliary.datastructs import PriorityQueue
from .planner import Planner


class PrioritizedSweeping(Planner):
    """

    """
    class DecisionState(Planner.DecisionState):
        """

        """
        class Action(Planner.DecisionState.Action, Listener):
            """

            """
            def __init__(self, token, parent, model):
                super(PrioritizedSweeping.DecisionState.Action, self).__init__(token, parent, model)

                self._parent = parent
                """:type: PrioritizedSweepingStateAction"""

            def notify(self, event):
                super(PrioritizedSweeping.DecisionState.Action, self).notify(event)
                self._parent._child_bound(self, self._errorbound)

        # -----------------------------
        # DecisionState
        # -----------------------------
        def __init__(self, token, planner, model):
            super(PrioritizedSweeping.DecisionState, self).__init__(token, planner, model)

            self._planner = planner
            """:type: PrioritizedSweeping"""
            self._heapindex = -1

        def __del__(self):
            super(PrioritizedSweeping.DecisionState, self).__del__()
            assert self._heapindex < 0

        def zero_bound(self):
            """Reset the bound on Bellman error to zero. May only be called
            if this DecisionState is at the head of the priority queue.
            Removes this DecisionState from the priority queue.

            """
            assert self._heapindex == 0
            self._errorbound = 0.0
            self._planner._pqueue.pop()

        def _child_bound(self, child, bound):
            """Informs this DecisionState that one of its child Action
            objects updated its error bound. This may change the error
            bound for this DecisionState, possibly adding this DecisionState
            to the planner's priority queue or increasing its priority in the queue.

            Parameters
            ----------
            child : PrioritizedSweepingDecisionStateAction
                An action object in the DecisionState object's action_value container
            bound : float
                The error bound for this child

            Returns
            -------

            """
            if child != self._max.dsa:
                bound -= self.value - child.q
            if self._errorbound < bound:
                self._errorbound = bound
                if self._errorbound > self._planner._epsilon:
                    if self._heapindex < 0:
                        self._planner._pqueue.push(self, self._errorbound)

    # -----------------------------
    # PrioritizedSweeping
    # -----------------------------
    def __init__(self, mdp, terminal, goal, gamma=None, epsilon=None):
        super(PrioritizedSweeping, self).__init__(mdp, terminal, goal, gamma)

        self._epsilon = epsilon if epsilon is not None else 0.01

        self._pqueue = PriorityQueue()
        """:type: PriorityQueue : This binary heap contains all the DecisionState
        objects created by the Planner that have (Bellman) error greater than
        epsilon. This error is used to prioritize value backups, which eliminate
        this error."""

    def _propagate_changes(self):
        while len(self._pqueue) > 0:
            ds = self._pqueue.front()
            assert ds
            ds.zero_bound()
            ds.propagate_value_change()
