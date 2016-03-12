import weakref

from .planner import Planner


class ValueIteration(Planner):

    class DecisionState(Planner.DecisionState):

        def __init__(self, token, planner, model):
            super(ValueIteration.DecisionState, self).__init__(token, planner, model)

        def backup_values(self):
            """Performs a value update for each action

            Returns
            -------
            float :
                The change in value for this state.

            """
            oldvalue = self.value

            original_action = self._max.action

            it = self._action_values.iteritems()

            act, dsa = it.next()
            dsa.compute_value()
            self._max = Planner.DecisionState.MaxAction(act, dsa)

            while True:
                try:
                    act, dsa = it.next()
                    dsa.compute_value()
                    if self._max.dsa.q < dsa.q:
                        self._max = Planner.DecisionState.MaxAction(act, dsa)
                except StopIteration:
                    break

            if self._max.action != original_action:
                # this is a new policy action
                self._planner._outbox[self] = original_action

            if self._max.dsa.q != self.value:
                self._set_value(self._max.dsa.q)
            return self.value - oldvalue

    # -----------------------------
    # ValueIteration
    # -----------------------------
    def __init__(self, mdp, terminal, goal, gamma=None, epsilon=None):
        super(ValueIteration, self).__init__(mdp, terminal, goal, gamma)

        self._epsilon = epsilon if epsilon is not None else 0.01

    def _propagate_changes(self):
        while True:
            norm_residual = 0

            for i, ds in self._nonterminals:
                if ds is not None:
                    change = ds.backup_values()
                    change = -change if change < 0 else change
                    norm_residual = change if norm_residual < change else norm_residual

            if norm_residual >= self._epsilon:
                break
