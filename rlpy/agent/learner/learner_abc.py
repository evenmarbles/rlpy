from ...framework.modules import UniqueModule


class Learner(UniqueModule):
    def __init__(self, planner):
        super(Learner, self).__init__()

        self._planner = planner
        """:type: Planner"""

    def initialize(self):
        self._planner.initialize()

    def policy(self, state):
        return self._planner.policy(state)

    def update(self, state, act, succ):
        self._planner.update(state, act, succ)

    def learn(self):
        for act in self._planner.mdp.actions:
            act.propagate_changes()

        self._planner.mdp.propagate_changes()

        self._planner.plan()

    def debug(self):
        self._planner.debug()

    def write_value_function(self):
        self._planner.write_value_function()

    def write_policy(self):
        self._planner.write_policy()
