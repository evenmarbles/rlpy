from ..mdp import MDP


class Casml(MDP):
    def __init__(self, actions, approximator):
        super(Casml, self).__init__(actions, approximator)
