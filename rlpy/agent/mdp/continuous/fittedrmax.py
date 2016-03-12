from ..mdp import MDP


class FittedRMax(MDP):

    def __init__(self, actions, approximator):
        super(FittedRMax, self).__init__(actions, approximator)
