from .funcapprox import FunctionApproximator
from ..knowledgerep.cbr.engine import CaseBase


class CaseBaseApproximator(FunctionApproximator):
    """

    """
    class Approximation(FunctionApproximator.Approximation):
        """

        """
        def __init__(self, approximator, state):
            super(CaseBaseApproximator.Approximation, self).__init__(state)

            self._approximator = approximator
            """:type: CaseBaseApproximator"""

            self._compute_weights()

        def _compute_weights(self):
            pass

    # -----------------------------
    # CaseBaseApproximator
    # -----------------------------
    def __init__(self, case_metadata, reuse_method=None, revision_method=None, retention_method=None,
                 plot_retrieval_method=None, plot_retrieval_params=None):
        super(CaseBaseApproximator, self).__init__()

        self._cb = CaseBase(case_metadata, reuse_method, revision_method, retention_method,
                            plot_retrieval_method, plot_retrieval_params)

    def approximate(self, state, act):
        pass
