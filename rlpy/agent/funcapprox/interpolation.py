import copy
import weakref
import numpy as np

from .funcapprox import FunctionApproximator
from ..mdp.state import MDPState


class InterpolationApproximator(FunctionApproximator):
    """An Approximator that uses multilinear interpolation to approximate
    states. It maps each query state to a scaled Euclidean space, where the
    query is approximated using a uniform grid with spacing equal to
    2^resolutionfactor points per unit distance. Where d is the number of
    dimensions, each query is approximated as the Approximation of exactly
    2^d points in this grid. Note that each of these points has one of exactly
    two values for each dimension. For each of these points, the weight is the
    product of the weights for the d linear interpolations of each dimension of
    the query using the two grid values in that dimension.

    Note that this Approximator generates and maintains its own set of basis
    vectors. It allocates the vectors lazily. These vectors will only have
    nonzero values for those indices specified by the dimensions parameter to
    the constructor.

    This Approximator generates Approximation objects with the property that all
    the weights sum to 1.

    Parameters
    ----------
    resolutionfactor : float
        Controls the spacing of the uniform grid used as the basis set
    scale : list[float]
        The weights used to obtain the scaled Euclidean space. Unused
        dimensions are set to 0 in the basis vectors.

    """
    class Approximation(FunctionApproximator.Approximation):
        """Approximation used by InterpolationApproximator.

        Parameters
        ----------
        approximator : InterpolationApproximator
            The InterpolationApproximator asked to approximate state
        state : MDPState
            The state to approximate.
        is_basis : bool
            Whether state is a basis state.

        """
        def __init__(self, approximator, state, is_basis):
            super(InterpolationApproximator.Approximation, self).__init__(state)

            self._approximator = approximator
            """:type: InterpolationApproximator"""

            self._is_basis = is_basis
            """:type: bool"""

            self._compute_weights()

        def __del__(self):
            if not self._is_basis:
                assert self.state in self._approximator._states
                # not a basis safe to delete entirely
                self._approximator._states.remove(self.state)

        def set_basis(self):
            self._is_basis = True

        def _compute_weights(self):
            """Compute the approximation weights from the set of basis
            neighbors, which are pruned according to the
            InterpolationApproximator's parameters.

            """
            scale = np.asarray(self._approximator._scale)
            res = self._approximator._res
            dim = len(scale)

            increment = np.exp2(-res)

            element = self.state * scale
            intermediate = np.floor(element * np.exp2(res))
            floor = intermediate * np.exp2(-res)
            alpha = (element - floor) / increment

            numsucc = 1 << dim
            for bitvector in range(numsucc):
                successor = np.zeros(dim)
                weight = 1.0
                for i, s in enumerate(scale):
                    if 0 == (bitvector & (1 << i)):
                        successor[i] = floor[i] / s
                        weight *= 1.0 - alpha[i]
                    else:
                        successor[i] = (floor[i] + increment) / s
                        weight *= alpha[i]

                if weight > 0:
                    succ = MDPState.create(successor)
                    try:
                        approx = self._approximator._queries[succ]
                    except KeyError:
                        self._approximator._states.add(succ)
                    else:
                        # make sure that any existing Approximation approximating
                        # successor knows that successor is a basis state
                        approx.set_basis()

                    self._weights[succ] = (weight, weight)

    # -----------------------------
    # InterpolationApproximator
    # -----------------------------
    def __init__(self, resolutionfactor, scale):
        super(InterpolationApproximator, self).__init__()

        self._res = resolutionfactor
        self._scale = scale

        #: Invariant: contains all the existing InterpolationApproximations
        #: that InterpolationApproximator has created. the keys include both
        #: bases and queries, so some keys have a None value.
        self._queries = weakref.WeakValueDictionary()
        """:type: dict[MDPState, Approximation]"""
        #: Invariant: contains all the InterpolationApproximations that
        #: InterpolationApproximator has created.
        self._states = set()
        """:type: set[MDPState]"""

    def approximate(self, state, *args, **kwargs):
        """Approximates a given state using an Approximation.

        Parameters
        ----------
        state : MDPState
            The state to approximate.

        Returns
        -------
        Approximation :
            The Approximation approximating state.

        """
        try:
            approx = self._queries[state]
        except KeyError:
            was_inserted = True
            if state in self._states:
                was_inserted = False
            self._states.add(copy.deepcopy(state))

            approx = InterpolationApproximator.Approximation(self, state, not was_inserted)
            self._queries[state] = approx
        return approx
