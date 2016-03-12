import bisect
import weakref
import numpy as np

from .funcapprox import FunctionApproximator
from ..mdp.state import MDPState

from ...libs import classifier


class KernelApproximator(FunctionApproximator):
    """An Approximator that uses Gaussian kernels to approximate states. In
    particular, for any given query state s, it assigns to each basis state x
    the weight by the kernel function. Note that all weights therefore lie
    in the interval (0,1]. For efficiency reasons, a weight is set to zero if
    it falls below a certain threshold or if it is smaller than a certain
    fraction of the sum of the larger weights. (The tail of the distribution
    is truncated since it has a small effect on the approximation.)

    Parameters
    ----------
    kernelfn : callable
        The kernel function used for weight assignment.
    minfraction : float
        Weights smaller than this fraction of the sum of larger
        weights are set to zero.
    maxd : float
        The maximum allowed distance for neighboring states.
    scale : list[float]
        The weights used to obtain the scaled Euclidean space. Unused
        dimensions are set to 0 in the basis vectors.

    """

    class Approximation(FunctionApproximator.Approximation):
        """Approximation used by KernelApproximator.

        Parameters
        ----------
        approximator : KernelApproximator
            The KernelApproximator asked to approximate state
        state : MDPState
            The state to approximate.
        kernelfn : callable
            The kernel function used for weight assignment.

        """

        def __init__(self, approximator, state, kernelfn):
            super(KernelApproximator.Approximation, self).__init__(state)

            self._approximator = approximator
            """:type: KernelApproximator"""

            self._kernelfn = kernelfn

            self._sum = 0.0

            #: A sorted array of the basis states near the given state, sorted
            #: in ascending order of distance. should contain all the states
            #: within self._approximator._maxd of the given state, except the
            #: ones pruned according to self._approximator._minfraction
            self._neighbors = [(d, MDPState.create(s)) for d, s in
                               sorted(self._approximator._basistree.neighbors(state.features, self._approximator._maxd),
                                      key=lambda x: x[0])]
            """list[tuple[float, ndarray]]"""

            self._compute_weights()

        def __del__(self):
            assert self.state not in self._approximator._queries
            if self.state not in self._approximator._bases:
                self._approximator._querytree.remove(self._state.features)

        def include(self, d, state):
            """Potentially update this KernelApproximation to include
            a new basis.

            Parameters
            ----------
            d : float
                The distance from state to this KernelApproximation's state
                according to the KernelApproximator's Distance function
            state : MDPState
                The new basis state.

            """
            assert d >= 0.0

            val = (d, state)
            if len(self._neighbors) <= 0 or val < self._neighbors[-1]:
                bisect.insort(self._neighbors, val)
                self._compute_weights()
                self.dispatch('average_change')
            else:
                assert self._sum > 0.0
                w = self._kernelfn(d)
                if w / self._sum >= self._approximator._minfraction:
                    self._neighbors.append(val)
                    self._compute_weights()
                    self.dispatch('average_change')

        def _compute_weights(self):
            """Compute the approximation weights from the set of basis
            neighbors, which are pruned according to the
            KernelApproximator's parameters.

            """
            self._weights.clear()
            self._sum = 0.0

            i = 0
            for d, succ in self._neighbors:
                w = self._kernelfn(d)
                if self._sum == 0.0 or w / self._sum >= self._approximator._minfraction:
                    self._weights[succ] = (w, w)
                    self._sum += w
                    i += 1
                else:
                    break
            print(i)
            del self._neighbors[i:]
            assert len(self._weights) == len(self._neighbors)

            for succ, (w, p) in self._weights.iteritems():
                self._weights[succ] = (w, p / self._sum)
            pass

    # -----------------------------
    # KernelApproximator
    # -----------------------------
    def __init__(self, minfraction, maxd, scale, kernelfn, inclusionfn=None):
        super(KernelApproximator, self).__init__()

        self._minfraction = minfraction
        self._maxd = maxd
        self._scale = scale
        self._kernelfn = kernelfn
        self._inclusionfn = inclusionfn

        #: Contains all the existing KernelAppoximations created by
        #: this KernelApproximator. The keys serve as both queries and
        #: bases (queries are a superset of bases), so a datum may be
        #: None if the associated key is just a basis, not a query.
        self._queries = weakref.WeakValueDictionary()
        """:type: dict[MDPState, Approximation]"""
        #: The subset of keys of queries that are also bases.
        self._bases = set()
        """:type: set[MDPState]"""

        #: Invariant: contains all the keys in queries
        self._querytree = classifier.CoverTree(np.asarray(self._scale))
        """:type: CoverTree"""
        #: Invariant: for each state given to add_basis as an argument,
        #: contains exactly one reference to a state equivalent on the
        #: dimensions used by this Approximator.
        self._basistree = classifier.CoverTree(np.asarray(self._scale))
        """:type: CoverTree"""
        self._num_bases = 0
        self._not_add_bases = 0

    def add_basis(self, state, *args, **kwargs):
        """Adds a state to the set of bases used to approximate query
        states.

        Parameters
        ----------
        state : MDPState
            The state to add

        Returns
        -------
        MDPState :
            The approximated state.

        """
        query = MDPState.create(self._querytree.insert(state.features))

        if query not in self._bases:
            if self._inclusionfn is None or self._inclusionfn(self, query):
                # new basis
                self._bases.add(query)
                basis = MDPState.create(self._basistree.insert(query.features))
                assert basis == query
                self._num_bases += 1

                neighbors = self._querytree.neighbors(query.features, self._maxd)

                for d, s in neighbors:
                    try:
                        approx = self._queries[MDPState.create(s)]
                    except KeyError:
                        pass
                    else:
                        approx.include(d, basis)
        else:
            self._not_add_bases += 1

        return query

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
        query = MDPState.create(self._querytree.insert(state.features))
        try:
            approx = self._queries[query]
        except KeyError:
            approx = KernelApproximator.Approximation(self, query, self._kernelfn)
            self._queries[query] = approx
        return approx
