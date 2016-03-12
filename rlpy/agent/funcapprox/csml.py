from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import time
import bisect
import weakref
import numpy as np
from hmmlearn.hmm import GaussianHMM

from .funcapprox import FunctionApproximator
from ..mdp.state import MDPState

from ...libs import classifier


class CsmlApproximator(FunctionApproximator):
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
            super(CsmlApproximator.Approximation, self).__init__(state)

            self._approximator = approximator
            """:type: KernelApproximator"""

            self._kernelfn = kernelfn
            self._sum = 0.0

            #: A sorted array of the basis states near the given state, sorted
            #: in ascending order of distance. should contain all the states
            #: within self._approximator._maxd of the given state, except the
            #: ones pruned according to self._approximator._minfraction
            self._neighbors = [(d, MDPState.create(s), self._approximator._bases[MDPState.create(s)]) for d, s in sorted(
                self._approximator._basistree.neighbors(state.features, self._approximator._maxd), key=lambda x: x[0])]
            """:type: list[tuple[float, MDPState, ndarray]]"""

            self._compute_weights()

        def __del__(self):
            assert self.state not in self._approximator._queries
            if self.state not in self._approximator._bases:
                self._approximator._querytree.remove(self._state.features)

        def include(self, d, state, delta):
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

            val = (d, state, delta)
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
            total = 0

            i = 0
            for d, succ, delta in self._neighbors:
                w = self._kernelfn(d)
                if self._sum == 0.0 or w / self._sum >= self._approximator._minfraction:
                    sequence = [np.asarray(self._state.features), np.asarray(self._state.features + delta)]

                    proba = np.exp(self._approximator._hmm.score(sequence))
                    self._weights[succ] = (w, proba)
                    self._sum += w
                    total += proba
                    i += 1
                else:
                    break
            del self._neighbors[i:]
            assert len(self._weights) == len(self._neighbors)

            for succ, (w, p) in self._weights.iteritems():
                self._weights[succ] = (w, p / total)
            pass

    # -----------------------------
    # CsmlApproximator
    # -----------------------------
    def __init__(self, minfraction, maxd, scale, kernelfn, ncomponents=1, n_iter=1, inclusionfn=None):
        super(CsmlApproximator, self).__init__()

        self._minfraction = minfraction
        self._maxd = maxd
        self._scale = scale
        self._kernelfn = kernelfn
        self._inclusionfn = inclusionfn
        self._new_sequence = True

        #: Contains all the existing KernelAppoximations created by
        #: this KernelApproximator. The keys serve as both queries and
        #: bases (queries are a superset of bases), so a datum may be
        #: None if the associated key is just a basis, not a query.
        self._queries = weakref.WeakValueDictionary()
        """:type: dict[MDPState, Approximation]"""
        #: The subset of keys of queries that are also bases.
        self._bases = {}
        """:type: dict[MDPState, ndarray]"""
        self._fit_X = []
        """:type: list[ndarray]"""

        #: Invariant: contains all the keys in queries
        self._querytree = classifier.CoverTree(np.asarray(self._scale))
        """:type: CoverTree"""
        #: Invariant: for each state given to add_basis as an argument,
        #: contains exactly one reference to a state equivalent on the
        #: dimensions used by this Approximator.
        self._basistree = classifier.CoverTree(np.asarray(self._scale))
        """:type: CoverTree"""
        #: The hidden Markov model maintaining the observations in the form
        #:     seq = <s_{i}, s_{i+1}>
        #: to reason on the transition probabilities of successor states.
        self._hmm = GaussianHMM(ncomponents, n_iter=n_iter)  # , covariance_type='full'
        """:type: GaussianHMM"""

        self._num_bases = 0
        self._not_add_bases = 0
        self._not_add_count = 0

    def initialize(self):
        """Prepare for a new episode."""
        self._new_sequence = True

    def add_basis(self, state, act, succ=None):
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
        if succ is None:
            succ = state

        # update the hmm with the new sequence
        self._fit_hmm(state, succ)

        query = MDPState.create(self._querytree.insert(state.features))

        if query not in self._bases:
            delta = succ - state
            if self._inclusionfn is None or self._inclusionfn(self, query, delta):
                # new basis
                self._bases[query] = delta
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
                        approx.include(d, basis, delta)
            else:
                self._not_add_count += 1
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
            approx = CsmlApproximator.Approximation(self, query, self._kernelfn)
            self._queries[query] = approx
        return approx

    def _fit_hmm(self, state, succ):
        if self._new_sequence:
            self._new_sequence = False
            self._fit_X.append([])
            self._fit_X[-1].append(state.features)

        self._fit_X[-1].append(succ.features)
        self._hmm.fit(np.concatenate(self._fit_X), lengths=[len(x) for x in self._fit_X])
