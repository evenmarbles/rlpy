from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import weakref
import bisect
import numpy as np
from hmmlearn.hmm import GaussianHMM

from .funcapprox import FunctionApproximator
from ..knowledgerep.cbr.engine import CaseBase
from ..knowledgerep.cbr.methods import RetentionMethod
from ..mdp.state import MDPState
from ...auxiliary.misc import Hashable
# from ...stats.dbn.hmm import GaussianHMM


class CasmlApproximator(FunctionApproximator):
    """

    """

    class _RetentionMethod(RetentionMethod):
        """The retention method for the transition case base implementation for :class:`Casml`.

        When the new problem-solving experience can be stored or not stored in memory,
        depending on the revision outcomes and the CBR policy regarding case retention.

        Parameters
        ----------
        owner : CaseBase
            A pointer to the owning case base.
        tau : float, optional
            The maximum permitted error when comparing most similar solution.
            Default is 0.8.
        sigma : float, optional
            The maximum permitted error when comparing actual with estimated
            transitions. Default is 0.2
        plot_retention_method : callable, optional
            Callback function plotting the retention step. Default is None.

        Notes
        -----
        The Casml retention method for the transition case base considers query cases as
        predicted correctly if both:

        1. the difference between the actual and the estimated transitions are less
           than or equal to the permitted error :math:`\\sigma`:

           .. math::

              d(\\text{case}.\\Delta_\\text{state}, T(s_{i-1}, a_{i-1}) <= \\sigma

        2. and the query case is within the maximum permitted error :math:`\\tau` of
           the most similar solution case:

           .. math::

              d(\\text{case}, 1\\text{NN}(C_T, \\text{case})) <= \\tau

        """

        def __init__(self, owner, tau=None, sigma=None, plot_retention_method=None):
            super(CasmlApproximator._RetentionMethod, self).__init__(owner, plot_retention_method,
                                                                     {'tau': tau, 'sigma': sigma})

            self._tau = tau if tau is not None else 0.8
            """:type: float"""

            self._sigma = sigma if sigma is not None else 0.2
            """:type: float"""

        def execute(self, features, matches, plot=True):
            """Execute the retention step.

            Parameters
            ----------
            features : list[tuple[str, ndarray]]
                A list of features of the form (`feature_name`, `data_points`).
            matches : dict[str, dict[int, tuple[float, ndarray]]]
                The solution identified through the similarity measure.
            plot: bool, optional
                Plot the data during the retention step.

            Returns
            -------
        int :
            The case id if the case was retained, -1 otherwise.

            """
            f = dict(features)

            do_add = True
            if matches:
                for id_, val in matches['state'].iteritems():
                    delta_error = np.linalg.norm(self._owner.get_feature('delta_state', id_).value - f['delta_state'])
                    if delta_error <= self._sigma:
                        # At least one of the cases in the case base correctly estimated the query case,
                        # the query case does not add any new information, do not add.
                        do_add = False
                        break

            basis_id = -1
            if do_add or matches['state'].values()[0][0] > self._tau:
                basis_id = self._owner.insert(features, matches)

            if plot:
                self.plot_data(features, matches)

            return basis_id

    class Approximation(FunctionApproximator.Approximation):
        """

        """

        def __init__(self, approximator, state, act, kernelfn):
            super(CasmlApproximator.Approximation, self).__init__(state)

            self._act = act

            self._approximator = approximator
            """:type: CasmlApproximator"""

            self._kernelfn = kernelfn
            self._sum = 0.0

            self._neighbors = []
            """:type: list"""
            self._deltas = []
            """:type: list"""

            self.update(state.features, act.features)

        def __del__(self):
            assert (self.state, Hashable(self._act.features)) not in self._approximator._queries
            # noinspection PyTypeChecker
            # if not next((True for elem in self._approximator._fit_X if np.all(elem == self.state.features)), False):
            if (self.state, Hashable(self._act.features)) not in self._approximator._bases:
                self._approximator._querycb.remove([('state', self.state.features), ('act', self._act.features)])

        def include(self, d, state, delta):
            assert d >= 0

            val = (d, state)
            if len(self._neighbors) <= 0 or val < self._neighbors[-1]:
                # noinspection PyTypeChecker
                # if not next((True for (dist, v) in self._neighbors if dist == d and np.all(v == state)), False):
                ind = bisect.bisect_left(self._neighbors, val)
                bisect.insort(self._neighbors, val)
                self._deltas.insert(ind, delta)
                self._compute_weights()
                self.dispatch('average_change')
            else:
                assert self._sum > 0.0
                w = self._kernelfn(d)
                if w / self._sum >= self._approximator._minfraction:
                    self._neighbors.append(val)
                    self._deltas.append(delta)
                    self._compute_weights()
                    self.dispatch('average_change')

        def update(self, state, act):
            neighbors = dict(self._approximator._basiscb.retrieve([('state', state), ('act', act)]))
            if 'state' in neighbors:
                self._deltas = [self._approximator._basiscb.get_feature('delta_state', id_).value for id_ in
                                neighbors['state'].iterkeys()]
                self._neighbors = neighbors['state'].values()

            self._compute_weights()

        def _compute_weights(self):
            self._weights.clear()
            self._sum = 0.0

            i = 0
            total = 0

            # calculate successor states from the current state and solution delta state
            for (d, succ), delta in zip(self._neighbors, self._deltas):
                w = self._kernelfn(d)
                if self._sum == 0.0 or w / self._sum >= self._approximator._minfraction:
                    sequence = [np.asarray(self._state.features), np.asarray(self._state.features + delta)]

                    proba = np.exp(self._approximator._hmm.score(sequence))
                    self._weights[MDPState.create(succ)] = (w, proba)       # proba
                    self._sum += w
                    total += proba
                    i += 1
                else:
                    break
            del self._neighbors[i:]
            del self._deltas[i:]

            for succ, (w, p) in self._weights.iteritems():
                self._weights[succ] = (w, p / total)        # total
            pass

            # sequences = np.zeros((len(self._neighbors), 2, len(self._state)), dtype=float)
            #
            # for i, delta in enumerate(self._deltas):
            #     sequences[i, 0] = np.array(self._state.features)
            #     sequences[i, 1] = np.asarray(self._state.features + delta)
            #
            # # use HMM to calculate probability for observing sequence <current_state, next_state>
            # # noinspection PyTypeChecker
            # weights = np.exp(self._approximator._hmm.score(sequences))
            # for (_, succ), w in zip(self._neighbors, weights):
            #     self._weights[MDPState.create(succ)] = w
            #
            # sum_ = weights.sum()
            # for (_, succ), w in zip(self._neighbors, weights):
            #     if len(weights) <= 1:
            #         w *= 0.9
            #     self._weights[MDPState.create(succ)] = w / sum_

    # -----------------------------
    # CasmlApproximator
    # -----------------------------
    def __init__(self, feature_metadata, minfraction, scale, kernelfn, tau=None, sigma=None, ncomponents=1, n_iter=1):
        super(CasmlApproximator, self).__init__()

        self._minfraction = minfraction
        self._scale = scale
        self._kernelfn = kernelfn
        self._new_sequence = True

        #: Contains all the existing CasmlAppoximations created by
        #: this CasmlApproximator. The keys serve as both queries and
        #: bases (queries are a superset of bases), so a datum may be
        #: None if the associated key is just a basis, not a query.
        self._queries = weakref.WeakValueDictionary()
        """:type: dict[tuple[MDPState, MDPAction], Approximation]"""
        #: The subset of keys of queries that are also bases.
        #: The order in which the bases have been received is preserved
        self._bases = set()
        """:type: set[tuple[MDPState, Hashable]"""
        self._fit_X = []
        """:type: list[ndarray]"""

        #: The case base maintaining the observations in the form
        #:     c = <s, a, ds>, where ds = s_{i+1} - s_i
        #: to identify possible successor states.
        self._basiscb = CaseBase(feature_metadata,
                                 retention_method=self._RetentionMethod,
                                 retention_method_params=(tau, sigma), name='basiscb')
        """:type: CaseBase"""
        del feature_metadata['delta_state']
        #: Invariant: contains all the keys in queries
        self._querycb = CaseBase(feature_metadata, name='querycb')
        """:type: CaseBase"""
        #: The hidden Markov model maintaining the observations in the form
        #:     seq = <s_{i}, s_{i+1}>
        #: to reason on the transition probabilities of successor states.
        self._hmm = GaussianHMM(ncomponents, n_iter=n_iter)  # , covariance_type='full'
        # self._hmm = GaussianHMM(ncomponents)
        """:type: GaussianHMM"""

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
        act : MDPAction
            The action performed in that state
        succ : MDPState:
            The successor state.

        Returns
        -------
        MDPState :
            The approximated state.

        """
        # update the hmm with the new sequence
        self._fit_hmm(state, succ)

        # retain the case in the query case base
        features = [('state', state.features), ('act', act.features)]
        self._querycb.retain(features)

        a = Hashable(act.features)
        if (state, a) in self._bases:
            self._not_add_bases += 1
            return state

        self._bases.add((state, a))

        # retain the case in the basis case base
        if succ is None:
            succ = state
        delta = succ - state
        features.append(('delta_state', delta))
        basis_id = self._basiscb.run(features)

        if basis_id <= -1:
            self._not_add_count += 1

        if basis_id >= 0:
            if self._querycb.similarity_uses_knn:
                for c in self._querycb.itervalues():
                    try:
                        approx = self._queries[(MDPState.create(c['state'].value), Hashable(c['act'].value))]
                    except KeyError:
                        pass
                    else:
                        approx.update(c['state'].value, c['act'].value)
            else:
                neighbors = dict(self._querycb.retrieve([('state', state.features), ('act', act.features)]))
                for id_, (d, s) in neighbors['state'].iteritems():
                    try:
                        approx = self._queries[(MDPState.create(s), Hashable(neighbors['act'][id_][1]))]
                    except KeyError:
                        pass
                    else:
                        approx.include(d, state.features, delta)

        return state

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
        self._querycb.retain([('state', state.features), ('act', act.features)])

        a = Hashable(act.features)
        try:
            approx = self._queries[(state, a)]
        except KeyError:
            approx = CasmlApproximator.Approximation(self, state, act, self._kernelfn)
            self._queries[(state, a)] = approx
        return approx

    def _fit_hmm(self, state, succ):
        # try:
        #     x = self._hmm._fit_X.copy()
        # except AttributeError:
        #     x = np.zeros(1, dtype=np.object)
        # else:
        #     if self._new_sequence:
        #         x = self._hmm._fit_X.tolist()
        #         x.append(np.zeros(1))
        #         x = np.array(x)
        #
        # if self._new_sequence:
        #     self._new_sequence = False
        #     x[-1] = np.hstack([np.reshape(state.features, (-1, state._nfeatures)).T])
        #
        # x[-1] = np.hstack([x[-1].tolist(), np.reshape(succ.features, (-1, succ._nfeatures)).T])
        # self._hmm.fit(x, n_init=1)

        if self._new_sequence:
            self._new_sequence = False
            self._fit_X.append([])
            self._fit_X[-1].append(state.features)

        if succ is not None:
            self._fit_X[-1].append(succ.features)
            self._hmm.fit(np.concatenate(self._fit_X), lengths=[len(x) for x in self._fit_X])
