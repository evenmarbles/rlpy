from __future__ import division, print_function, absolute_import

import math
import copy

import numpy as np

from ....auxiliary.collection_ext import listify
from ...knowledgerep.cbr.methods import DefaultRetentionMethod
from .features import Feature, FEATURE_MAPPING
from .similarity_old import Similarity, SIMILARITY_MAPPING


class CaseBase(object):
    # noinspection PyTypeChecker
    """The case base engine.

    The case base engine maintains the a database of all cases entered
    into the case base. It manages retrieval, revision, reuse, and retention
    of cases.

    Parameters
    ----------
    metadata: dict
        The template from which to create a new case.

        :Example:

            An example template for a feature named ``state`` with the
            specified feature parameters. ``data`` is the data from which
            to extract the case from. In this example it is expected that
            ``data`` has a member variable ``state``.

            ::

                {
                    "state": {
                        "type": "float",
                        "value": "data.state",
                        "is_index": True,
                        "retrieval_method": "radius-n",
                        "retrieval_method_params": 0.01
                    },
                    "delta_state": {
                        "type": "float",
                        "value": "data.next_state - data.state",
                        "is_index": False,
                    }
                }

    reuse_method : ReuseMethod, optional
        The reuse method to be used during the reuse step. Default is None.
    revision_method : RevisionMethod, optional
        The revision method to be used during the revision step. Default is None.
    retention_method : RetentionMethod, optional
        The retention method to be used during the retention step. Default is
        `DefaultRetentionMethod`.
    plot_retrieval_method : callable, optional
        Callback function plotting the result of the revision method. Default is None.
        The callback should have the following signature:
            cb(case, case_id_list, names)
    plot_retrieval_params : dict
        Parameters used for plotting. Valid parameter keys are:

            'names' : str or list[str]
                The names of the feature which to plot.

    Examples
    --------
    Create a case base:

    >>> from mlpy.auxiliary.io import load_from_file
    >>>
    >>> template = {}
    >>> cb = CaseBase(template)

    Fill case base with data read from file:

    >>> from mlpy.mdp.stateaction import Experience, MDPState, MDPAction
    >>>
    >>> data = load_from_file("data/jointsAndActionsData.pkl")
    >>> for i in xrange(len(data.itervalues().next())):
    ...     for j in xrange(len(data.itervalues().next()[0][i]) - 1):
    ...         if not j == 10:  # exclude one experience as test case
    ...             experience = Experience(MDPState(data["states"][i][:, j]),
    ...                                     MDPAction(data["actions"][i][:, j]),
    ...                                     MDPState(data["states"][i][:, j + 1]))
    ...             cb.run(cb.case_from_data(experience))


    Loop over all cases in the case base:

    >>> for i in len(cb):
    ...     pass

    Retrieve case with ``id=0``:

    >>> case = cb[0]

    """

    class Match(object):
        """Case match information.

        Parameters
        ----------
        case : Case
            The matching case.
        similarity :
            A measure for the similarity to the query case.

        Attributes
        ----------
        is_solution : bool
            Whether this case match is a solution to the query case or not.
        error : float
            The error of the prediction.
        predicted : bool
            Whether the query case could be correctly predicted using this
            case match.

        """
        __slots__ = ('_case', '_similarity', 'is_solution', 'error', 'predicted')

        @property
        def case(self):
            """The case that matches the query case.

            Returns
            -------
            Case :
                The case matching the query.
            """
            return self._case

        # noinspection PyShadowingNames
        def __init__(self, case, key, similarity=None):
            self._case = case

            self._similarity = {
                key: similarity
            }
            self.is_solution = False
            self.error = np.inf
            self.predicted = False

        def __repr__(self):
            solution = 'True' if self.is_solution else 'False'
            return 'case={0} similarity={1} is_solution={2} error={3} predicted={4}'.format(self._case.id,
                                                                                            self._similarity,
                                                                                            solution, self.error,
                                                                                            self.predicted)

        def get_similarity(self, key):
            """Retrieve the similarity measure for the feature identified by the key.

            Parameters
            ----------
            key : str or list[str]
                Feature identifying key.

            Returns
            -------
            float :
                The similarity measure.
            """
            return self._similarity[key]

        def set_similarity(self, key, value):
            """Set the similarity measure for the feature identified by the key.

            Parameters
            ----------
            key : str or list[str]
                Feature identifying key.
            value : float
                The similarity value.

            """
            self._similarity[key] = value

    class Case(object):
        """The representation of a case in the case base.

        A case is composed of one or more :class:`Feature`.

        Parameters
        ----------
        cid : int
            The case's unique identifier.
        name : str
            The name for the case.
        description : str
            Text describing the case, optional.
        features : dict
            A list of features describing the case.

        """
        __slots__ = ('_id', '_name', '_description', '_features', 'ix')

        @property
        def id(self):
            """
            The case's unique identifier.

            Returns
            ------
            int : The case id.

            """
            return self._id

        def __init__(self, cid, name=None, description=None, features=None):
            self._id = cid
            self._name = "" if name is None else name
            self._description = "" if description is None else description
            self._features = {} if features is None else copy.copy(features)

        def __repr__(self):
            return 'id={0} features={1}'.format(self.id, self._features)

        def __getitem__(self, key):
            return self._features[key].value

        def __setitem__(self, key, value):
            self._features[key].value = value

        def __len__(self):
            return len(self._features)

        def __iter__(self):
            self.ix = 0
            return self

        def next(self):
            if self.ix == len(self._features):
                raise StopIteration
            item = self._features[self.ix][1]
            self.ix += 1
            return item

        def add_feature(self, feature):
            """Add a new feature.

            Parameters
            ----------
            feature : Feature
                The feature to add

            """
            self._features[feature.name] = feature

        # noinspection PyShadowingNames
        def get_indexed(self):
            """Return sorted collection of all indexed features.

            Returns
            -------
            list :
                The names of the indexed features in ascending order.

            """
            names = [x[1].name for x in self._features.items() if x[1].is_index]
            if len(names) == 1:
                return names[0]
            return names

        # noinspection PyShadowingNames
        def get_features(self, names):
            """Return sorted collection of features with the specified name.

            Parameters
            ----------
            names : str or list[str]
                The name(s) of the features to retrieve.

            Returns
            -------
            list or int or str or bool or float :
                List of features with the specified names(s)

            """
            if isinstance(names, list):
                return [x[1].value for x in self._features.items() if x[1].name in names]

            return self._features[names].value

        def compute_similarity(self, other):
            """Computes how similar two cases are.

            Parameters
            ----------
            other : Case
                The other case this case is compared to.

            Returns
            -------
            float :
                The similarity measure between the two cases.

            """
            total_similarity = 0.0

            for key, sfeature in self._features.iteritems():
                ofeature = other[key]

                if sfeature.is_index and ofeature.is_index:
                    weight = sfeature.weight * ofeature.weight
                    total_similarity += weight * math.pow(sfeature.compare(ofeature), 2)

            return math.sqrt(total_similarity)

    class _Metadata(object):
        """

        """

        class _SimilarityWrapper(object):
            """The similarity wrapper class.

            The entry maintains a similarity model from which similar cases can
            be derived.

            Internally the similarity model maintains an indexing structure
            dependent on the similarity model type for efficient computation
            of the similarity between cases. The case base is responsible for
            updating the indexing structure as cases are added and removed.

            Parameters
            ----------
            model : Similarity
                The similarity model.

            Attributes
            ----------
            dirty : bool
                A flag which identifies whether the model needs to be rebuild.

                The indexing structure of the similarity model is only rebuild if the dirty
                flag is set. This behavior can be circumvented by turing checking the dirty
                flag off.

            """
            __slots__ = ('dirty', '_similarity')

            def __init__(self, model):
                self.dirty = True

                self._similarity = model

            def compute_similarity(self, data_point, cases, name, check_dirty=True, case_matches=None):
                """Computes the similarity.

                Computes the similarity between the data point and each
                entry in the similarity model's indexing structure.

                Parameters
                ----------
                data_point : ndarray
                    The data point each entry in the similarity model is compared to.
                cases : dict[int, Case]
                    The collection of cases from which to build the indexing structure used
                    by the similarity model.
                name : str
                    The feature name relevant for the similarity computation.
                check_dirty : bool
                    This flag controls whether the dirty flag is being checked before
                    determining whether to rebuild the model or not.
                case_matches : dict[int, CaseMatch]
                    The solution to the problem-solving experience.

                Returns
                -------
                dict[int, Case] :
                    The similarity statistics of all entries in the model's indexing structure.

                """
                self._build_indexing_structure(cases, name, check_dirty)

                stats = self._similarity.compute_similarity(data_point)

                c = {}
                for s in stats:
                    c[s.case_id] = cases[s.case_id]
                    # merge case matches
                    if s.case_id not in case_matches:
                        case_matches[s.case_id] = CaseBase.Match(cases[s.case_id], name, s.similarity)
                    else:
                        case_matches[s.case_id].set_similarity(name, s.similarity)
                return c

            def _build_indexing_structure(self, cases, name, check_dirty):
                """Build the indexing structure.

                Build the indexing structure of the similarity model for specific feature name
                or alternatively for the cases provided in the `data` field.

                Parameters
                ----------
                cases : dict[int, Case]
                    The complete case base from which to build the indexing structure used
                    by the similarity model.
                name : str or list[str]
                    The feature name(s) relevant for the similarity computation. This field
                    is only required if the cases for building the indexing structure comes
                    from the `cases` field.
                check_dirty : bool
                    This flag controls whether the dirty flag is being checked before
                    determining whether to rebuild the model or not.

                """
                if not check_dirty or self.dirty:
                    data = []
                    id_map = {}

                    for i, c in enumerate(cases.itervalues()):
                        feature_list = c.get_features(name)

                        data.append(feature_list)
                        id_map[i] = c.id

                    self._similarity.build_indexing_structure(np.asarray(data), id_map)
                    self.dirty = False

        # -----------------------------
        # _Metadata
        # -----------------------------
        @property
        def feature_type(self):
            return self._type

        @property
        def feature_weight(self):
            return self._weight

        @property
        def feature_is_index(self):
            return self._is_index

        @property
        def similarity(self):
            return self._similarity

        @property
        def retrieval_method(self):
            return self._retrieval_method

        @property
        def retrieval_method_param(self):
            return self._retrieval_method_param

        def __init__(self, metadata):
            self._type = FEATURE_MAPPING[metadata['type']]
            """:type: Feature"""

            self._is_index = metadata['is_index'] if 'is_index' in metadata else True
            """:type: bool"""

            self._weight = metadata['weight'] if 'weight' in metadata else 1.0

            self._similarity = None
            """:type: _SimilarityWrapper"""

            try:
                self._retrieval_method = metadata['retrieval_method']
            except KeyError:
                self._retrieval_method = None
            else:
                try:
                    retrieval_method_params = metadata['retrieval_method_params']
                except KeyError:
                    if self._retrieval_method == 'knn' or self._retrieval_method == 'radius-n':
                        raise ValueError("For 'knn' retrieval method, the parameter 'n-neighbors' must be provided.")
                    if self._retrieval_method == 'radius-n':
                        raise ValueError("For 'radius-n' retrieval method, the parameter 'radius' must be provided.")
                    self._retrieval_method_param = None
                else:
                    if self._retrieval_method == 'knn' or self._retrieval_method == 'radius-n' and isinstance(
                            retrieval_method_params, dict):

                        default_params = {'n_neighbors': None, 'radius': None, 'algorithm': 'kd_tree',
                                          'metric': 'minkowski', 'metric_params': 2}
                        params = []
                        for n, v in default_params.iteritems():
                            if n == 'n_neighbors' or 'radius':
                                try:
                                    params.append(retrieval_method_params[n])
                                except KeyError:
                                    continue
                            params.append(retrieval_method_params.get(n, v))
                    else:
                        params = listify(retrieval_method_params)

                    self._retrieval_method_param = params[0]
                    self._similarity = self._SimilarityWrapper(SIMILARITY_MAPPING[self._retrieval_method](*params))

    # -----------------------------
    # CaseBase
    # -----------------------------
    @property
    def counter(self):
        """The case counter.

        The counter is increased with every case added to the case base.

        Returns
        -------
        int :
            The current count.

        """
        return self._counter

    def __init__(self, metadata, reuse_method=None, revision_method=None, retention_method=None,
                 plot_retrieval_method=None, plot_retrieval_params=None):
        self._counter = 0
        """:type: int"""

        #: Collection of the unadulterated cases.
        self._cases = {}
        """:type: dict[int, Case]"""

        #: The cases database metadata including the similarity model for the feature
        self._metadata = {}
        """:type: dict[str, _Metadata]"""
        for feature_name, info in metadata.iteritems():
            self._metadata[feature_name] = self._Metadata(info)

        self._reuse_method = reuse_method
        """:type: ReuseMethod"""

        self._revision_method = revision_method
        """:type: RevisionMethod"""

        self._retention_method = retention_method if retention_method is not None else DefaultRetentionMethod(self)
        """:type: _RetentionMethod"""

        self._plot_retrieval_method = plot_retrieval_method
        """:type: callable"""
        if plot_retrieval_params is not None:
            for key, value in plot_retrieval_params.iteritems():
                if key not in ['names']:
                    raise ValueError(
                        "%s is not a valid plot parameter for retrieval method" % key)
        self._plot_retrieval_params = plot_retrieval_params

    def __repr__(self):
        return 'num_cases={0}'.format(self._counter)

    def __getitem__(self, key):
        return self._cases[key]

    def __contains__(self, item):
        return item in self._cases

    def __len__(self):
        return len(self._cases)

    def __iter__(self):
        self.ix = 0
        return self

    def next(self):
        if self.ix == len(self._cases):
            raise StopIteration
        item = self._cases[self.ix]
        self.ix += 1
        return item

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def feature_type(self, name):
        return self._metadata[name].feature_type

    def feature_weight(self, name):
        return self._metadata[name].feature_weight

    def feature_is_index(self, name):
        return self._metadata[name].feature_is_index

    def feature_retrieval_method(self, name):
        return self._metadata[name].retrieval_method

    def feature_retrieval_method_param(self, name):
        return self._metadata[name].retrieval_method_param

    def add(self, data_points, names):
        """Add a new case without any checks.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.

        """
        self._cases[self._counter] = self.make_case(data_points, names)
        self._counter += 1
        for m in self._metadata.itervalues():
            if m.similarity is not None:
                m.similarity.dirty = True

    def remove(self, case_id):
        """Remove the case with the given case id from the case base.

        No checks are being performed

        Parameters
        ----------
        case_id : int
            The id of the case to be removed.

        """
        del self._cases[case_id]
        self._counter -= 1
        for m in self._metadata.itervalues():
            if m.similarity is not None:
                m.similarity.dirty = True

    def run(self, data_points, names, plot_retrieval=False, plot_reuse=False, plot_revision=False,
            plot_retention=False):
        """Run the case base.

        Run the case base using the CBR methods retrieve, reuse,
        revision and retention.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        plot_retrieval : bool, optional
            Plot the data during the retrieval step.
        plot_reuse: bool, optional
            Plot the data during the reuse step.
        plot_revision: bool, optional
            Plot the data during the revision step.
        plot_retention: bool, optional
            Plot the data during the retention step.

        Returns
        -------
        bool :
            Whether the case was inserted into the case base or not.

        """
        assert len(data_points) == len(names)
        case_matches = self.retrieve(data_points, names, plot=plot_retrieval)
        solution = self.reuse(data_points, names, case_matches, plot=plot_reuse)
        solution = self.revision(data_points, names, solution, plot=plot_revision)
        return self.retain(data_points, names, solution, plot=plot_retention)

    def retrieve(self, data_points, names=None, plot=False):
        """Retrieve cases similar to the query case.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to retrieve from the case base.
        names : str or list[str]
            The name(s) of the features for which to retrieve similar cases.
        plot: bool, optional
            Plot the data during the retrieval step.

        Returns
        -------
        dict[int, Match] :
            The solution to the problem-solving experience.

        """
        if len(self._cases) == 0:
            return {}

        names = listify(names)
        assert len(data_points) == len(names)

        cases = self._cases
        check_dirty = True
        case_matches = {}

        for name, data_point in zip(names, data_points):
            try:
                similiarty = self._metadata[name].similarity
            except KeyError:
                pass
            else:
                cases = similiarty.compute_similarity(data_point, cases, name, check_dirty, case_matches)
            check_dirty = False

        # check if case match is solution
        for cm in case_matches.itervalues():
            num_keys = 0
            for key in names:
                try:
                    cm.get_similarity(key)
                except KeyError:
                    break
                else:
                    num_keys += 1
            if num_keys >= len(names):
                cm.is_solution = True

        if plot:
            self.plot_retrieval(data_points, names, case_matches)

        return case_matches

    def reuse(self, data_points, names, case_matches, plot=True):
        """Performs the reuse step

        Performs new generalizations and specializations as a consequence of
        the solution transformation.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The solution to the problem-solving experience.
        plot: bool, optional
            Plot the data during the reuse step.

        Returns
        -------
        dict[int, Match] :
            The revised solution to the problem-solving experience.

        """
        if not case_matches:
            return {}

        names = listify(names)
        assert len(data_points) == len(names)

        return self._reuse_method.execute(data_points, names, case_matches,
                                          plot) if self._reuse_method is not None else case_matches

    def revision(self, data_points, names, case_matches, plot=True):
        """Evaluate solution provided by problem-solving experience.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The revised solution to the problem-solving experience.
        plot: bool, optional
            Plot the data during the revision step.

        Returns
        -------
        dict[int, Match] :
            The corrected solution.

        """
        if not case_matches:
            return {}

        names = listify(names)
        assert len(data_points) == len(names)

        return self._revision_method.execute(data_points, names, case_matches,
                                             plot) if self._revision_method is not None else case_matches

    def retain(self, data_points, names, case_matches, plot=True):
        """Retain new case.

        Retain new case depending on the revise outcomes and the
        CBR policy regarding case retention.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The corrected solution
        plot: bool, optional
            Plot the data during the retention step.

        Returns
        -------
        bool :
            Whether the case was inserted into the case base or not

        """
        names = listify(names)
        assert len(data_points) == len(names)

        return self._retention_method.execute(data_points, names, case_matches, plot)

    def plot_retrieval(self, data_points, names, case_matches):
        """Plot the retrieved data.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The solution to the problem-solving experience.

        """
        if self._plot_retrieval_method:
            self._plot_retrieval_method(data_points, names, self._cases, case_matches, self._plot_retrieval_params)

    def plot_reuse(self, data_points, names, case_matches):
        """Plot the reuse result.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The solution to the problem-solving experience.

        """
        self._reuse_method.plot_data(data_points, names, case_matches)

    def plot_revision(self, data_points, names, case_matches):
        """Plot revision results.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The revised solution to the problem-solving experience.

        """
        self._revision_method.plot_data(data_points, names, case_matches)

    def plot_retention(self, data_points, names, case_matches):
        """Plot the retention result.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.
        case_matches : dict[int, Match]
            The corrected solution

        """
        self._retention_method.plot_data(data_points, names, case_matches)

    def make_case(self, data_points, names):
        """Convert data into a case using the case template.

        Parameters
        ----------
        data_points : ndarray or list[ndarray
            The data points to run against the case base
        names : str or list[str]
            The name(s) of the features associated with the data points.

        Returns
        -------
        Case :
            The case extracted from the data.
        """
        feature_list = {}
        for n, v in zip(names, data_points):
            feature_list[n] = self.feature_type(n)(n, v, self.feature_weight(n), self.feature_is_index(n))
        return CaseBase.Case(self.counter, features=feature_list)
