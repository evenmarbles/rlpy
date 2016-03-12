from __future__ import division, print_function, absolute_import

import weakref
from collections import defaultdict
from collections import OrderedDict

from ....auxiliary.collection_ext import listify
from .features import Feature, FEATURE_MAPPING
from .similarity import SIMILARITY_MAPPING


class CaseBase(object):
    """

    """

    class _FeatureMetadata(object):

        @property
        def type(self):
            return self._type

        @property
        def key(self):
            """The features identifying key.

            Returns
            -------
            str :
                The key of the feature.

            """
            return self._key

        @property
        def weight(self):
            """The weights given to each feature value.

            Returns
            -------
            float or list[float] :
                The feature weights.

            """
            return self._weight

        @property
        def is_index(self):
            """Flag indicating whether this feature is an index.

            Returns
            -------
            bool :
                Whether the feature is an index.

            """
            return self._is_index

        @property
        def order(self):
            """Identifies the order of all features.

            Returns
            -------
            int :
                The order among all features.

            """
            return self._order

        @property
        def similarity(self):
            return self._similarity

        def __init__(self, key, metadata):
            self._type = FEATURE_MAPPING[metadata['type']]
            """:type: Feature"""

            self._key = key
            """:type: str"""

            self._is_index = metadata['is_index'] if 'is_index' in metadata else True
            """:type: bool"""

            self._weight = metadata['weight'] if 'weight' in metadata else 1.
            """:type: float"""

            self._order = metadata['order'] if 'order' in metadata else 0
            """:type: int"""

            self._similarity = None
            """:type: Similarity"""

            try:
                retrieval_method = metadata['retrieval_method']
            except KeyError:
                pass
            else:
                params = []
                try:
                    retrieval_method_params = metadata['retrieval_method_params']
                except KeyError:
                    if retrieval_method == 'knn' or retrieval_method == 'radius-n':
                        raise ValueError("For 'knn' retrieval method, the parameter 'n-neighbors' must be provided.")
                    if retrieval_method == 'radius-n':
                        raise ValueError("For 'radius-n' retrieval method, the parameter 'radius' must be provided.")
                else:
                    if isinstance(retrieval_method_params, dict) and (
                                retrieval_method == 'knn' or retrieval_method == 'radius-n'):

                        default_params = OrderedDict()
                        default_params['scale'] = None
                        default_params['n_neighbors'] = None
                        default_params['radius'] = None

                        for n, v in default_params.iteritems():
                            if n == 'scale':
                                if v is None:
                                    raise ValueError(
                                        'For \'knn\' or \'radius-n\' retrieval method, \'scale\' must be provided.')
                            if n == 'n_neighbors' or 'radius':
                                try:
                                    params.append(retrieval_method_params[n])
                                except KeyError:
                                    continue
                            params.append(retrieval_method_params.get(n, v))
                    else:
                        params = listify(retrieval_method_params)
                self._similarity = SIMILARITY_MAPPING[retrieval_method](*params)

    # -----------------------------
    # CaseBase
    # -----------------------------
    @property
    def similarity_uses_knn(self):
        return self._similarity_uses_knn

    def __init__(self, feature_metadata,
                 reuse_method=None, reuse_method_params=None,
                 revision_method=None, revision_method_params=None,
                 retention_method=None, retention_method_params=None, name=None):
        self._name = name if name is not None else ''

        self._similarity_uses_knn = False

        self._cases = {}
        """:type: dict[str, dict[ndarray, list[Feature]]]"""
        self._id_map = {}
        """:type: dict[str, dict[int, Feature]]"""

        self._metadata = {}
        """:type: dict[str, _FeatureMetadata]"""
        for key, info in feature_metadata.iteritems():
            self._cases[key] = defaultdict(list)
            self._id_map[key] = weakref.WeakValueDictionary()
            self._metadata[key] = self._FeatureMetadata(key, info)

            similarity = self._metadata[key].similarity
            if similarity is not None:
                if self._similarity_uses_knn or similarity.name == 'knn':
                    self._similarity_uses_knn = True

        self._dependency = zip(*sorted(self._metadata.items(), key=lambda x: x[1].order))[0]
        """:type: list[str]"""

        self._reuse_method = self._create_method(reuse_method, reuse_method_params)
        """:type: ReuseMethod"""
        self._revision_method = self._create_method(revision_method, revision_method_params)
        """:type: RevisionMethod"""
        self._retention_method = self._create_method(retention_method, retention_method_params)
        """:type: _RetentionMethod"""

        # self._plot_retrieval_method = plot_retrieval_method
        # """:type: callable"""
        # if plot_retrieval_params is not None:
        #     for key, value in plot_retrieval_params.iteritems():
        #         if key not in ['names']:
        #             raise ValueError(
        #                 "%s is not a valid plot parameter for retrieval method" % key)
        # self._plot_retrieval_params = plot_retrieval_params

    def __repr__(self):
        return '#cases={0}'.format(len(self))

    # def __getitem__(self, key):
    #     return self._cases[key]
    #
    # def __contains__(self, key):
    #     return key in self._cases

    def __len__(self):
        return len(next(iter(self._id_map.values())))

    def __iter__(self):
        return iter((id_, self.get_case(id_))
                    for id_, f in self._id_map[self._dependency[0]].iteritems())

    def iterkeys(self):
        return self._id_map[self._dependency[0]].iterkeys()

    def itervalues(self):
        for id_ in self._id_map[self._dependency[0]].iterkeys():
            features = {}
            for k in self._dependency:
                features[k] = self._id_map[k][id_]
            yield features

    def iteritems(self):
        return iter((id_, self.get_case(id_))
                    for id_, f in self._id_map[self._dependency[0]].iteritems())

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def get_case(self, item):
        """

        Parameters
        ----------
        item : int or list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).

        Returns
        -------
        case : dict[str, Feature]
            The case containing all features.

        """
        case = {}
        if isinstance(item, int):
            for k in self._dependency:
                case[k] = self._id_map[k][item]
        elif isinstance(item, list):
            f = dict(item)

            prev_ids = set()
            for key, v in f.iteritems():
                fhash = Feature.hash(v)
                if fhash not in self._cases[key]:
                    return {}

                ids = set([f.cid for f in self._cases[key][fhash]])
                if prev_ids:
                    prev_ids = prev_ids.intersection(ids)
                else:
                    prev_ids = ids

            # prev_cases = {}
            # for key, v in f.iteritems():
            #     fhash = Feature.hash(v)
            #     if fhash not in self._cases[key]:
            #         return {}
            #
            #     cases = {f.cid: {key: f} for f in self._cases[key][fhash]}
            #     if prev_cases:
            #         for id_ in cases.keys():
            #             if id_ not in prev_cases:
            #                 del cases[id_]
            #         for id_ in prev_cases.keys():
            #             if id_ not in cases:
            #                 del prev_cases[id_]
            #     else:
            #         prev_cases = cases

            assert len(prev_ids) <= 1
            if len(prev_ids) == 1:
                case = self.get_case(next(iter(prev_ids)))
        return case

    def get_feature(self, key, id_):
        return self._id_map[key][id_]

    def insert(self, features, matches=None):
        """Add a new case without any checks.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : dict[str, dict[int, [float, ndarray]]] :
            The matches identified by the appropriate similarity measure.

        """
        assert len(features) == len(self._dependency)

        if self.get_case(features):
            return -1

        features = dict(features)
        matches = dict(matches) if matches is not None else {}
        basis_id = max(next(iter(self._id_map.values())).keys()) + 1 if len(self) > 0 else 0

        prev_nbors = None
        neighbors = []
        m = []

        for key in self._dependency:
            basis = self._metadata[key].type(basis_id, features[key])

            fhash = Feature.hash(basis.value)
            self._cases[key][fhash].append(basis)
            self._id_map[key][basis_id] = basis

            similarity = self._metadata[key].similarity
            if similarity is not None:
                if matches and key in matches:
                    matches[key][basis_id] = (0., basis.value)
                    m.append((key, OrderedDict(sorted(matches[key].items(), key=lambda x: x[1][0]))))

                nbors = self._find_neighbors(key, basis.value, similarity, prev_nbors, neighbors, basis_id)
                if not nbors:
                    del neighbors[:]
                    break

                neighbors.append((key, nbors))
                prev_nbors = neighbors[-1][1]

        for key, nbors in neighbors:
            self._id_map[key][basis_id].neighbors = nbors

        # update neighbors
        if not self._similarity_uses_knn:
            # similarity != 'knn'
            for key, nbors in neighbors:
                basis = self._id_map[key][basis_id].value
                for id_, (d, v) in nbors.iteritems():
                    if id_ == basis_id:
                        continue
                    self._id_map[key][id_].include(d, basis, basis_id)
            return basis_id

        # similarity == 'knn'
        for id_ in self._id_map[self._dependency[0]].keys():
            if id_ == basis_id:
                continue
            f = [(k, self._id_map[k][id_].value) for k in self._dependency]

            prev_nbors = None
            neighbors = []

            features = dict(f)
            for key in self._dependency:
                similarity = self._metadata[key].similarity
                if similarity is not None:
                    nbors = self._find_neighbors(key, features[key], similarity, prev_nbors, neighbors)
                    if not nbors:
                        del neighbors[:]
                        break
                    neighbors.append((key, nbors))
                    prev_nbors = neighbors[-1][1]

            for key, nbors in neighbors:
                self._id_map[key][id_].neighbors = nbors

        return basis_id

    def remove(self, features):
        """Remove the case with the given case id from the case base.

        No checks are being performed

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).

        """
        features = dict(features)

        prev_ids = set()
        for key, v in features.iteritems():
            fhash = Feature.hash(v)
            if fhash not in self._cases[key]:
                return -1

            ids = set([f.cid for f in self._cases[key][fhash]])
            if prev_ids:
                prev_ids = prev_ids.intersection(ids)
            else:
                prev_ids = ids

        assert len(self._cases['state'][Feature.hash(features['state'])]) < 4
        assert len(prev_ids) == 1
        basis_id = next(iter(prev_ids))

        for key in self._dependency:
            value = self._id_map[key][basis_id].value
            fhash = Feature.hash(value)

            similarity = self._metadata[key].similarity
            if similarity is not None:
                similarity.remove(basis_id, value)

            # update neighbors
            if not self._similarity_uses_knn:
                nbors = self._id_map[key][basis_id].neighbors
                for id_ in nbors.iterkeys():
                    del self._id_map[key][id_].neighbors[basis_id]

            i = -1
            for i, f in enumerate(self._cases[key][fhash]):
                if f.cid == basis_id:
                    break
            del self._id_map[key][basis_id]
            del self._cases[key][fhash][i]
            if not self._cases[key][fhash]:
                del self._cases[key][fhash]

        # similarity == 'knn'
        if self._similarity_uses_knn:
            for id_, f in self._id_map[self._dependency[0]].iteritems():
                if id_ == basis_id:
                    continue

                if basis_id in f.neighbors:
                    del f.neighbors[basis_id]

    def run(self, features):
        matches = self.retrieve(features)
        matches = self.reuse(features, matches)
        matches = self.revise(features, matches)
        return self.retain(features, matches)

    def retrieve(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------
        dict[str, dict[int, [float, ndarray]]] :
            The neighbors of the features

        """
        if len(self._cases.itervalues().next()) <= 0:
            return {}

        prev_nbors = None
        neighbors = []

        features = dict(features)

        for key in self._dependency:
            similarity = self._metadata[key].similarity
            if similarity is not None:
                fhash = Feature.hash(features[key])
                if fhash in self._cases[key]:
                    this_feature = self._cases[key][fhash]

                    nbors = {}
                    for f in this_feature:
                        nbors.update(f.neighbors)
                    nbors = OrderedDict(sorted(nbors.items(), key=lambda x: x[1][0]))

                    if prev_nbors is not None:
                        prev_ids = prev_nbors.keys()
                        nbors = prev_nbors.__class__([(id_, v) for id_, v in nbors.items() if id_ in prev_ids])

                        valid_ids = nbors.keys()
                        for i, n in reversed(list(enumerate(neighbors))):
                            for id_ in n[1].iterkeys():
                                if id_ not in valid_ids:
                                    del neighbors[i][1][id_]
                else:
                    nbors = self._find_neighbors(key, features[key], similarity, prev_nbors, neighbors)
                    if not nbors:
                        del neighbors[:]
                        break

                neighbors.append((key, nbors))
                prev_nbors = neighbors[-1][1]

        return dict(neighbors)

    def reuse(self, features, matches):
        if self._reuse_method is None or not matches:
            return matches
        return self._reuse_method.execute(features, matches)

    def revise(self, features, matches):
        if self._revision_method is None or not matches:
            return matches
        return self._revision_method.execute(features, matches)

    def retain(self, features, matches=None):
        if self._retention_method is None:
            return self.insert(features, matches)
        return self._retention_method.execute(features, matches)

    def _create_method(self, method, params):
        params = params if params is not None else {}
        try:
            if isinstance(params, dict):
                method = method.create(self, **params)
            else:
                method = method.create(self, *listify(params))
        except AttributeError:
            method = None
        return method

    def _find_neighbors(self, key, feature, similarity, prev_nbors, neighbors, cid=None):
        """

        Parameters
        ----------
        key : str
        feature : ndarray]
            The features of the query.
        similarity : Similarity
        prev_nbors : dict[int, tuple[float, ndarray]]
        neighbors : list[tuple[str, dict[tuple[float, dict]]]
        cid : int

        Returns
        -------

        """
        if prev_nbors is not None:
            similarity.clear()
            for id_ in prev_nbors.iterkeys():
                similarity.insert(id_, self._id_map[key][id_].value)

        if cid is not None:
            similarity.insert(cid, feature)
        nbors = similarity.compute_similarity(feature)

        if prev_nbors is not None and nbors:
            valid_ids = nbors.keys()
            for i, n in reversed(list(enumerate(neighbors))):
                for id_ in n[1].iterkeys():
                    if id_ not in valid_ids:
                        del neighbors[i][1][id_]
        return nbors
