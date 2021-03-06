from __future__ import division, print_function, absolute_import

import math
import warnings
import numpy as np

from abc import ABCMeta, abstractmethod

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors.dist_metrics import METRIC_MAPPING as SKLEARN_METRIC_MAPPING
from scipy.spatial import distance


METRIC_MAPPING = {
   'braycurtis': distance.braycurtis,
   'canberra': distance.canberra,
   'chebyshev': distance.chebyshev,
   'cityblock': distance.cityblock,
   'correlation': distance.correlation,
   'cosine': distance.cosine,
   'dice': distance.dice,
   'euclidean': distance.euclidean,
   'hamming': distance.hamming,
   'jaccard': distance.jaccard,
   'kulsinski': distance.kulsinski,
   'mahalanobis': distance.mahalanobis,
   'matching': distance.matching,
   'minkowski': distance.minkowski,
   'rogerstanimoto': distance.rogerstanimoto,
   'russellrao': distance.russellrao,
   'seuclidean': distance.seuclidean,
   'sokalmichener': distance.sokalmichener,
   'sokalsneath': distance.sokalsneath,
   'sqeuclidean': distance.sqeuclidean,
   'wminkowski': distance.wminkowski,
   'yule': distance.yule
}
METRIC_MAPPING.update(SKLEARN_METRIC_MAPPING)


class Stat(object):
    """The similarity statistics container.

    The similarity statistics is a container to pass the
    calculated measure of similarity between the case
    identified by the case id and the query case between
    functions.

    Parameters
    ----------
    case_id : int
        The case's id.
    similarity : float
        The similarity measure.

    """
    __slots__ = ('_case_id', '_similarity')

    @property
    def case_id(self):
        """The case's id.

        Returns
        -------
        int :
            The case's id

        """
        return self._case_id

    @property
    def similarity(self):
        """The similarity measure.

        Returns
        -------
        float :
            The similarity measure.

        """
        return self._similarity

    def __init__(self, case_id, similarity=None):
        self._case_id = case_id
        self._similarity = similarity


class Similarity(object):
    """The similarity model interface.

    The similarity model keeps an internal indexing structure of
    the relevant case data to efficiently computing the similarity
    measure between data points.

    Notes
    -----
    All similarity models must inherit from this class.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        #: The indexing structure
        self._indexing_structure = None
        #: The mapping of the data points to their case ids
        self._id_map = None
        """:ivar: dict"""

    @abstractmethod
    def build_indexing_structure(self, data, id_map):
        """Build the indexing structure.

        Parameters
        ----------
        data : ndarray[ndarray[float]]
            The raw data points to be indexed.
        id_map : dict[int, int]
            The mapping from the data points to their case ids.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    @abstractmethod
    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data in
        the indexing structure returning the results in a collection of
        similarity statistics (:class:`Stat`).

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points stored in the
            indexing structure.

        Returns
        -------
        list[Stat] :
            A collection of similarity statistics.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


class NeighborSimilarity(Similarity):
    """The neighborhood similarity model.

    The neighbor similarity model determines similarity between the data
    in the indexing structure and the query data by using the nearest
    neighbor algorithm :class:`sklearn.neighbors.NearestNeighbors`.

    Both a k-neighbors classifier and a radius-neighbor-classifier are implemented.
    To choose between the classifiers either `n_neighbors` or `radius` must be
    specified.

    Parameters
    ----------
    n_neighbors : int
        The number of data points considered to be closest neighbors.
    radius : int
        The radius around the query data point, within which the data points
        are considered closest neighbors.
    algorithm : str
        The internal indexing structure of the training data. Defaults to
        `kd-tree`.
    metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid
        identifiers. Default is `euclidean`.
    metric_params : dict
        Parameters relevant to the specified metric.

    Raises
    ------
    UserWarning :
        If the either both or none of `n_neighbors` and `radius` are given.

    See Also
    --------
    :class:`sklearn.neighbors.KNeighborsClassifier`, :class:`sklearn.neighbors.RadiusNeighborsClassifier`

    """
    def __init__(self, n_neighbors=None, radius=None, algorithm='kd_tree', metric='minkowski', metric_params=2):
        super(NeighborSimilarity, self).__init__()

        if (n_neighbors is not None and radius is not None) or not (n_neighbors is None or radius is None):
            raise UserWarning("Exactly one of n_neighbors or radius must be initialized.")

        self._n_neighbors = n_neighbors
        self._radius = radius

        if self._n_neighbors is not None:
            if math.floor(self._n_neighbors) != self._n_neighbors:
                warnings.warn("k has been truncated to %d." % int(self._n_neighbors))
            self._n_neighbors = int(self._n_neighbors)

        if algorithm not in ["ball_tree", "kd_tree", "brute", "auto"]:
            raise ValueError("%s is not a valid retrieval algorithm" % algorithm)
        self._algorithm = algorithm

        if metric not in METRIC_MAPPING:
            raise ValueError("%s is not a valid retrieval metric" % metric)
        self._metric = metric

        self._metric_params = metric_params

    def build_indexing_structure(self, data, id_map):
        """Build the indexing structure.

        Build the indexing structure by fitting the data according to the
        specified algorithm.

        Parameters
        ----------
        data : ndarray[ndarray[float]]
            The raw data points to be indexed.
        id_map : dict[int, int]
            The mapping from the data points to their case ids.
        """
        self._id_map = id_map

        if self._n_neighbors is not None:
            self._indexing_structure = NearestNeighbors(n_neighbors=self._n_neighbors, algorithm=self._algorithm,
                                                        metric=self._metric, p=self._metric_params).fit(data)
        else:
            self._indexing_structure = NearestNeighbors(radius=self._radius, algorithm=self._algorithm,
                                                        metric=self._metric, p=self._metric_params).fit(data)

    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data in
        the indexing structure using the :class:`sklearn.neighbors.NearestNeighbors`
        algorithm. The results are returned in a collection of similarity statistics
        (:class:`Stat`).

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points stored in the
            indexing structure.

        Returns
        -------
        list[Stat] :
            A collection of similarity statistics.

        """
        if self._n_neighbors is not None:
            # noinspection PyProtectedMember
            raw_data = self._indexing_structure._fit_X
            if len(raw_data) < self._n_neighbors:
                result = []
                for i, feat in enumerate(raw_data):
                    dist = np.linalg.norm(np.asarray(data_point) - np.asarray(feat))
                    result.append(Stat(self._id_map[i], dist))

                # noinspection PyShadowingNames
                result = sorted(result, key=lambda x: x.similarity)
            else:
                d, key_lists = self._indexing_structure.kneighbors(data_point)
                result = [Stat(self._id_map[x], d[0][i]) for i, x in enumerate(key_lists[0])]

        else:
            d, key_lists = self._indexing_structure.radius_neighbors(data_point)
            result = [Stat(self._id_map[x], d[0][i]) for i, x in enumerate(key_lists[0])]
        return result


class KMeansSimilarity(Similarity):
    """The KMeans similarity model.

    The KMeans similarity model determines similarity between the data in the
    indexing structure and the query data by using the :class:`sklearn.cluster.KMeans`
    algorithm.

    Parameters
    ----------
    n_cluster : int
        The number of clusters to fit the raw data in.

    """
    def __init__(self, n_cluster=None):
        super(KMeansSimilarity, self).__init__()

        self._n_cluster = n_cluster if n_cluster is None else 8

    def build_indexing_structure(self, data, id_map):
        """Build the indexing structure.

        Build the indexing structure by fitting the data into `n_cluster`
        clusters.

        Parameters
        ----------
        data : ndarray[ndarray[float]]
            The raw data points to be indexed.
        id_map : dict[int, int]
            The mapping from the data points to their case ids.
        """
        self._id_map = id_map
        self._indexing_structure = KMeans(init='k-means++', n_clusters=self._n_cluster, n_init=10).fit(data)

    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data in
        the indexing structure using the :class:`sklearn.cluster.KMeans`
        clustering algorithm. The results are returned in a collection
        of similarity statistics (:class:`Stat`).

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points stored in the
            indexing structure.

        Returns
        -------
        list[Stat] :
            A collection of similarity statistics.

        """
        label = self._indexing_structure.predict(data_point)

        result = []
        try:
            # noinspection PyTypeChecker,PyUnresolvedReferences
            key_lists = np.nonzero(self._indexing_structure.labels_ == label[0])[0]
            result = [Stat(self._id_map[x]) for x in key_lists]
        except IndexError:
            pass

        return result


class ExactMatchSimilarity(Similarity):
    """The exact match similarity model.

    The exact match similarity model considered only exact matches between
    the data in the indexing structure and the query data as similar.

    """
    # noinspection PyUnusedLocal
    def __init__(self, **kwargs):
        super(ExactMatchSimilarity, self).__init__()

    def build_indexing_structure(self, data, id_map):
        """Build the indexing structure.

        To determine exact matches a brute-force algorithm is used thus
        the data remains as is and no special indexing structure is
        implemented.

        Parameters
        ----------
        data : ndarray[ndarray[float]]
            The raw data points to be indexed.
        id_map : dict[int, int]
            The mapping from the data points to their case ids.

        .. todo::
            It might be worth looking into a more efficient way of determining
            exact matches.

        """
        self._id_map = id_map
        self._indexing_structure = data

    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data in
        the indexing structure identifying exact matches. The results are
        returned in a collection of similarity statistics (:class:`Stat`).

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points stored in the
            indexing structure.

        Returns
        -------
        list[Stat] :
            A collection of similarity statistics.

        """
        result = []

        for i, feat in enumerate(self._indexing_structure):
            total = 0
            for j, val in enumerate(data_point):
                total += math.pow(val - feat[j], 2)

            if total == 0.0:
                result.append(Stat(self._id_map[i], 1.0))

        return result


class CosineSimilarity(Similarity):
    """The cosine similarity model.

    Cosine similarity is a measure of similarity between two vectors of an inner
    product space that measures the cosine of the angle between them. The cosine
    of 0 degree is 1, and it is less than 1 for any other angle. It is thus a
    judgement of orientation and not magnitude: tow vectors with the same
    orientation have a cosine similarity of 1, two vectors at 90 degrees have a
    similarity of 0, and two vectors diametrically opposed have a similarity of -1,
    independent of their magnitude [1]_.

    The cosine model employs the
    `cosine_similarity <http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity>`_
    function from the :mod:`sklearn.metrics.pairwise` module to determine similarity.

    .. seealso::
        `Machine Learning::Cosine Similarity for Vector Space Models (Part III)
        <http://blog.christianperone.com/?p=2497>`_

    References
    ----------
    .. [1] `Wikipidia::cosine_similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_

    """
    # noinspection PyUnusedLocal
    def __init__(self, threshold=None):
        super(CosineSimilarity, self).__init__()

        self._threshold = threshold if threshold is not None else 0.97

    def build_indexing_structure(self, data, id_map):
        """Build the indexing structure.

        The cosine_similarity function from :mod:`sklearn.metrics.pairwise` takes
        the raw data as input. Thus the data remains as is and no special indexing
        structure is implemented.

        Parameters
        ----------
        data : ndarray[ndarray[float]]
            The raw data points to be indexed.
        id_map : dict[int, int]
            The mapping from the data points to their case ids.

        """
        self._id_map = id_map
        self._indexing_structure = data

    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data in
        the indexing structure using the function :func:`cosine_similarity` from
        :mod:`sklearn.metrics.pairwise`.

        The resulting similarity ranges from -1 meaning exactly opposite, to 1
        meaning exactly the same, with 0 indicating orthogonality (decorrelation),
        and in-between values indicating intermediate similarity or dissimilarity.
        The results are returned in a collection of similarity statistics (:class:`Stat`).

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points stored in the
            indexing structure.

        Returns
        -------
        list[Stat] :
            A collection of similarity statistics.

        """
        similarity = cosine_similarity(data_point, self._indexing_structure)
        if not np.any(data_point):
            similarity = np.array([[float(np.array_equal(data_point, m)) for m in np.array(self._indexing_structure)]])

        return [Stat(self._id_map[i], x) for i, x in enumerate(similarity[0]) if x >= self._threshold]


SIMILARITY_MAPPING = {
    "knn": NeighborSimilarity,
    "radius-n": NeighborSimilarity,
    "kmeans": KMeansSimilarity,
    "exact-match": ExactMatchSimilarity,
    "cosine": CosineSimilarity,
}
