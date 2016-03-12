from __future__ import division, print_function, absolute_import

import math
import warnings
import numpy as np
from collections import defaultdict
from collections import OrderedDict

from ....libs import classifier
from ....auxiliary.misc import Hashable


class Similarity(object):
    """

    """

    # -----------------------------
    # Similarity
    # -----------------------------
    @property
    def name(self):
        return self._name

    def __init__(self):
        self._name = ''
        self._bases = defaultdict(set)
        """:type: dict[ndarray]"""

    def insert(self, uid, data_point):
        """

        Parameters
        ----------
        uid : int
        data_point : ndarray

        Returns
        -------
        data_point : ndarray

        was_inserted : bool
            True if the data point was inserted, False otherwise.

        """
        hashed = Hashable(data_point)
        was_inserted = False
        if hashed not in self._bases:
            was_inserted = True
        self._bases[hashed].add(uid)
        return data_point, was_inserted

    def remove(self, uid, data_point):
        """

        Parameters
        ----------
        uid : int
        data_point : ndarray

        Returns
        -------
        bool :
            True if the element was removed, False otherwise.

        """
        was_removed = False
        hashed = Hashable(data_point)
        if hashed in self._bases:
            if uid in self._bases[hashed]:
                self._bases[hashed].remove(uid)
                was_removed = True
            if not self._bases[hashed]:
                del self._bases[hashed]
        return was_removed

    def clear(self):
        self._bases.clear()

    def compute_similarity(self, data_point):
        """Computes the similarity.

        Computes the similarity between the data point and the data.

        Parameters
        ----------
        data_point : ndarray
            The raw data point to compare against the data points.

        Returns
        -------
        dict[int, tuple[float, ndarray]] :
            A collection of similarity statistics.

        """
        raise NotImplementedError


class NeighborSimilarity(Similarity):
    """

    """

    def __init__(self, scale):
        super(NeighborSimilarity, self).__init__()

        self._scale = scale
        self._covertree = classifier.CoverTree(np.asarray(self._scale))

    def insert(self, uid, data_point):
        query, was_inserted = super(NeighborSimilarity, self).insert(uid, data_point)
        if not was_inserted:
            return query
        return self._covertree.insert(data_point)

    def remove(self, uid, data_point):
        if super(NeighborSimilarity, self).remove(uid, data_point):
            return self._covertree.remove(data_point)
        return False


class KnnSimilarity(NeighborSimilarity):
    """

    """

    def __init__(self, scale, n_neighbors):
        super(KnnSimilarity, self).__init__(scale)

        self._name = 'knn'

        self._n_neighbors = n_neighbors
        if math.floor(self._n_neighbors) != self._n_neighbors:
            warnings.warn("k has been truncated to %d." % int(self._n_neighbors))
        self._n_neighbors = int(self._n_neighbors)

    def compute_similarity(self, data_point):
        neighbors = self._covertree.nearest(data_point, self._n_neighbors)

        nbors = OrderedDict()
        for d, v in neighbors:
            for uid in self._bases[Hashable(v)]:
                nbors[uid] = (d, v)
        return nbors


class RadiusSimilarity(NeighborSimilarity):
    """

    """

    def __init__(self, scale, radius):
        super(RadiusSimilarity, self).__init__(scale)

        self._name = 'radius-n'
        self._radius = radius

    def compute_similarity(self, data_point):
        neighbors = sorted(self._covertree.neighbors(data_point, self._radius), key=lambda x: x[0])

        nbors = OrderedDict()
        for d, v in neighbors:
            for uid in self._bases[Hashable(v)]:
                nbors[uid] = (d, v)
        return nbors


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

        self._name = 'kmeans'

        from sklearn.cluster import KMeans
        self._esitmator = KMeans(init='k-means++', n_clusters=self._n_cluster, n_init=10)

        self._n_cluster = n_cluster if n_cluster is None else 8
        self._dirty = False

    def insert(self, uid, data_point):
        query, was_inserted = super(KMeansSimilarity, self).insert(uid, data_point)
        if was_inserted:
            self._dirty = True
        return query

    def remove(self, uid, data_point):
        was_removed = super(KMeansSimilarity, self).remove(uid, data_point)
        if was_removed:
            self._dirty = True
        return was_removed

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
        dict[int, tuple[float, ndarray]] :
            A collection of similarity statistics.

        """
        data, uid = zip(*self._bases.items())
        data = [d.value for d in data]

        if self._dirty:
            # compute k-means clustering
            self._esitmator.fit(data)

        labels = self._esitmator.predict(data_point)

        nbors = OrderedDict()
        try:
            # noinspection PyTypeChecker,PyUnresolvedReferences
            labels = np.nonzero(self._estimator.labels_ == labels[0])[0]

            for x in labels:
                nbors[uid[x]] = (1., data[x])
        except IndexError:
            pass
        return nbors


class ExactMatchSimilarity(Similarity):
    """The exact match similarity model.

    The exact match similarity model considered only exact matches between
    the data in the indexing structure and the query data as similar.

    """
    # noinspection PyUnusedLocal
    def __init__(self):
        super(ExactMatchSimilarity, self).__init__()

        self._name = 'exact-match'

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
        dict[int, tuple[float, ndarray]] :
            A collection of similarity statistics.

        """

        nbors = OrderedDict()
        for uid in self._bases[Hashable(data_point)]:
            nbors[uid] = (1., data_point)
        return nbors


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

        self._name = 'cosine'

        from sklearn.metrics.pairwise import cosine_similarity
        self._cosine_similarity = cosine_similarity

        self._threshold = threshold if threshold is not None else 0.97

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
        dict[int, tuple[float, ndarray]] :
            A collection of similarity statistics.

        """
        data, uid = zip(*self._bases.items())
        data = [d.value for d in data]

        if not np.any(data_point):
            similarity = np.array([[float(np.array_equal(data_point, d)) for d in np.array(data)]])
        else:
            similarity = self._cosine_similarity(data_point, data)

        nbors = OrderedDict()
        for x, d in enumerate(similarity[0]):
            if d >= self._threshold:
                nbors[uid[x]] = (d, data[x])
        return nbors


SIMILARITY_MAPPING = {
    "knn": KnnSimilarity,
    "radius-n": RadiusSimilarity,
    "kmeans": KMeansSimilarity,
    "exact-match": ExactMatchSimilarity,
    "cosine": CosineSimilarity,
}
