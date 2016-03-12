import numpy as np
from copy import deepcopy


def _process_vector(features):
    features = np.asarray(features)

    if features.ndim < 1:
        features.shape = (1,)

    if features.ndim != 1:
        raise ValueError("Array 'features' must be one-dimensional,"
                         " but features.ndim = %d" % features.ndim)
    return features


class Primitive(object):
    """

    """
    _dtype = np.float

    @property
    def features(self):
        return self._features

    @property
    def name(self):
        return self._name

    def __init__(self, features, name=None):
        self._features = _process_vector(features)

        features_str = np.array_str(self._features)
        self._name = '{0}: {1}'.format(name, features_str) if name else features_str
        if not isinstance(self._name, basestring):
            raise ValueError("'name' must be a string, but got %s" % str(type(self._name)))

    # noinspection PyUnusedLocal
    def __get__(self, instance, owner):
        return self._features

    def __set__(self, instance, value):
        self._features = _process_vector(value)

    def __len__(self):
        return len(self._features)

    def __contains__(self, item):
        return item in self._features

    def __hash__(self):
        return hash(tuple(self._features)) if self._features is not None else None

    def __eq__(self, other):
        return np.array_equal(other.features, self.features)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self._features + other.features
        return self._features + other

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self._features - other.features
        return self._features - other

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self._features * other.features
        return self._features * other

    def __rmul__(self, other):
        if isinstance(other, type(self)):
            return other.features + self._features
        return other * self._features

    def __imul__(self, other):
        if isinstance(other, type(self)):
            self._features *= other.features
        else:
            self._features *= other
        return self

    def __iter__(self):
        self._ix = 0
        return self

    def __str__(self):
        return "\'" + self._name + "\'"

    def __repr__(self):
        return "\'" + self._name + "\'"

    def next(self):
        if self._ix == len(self):
            raise StopIteration
        item = self._features[self._ix]
        self._ix += 1
        return item

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.iteritems():
            try:
                setattr(result, k, deepcopy(v, memo))
            except AttributeError:
                pass
        return result

    def __getstate__(self):
        data = self.__dict__.copy()
        del data['_ix']
        return data

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)

    def tolist(self):
        """Returns the feature array as a list.

        Returns
        -------
        list :
            The features list.

        """
        return self._features.tolist()


class MDPPrimitive(Primitive):
    """

    """
    _instance = object()
    _nfeatures = 0
    _feature_limits = []

    @property
    def nfeatures(self):
        return self._nfeatures

    def __init__(self, token, features, name=None):
        if token is not self._instance:
            raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))
        super(MDPPrimitive, self).__init__(features, name)

    def __set__(self, instance, value):
        features = self._process_parameters(value, None)
        super(MDPPrimitive, self).__set__(instance, features)

    @classmethod
    def create(cls, features, name=None, feature_limits=None):
        features = cls._process_parameters(features, feature_limits)
        return cls(cls._instance, features, name)

    @classmethod
    def set_feature_limits(cls, feature_limits):
        if 0 < cls._nfeatures != len(feature_limits):
            raise ValueError("Dimension mismatch: array 'feature' is a vector of length %d,"
                             " but only %d feature limits are provided" % (cls._nfeatures, len(feature_limits)))
        cls._feature_limits = deepcopy(feature_limits)

    @classmethod
    def _process_parameters(cls, features, feature_limits):
        if not hasattr(features, '__len__') and not isinstance(features, basestring):
            features = _process_vector(features)

        n = len(features)
        if n == 1:
            features = _process_vector(np.asarray(features))
            if features.dtype != np.int and features.dtype != np.float:
                raise ValueError("Array 'features' only supports integer and float features")
        elif n == 2:
            feat1 = _process_vector(features[0])
            feat2 = _process_vector(features[1])

            if feat1.dtype != feat2.dtype:
                dtype = []
                if feat1.dtype == np.int:
                    dtype.append(('int', '%di4' % feat1.shape[0]))
                else:
                    if feat1.dtype != np.float:
                        raise ValueError("Array 'features' only supports integer and float features")
                    dtype.append(('float', '%df4' % feat1.shape[0]))

                if feat2.dtype == np.int:
                    dtype.append(('int', '%di4' % feat2.shape[0]))
                else:
                    if feat2.dtype != np.float:
                        raise ValueError("Array 'features' only supports integer and float features")
                    dtype.append(('float', '%df4' % feat2.shape[0]))

                features = np.array([(feat1, feat2)], dtype=dtype)
            else:
                features = np.concatenate((feat1, feat2), axis=1)
                if features.dtype != np.int and features.dtype != np.float:
                    raise ValueError("Array 'features' only supports integer and float features")
        else:
            raise ValueError("Array 'features' must be a vector of at most length 2,"
                             " and only supports integer and float features")

        if cls._nfeatures == 0:
            cls._dtype = features.dtype
            cls._nfeatures = features.shape[0]
        else:
            if features.dtype != cls._dtype:
                raise ValueError("Type mismatch: array 'features' is of type %s,"
                                 " but expected is type %s" % (features.dtype, cls._dtype))
            if cls._dtype != np.int and cls._dtype != np.float:
                nfeatures = features['int'][0].shape[0] + features['float'][0].shape[0]
                if nfeatures != cls._nfeatures:
                    raise ValueError("Dimension mismatch: array 'features' is a vector of length %d,"
                                     " but expected number of features is %d" % (nfeatures, cls._nfeatures))
            elif features.shape[0] != cls._nfeatures:
                    raise ValueError("Dimension mismatch: array 'features' is a vector of length %d,"
                                     " but expected number of features is %d" % (features.shape[0], cls._nfeatures))

        if feature_limits is not None and not cls._feature_limits:
            cls.set_feature_limits(feature_limits)

        if cls._feature_limits and len(cls._feature_limits) != cls._nfeatures:
            raise ValueError("Dimension mismatch: array 'feature' is a vector of length %d,"
                             " but only %d feature limits are provided" % (cls._nfeatures, len(feature_limits)))

        return features
