from __future__ import division, print_function, absolute_import

import numpy as np
from abc import ABCMeta

from ....auxiliary.collection_ext import listify


class Feature(object):
    """The abstract feature class.

    A feature consists of one or more feature values.

    Parameters
    ----------
    cid : int
        The id of the case the feature belongs to.
    value : bool or string or int or float or list
        The feature value.

    """
    __metaclass__ = ABCMeta

    __slots__ = ('_cid', '_value', '_neighbors')

    @property
    def cid(self):
        """The case id this feature belongs to.

        Returns
        -------
        int :
            The case id.

        """
        return self._cid

    @property
    def value(self):
        """The feature value.

        Returns
        -------
        bool or string or int or float :
            The feature value.

        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value):
        self._neighbors = value

    def __init__(self, cid, value):
        self._cid = cid
        """:type: int"""
        self._value = np.asarray(listify(value))
        """:type: ndarray"""

        self._neighbors = None
        """:type: dict[int, tuple[float, int]]"""

    def __repr__(self):
        s = 'cid={0} value={1}'.format(self._cid, self._value)
        return s

    @staticmethod
    def hash(value):
        return hash(tuple(value))

    def include(self, d, basis, cid):
        nbors = self._neighbors.__class__()
        for id_, val in self._neighbors.items():
            if d < val[0]:
                nbors[cid] = (d, basis)
            nbors[id_] = val

        if cid not in nbors:
            nbors[cid] = (d, basis)

        self._neighbors.clear()
        self._neighbors.update(nbors)


class BoolFeature(Feature):
    """The boolean feature.

    The boolean feature is either represented by a scalar
    or by a list of booleans.

    Parameters
    ----------
    cid : int
        The id of the case the feature belongs to.
    value : bool or list[bool]
        The boolean feature value(s).

    Raises
    ------
    ValueError :
        If the feature values are not of type `boolean`.

    """

    def __init__(self, cid, value):
        super(BoolFeature, self).__init__(cid, value)

        if not self._value.dtype == bool:
            raise ValueError("The feature value is not of type `bool`.")


class StringFeature(Feature):
    """The string feature.

    The string feature is either represented by a single string
    or by a list of strings.

    Parameters
    ----------
    cid : int
        The id of the case the feature belongs to.
    value : bool or string or int or float or list
        The feature value.

    Raises
    ------
    ValueError
        If the feature values is not of type `string`.

    """

    def __init__(self, cid, value):
        super(StringFeature, self).__init__(cid, value)

        if not self._value.dtype == '|S5':
            raise ValueError("The feature value is not of type `string`.")


class IntFeature(Feature):
    """The integer feature.

    The integer feature is either represented by a scalar
    or by a list of integers.

    Parameters
    ----------
    cid : int
        The id of the case the feature belongs to.
    value : bool or string or int or float or list
        The feature value.

    Raises
    ------
    ValueError
        If not all feature values are of type `integer`.

    """

    def __init__(self, cid, value):
        super(IntFeature, self).__init__(cid, value)

        if self._value.dtype not in [np.int]:
            raise ValueError("The feature value is not of type `integer`.")


class FloatFeature(Feature):
    """The float feature.

    The float feature is either represented by a scalar
    or by a list of floats.

    Parameters
    ----------
    cid : int
        The id of the case the feature belongs to.
    value : bool or string or int or float or list
        The feature value.

    Raises
    ------
    ValueError
        If not all feature values are of type `float`.

    """

    def __init__(self, cid, value):
        super(FloatFeature, self).__init__(cid, value)

        if self._value.dtype not in [np.float, np.int]:
            raise ValueError("The feature value is not of type `float` or `int`.")
        if self._value.dtype == np.int:
            self._value = self._value.astype(np.float)


FEATURE_MAPPING = {
    "bool": BoolFeature,
    "string": StringFeature,
    "int": IntFeature,
    "float": FloatFeature,
}
