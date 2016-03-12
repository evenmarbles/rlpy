from __future__ import division, print_function, absolute_import

import weakref
import numpy as np


# -------------------------------------
# Dictionary
# -------------------------------------
class LazyDict(dict):
    def get(self, k, d=None, *args):
        return self[k] if k in self else d(*args) if callable(d) else d

    def setdefault(self, k, d=None, *args):
        return self[k] if k in self else dict.setdefault(self, k, d(*args) if callable(d) else d)


class LazyWeakValueDictionary(weakref.WeakValueDictionary):
    def get(self, k, d=None, *args):
        try:
            wr = self.data[k]
        except KeyError:
            if callable(d):
                d = d(*args)
            return d
        else:
            o = wr()
            if o is None:
                # This should only happen
                if callable(d):
                    d = d(*args)
                return d
            else:
                return o

    # noinspection PyUnresolvedReferences
    def setdefault(self, k, d=None, *args):
        try:
            wr = self.data[k]
        except KeyError:
            if self._pending_removals:
                self._commit_removals()
            if callable(d):
                d = d(*args)
            from weakref import KeyedRef
            self.data[k] = KeyedRef(d, self._remove, k)
            return d
        else:
            return wr()


def remove_key(d, key):
    """Safely remove the `key` from the dictionary.

    Safely remove the `key` from the dictionary `d` by first
    making a copy of dictionary. Return the new dictionary together
    with the value stored for the `key`.

    Parameters
    ----------
    d : dict
        The dictionary from which to remove the `key`.
    key :
        The key to remove

    Returns
    -------
    v :
        The value for the key
    r : dict
        The dictionary with the key removed.

    """
    r = dict(d)
    v = r.pop(key, None)
    return v, r


# -------------------------------------
# List
# -------------------------------------
def listify(obj):
    """Ensure that the object `obj` is of type list.

    If the object is not of type `list`, the object is
    converted into a list.

    Parameters
    ----------
    obj :
        The object.

    Returns
    -------
    list :
        The object inside a list.

    """
    if obj is None:
        return []

    return obj if isinstance(obj, (list, tuple, np.ndarray, type(None))) else [obj]
