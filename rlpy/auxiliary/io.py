from __future__ import division, print_function, absolute_import

import os
import pickle
import importlib

from .collection_ext import listify


def load_from_file(filename, import_modules=None):
    """Load data from file.

    Different formats are supported.

    Parameters
    ----------
    filename : str
        Name of the file to load data from.
    import_modules : str or list
        List of modules that may be required by the data
        that need to be imported.

    Returns
    -------
    dict or list :
        The loaded data. If any errors occur, ``None`` is returned.

    """
    for m in listify(import_modules):
        vars()[m] = importlib.import_module(m)

    try:
        with open(filename, 'rb') as f:
            try:
                data = eval(f.read())
            except TypeError:
                f.seek(0, 0)
                data = pickle.load(f)
            except SyntaxError:
                f.seek(0, 0)
                data = []
                for line in f:
                    data.append(eval(line.strip('\r\n')))
            finally:
                f.seek(0, 0)
        return data
    except TypeError:
        return None


def save_to_file(filename, data):
    """Saves data to file.

    The data can be a dictionary or an object's
    state and is saved in :mod:`pickle` format.

    Parameters
    ----------
    filename : str
        Name of the file to which to save the data to.
    data : dict or object
        The data to be saved.

    """
    if filename is None:
        return

    path = os.path.dirname(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except TypeError:
        return
