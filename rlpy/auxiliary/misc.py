"""
.. module:: mlpy.auxiliary.misc
   :platform: Unix, Windows
   :synopsis: Utility functions.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import os
import sys
import time
import numpy as np
from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    """Preventing a C shared library to print on stdout.

    Examples
    --------
    >>> import os
    >>>
    >>> with stdout_redirected(to="filename"):
    >>>    print("from Python")
    >>>    os.system("echo non-Python applications are also supported")

    .. note::
        | Project: Code from `StackOverflow <http://stackoverflow.com/a/17954769>`_.
        | Code author: `J.F. Sebastian <http://stackoverflow.com/users/4279/j-f-sebastian>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    fd = sys.stdout.fileno()

    # noinspection PyShadowingNames
    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)    # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')     # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as f:
            _redirect_stdout(to=f)
        try:
            yield   # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)     # restore stdout, buffering and flags such as CLOEXEC may be different


def columnize(table_row, justify="R", column_width=0, header=None, sep=None):
    """Pretty print a table in tabular format a row at a time.

    Parameters
    ----------
    table_row : list or tuple
        A row of a table
    justify : {"R", "L", "C"}, optional
        Justification of the columns. Default is "R".
    column_width : int, optional
        The width of the column if `0`, the column width is determined from
        the elements in the header and the table row. The column width must
        be greater than max column width in the table. Default is 0.
    header : list or tuple
        The header elements for the table. If provided a header is created
        for the table. Should only be passed in for the first row of the table.
        Default is None.
    sep : str
        The separator used for each column. Default is " ".

    Returns
    -------
    out : str
        The formatted row.
    column_width : int
        The maximum width of the column

    """
    sep = sep if sep is not None else " "
    if column_width == 0:
        def find_width(row, width):
            for col in row:
                w = len(str(col))
                if w > width:
                    width = w
            return width

        if header is not None:
            column_width = find_width(header, column_width)
        column_width = find_width(table_row, column_width) + 2

    def format(r, j, w, s):
        l = []
        for c in r:
            if j == "R":
                l.append(str(c).rjust(w))
            elif j == "L":
                l.append(str(c).ljust(w))
            elif j == "C":
                l.append(str(c).center(w))
        return s.join(l)

    out = ""
    if header is not None:
        out += format(header, justify, column_width, sep) + "\n"
        sep_list = []
        # noinspection PyTypeChecker
        for _ in range(len(header)):
            sep_list.append('='*column_width)
        out += sep.join(sep_list) + "\n"
    out += format(table_row, justify, column_width, sep)
    return out, column_width


class Timer(object):
    """Timer class for timing sections of code.

    The timer class follows the context management protocol and
    thus is used with the `with` statement.

    Examples
    --------
    >>> with Timer() as t:
    ...     # code to time here
    >>> print('Request took %.03f sec.' % t.time)

    """

    def __enter__(self):
        self.s0 = time.clock()
        return self

    def __exit__(self, *args):
        self.s1 = time.clock()
        self.time = self.s1 - self.s0


class Hashable(object):
    @property
    def value(self):
        return self._value

    def __init__(self, value):
        self._value = value
        """:type: ndarray"""

    def __repr__(self):
        return np.array_str(self._value)

    def __eq__(self, other):
        # noinspection PyTypeChecker
        return np.all(self.value == other.value)

    def __hash__(self):
        return hash(tuple(self.value))
