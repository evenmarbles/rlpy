from __future__ import division, print_function, absolute_import

from abc import ABCMeta


class RegistryInterface(type):
    """Metaclass registering all subclasses derived from a given class.

    The registry interface adds every class derived from a given class
    to its registry dictionary. The `registry` attribute is a class
    variable and can be accessed anywhere. Therefore, this interface can
    be used to find all subclass of a given class.

    One use case are factory classes.

    Attributes
    ----------
    registry : list
        List of all classes deriving from a registry class.

    Methods
    -------
    __init__

    Examples
    --------
    Create a registry class:

    >>> from rlpy.framework.registry import RegistryInterface
    >>> class MyRegistryClass(object):
    ...     __metaclass__ = RegistryInterface

    .. note::
        | Project: Code from `A Primer on Python Metaclasses <https://jakevdp.github.io/blog/2012/12/01/a-primer-on-python-metaclasses/>`_.
        | Code author: `Jake Vanderplas <http://www.astro.washington.edu/users/vanderplas/>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    __metaclass__ = ABCMeta

    def __init__(cls, name, bases, dct):
        """Register the deriving class on instantiation."""
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        else:
            cls.registry[name.lower()] = cls

        super(RegistryInterface, cls).__init__(name, bases, dct)
