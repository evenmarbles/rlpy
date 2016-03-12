from __future__ import division, print_function, absolute_import


class Singleton(type):
    """
    Metaclass ensuring only one instance of the class exists.

    The singleton pattern ensures that a class has only one instance
    and provides a global point of access to that instance.

    Methods
    -------
    __call__

    Notes
    -----
    To define a class as a singleton include the :data:`__metaclass__`
    directive.

    See Also
    --------
    :class:`Borg`

    Examples
    --------
    Define a singleton class:

    >>> from mlpy.modules.patterns import Singleton
    >>> class MyClass(object):
    >>>     __metaclass__ = Singleton

    .. note::
        | Project: Code from `StackOverflow <http://stackoverflow.com/q/6760685>`_.
        | Code author: `theheadofabroom <http://stackoverflow.com/users/655372/theheadofabroom>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    _instance = {}

    def __call__(cls, *args, **kwargs):
        """Returns instance to object."""
        if cls not in cls._instance:
            # noinspection PyArgumentList
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[cls]


class Borg(object):
    """Class ensuring that all instances share the same state.

    The borg design pattern ensures that all instances of a class share
    the same state  and provides a global point of access to the shared state.

    Rather than enforcing that only ever one instance of a class exists,
    the borg design pattern ensures that all instances share the same state.
    That means every the values of the member variables are the same for every
    instance of the borg class.

    The member variables which are to be shared among all instances must be
    declared as class variables.

    See Also
    --------
    :class:`Singleton`

    Notes
    -----
    One side effect is that if you subclass a borg, the objects all have the
    same state, whereas subclass objects of a singleton have different states.

    Examples
    --------
    Create a borg class:

    >>> from mlpy.modules.patterns import Borg
    >>> class MyClass(Borg):
    >>>     shared_variable = None

    .. note::
        | Project: Code from `ActiveState <http://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/>`_.
        | Code author: `Alex Naanou <http://code.activestate.com/recipes/users/104183/>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    _shared_state = {}

    def __new__(cls, *p, **k):
        # noinspection PyArgumentList
        self = object.__new__(cls, *p, **k)
        self.__dict__ = cls._shared_state
        return self
