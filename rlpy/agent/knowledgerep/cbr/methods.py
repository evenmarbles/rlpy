from __future__ import division, print_function, absolute_import

import weakref
from abc import abstractmethod
from ....framework.registry import RegistryInterface


class CbrMethod(object):
    """The method interface.

    This is the interface for reuse, revision, and retention methods and handles
    registration of all subclasses.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.
    plot_method : callable
        A callback function performing plotting of the data with the following
        signature:
            cb(features, matches, case_base, *args, **kwargs)
    plot_method_params: dict
        Parameters used for plotting.

    Notes
    -----
    All case base reasoning method must inherit from this class.

    """
    __metaclass__ = RegistryInterface

    def __init__(self, owner, plot_method=None, plot_method_params=None):
        self._owner = weakref.proxy(owner)
        self._plot_method = plot_method
        self._plot_method_params = plot_method_params

    @classmethod
    def create(cls, owner, *args, **kwargs):
        return cls(owner, *args, **kwargs)

    @abstractmethod
    def execute(self, features, matches, plot=True):
        """Execute reuse step.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : dict[int, tuple[float, ndarray]]
            The solution identified through the similarity measure.
        plot: bool, optional
            Plot the data.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def plot_data(self, features, matches):
        """Plot the data.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : list[tuple[float, int]]
            The solution identified through the similarity measure.

        """
        if self._plot_method is not None:
            self._plot_method(features, matches, self._owner, **self._plot_method_params)


class ReuseMethod(CbrMethod):
    """The reuse method interface.

    The solutions of the best (or set of best) retrieved cases are used to construct
    the solution for the query case; new generalizations and specializations may occur
    as a consequence of the solution transformation.

    Notes
    -----
    All reuse method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, features, matches, plot=True):
        """Execute reuse step.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : list[tuple[float, int]]
            The solution identified through the similarity measure.
        plot: bool, optional
            Plot the data during the reuse step.

        Returns
        -------
        dict[int, Match] :
            The revised solution.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


class RevisionMethod(CbrMethod):
    """The revision method interface.

    The solutions provided by the query case is evaluated and information about whether the solution
    has or has not provided a desired outcome is gathered.

    Notes
    -----
    All revision method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, features, matches, plot=True):
        """Execute the revision step.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : list[tuple[float, int]]
            The solution identified through the similarity measure.
        plot: bool, optional
            Plot the data during the revision step.

        Returns
        -------
        dict[int, Match] :
            The corrected solution.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


class RetentionMethod(CbrMethod):
    """The retention method interface.

    When the new problem-solving experience can be stored or not stored in memory, depending on
    the revision outcomes and the case base reasoning policy regarding case retention.

    Notes
    -----
    All retention method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, features, matches, plot=True):
        """Execute retention step.

        Parameters
        ----------
        features : list[tuple[str, ndarray]]
            A list of features of the form (`feature_name`, `data_points`).
        matches : list[tuple[float, int]]
            The solution identified through the similarity measure.
        plot: bool, optional
            Plot the data during the retention step.

        Returns
        -------
        int :
            The case id if the case was retained, -1 otherwise.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError
