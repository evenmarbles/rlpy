from __future__ import division, print_function, absolute_import

import weakref
from abc import ABCMeta, abstractmethod

from .modules import UniqueModule


class Observable(UniqueModule):
    """The observable base class.

    The observable keeps a record of all listeners and notifies them
    of the events they have subscribed to by calling :meth:`Listener.notify`.

    The listeners are notified by calling :meth:`dispatch`. Listeners are notified
    if either the event that is being dispatched is ``None`` or the listener has
    subscribed to a ``None`` event, or the name of the event the listener has subscribed
    to is equal to the name of the dispatching event.

    An event is an object consisting of the `source`; i.e. the observable, the event
    `name`, and the event `data` to be passed to the listener.

    Parameters
    ----------
    mid : str
        The module's unique identifier

    Methods
    -------
    dispatch
    load
    save
    subscribe
    unsubscribe

    Examples
    --------
    >>> from mlpy.modules.patterns import Observable
    >>>
    >>> class MyObservable(Observable):
    >>>     pass
    >>>
    >>> o = MyObservable()

    This defines the observable `MyObservable` and creates
    an instance of it.

    >>> from mlpy.modules.patterns import Listener
    >>>
    >>> class MyListener(Listener):
    >>>
    >>>     def notify(self, event):
    >>>         print "I have been notified!"
    >>>
    >>> l = MyListener(o, "test")

    This defines the listener `MyListener` that when notified will print
    the same text to the console regardless of which event has been thrown
    (as long as the listener has subscribed to the event). Then an instance
    of MyListener is created that subscribes to the event `test` of `MyObservable`.

    When the event `test` is dispatched by the observable, the listener is notified
    and the text is printed on the stdout:

    >>> o.dispatch("test", **{})
    I have been notified!

    """

    class Event(object):
        """Event being dispatched by the observable.

        Parameters
        ----------
        source : Observable
            The observable instance.
        name : str
            The name of the event.
        data : dict
            The information to be send.

        """

        def __init__(self, source, name, **attrs):
            self.source = source
            self.name = name
            self.__dict__.update(**attrs)

    def __init__(self, mid=None):
        super(Observable, self).__init__(mid)

        self._observers = weakref.WeakKeyDictionary()
        """dict[Listener, list[str] or dict[str,]]: A map of observer objects to the
        events they are listening to which can optionally include event specific data for the
        listener"""

    def subscribe(self, observer, name=None, event_data=None):
        """Subscribe to the observable.

        Parameters
        ----------
        observer : Listener
            The listener instance.
        name : str, optional
            The names of the event the observer wants to be notified about. Default is 'default'.
        event_data : dict or None, optional
            The following parameters are available:

            'data' : dict[str]
                A list of name-value pairs of data to be send to the observer.

            'func': dict[str, callable]
                A function to be executed before dispatch. This function may be
                used to update 'data'.

                    'attrib':
                        The name of the variable the return value is stored in.

                    'callable' : callable
                        The function to be executed

        """
        name = name if name is not None else 'default'
        if name == 'default' and name in self._observers[observer]:
            raise ValueError('Only one default event_data can be specified')

        self._observers[observer] = {}
        self._observers[observer][name] = event_data if event_data is not None else {}

    def unsubscribe(self, observer, name=None):
        """Unsubscribe from the observable.

        The observer is removed from the list of listeners.

        Parameters
        ----------
        observer : Listener
            The observer instance.
        name : str, optional
            The names of the event the observer wants to remove. Default is 'default'.

        """
        if name is None:
            name = 'default'
        if observer in self._observers:
            del self._observers[observer][name]

    def dispatch(self, name, *args, **kwargs):
        """Dispatch the event to all listeners.

        Parameters
        ----------
        name : str
            The name of the event to dispatch.
        kwargs : dict
            The information send to the listeners.

        """
        # Notify all listeners of this event
        for listener, events in self._observers.iteritems():
            name2 = name if name in events else 'default' if 'default' in events else None
            if name2 is not None:
                e = Observable.Event(self, name)
                try:
                    e.__dict__.setdefault(events[name2]['func']['attrib'],
                                          events[name2]['func']['callable'](*args))
                except KeyError:
                    pass

                e.__dict__.update(kwargs)

                try:
                    # Create the event to send
                    listener.notify(e)
                except Exception as ex:
                    import sys
                    import traceback
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)
                    sys.exit(1)


class Listener(object):
    """The listener interface.

    A listener subscribes to an observable identifying the events the listener is
    interested in. The observable calls :meth:`notify` to send relevant event information.

    Parameters
    ----------
    o : Observable, optional
        The observable instance.
    events : str or list[str], optional
        The event names the listener wants to be notified about.

    Notes
    -----
    Every class inheriting from Listener must implement :meth:`notify`, which
    defines what to do with the information send by the observable.

    Examples
    --------
    >>> from mlpy.modules.patterns import Observable
    >>>
    >>> class MyObservable(Observable):
    >>>     pass
    >>>
    >>> o = MyObservable()

    This defines the observable `MyObservable` and creates
    an instance of it.

    >>> from mlpy.modules.patterns import Listener
    >>>
    >>> class MyListener(Listener):
    >>>
    >>>     def notify(self, event):
    >>>         print "I have been notified!"
    >>>
    >>> l = MyListener(o, "test")

    This defines the listener `MyListener` that when notified will print
    the same text to the console regardless of which event has been thrown
    (as long as the listener has subscribed to the event). Then an instance
    of MyListener is created that subscribes to the event `test` of `MyObservable`.

    When the event `test` is dispatched by the observable, the listener is notified
    and the text is printed on the stdout:

    >>> o.dispatch("test", **{})
    I have been notified!

    """
    __metaclass__ = ABCMeta

    def __init__(self, o=None, events=None):
        if o is not None:
            o.subscribe(self, events)

    @abstractmethod
    def notify(self, event):
        """Notification from the observable.

        Parameters
        ----------
        event : Observable.Event
            The event object dispatched by the observable consisting of `source`;
            i.e. the observable, the event `name`, and the event `data`.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        This is an abstract method and *must* be implemented by its deriving class.

        """
        raise NotImplementedError
