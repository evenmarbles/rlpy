import weakref
import numpy as np
from collections import namedtuple

from itertools import count

from ...framework.observer import Observable, Listener


class Planner(object):
    """

    """

    class ValueState(Observable):
        """

        """

        @property
        def value(self):
            return self._value

        def __init__(self, value=None):
            super(Planner.ValueState, self).__init__()

            self._value = value if value is not None else 0.0

        def _set_value(self, value):
            """Changes the value of this ValueState nd sends a notification
            to all of the observers of this ValueState.

            Parameters
            ----------
            value : float
                The new value of this ValueState

            """
            change = value - self._value
            self._value = value
            self.dispatch('value_change', change)

    class DecisionState(ValueState, Listener):
        """

        """
        _instance = object()

        MaxAction = namedtuple('MaxAction', ['action', 'dsa'])

        class Action(Listener):
            """

            """
            _ids = count(0)
            _instance = object()

            ValueStateProbability = namedtuple('ValueStateProbability', ['vs', 'proba'])

            @property
            def q(self):
                """float : The q-value for each state-action."""
                return self._q

            @property
            def mdp(self):
                return self._model

            def __init__(self, token, parent, model):
                if token is not self._instance:
                    raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))

                super(Planner.DecisionState.Action, self).__init__()

                self._parent = parent
                """:type: DecisionState"""
                self._model = model
                """StateAction"""

                self._q = 0.0
                self._errorbound = 0.0

                self._successors = {}
                """dict[MDPState, ValueStateProbability]"""

                self._mid = "%s.%s:%i" % (self.__class__.__module__, self.__class__.__name__, next(self._ids))

            def __repr__(self):
                return self._mid

            @classmethod
            def create(cls, parent, model):
                result = cls(cls._instance, parent, model)
                parent._action_values[model._action] = result
                result.compute_successors()
                result.compute_value()
                return result

            def debug(self):
                for s, vsp in self._successors.iteritems():
                    vs = vsp.vs
                    if vsp.vs is None:
                        vs = self._parent
                    print("\t\t%f: %s(%f)" % (vsp.proba, s, vs.value))
                print("\t\tr = %f, Q = %f" % (self._model.reward, self.q))

            def notify(self, event):
                if event.name != 'value_change':
                    return

                change = event.change
                if change < 0:
                    change = -change
                change *= self._parent._planner._gamma
                self._errorbound += change

            # TODO: need to move to Prioritized Sweeping?
            def compute_successors(self):
                model_succs = self._model.successor_probabilities

                # remove erstwhile successors
                for s, vsp in self._successors.items():
                    if s not in model_succs.keys():
                        if vsp.vs is not None:
                            vsp.vs.unsubscribe(self, 'value_change')
                        else:
                            self._parent.unsubscribe(self, 'value_change')
                        del self._successors[s]

                for s, proba in model_succs.iteritems():
                    if s not in self._successors:
                        succ = self._parent._planner._successor_value(s)
                        assert succ
                        succ.subscribe(self, 'value_change', {
                            'func': {
                                'attrib': 'change',
                                'callable': lambda x: proba * x
                            }
                        })

                        # use None reference as a proxy for parent to avoid creating
                        # a cycle of strong references
                        if succ == self._parent:
                            succ = None
                        self._successors[s] = Planner.DecisionState.Action.ValueStateProbability(
                            succ if succ._mid != self._parent._mid else weakref.proxy(succ), proba)
                    elif self._successors[s].proba != proba:
                        # update existing successor
                        vs = self._successors[s].vs
                        if vs is not None:
                            vs.subscribe(self, 'value_change', {
                                'func': {
                                    'attrib': 'change',
                                    'callable': lambda x: proba * x
                                }
                            })
                            self._successors[s] = self._successors[s]._replace(proba=proba)
                        else:
                            self._parent.subscribe(self, 'value_change', {
                                'func': {
                                    'attrib': 'change',
                                    'callable': lambda x: proba * x
                                }
                            })
                            self._successors[s] = self._successors[s]._replace(vs=self._parent, proba=proba)
                self._errorbound = np.infty

            def compute_value(self):
                self._q = 0.0
                for vsp in self._successors.itervalues():
                    succ = vsp.vs if vsp.vs is not None else self._parent
                    self._q += vsp.proba * succ.value

                self._q *= self._parent._planner._gamma
                self._q += self._model.reward
                self._errorbound = 0.0

            def update_value(self):
                if self._errorbound > 0:
                    self.compute_value()

        # -----------------------------
        # DecisionState
        # -----------------------------
        @property
        def state(self):
            return self._model.state

        @property
        def policy_action(self):
            return self._max.action

        @property
        def policy_model(self):
            return self._model

        def __init__(self, token, planner, model):
            if token is not self._instance:
                raise ValueError("Use 'create' to construct {0}".format(self.__class__.__name__))

            super(Planner.DecisionState, self).__init__()

            self._planner = planner
            """:type: Planner"""
            self._model = model
            """:type: StateData"""

            self._action_values = {}
            """dict[MDPAction, Action]"""
            self._max = Planner.DecisionState.MaxAction(None, None)
            """MaxAction"""

            self._errorbound = 0.0

            self._inbox = weakref.WeakSet()
            """:type: set[]"""

            # print("Planner.DecisionState: %s" % self._model.state)

        def __del__(self):
            pass
            # print("Planner.DecisionState.__del__: %s" % self._model.state)
            # for a in self._action_values.itervalues():
            #     for vsp in a._successors.itervalues():
            #         if vsp.vs is not None:
            #             del vsp

        def __str__(self, level=0):
            ret = "\t" * level + repr(self._model.state) + "\n"
            for av in self._action_values.itervalues():
                for vsp in av._successors.itervalues():
                    if vsp.vs._mid != self._mid:
                        ret += vsp.vs.__str__(level + 1)
            return ret

        def __repr__(self):
            return self._mid

        @classmethod
        def create(cls, planner, model, is_completion):
            result = cls(cls._instance, planner, model)
            if is_completion:
                planner._completions[model.state] = result
            planner._nonterminals[model.state] = result
            result._model.subscribe(result, 'mdp_change')
            result.initialize()
            return result

        def debug(self):
            print("State %s" % self.state)
            for a, dsa in self._action_values.iteritems():
                print("\tAction %s" % a)
                dsa.debug()
            print

        def notify(self, event):
            self._inbox.add(event.action)
            self._planner._inbox.add(self)

        def initialize(self):
            """Compute the initial value of this state.

            Note that this method is separate from the constructor, since otherwise
            a cycle in the MDP structure might cause the planner to try to create a
            new DecisionState object for a state in which another DecisionState object
            is still being constructed. This way, we can update the hash table mapping
            states of DecisionState objects after construction but before initialization,
            since the initialization is what causes the cycle.

            """
            for act, model in self._model.state_actions.iteritems():
                if len(self._action_values) <= 0 or act not in self._action_values:
                    # this can lead to the construction and initialization of other
                    # DecisionState objects
                    dsa = type(self).Action.create(weakref.proxy(self), model)
                    assert act in self._action_values
                    assert dsa == self._action_values[act]

                if self._max.dsa is None or self._max.dsa.q < dsa.q:
                    self._max = Planner.DecisionState.MaxAction(act, dsa)

            self._set_value(self._max.dsa.q)

        def propagate_mdp_change(self):
            for i in self._inbox:
                self._action_values[i].compute_successors()

            self._inbox.clear()
            self.propagate_value_change()

        def propagate_value_change(self):
            original_action = self._max.action

            assert self._action_values
            it = self._action_values.iteritems()

            act, dsa = it.next()
            dsa.update_value()
            self._max = Planner.DecisionState.MaxAction(act, dsa)

            while True:
                try:
                    act, dsa = it.next()
                    dsa.update_value()
                    if self._max.dsa.q < dsa.q:
                        self._max = Planner.DecisionState.MaxAction(act, dsa)
                except StopIteration:
                    break

            if self._max.action != original_action:
                # this is a new policy action
                self._planner._outbox[self] = original_action

            if self._max.dsa.q != self.value:
                self._set_value(self._max.dsa.q)

        def propagate_policy_change(self, original):
            if original != self._max.action:
                self.dispatch('policy_change')

    # -----------------------------
    # Planner
    # -----------------------------
    @property
    def mdp(self):
        return self._mdp

    def __init__(self, mdp, terminal, goal, gamma=None):
        self._mdp = mdp
        """:type: MDP"""

        self._terminal = terminal
        """:type: callable"""
        self._goal = goal
        """:type: callable"""

        self._gamma = gamma if gamma is not None else 1.0

        self._nonterminals = weakref.WeakValueDictionary()
        """:type: dict[MDPState, DecisionState] : Each key is the return value of the state()
        method for an MDPStateData. Each datum is the DecisionState object constructed
        with that MDPStateData"""

        self._completions = weakref.WeakValueDictionary()
        """:type: dict[MDPState, ValueState] : Each key is a basis state that appears
        in the value of successors() for sine MDPStateAction object. The data
        pointers are either DecisionState objects in nonterminals or ValueState
        objects representing terminal states"""

        self._inbox = weakref.WeakSet()
        """:type: set[DecisionState] : Includes all DecisionState objects that have not
        executed propagate_value_change since their observed MDPStateData object
        changed"""
        self._outbox = weakref.WeakKeyDictionary()
        """:type: dict[DecisionState, MDPAction] : Each key is a DecisionState object that
        has changed their policy action. The data is the original policy action
        before the first change. (The data can be used to avoid sending policy
        change notifications unnecessarily, when a DecisionState object switches
        back to its original policy action.)"""

    def initialize(self):
        self._mdp.initialize()

    def policy(self, state):
        """Outputs (after possibly computing) the optimal policy for
        the MDP.

        Parameters
        ----------
        state : MDPState
            The state at which to evaluate the optimal policy

        Returns
        -------
        DecisionState:
            An object specifying the optimal child action, as well as
            giving access to the StateData describing that state-action's
            behavior.

        """
        ds = self._policy(self.mdp.state_data(state))
        self.plan()
        return ds

    def update(self, state, act, succ):
        self._mdp.update(state, act, succ)

    def plan(self):
        for ds in self._inbox:
            ds.propagate_mdp_change()
        self._inbox.clear()

        self._propagate_changes()

        for ds, act in self._outbox.iteritems():
            if ds is not None:
                ds.propagate_policy_change(act)
        self._outbox = {}

    def debug(self):
        print("model:")
        for ds in self._nonterminals.itervalues():
            ds.debug()
        print("end model")

        print("value function:")
        self.write_value_function()
        print("end value function")

        print("policy:")
        self.write_policy()
        print("end policy")

    def write_value_function(self):
        for s, ds in self._nonterminals.iteritems():
            print("{0} {1}".format(s, ds.value))

    def write_policy(self):
        policy = {}
        for s, ds in self._nonterminals.iteritems():
            states = policy.setdefault(ds.policy_action, [])
            states.append(ds.state)

        for a, states in policy.iteritems():
            print("# {0}:".format(a))
            for s in states:
                print(s)
        print('\n')

    def _propagate_changes(self):
        pass

    def _policy(self, state_data, is_completion=False):
        """

        Parameters
        ----------
        state_data : MDPState.StateData
            The state at which the policy should be evaluated.

        Returns
        -------
        DecisionState

        """
        try:
            ds = self._nonterminals[state_data.state]
        except KeyError:
            ds = type(self).DecisionState.create(self, state_data, is_completion)
            assert state_data.state in self._nonterminals
            assert ds == self._nonterminals[state_data.state]
        return ds

    def _successor_value(self, successor):
        """

        Parameters
        ----------
        successor : MDPState
            A basis state that appears in teh value of successor_probabilities()
            for some MDPStateAction object

        Returns
        -------
        ValueState

        """
        try:
            vs = self._completions[successor]
        except KeyError:
            if self._terminal(successor):
                vs = Planner.ValueState(self._goal(successor))
            else:
                vs = self._policy(self.mdp.state_data(successor), is_completion=True)
                assert successor in self._completions
                assert vs == self._completions[successor]
        return vs
