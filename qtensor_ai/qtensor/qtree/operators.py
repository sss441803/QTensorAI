"""
This module implements quantum gates from the CMON set of Google
"""
import numpy as np

from functools import partial

import uuid


class Gate:
    """
    Base class for quantum gates.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).


    Methods
    -------
    gen_tensor(): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    dagger():
            Class method that returns a daggered class

    dagger_me():
            Changes the instance's gen_tensor inplace

    is_parametric(): bool
            Returns False for gates without parameters
    """

    def __init__(self, *qubits):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = { }
        self._check_qubit_count(qubits)
        self.name = type(self).__name__

    def _check_qubit_count(self, qubits):
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits for gate {}:\n"
                "{}, required: {}".format(
                    self.name, len(qubits), n_qubits))

    @classmethod
    def dag_tensor(cls, inst):
        return cls.gen_tensor(inst).conj().T

    @classmethod
    def dagger(cls):
        # This thing modifies the base class itself.
        orig = cls.gen_tensor
        def conj_tensor(self):
            t = orig(self)
            return t.conj().T
        cls.gen_tensor = conj_tensor
        cls.__name__ += '.dag'
        return cls

    def dagger_me(self):
        # Maybe the better way is to create a separate object
        # Warning: dagger().dagger().dagger() will define many things
        self.gen_tensor = partial(self.dag_tensor, self)
        self.name += '+'
        return self


    def gen_tensor(self):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self._parameters

    def is_parametric(self):
        return len(self.parameters) > 0

    @property
    def qubits(self):
        return self._qubits

    @property
    def changed_qubits(self):
        return tuple(self._qubits[idx] for idx in self._changes_qubits)

    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )

    def __repr__(self):
        return self.__str__()


class ParametricGate(Gate):
    """
    Gate with parameters.

    Attributes
    ----------
    name: str
            The name of the gate

    parameters: dict
             Parameters used by the gate (may be empty)

    qubits: tuple
            Qubits the gate acts on

    changed_qubits : tuple
            Tuple of ints which states what qubit's bases are changed
            (along which qubits the gate is not diagonal).

    Methods
    -------
    gen_tensor(\\**parameters={}): numpy.array
            The gate tensor. For each qubit a gate
            either introduces a new variable (non-diagonal gate, like X)
            or does not (diagonal gate, like T). Multiqubit gates
            can be diagonal on some of the variables, and not diagonal on
            others (like ccX). The order of dimensions IS ALWAYS
            (new_a, a, b_new, b, c, d_new, d, ...)

    is_parametric(): bool
            Returns True
    """
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = parameters
        self._check_qubit_count(qubits)
        self.name = type(self).__name__

    def _check_qubit_count(self, qubits):
        # fill parameters and save a copy
        filled_parameters = {}
        for par, value in self._parameters.items():
            if isinstance(value, placeholder):
                filled_parameters[par] = np.zeros(value.shape)
            else:
                filled_parameters[par] = value
        parameters = self._parameters

        # substitute parameters by filled parameters
        # to evaluate tensor shape
        self._parameters = filled_parameters
        n_qubits = len(self.gen_tensor().shape) - len(
            self._changes_qubits)
        # return back the saved version
        self._parameters = parameters

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits: {}, required: {}".format(
                    len(qubits), n_qubits))

    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            return self._gen_tensor(**self._parameters)
        else:
            return self._gen_tensor(**parameters)

    def __str__(self):
        par_str = (",".join("{}={}".format(
            param_name,
            '?.??' if isinstance(param_value, placeholder)
            else '{:.2f}'.format(float(param_value)))
                            for param_name, param_value in
                            sorted(self._parameters.items(),
                                   key=lambda pair: pair[0])))

        return ("{}".format(self.name) + "[" + par_str + "]" +
                "({})".format(','.join(map(str, self._qubits))))


def op_scale(factor, operator, name):
    """
    Scales a gate class by a scalar. The resulting class
    will have a scaled tensor

    It is not recommended to use this many times because of
    possibly low performance

    Parameters
    ----------
    factor: float
          scaling factor
    operator: class
          operator to modify
    name: str
          Name of the new class
    Returns
    -------
    class
    """
    def gen_tensor(self):
        return factor * operator.gen_tensor(operator)

    attr_dict = {attr: getattr(operator, attr) for attr in dir(operator)}
    attr_dict['gen_tensor'] = gen_tensor

    return type(name, (operator, ), attr_dict)



class placeholder:
    """
    Class for placeholders. Placeholders are used to implement
    symbolic computation. This class is very similar to the
    Tensorflow's placeholder class.

    Attributes
    ----------
    name: str, default None
          Name of the placeholder (for clarity)
    shape: tuple, default None
          Shape of the tensor the placeholder represent
    """
    def __init__(self, name=None, shape=()):
        self._name = name
        self._shape = shape
        self._uuid = uuid.uuid4()

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def uuid(self):
        return self._uuid


def _flatten(l):
    """
    https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    Parameters
    ----------
    l: iterable
        arbitrarily nested list of lists

    Returns
    -------
    generator of a flattened list
    """
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from _flatten(el)
        else:
            yield el
