from functools import partial
import torch

class OpFactory:
    pass


def torch_j_exp(z):
    """
    https://discuss.pytorch.org/t/complex-functions-exp-does-not-support-automatic-differentiation-for-outputs-with-complex-dtype/98039/3
    """

    z = -1j*z # this is a workarond to fix torch complaining
    # on different types of gradient vs inputs.
    # Just make input complex
    return torch.cos(z) + 1j * torch.sin(z)


class CircuitBuilder:
    """ ABC for creating a circuit."""
    operators = OpFactory

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.reset()
        self.qubits = self.get_qubits()

    def get_qubits(self):
        raise NotImplementedError

    def reset(self):
        """ Initialize new circuit """
        raise NotImplementedError

    def inverse(self):
        if not hasattr(self, '_warned'):
            #print('Warning: conjugate is not implemented. Returning same circuit, in case you only care about circuit structure')
            self._warned = True
        return self._circuit

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(**params), *qubits)

    @property
    def circuit(self):
        return self._circuit
    @circuit.setter
    def circuit(self, circuit):
        self._circuit = circuit

    def view(self):
        other = type(self)(self.n_qubits)
        other.circuit = self.circuit
        return other

    def copy(self):
        other = type(self)(self.n_qubits)
        other.circuit = self.circuit.copy()
        return other


class QtreeBuilder(CircuitBuilder):

    def get_qubits(self):
        return list(range(self.n_qubits))

    def reset(self):
        self._circuit = []

    def apply_gate(self, gate, *qubits, **params):
        self._circuit.append(gate(*qubits, **params))

    def inverse(self):
        # --
        # this in-place creature is to be sure that
        # a new gate is creaded on inverse
        def dagger_gate(gate):
            if hasattr(gate, '_parameters'):
                params = gate._parameters
            else:
                params = {}
            new = type(gate)(*gate._qubits, **params)
            new.name = gate.name + '+'
            new.gen_tensor = partial(gate.dag_tensor, gate)
            return new
        #--

        self._circuit = list(reversed([dagger_gate(g) for g in self._circuit]))