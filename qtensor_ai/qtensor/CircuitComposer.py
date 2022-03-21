from .OpFactory import CircuitBuilder

class CircuitComposer():
    """ Director for CircuitBuilder, but with a special way to get the builder"""
    Bulider = CircuitBuilder
    def __init__(self, *args, **params):
        self.params = params
        self.builder = self._get_builder()
        self.n_qubits = self.builder.n_qubits

    #-- Setting up the builder
    def _get_builder_class(self):
        raise NotImplementedError

    def _get_builder(self):
        return self._get_builder_class()()


    #-- Mocking some of bulider behaviour
    @property
    def operators(self):
        return self.builder.operators

    @property
    def circuit(self):
        return self.builder.circuit
    @circuit.setter
    def circuit(self, circuit):
        self.builder.circuit = circuit

    @property
    def qubits(self):
        return self.builder.qubits
    @qubits.setter
    def qubits(self, qubits):
        self.builder.qubits = qubits

    def apply_gate(self, gate, *qubits, **params):
        self.builder.apply_gate(gate, *qubits, **params)

    def conjugate(self):
        # changes builder.circuit, hence self.circuit()
        self.builder.conjugate()
