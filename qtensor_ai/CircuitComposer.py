from .qtensor.CircuitComposer import CircuitComposer
from .OpFactory import ParallelTorchBuilder, ParallelParametricGate

class ParallelComposer(CircuitComposer):
    """
    Base class of circuit composers that are compatible with batch parallelism.
    Circuit composers are where the logic for describing where what gates with
    what parameters should go in the circuit is contained.

    Attributes
    ----------
    final_circuit: list
            List that contains all the gate operations that will be used for
            simulation. This is maintained as static throughout param updates.

    Methods
    -------
    apply_gate(gate, *qubits, **parameters):
            Applies the gate to the qubits with parameters.
            Call this function for building the custom circuit in
            the custom circuit composers.
            This adds the gate to the self.builder.circuit attribute, and
            NOT self.final_circuit.

    update_full_circuit(**parameters): list
            This function needs to be implemented for custom circuit composers.
            Create a copy of the desired circuit based on parameters and returns
            a list of its gates.
            Need to use apply_gate and then access the built circuit with
            the self.builder.circuit attribute. See more methods of builders in
            qtensor_ai.OpFactory.ParallelTorchFactory.
            

    produce_circuit(**parameters):
            Change self.final_circuit according to the new parameters.
            Call this function when wants to update the tensors of the simulated
            circuit according to parameters.

    name(): str
            Returns the name of the circuit composer. Needs to write the name
            function for any custom circuit composers.
    """
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.device = 'cpu'
        self.circuit_initialized = False

        '''This stores whether tensor values of gates are already assigned.
        This parameter is passed to apply_gate and tells it to create a
        new tensor when false. This is important for the model to be a
        static graph and compatible with CUDAGraph'''
        self.final_circuit = []

        super().__init__()
    
    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)
    
    def _get_builder_class(self):
        return ParallelTorchBuilder

    def apply_gate(self, gate, *qubits, **parameters):
        return super().apply_gate(gate, *qubits, **parameters, n_batch=self.n_batch, device=self.device, is_placeholder = self.circuit_initialized)

    '''This is NOT the circuit used in simulation. If self.circuit_initialized
    is true, then the returned circuit would not create new tensor values. 
    The stored parameters are used to update tensor values of self.final_circuit.'''
    def updated_full_circuit(self, **parameters):
        raise NotImplementedError

    def produce_circuit(self, **parameters):
        self.device = list(parameters.values())[0].device
        self.n_batch = list(parameters.values())[0].shape[0]
        new_circuit = self.updated_full_circuit(**parameters)
        if len(self.final_circuit) != 0:
            for self_op, new_op in zip(self.final_circuit, new_circuit):
                if isinstance(self_op, ParallelParametricGate):
                    assert self_op._parameters.keys() == new_op._parameters.keys(), "The gate {} from the new circuit and the gate {} from the old circuit do not have the same parameters: {} and {}".format(new_op, self_op, self_op._parameters.keys(), new_op._parameters.keys())
                    for parameter_key in new_op._parameters.keys():
                        self_op.parameters[parameter_key] = new_op._parameters[parameter_key]
                else:
                    assert not isinstance(new_op, ParallelParametricGate), "The gate {} from the new circuit is parametric while the gate {} from the old circuit is non-parametric".format(new_op, self_op)
                    self_op._parameters['n_batch'] = new_op._parameters['n_batch']
                
        else:
            '''Initializing circuit'''
            self.final_circuit = new_circuit
            self.circuit_initialized = True

    def name(self):
        raise NotImplementedError