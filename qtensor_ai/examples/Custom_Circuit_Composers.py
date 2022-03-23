from qtensor_ai import ParallelComposer

'''Implements a quantum circuit that can be used as a quantum neural network. Abbas et. al., arXiv:2011.00027'''
class QNNComposer(ParallelComposer):
    
    def __init__(self, n_qubits, n_layers, higher_order=False):
        '''higher order encoding encodes the products of data points as rotation angles. Depth is N^2.'''
        self.n_layers = n_layers
        self.higher_order = higher_order
        super().__init__(n_qubits)
    
    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q)
    
    def entangling_layer(self):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit)
        control_qubit = self.qubits[-1]
        target_qubit = self.qubits[0]
    
    def encoding_circuit(self, data):
        self.layer_of_Hadamards()
        for i, qubit in enumerate(self.qubits):
            self.apply_gate(self.operators.ZPhase, qubit, alpha=data[:, i])
        if self.higher_order:
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    control_qubit = self.qubits[i]
                    target_qubit = self.qubits[j]
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit)
                    self.apply_gate(self.operators.ZPhase, target_qubit, alpha=data[:, i]*data[:, j])
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit)
    
    # A single layer of rotation gates depending on trainable parameters
    def variational_layer(self, layer, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(self.operators.YPhase, qubit, alpha=layer_params[:, i])
     
    def cost_operator(self):
        for qubit in self.qubits:
            self.apply_gate(self.operators.Z, qubit)
    
    # Building circuit that needs to be measured
    def forward_circuit(self, data, params):

        """
        Parameters
        ----------
        data: torch.Tensor
                Has dimension (n_batch, n_qubits). It contains data to be encoded.

        params: torch.Tensor
                Has dimension (n_batch, n_qubits, n_layers). It stores rotation angles that will be learned.
        """
        
        self.n_batch = data.shape[0]
        self.encoding_circuit(data)
        self.entangling_layer()
        for layer in range(self.n_layers):
            self.variational_layer(layer, params[:, :, layer])
            self.entangling_layer()
            
    '''This function MUST be written for all custom circuit Composers.
    Building circuit whose first amplitude is the expectation value of
    the measured circuit w.r.t. the cost_operator'''
    def updated_full_circuit(self, **parameters):
        data = parameters['data']
        params = parameters['params']
        self.builder.reset() # Clear builder.circuit
        self.forward_circuit(data, params) # Set builder.circuit to the forward circuit according to data and params
        self.cost_operator() # Add the cost operators to builder.circuit
        first_part = self.builder.circuit # Extract builder.circuit at this stage for later use
        self.builder.reset() # Clear builder.circuit
        self.forward_circuit(data, params) # Set builder.circuit to the forward circuit according to data and params
        self.builder.inverse() # Change builder.circuit to it's reverse, which is the forward circuit in reverse
        second_part = self.builder.circuit # Extract the inverse circuit
        self.builder.reset() # Clear builder circuit
        '''The final circuit is forward + cost + inverse.
        The first amplitude is the expectation value of the cost operator
        for the forward circuit initialized with the 0 state.'''
        return first_part + second_part

    '''This function MUST be written for all custom circuit Composers.
    Returns the name of the circuit composer'''
    def name(self):
        return 'QNN'


'''Circuit for learnable quantum embeddings. Lloyd et. al., arXiv:2001.03622'''
class MetricLearningComposer(ParallelComposer):

    def __init__(self, n_qubits, n_layers):
        self.n_layers = n_layers
        super().__init__(n_qubits)
    
    def zz_layer(self, zz_params):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit])
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit])
    
    # A single layer of rotation gates depending on trainable parameters
    def variational_layer(self, gate, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(gate, qubit, alpha=layer_params[:, i])
    
    # Building circuit that needs to be measured#
    def forward_circuit(self, inputs, zz_params, y_params):

        """
        Parameters
        ----------
        inputs: torch.Tensor
                Has dimension (n_batch, n_qubits). It contains data to be encoded.

        zz_params: torch.Tensor
                Has dimension (n_batch, n_qubits-1, n_layers). It stores ZZ angles.

        y_params: torch.Tensor
                Has dimension (n_batch, n_qubits, n_layers). It stores RY angles.
        """

        self.n_batch = inputs.shape[0]
        for layer in range(self.n_layers):
            self.variational_layer(self.operators.XPhase, inputs)
            layer_zz_params = zz_params[:, :, layer]
            self.zz_layer(layer_zz_params)
            layer_y_params = y_params[:, :, layer]
            self.variational_layer(self.operators.YPhase, layer_y_params)
        self.variational_layer(self.operators.XPhase, inputs)



    '''This function MUST be written for all custom circuit Composers.
    Building circuit whose first amplitude is the inner product'''
    def updated_full_circuit(self, **parameters):
        inputs1, inputs2, zz_params, y_params = parameters['inputs1'], parameters['inputs2'], parameters['zz_params'], parameters['y_params']
        self.builder.reset()
        self.device = inputs1.device
        self.forward_circuit(inputs1, zz_params, y_params)
        first_part = self.builder.circuit
        self.builder.reset()
        self.forward_circuit(inputs2, zz_params, y_params)
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        '''There is no cost operator unlike in the previous circuit
        because we just want the overlap between embeddings'''
        return first_part + second_part

    '''This function MUST be written for all custom circuit Composers.
    Returns the name of the circuit composer'''
    def name(self):
        return 'MetricLearning'