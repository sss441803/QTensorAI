import torch
import numpy as np
from .qtensor import qtree


###########################################################################
#Changing the Var class in qtree.optimizer. The Var class gets initialized#
#to the wrong size when used in parallel mode. Here, size = 2 is forced   #
###########################################################################

class ParallelVar(object):
    """
    Index class. Primarily used to store variable id:size pairs
    """
    def __init__(self, identity, size=2, name=None):
        """
        Initialize the variable
        identity: int
              Index identifier. We use mainly integers here to
              make it play nicely with graphical models.
        size: int, optional
              Size of the index. Default 2
        name: str, optional
              Optional name tag. Defaults to "v[{identity}]"
        """
        size = 2
        self._identity = identity
        self._size = size
        if name is None:
            name = f"v_{identity}"
        self._name = name
        self.__hash = hash((identity, name, size))

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def identity(self):
        return self._identity

    def copy(self, identity=None, size=None, name=None):
        if identity is None:
            identity = self._identity
        if size is None:
            size = self._size

        return Var(identity, size, name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return int(self.identity)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.identity == other.identity
                and self.name == other.name)

    def __lt__(self, other):  # this is required for sorting
        return self.identity < other.identity
    
qtree.optimizer.Var = ParallelVar

######################################################################
#Defining the contraction backend used for parallel processing.      #
#1. get_einsum_expr is changed to add 'Z' before all strings for     #
#   tensors to indicate that the 0th dimension (batch) is in parallel#
#2. torch.sum at the end of process_bucket has axis=1 instead of 0   #
#3. In get_sliced_bucked, transpose_order is modified to             #
#   tensor_transpose_order to be used in data.permute. We add 0 to   #
#   the beginning of the permutation order to preserve the batches   #
#4. In get_sliced_bucked, slice_bounds is initialized with           #
#   slice(None). This is the slice for the batch dimension, telling  #
#   it to keep all the batches.                                      #
######################################################################

from .qtensor.qtree.utils import num_to_alpha
import numpy as np

def get_einsum_expr(idx1, idx2):
    """
    Takes two tuples of indices and returns an einsum expression
    to evaluate the sum over repeating indices

    Parameters
    ----------
    idx1 : list-like
          indices of the first argument
    idx2 : list-like
          indices of the second argument

    Returns
    -------
    expr : str
          Einsum command to sum over indices repeating in idx1
          and idx2.
    """
    result_indices = sorted(list(set(idx1 + idx2)))
    # remap indices to reduce their order, as einsum does not like
    # large numbers
    idx_to_least_idx = {old_idx: new_idx for new_idx, old_idx
                        in enumerate(result_indices)}

    str1 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx1)
    str2 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in idx2)
    str3 = ''.join(num_to_alpha(idx_to_least_idx[ii]) for ii in result_indices)
    return 'Z' + str1 + ',' + 'Z' + str2 + '->' + 'Z' + str3

from .qtensor.contraction_backends import ContractionBackend

class ParallelTorchBackend(ContractionBackend):

    def __init__(self, device = "cpu"):
        self.device = device
        self.cuda_available = torch.cuda.is_available()

    def process_bucket(self, bucket, no_sum=False):
        
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        
        for tensor in bucket[1:]:
            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )
            result_data = torch.einsum(expr, result_data, tensor.data)
            # Merge and sort indices and shapes
            result_indices = tuple(sorted(
                set(result_indices + tensor.indices),
                key=int)
            )

        if len(result_indices) > 0:
            if not no_sum:  # trim first index
                first_index, *result_indices = result_indices
            else:
                first_index, *_ = result_indices
            tag = first_index.identity
        else:
            tag = 'f'
            result_indices = []

        # reduce
        if no_sum:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=result_data)
        else:
            result = qtree.optimizer.Tensor(f'E{tag}', result_indices,
                                data=torch.sum(result_data, axis=1))
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                transpose_order = np.argsort(list(map(int, tensor.indices)))
                tensor_transpose_order = [0] + list(map(lambda x : x + 1, transpose_order))
                
                data = data_dict[tensor.data_key]
                #if self.cuda_available:
                #    cuda = torch.device('cuda')
                #    data = data.to(cuda)
                
                data = data.permute(tuple(tensor_transpose_order))
                # transpose indices
                indices_sorted = [tensor.indices[pp]
                                  for pp in transpose_order]

                # slice data
                slice_bounds = [slice(None)]
                for idx in indices_sorted:
                    try:
                        slice_bounds.append(slice_dict[idx])
                    except KeyError:
                        slice_bounds.append(slice(None))
                
                data = data[tuple(slice_bounds)]

                # update indices
                indices_sliced = [idx.copy(size=size) for idx, size in
                                  zip(indices_sorted, data.shape)]
                indices_sliced = [i for sl, i in zip(slice_bounds, indices_sliced) if not isinstance(sl, int)]
                assert len(data.shape) - 1 == len(indices_sliced)

                sliced_bucket.append(
                    tensor.copy(indices=indices_sliced, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return result.data


########################################################################
#Modifying gate classes to be compatible with parallelism              #
########################################################################

from .qtensor.qtree.operators import placeholder, ParametricGate
    
'''Batch parallel function for implementing complex conjugate tranpose'''
def ParallelConjT(tensor):
    n_dims = len(tensor.shape)
    permutation = [0] + [n_dims - i for i in range(1, n_dims)]
    return torch.permute(tensor, tuple(permutation)).conj()
    
class ParallelParametricGate(ParametricGate):

    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        self.name = type(self).__name__
        # supposedly unique id for an instance
        self._parameters = parameters
        self.is_inverse = False
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self._check_qubit_count(qubits)
    
    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            tensor = self._gen_tensor(**self._parameters)
        else:
            tensor = self._gen_tensor(**parameters)
        if self.is_inverse:
            tensor = ParallelConjT(tensor)
        return tensor
    
    def _check_qubit_count(self, qubits):
        # fill parameters and save a copy
        filled_parameters = {}
        for par, value in self._parameters.items():
            filled_parameters[par] = value

        # substitute parameters by filled parameters
        # to evaluate tensor shape
        self._parameters = filled_parameters
        ''' n_qubits has an additional -1 compared to non-parallel implementation,
            because the tensor would have an additional batch dimension'''
        n_qubits = len(self.gen_tensor().shape) - 1 - len(
            self._changes_qubits)
        # return back the saved version

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits for gate {}:\n"
                "{}, required: {}".format(
                    self.name, len(qubits), n_qubits))
            
    def __str__(self):
        # We will not print parameters such as n_batch and device because they are not gate parameters, but rather model parameters
        self_params_no_device = dict(filter(lambda key_value: key_value[0] not in ['n_batch', 'device', 'is_placeholder'], self._parameters.items()))
        par_str = (",".join("{}={}".format(
            param_name,
            '?.??' if isinstance(param_value, placeholder)
            else '{:.2f}'.format(float(param_value[0])))
                            for param_name, param_value in
                            sorted(self_params_no_device.items(),
                                   key=lambda pair: pair[0])))

        return ("{}".format(self.name) + "[" + par_str + "]" +
                "({})".format(','.join(map(str, self._qubits))))
    
class ParallelGate(ParametricGate):
    
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        self.name = type(self).__name__
        # supposedly unique id for an instance
        self.is_inverse = False
        self._parameters = parameters
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self._check_qubit_count(qubits)
        
    def gen_tensor(self, **parameters):
        if len(parameters) == 0:
            tensor = self._gen_tensor(**self._parameters)
        else:
            tensor = self._gen_tensor(**parameters)
        if self.is_inverse:
            tensor = ParallelConjT(tensor)
        return tensor
    
    def _check_qubit_count(self, qubits):
        ''' n_qubits has an additional -1 compared to non-parallel implementation,
            because the tensor would have an additional batch dimension.'''
        n_qubits = len(self.gen_tensor().shape) - 1 - len(
            self._changes_qubits)
        # return back the saved version

        if len(qubits) != n_qubits:
            raise ValueError(
                "Wrong number of qubits for gate {}:\n"
                "{}, required: {}".format(
                    self.name, len(qubits), n_qubits))
    
    def __str__(self):
        return ("{}".format(self.name) +
                "({})".format(','.join(map(str, self._qubits)))
        )

########################################################################
#Defining parallel operators with an batch dimension and the Parallel  #
#factory 'ParallelTorchFactory'                                        #
########################################################################
    
from .qtensor.OpFactory import torch_j_exp
    
class M(ParallelGate):
    name = 'M'
    _changes_qubits = (0, )
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the introduction of a variable in the graphical model
    """

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([[1, 0], [0, 1]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'], 1, 1)
    
qtree.operators.M = M
    
class H(ParallelGate):
    name = 'H'
    _changes_qubits = (0, )

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = 1/np.sqrt(2) * torch.tensor([[1,  1], [1, -1]]).to(device)
        super().__init__(*qubits, **parameters)
    
    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(parameters['n_batch'],1,1)
    
class Z(ParallelGate):
    name = 'Z'
    _changes_qubits = tuple()

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([1, -1]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'],1)

class X(ParallelGate):
    name = 'X'
    _changes_qubits = (0, )

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([[0, 1], [1, 0]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'],1,1)

class Y(ParallelGate):
    name = 'Y'
    _changes_qubits = (0, )

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([[0, -1j], [1j, 0]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'],1,1)

class cX(ParallelGate):
    name = 'cX'
    _changes_qubits=(1, )

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([[[1., 0.],
                                         [0., 1.]],
                                        [[0., 1.],
                                         [1., 0.]]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'],1,1,1)
    
class cZ(ParallelGate):
    name = 'cZ'
    _changes_qubits = tuple()

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.tensor = torch.tensor([[1, 1], [1, -1]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        return self.tensor.repeat(self._parameters['n_batch'],1,1)
    
class ZZ(ParallelParametricGate):
    name = 'ZZ'
    _changes_qubits=tuple()
    parameter_count=1
    
    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.t = torch.tensor([[-1, +1],[+1, -1]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        alpha = self.parameters['alpha']
        device = alpha.device
        return torch_j_exp(1j*alpha.unsqueeze(1)[:,None]*self.t*np.pi/2)
    
class ZPhase(ParallelParametricGate):
    name = 'ZPhase'
    _changes_qubits = tuple()
    parameter_count = 1

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.t = torch.tensor([0, 1]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        """Rotation along Z axis"""
        alpha = parameters['alpha']
        return torch_j_exp(1j*self.t*np.pi*alpha.unsqueeze(1))
    
class YPhase(ParallelParametricGate):
    name = 'YPhase'
    parameter_count = 1
    _changes_qubits = (0, )

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.ct = torch.tensor([[1,0],[0,1]]).to(device)
            self.st = torch.tensor([[0,-1],[1,0]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        """Rotation along Y axis"""
        alpha = parameters['alpha']
        c = torch.cos(np.pi * alpha / 2).unsqueeze(1)[:,None]*self.ct
        s = torch.sin(np.pi * alpha / 2).unsqueeze(1)[:,None]*self.st
        #g = torch_j_exp(1j * np.pi * alpha / 2).unsqueeze(1)[:,None]
        #return g*(c + s)
        return c+s
    
class XPhase(ParallelParametricGate):
    name = 'XPhase'
    _changes_qubits = (0, )
    parameter_count = 1

    def __init__(self, *qubits, **parameters):
        device = parameters['device']
        is_placeholder = parameters['is_placeholder']
        if not is_placeholder:
            self.ct = torch.tensor([[1,0],[0,1]]).to(device)
            self.st = torch.tensor([[0, -1j], [-1j, 0]]).to(device)
        super().__init__(*qubits, **parameters)

    def _gen_tensor(self, **parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']
        c = torch.cos(np.pi*alpha/2).unsqueeze(1)[:,None]*self.ct
        s = torch.sin(np.pi*alpha/2).unsqueeze(1)[:,None]*self.st
        g = torch_j_exp(1j*np.pi*alpha/2).unsqueeze(1)[:,None]
        return g*c + g*s
    
class ParallelTorchFactory:
    pass

ParallelTorchFactory.ZZ = ZZ
ParallelTorchFactory.XPhase = XPhase
ParallelTorchFactory.YPhase = YPhase
ParallelTorchFactory.ZPhase = ZPhase
ParallelTorchFactory.H = H
ParallelTorchFactory.Z = Z
ParallelTorchFactory.X = X
ParallelTorchFactory.Y = Y
ParallelTorchFactory.cZ = cZ
ParallelTorchFactory.cX = cX

########################################################################
#Defining parallel circuit composers.                                  #
########################################################################

from .qtensor.OpFactory import QtreeBuilder
from .qtensor.CircuitComposer import CircuitComposer

class ParallelTorchBuilder(QtreeBuilder):
    operators = ParallelTorchFactory

    def inverse(self):

        def dagger_gate(gate):
            if hasattr(gate, '_parameters'):
                params = gate._parameters
            else:
                params = {}
            new = type(gate)(*gate._qubits, **params)
            new.name = gate.name + '+'
            new.is_inverse = True
            return new

        self._circuit = list(reversed([dagger_gate(g) for g in self._circuit]))

    
class ParallelTorchQkernelComposer(CircuitComposer):
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        '''higher order encoding encodes the products of data points as rotation angles. Depth is N^2. 	arXiv:2011.00027'''
        self.higher_order = False
        self.device = 'cpu'
        self.expectation_circuit_initialized = False
        self.static_circuit = []
        super().__init__()
    
    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)
    
    def _get_builder_class(self):
        return ParallelTorchBuilder
    
    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
    
    def entangling_layer(self):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
        control_qubit = self.qubits[-1]
        target_qubit = self.qubits[0]
    
    def encoding_circuit(self, data):
        self.layer_of_Hadamards()
        for i, qubit in enumerate(self.qubits):
            self.apply_gate(self.operators.ZPhase, qubit, alpha=data[:, i], device=self.device, is_placeholder = self.expectation_circuit_initialized)
        if self.higher_order:
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    control_qubit = self.qubits[i]
                    target_qubit = self.qubits[j]
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
                    self.apply_gate(self.operators.ZPhase, target_qubit, alpha=data[:, i]*data[:, j], device=self.device, is_placeholder = self.expectation_circuit_initialized)
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
    
    '''A single layer of rotation gates depending on trainable parameters'''
    def variational_layer(self, layer, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(self.operators.YPhase, qubit, alpha=layer_params[:, i], device=self.device, is_placeholder = self.expectation_circuit_initialized)
     
    def cost_operator(self):
        #self.apply_gate(self.operators.Z, self.qubits[0], n_batch=self.n_batch)
        for qubit in self.qubits:
            self.apply_gate(self.operators.Z, qubit, n_batch=self.n_batch, device=self.device, is_placeholder = self.expectation_circuit_initialized)
    
    '''Building circuit that needs to be measured'''
    def forward_circuit(self, data, params):
        '''data is a np.ndarray that has dimension (n_batch, n_qubits). It contains data to be encoded'''
        '''params is a np.ndarray that has dimension (n_batch, n_qubits, layers). It stores rotation angles that will be learned'''
        self.n_batch = data.shape[0]
        layers = params.shape[2]
        self.encoding_circuit(data)
        self.entangling_layer()
        for layer in range(layers):
            self.variational_layer(layer, params[:, :, layer])
            self.entangling_layer()
            
    '''Building circuit whose first amplitude is the expectation value of the measured circuit wrt to the cost_operator'''
    def updated_expectation_circuit(self, data, params):
        self.builder.reset()
        self.device = data.device
        self.forward_circuit(data, params)
        self.cost_operator()
        first_part = self.builder.circuit
        self.builder.reset()
        self.forward_circuit(data, params)
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        return first_part + second_part

    def expectation_circuit(self, data, params):
        new_circuit = self.updated_expectation_circuit(data, params)
        if len(self.static_circuit) != 0:
            for self_op, new_op in zip(self.static_circuit, new_circuit):
                if isinstance(self_op, ParallelParametricGate):
                    self_op._parameters['alpha'] = new_op._parameters['alpha']
                else:
                    self_op._parameters['n_batch'] = new_op._parameters['n_batch']
        else:
            '''Initializing circuit'''
            self.static_circuit = new_circuit
            self.expectation_circuit_initialized = True

    def name():
        return 'Qkernel'



class MetricLearningCircuitComposer(ParallelTorchQkernelComposer):

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.device = 'cpu'
        self.expectation_circuit_initialized = False
        self.static_circuit = []
        super().__init__(n_qubits)
    
    def zz_layer(self, zz_params):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit], device=self.device, is_placeholder = self.expectation_circuit_initialized)
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.ZZ, control_qubit, target_qubit, alpha=zz_params[:,control_qubit], device=self.device, is_placeholder = self.expectation_circuit_initialized)
    
    '''A single layer of rotation gates depending on trainable parameters'''
    def variational_layer(self, gate, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(gate, qubit, alpha=layer_params[:, i], device=self.device, is_placeholder = self.expectation_circuit_initialized)
    
    '''Building circuit that needs to be measured'''
    def circuit(self, inputs, zz_params, y_params):
        '''data is a np.ndarray that has dimension (n_batch, n_qubits). It contains data to be encoded'''
        '''params is a np.ndarray that has dimension (n_batch, n_qubits, layers). It stores rotation angles that will be learned'''
        self.n_batch = inputs.shape[0]
        self.layers = zz_params.shape[2]
        for layer in range(self.layers):
            self.variational_layer(self.operators.XPhase, inputs)
            layer_zz_params = zz_params[:, :, layer]
            self.zz_layer(layer_zz_params)
            layer_y_params = y_params[:, :, layer]
            self.variational_layer(self.operators.YPhase, layer_y_params)
        self.variational_layer(self.operators.XPhase, inputs)

    '''Building circuit whose first amplitude is the expectation value of the measured circuit wrt to the cost_operator'''
    def updated_expectation_circuit(self, inputs1, inputs2, zz_params, y_params):
        self.builder.reset()
        self.device = inputs1.device
        self.circuit(inputs1, zz_params, y_params)
        first_part = self.builder.circuit
        self.builder.reset()
        self.circuit(inputs2, zz_params, y_params)
        self.builder.inverse()
        second_part = self.builder.circuit
        self.builder.reset()
        return first_part + second_part

    def expectation_circuit(self, inputs1, inputs2, zz_params, y_params):
        new_circuit = self.updated_expectation_circuit(inputs1, inputs2, zz_params, y_params)

        if len(self.static_circuit) != 0:
            for self_op, new_op in zip(self.static_circuit, new_circuit):
                if isinstance(self_op, ParallelParametricGate):
                    self_op._parameters['alpha'] = new_op._parameters['alpha']
                else:
                    self_op._parameters['n_batch'] = new_op._parameters['n_batch']
        else:
            '''Initializing circuit'''
            self.static_circuit = new_circuit
            self.expectation_circuit_initialized = True

    def name():
        return 'MetricLearning'


########################################################################
#Defining parallel tensornetwork class                                 #
#Modifying circ2buckets in qtree.optimizer                             #
#circ2buckets implements measurement gates M from qtree.optimizer.     #
#Originally, M is class Gate. We now need a batch dimension, so M is   #
#class ParallelGate with n_batch as a parameter input. We call M with  #
#this parameter here now.                                              #
########################################################################

import functools
import itertools
import random

from .qtensor.qtree import operators as ops
from .qtensor.qtree.optimizer import Var, Tensor
from .qtensor.optimisation.TensorNet import QtreeTensorNet

random.seed(0)


class ParallelQtreeTensorNet(QtreeTensorNet):

    def __init__(self, buckets, data_dict, bra_vars, ket_vars, **kwargs):
        self.measurement_circ = None
        self.measurement_op = None
        super().__init__(buckets, data_dict, bra_vars, ket_vars, **kwargs)

    def slice(self, slice_dict):
        sliced_buckets = self.backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        self.buckets = sliced_buckets
        return self.buckets

    @classmethod
    def circ2buckets(cls, qubit_count, circuit, measurement_circ=None, measurement_op=None, pdict={}, max_depth=None):
        # import pdb
        # pdb.set_trace()
        n_batch = circuit[0][0].gen_tensor().shape[0]
        device = circuit[0][0].gen_tensor().device
        
        if max_depth is None:
            max_depth = len(circuit)

        data_dict = {}

        # Let's build buckets for bucket elimination algorithm.
        # The circuit is built from left to right, as it operates
        # on the ket ( |0> ) from the left. We thus first place
        # the bra ( <x| ) and then put gates in the reverse order

        # Fill the variable `frame`
        layer_variables = [Var(qubit, name=f'o_{qubit}')
                        for qubit in range(qubit_count)]
        current_var_idx = qubit_count

        # Save variables of the bra
        bra_variables = [var for var in layer_variables]

        # Initialize buckets
        for qubit in range(qubit_count):
            buckets = [[] for qubit in range(qubit_count)]
        # Place safeguard measurement circuits before and after
        # the circuit
        if measurement_circ == None:
            measurement_circ = [[ops.M(qubit, n_batch=n_batch, device=device, is_placeholder=False) for qubit in range(qubit_count)]]
        else:
            for op in measurement_circ[0]:
                op._parameters['n_batch'] = n_batch

        combined_circ = functools.reduce(
            lambda x, y: itertools.chain(x, y),
            [measurement_circ, reversed(circuit[:max_depth])])

        # Start building the graph in reverse order
        for layer in combined_circ:
            for op in layer:
                # CUSTOM
                # build the indices of the gate. If gate
                # changes the basis of a qubit, a new variable
                # has to be introduced and current_var_idx is increased.
                # The order of indices
                # is always (a_new, a, b_new, b, ...), as
                # this is how gate tensors are chosen to be stored
                variables = []
                current_var_idx_copy = current_var_idx
                min_var_idx = current_var_idx
                for qubit in op.qubits:
                    if qubit in op.changed_qubits:
                        variables.extend(
                            [layer_variables[qubit],
                            Var(current_var_idx_copy)])
                        current_var_idx_copy += 1
                    else:
                        variables.extend([layer_variables[qubit]])
                    min_var_idx = min(min_var_idx,
                                    int(layer_variables[qubit]))

                # fill placeholders in parameters if any
                for par, value in op.parameters.items():
                    if isinstance(value, ops.placeholder):
                        op._parameters[par] = pdict[value]

                data_key = (op.name,
                            hash((op.name, tuple(op.parameters.items()))))
                # Build a tensor
                t = Tensor(op.name, variables,
                        data_key=data_key)

                # Insert tensor data into data dict
                data_dict[data_key] = op.gen_tensor()

                # Append tensor to buckets
                # first_qubit_var = layer_variables[op.qubits[0]]
                buckets[min_var_idx].append(t)

                # Create new buckets and update current variable frame
                for qubit in op.changed_qubits:
                    layer_variables[qubit] = Var(current_var_idx)
                    buckets.append(
                        []
                    )
                    current_var_idx += 1

        # Finally go over the qubits, append measurement gates
        # and collect ket variables
        ket_variables = []

        if measurement_op==None:
            measurement_op = ops.M(0, n_batch=n_batch, device=device, is_placeholder=False)  # create a single measurement gate object
        else:
            measurement_op._parameters['n_batch'] = n_batch
        data_key = (measurement_op.name, hash((measurement_op.name, tuple(measurement_op.parameters.items()))))
        data_dict.update({data_key: measurement_op.gen_tensor()})

        for qubit in range(qubit_count):
            var = layer_variables[qubit]
            new_var = Var(current_var_idx, name=f'i_{qubit}', size=2)
            ket_variables.append(new_var)
            # update buckets and variable `frame`
            buckets[int(var)].append(
                Tensor(measurement_op.name,
                    indices=[var, new_var],
                    data_key=data_key)
            )
            buckets.append([])
            layer_variables[qubit] = new_var
            current_var_idx += 1

        return buckets, data_dict, bra_variables, ket_variables, measurement_circ, measurement_op

    @classmethod
    def from_qtree_gates(cls, qc, measurement_circ=None, measurement_op=None, **kwargs):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        qtree_circuit = [[g] for g in qc]
        buckets, data_dict, bra_vars, ket_vars, measurement_circ, measurement_op = cls.circ2buckets(n_qubits, qtree_circuit, measurement_circ, measurement_op)
        tn = cls(buckets, data_dict, bra_vars, ket_vars, **kwargs)
        return tn, measurement_circ, measurement_op

########################################################################
#Defining parallel simulator.                                          #
########################################################################
from .qtensor.Simulate import QtreeSimulator

class ParallelQtreeSimulator(QtreeSimulator):

    def __init__(self, **kwargs):
        self.measurement_circ = None
        self.measurement_op = None
        super().__init__(**kwargs)

    def _create_buckets(self):
        self.tn, self.measurement_circ, self.measurement_op = ParallelQtreeTensorNet.from_qtree_gates(self.all_gates, self.measurement_circ, self.measurement_op, backend=self.backend)
        self.tn.backend = self.backend
    
    def prepare_buckets(self, qc, batch_vars=0, peo=None):
        self._new_circuit(qc)
        self._create_buckets()
        # Collect free qubit variables
        if isinstance(batch_vars, int):
            free_final_qubits = list(range(batch_vars))
        else:
            free_final_qubits = batch_vars

        self._set_free_qubits(free_final_qubits)
        if peo is None:
            self._optimize_buckets()
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')
        else:
            self.peo = peo

        all_indices = sum([list(t.indices) for bucket in self.tn.buckets for t in bucket], [])
        identity_map = {v.name: v for v in all_indices}
        self.peo = [identity_map[i.name] for i in self.peo]

        self._reorder_buckets()
        slice_dict = self._get_slice_dict()
        
        sliced_buckets = self.tn.slice(slice_dict)
        self.buckets = sliced_buckets

    def simulate_batch(self, qc, batch_vars=0, peo=None):
        self.prepare_buckets(qc, batch_vars, peo)
        result = qtree.optimizer.bucket_elimination(
            self.buckets, self.backend.process_bucket,
            n_var_nosum=len(self.tn.free_vars)
        )
        return self.backend.get_result_data(result).flatten()