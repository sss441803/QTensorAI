import qtensor
import torch
import numpy as np
import qtree


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

from qtensor.tools.lazy_import import torch
import qtree
from qtree.utils import num_to_alpha
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

from qtensor.contraction_backends import ContractionBackend
from qtensor.contraction_backends.torch import qtree2torch_tensor

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
#Modifying circ2buckets in qtree.optimizer                             #
#circ2buckets implements measurement gates M from qtree.optimizer.     #
#Originally, M is class Gate. We now need a batch dimension, so M is   #
#class ParallelGate with n_batch as a parameter input. We call M with  #
#this parameter here now.                                              #
########################################################################

import functools
import itertools
import random
import networkx as nx
import qtree.operators as ops
from qtree.optimizer import Var, Tensor

from qtree.logger_setup import log

random.seed(0)

def circ2buckets(qubit_count, circuit, pdict={}, max_depth=None):
    """
    Takes a circuit in the form of list of lists, builds
    corresponding buckets. Buckets contain Tensors
    defining quantum gates. Each bucket
    corresponds to a variable. Each bucket can hold tensors
    acting on it's variable and variables with higher index.

    Parameters
    ----------
    qubit_count : int
            number of qubits in the circuit
    circuit : list of lists
            quantum circuit as returned by
            :py:meth:`operators.read_circuit_file`
    pdict : dict
            Dictionary with placeholders if any parameteric gates
            were unresolved

    max_depth : int
            Maximal depth of the circuit which should be used
    Returns
    -------
    buckets : list of lists
            list of lists (buckets)
    data_dict : dict
            Dictionary with all tensor data
    bra_variables : list
            variables of the output qubits
    ket_variables: list
            variables of the input qubits
    """
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
    measurement_circ = [[ops.M(qubit, n_batch=n_batch, device=device) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit[:max_depth])])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # CUSTOM
            # Swap variables on swap gate 
            if isinstance(op, ops.SWAP):
                q1, q2 = op.qubits
                _v1 = layer_variables[q1]
                layer_variables[q1] = layer_variables[q2]
                layer_variables[q2] = _v1
                continue

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

    op = ops.M(0, n_batch=n_batch, device=device)  # create a single measurement gate object
    data_key = (op.name, hash((op.name, tuple(op.parameters.items()))))
    data_dict.update(
        {data_key: op.gen_tensor()})

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = Var(current_var_idx, name=f'i_{qubit}', size=2)
        ket_variables.append(new_var)
        # update buckets and variable `frame`
        buckets[int(var)].append(
            Tensor(op.name,
                   indices=[var, new_var],
                   data_key=data_key)
        )
        buckets.append([])
        layer_variables[qubit] = new_var
        current_var_idx += 1

    return buckets, data_dict, bra_variables, ket_variables

qtree.optimizer.circ2buckets = circ2buckets

########################################################################
#Modifying gate classes to be compatible with parallelism              #
########################################################################

from qtree.operators import placeholder, Gate, ParametricGate
    
'''Batch parallel function for implementing complex conjugate tranpose'''
def ParallelConjT(tensor):
    n_dims = len(tensor.shape)
    permutation = [0] + [n_dims - i for i in range(1, n_dims)]
    return torch.permute(tensor, tuple(permutation)).conj()
    
class ParallelParametricGate(ParametricGate):

    @classmethod
    def dag_tensor(cls, inst):
        return ParallelConjT(cls.gen_tensor(inst))

    @classmethod
    def dagger(cls):
        # This thing modifies the base class itself.
        orig = cls.gen_tensor
        def conj_tensor(self):
            t = orig(self)
            return ParallelConjT(t)
        cls.gen_tensor = conj_tensor
        cls.__name__ += '.dag'
        return cls 
    
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
        par_str = (",".join("{}={}".format(
            param_name,
            '?.??' if isinstance(param_value, placeholder)
            else '{:.2f}'.format(float(param_value[0])))
                            for param_name, param_value in
                            sorted(self._parameters.items(),
                                   key=lambda pair: pair[0])))

        return ("{}".format(self.name) + "[" + par_str + "]" +
                "({})".format(','.join(map(str, self._qubits))))
    
class ParallelGate(Gate):
    
    def __init__(self, *qubits, **parameters):
        self._qubits = tuple(qubits)
        # supposedly unique id for an instance
        self._parameters = parameters
        self._check_qubit_count(qubits)
        self.name = type(self).__name__
    
    @classmethod
    def dag_tensor(cls, inst):
        return ParallelConjT(cls.gen_tensor(inst))

    @classmethod
    def dagger(cls):
        # This thing modifies the base class itself.
        orig = cls.gen_tensor
        def conj_tensor(self):
            t = orig(self)
            return ParallelConjT(t)
        cls.gen_tensor = conj_tensor
        cls.__name__ += '.dag'
        return cls 
    
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
    
from qtensor.OpFactory import torch_j_exp
    
class M(ParallelGate):
    name = 'M'
    _changes_qubits = (0, )
    """
    Measurement gate. This is essentially the identity operator, but
    it forces the introduction of a variable in the graphical model
    """
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        #return torch.tensor([
        #                    [1, 0],
        #                    [0, 1]
        #                    ]).repeat(n_batch,1,1)
        return torch.tensor([[1, 0], [0, 1]]).to(device).repeat(n_batch, 1, 1)
    
qtree.operators.M = M
    
class H(ParallelGate):
    name = 'H'
    _changes_qubits = (0, ) 
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return 1/np.sqrt(2) * torch.tensor([[1,  1],
                                            [1, -1]
                                           ]).to(device).repeat(n_batch,1,1)
    
class Z(ParallelGate):
    name = 'Z'
    _changes_qubits = tuple()
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return torch.tensor([1, -1]).to(device).repeat(n_batch,1)

class X(ParallelGate):
    name = 'X'
    _changes_qubits = (0, )
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return torch.tensor([
            [0, 1],
            [1, 0]
        ]).to(device).repeat(n_batch,1,1)

class Y(ParallelGate):
    name = 'Y'
    _changes_qubits = (0, )
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return torch.tensor([
            [0, -1j],
            [1j, 0]
        ]).to(device).repeat(n_batch,1,1)

class cX(ParallelGate):
    name = 'cX'
    _changes_qubits=(1, )
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return torch.tensor([[[1., 0.],
                              [0., 1.]],
                             [[0., 1.],
                              [1., 0.]]]).to(device).repeat(n_batch,1,1,1)
    
class cZ(ParallelGate):
    name = 'cZ'
    _changes_qubits = tuple()
    def gen_tensor(self):
        n_batch = self.parameters['n_batch']
        device = self.parameters['device']
        return torch.tensor([[1, 1],
                             [1, -1]
                            ]).to(device).repeat(n_batch,1,1)
    
class ZZ(ParallelParametricGate):
    name = 'ZZ'
    _changes_qubits=tuple()
    parameter_count=1
    def gen_tensor(self):
        alpha = self.parameters['alpha']
        device = alpha.device
        tensor = torch.tensor([
            [-1, +1],
            [+1, -1]
        ]).to(device)
        return torch_j_exp(1j*alpha.unsqueeze(1)[:,None]*tensor*np.pi/2)
    
class ZPhase(ParallelParametricGate):
    name = 'ZPhase'
    _changes_qubits = tuple()
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Z axis"""
        alpha = parameters['alpha']
        device = alpha.device
        t_ = torch.tensor([0, 1]).to(device)
        return torch_j_exp(1j*t_*np.pi*alpha.unsqueeze(1))
    
class YPhase(ParallelParametricGate):
    name = 'YPhase'
    parameter_count = 1
    _changes_qubits = (0, )

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along Y axis"""
        alpha = parameters['alpha']
        device = alpha.device
        c = torch.cos(np.pi * alpha / 2).unsqueeze(1)[:,None]*torch.tensor([[1,0],[0,1]]).to(device)
        s = torch.sin(np.pi * alpha / 2).unsqueeze(1)[:,None]*torch.tensor([[0, -1],[1,0]]).to(device)
        g = torch_j_exp(1j * np.pi * alpha / 2).unsqueeze(1)[:,None]
        return g*(c + s)
    
class XPhase(ParallelParametricGate):
    name = 'XPhase'
    _changes_qubits = (0, )
    parameter_count = 1

    @staticmethod
    def _gen_tensor(**parameters):
        """Rotation along X axis"""
        alpha = parameters['alpha']
        device = alpha.device
        c = torch.cos(np.pi*alpha/2).unsqueeze(1)[:,None]*torch.tensor([[1,0],[0,1]]).to(device)
        s = torch.sin(np.pi*alpha/2).unsqueeze(1)[:,None]*torch.tensor([[0, -1j], [-1j, 0]]).to(device)
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

from qtensor.OpFactory import QtreeBuilder
from qtensor.CircuitComposer import CircuitComposer

class ParallelTorchBuilder(QtreeBuilder):
    operators = ParallelTorchFactory
    
class ParallelTorchQkernelComposer(CircuitComposer):
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        '''higher order encoding encodes the products of data points as rotation angles. Depth is N^2. 	arXiv:2011.00027'''
        self.higher_order = False
        self.device = 'cpu'
        super().__init__()
    
    def _get_builder(self):
        return self._get_builder_class()(self.n_qubits)
    
    def _get_builder_class(self):
        return ParallelTorchBuilder
    
    def layer_of_Hadamards(self):
        for q in self.qubits:
            self.apply_gate(self.operators.H, q, n_batch=self.n_batch, device=self.device)
    
    def entangling_layer(self):
        for i in range(self.n_qubits//2):
            control_qubit = self.qubits[2*i]
            target_qubit = self.qubits[2*i+1]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device)
        for i in range((self.n_qubits+1)//2-1):
            control_qubit = self.qubits[2*i+1]
            target_qubit = self.qubits[2*i+2]
            self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device)
        control_qubit = self.qubits[-1]
        target_qubit = self.qubits[0]
    
    def encoding_circuit(self, data):
        self.layer_of_Hadamards()
        for i, qubit in enumerate(self.qubits):
            self.apply_gate(self.operators.ZPhase, qubit, alpha=data[:, i], device=self.device)
        if self.higher_order:
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    control_qubit = self.qubits[i]
                    target_qubit = self.qubits[j]
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device)
                    self.apply_gate(self.operators.ZPhase, target_qubit, alpha=data[:, i]*data[:, j], device=self.device)
                    self.apply_gate(self.operators.cX, control_qubit, target_qubit, n_batch=self.n_batch, device=self.device)
    
    '''A single layer of rotation gates depending on trainable parameters'''
    def variational_layer(self, layer, layer_params):
        for i in range(self.n_qubits):
            qubit = self.qubits[i]
            self.apply_gate(self.operators.YPhase, qubit, alpha=layer_params[:, i], device=self.device)
     
    def cost_operator(self):
        #self.apply_gate(self.operators.Z, self.qubits[0], n_batch=self.n_batch)
        for qubit in self.qubits:
            self.apply_gate(self.operators.Z, qubit, n_batch=self.n_batch, device=self.device)
    
    '''Building circuit that needs to be measured'''
    def circuit(self, data, params):
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
    def energy_expectation(self, data, params):
        self.device = data.device
        self.circuit(data, params)
        self.cost_operator()
        first_part = self.builder.circuit
        self.builder.reset()

        self.circuit(data, params)
        self.builder.inverse()
        second_part = self.builder.circuit

        self.circuit = first_part + second_part
            
########################################################################
#Defining parallel simulator.                                          #
########################################################################

ParallelSimulator = qtensor.QtreeSimulator(backend=ParallelTorchBackend())