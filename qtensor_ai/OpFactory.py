########################################################################
#Modifying gate classes to be compatible with parallelism              #
########################################################################

import torch
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
#Defining parallel operators with an batch dimension                   #
########################################################################
    
from .qtensor.OpFactory import torch_j_exp
from .qtensor import qtree
import numpy as np
    
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



########################################################################
#Defining the factory of operators and the circuit builder using the   #
#factory 'ParallelTorchFactory'                                        #
########################################################################

class ParallelTorchFactory:
    pass

ParallelTorchFactory.M = M
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



from .qtensor.OpFactory import QtreeBuilder

class ParallelTorchBuilder(QtreeBuilder):
    """
    Base class of circuit builder that are compatible with batch parallelism.
    Circuit builders store available gates and other methods.

    Attributes
    ----------
    circuit: list
            List that contains all the gates for the circuit.

    Methods
    -------
    inverse():
            Change self.circuit into the inverse circuit.
    reset():
            Clear self.circuit.
    """
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