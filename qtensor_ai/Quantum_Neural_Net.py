import torch
from torch import nn

from .ParallelQTensor import MetricLearningCircuitComposer, ParallelTorchQkernelComposer, ParallelQtreeSimulator, ParallelQtreeTensorNet, ParallelTorchBackend
from .qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer

import os
import pickle
        
'''This is a drop-in replacement of linear layers.'''
class QNN(nn.Module):
    
    def __init__(self, in_features, out_features, variational_layers=1, higher_order=False, optimizer=DefaultOptimizer()):
        super(QNN, self).__init__()
                
        '''Initializing module parameters'''
        self.higher_order = higher_order
        self.in_features = in_features
        self.out_features = out_features
        self.variational_layers = variational_layers
        self.higher_order = higher_order
        
        '''Initializing simulator'''
        self.sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
        
        '''Tree optimization'''
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        init_inputs = torch.zeros(1, in_features, requires_grad=False).to(device)
        init_params = torch.zeros(1, in_features, variational_layers, requires_grad=False).to(device)
        self.com = ParallelTorchQkernelComposer(in_features)
        self.com.higher_order = higher_order
        self.com.expectation_circuit(init_inputs, init_params)
        tn, measurement_circ, measurement_op = ParallelQtreeTensorNet.from_qtree_gates(self.com.static_circuit)
        
        '''peo is the tensor network contraction order'''
        self.peo = circuit_optimization(in_features, variational_layers, tn, optimizer, ParallelTorchQkernelComposer)
        
        '''self.weight are model weights'''
        self.weight = nn.Parameter(torch.randn(out_features, in_features, variational_layers, dtype=torch.float32))

    def forward(self, x):
        n_batch, in_features = x.shape # (n_batch, in_features)
        out_features, in_features, variational_layers = self.weight.shape
        x = x.repeat(out_features, 1) # (out_features*n_batch, in_features)
        params = self.weight.unsqueeze(1) # (out_features, 1, in_features, variational_layers)
        params = params.expand(-1, n_batch, -1, -1) # (out_features, n_batch, in_features, variational_layers)
        params = params.reshape(out_features*n_batch, in_features, variational_layers) # (out_features*n_batch, in_features, variational_layers)
        self.com.expectation_circuit(x, params)
        out = torch.real(self.sim.simulate_batch(self.com.static_circuit, peo=self.peo)) # (out_features*n_batch)
        out = out.reshape(out_features, n_batch) # (out_features, n_batch)
        out = out.permute(0, 1) # (n_batch, out_features)
        return out
    
        
'''This is an example for 1D convolution. The filter is replaced with the QNN.'''
class QConv1D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, variational_layers=1, higher_order=False, optimizer=DefaultOptimizer(), dilation=1, padding=0, stride=1):
        super().__init__()
                
        '''Initializing module parameters'''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = in_channels * kernel_size
        
        '''Defining unfold operation for convolution'''
        self.unfold = nn.Unfold(kernel_size=(kernel_size,1), dilation=(dilation,1), padding=(padding,0), stride=(stride,1))
        
        '''Defining multichannel filter to be convolved'''
        self.filter = QNN(self.n_qubits, out_channels, variational_layers, higher_order, optimizer)
        

    '''This function transforms batched, multichannel sequences into parallel values of convolution kernel inputs'''
    def memory_strided_im2col(self, x):
        # x has dimension (n_batch, in_channels, length)
        x = x.unsqueeze(-1)
        out = self.unfold(x)
        out = torch.transpose(out, 1, 2)
        # out has dimension (n_batch, L, kernel_size*in_channels=n_qubits)
        return out
    
    def forward(self, x):
        n_batch = x.size(0) # (n_batch, in_channels, length)
        x = self.memory_strided_im2col(x) # (n_batch, L, kernel_size*in_channels=n_qubits)
        x = x.reshape(-1, self.kernel_size*self.in_channels) # (n_batch*L, kernel_size*in_channels=n_qubits)
        output = self.filter(x) # (n_batch*L, out_channels)  
        output = output.reshape(n_batch, -1, self.out_channels) # (n_batch, L, out_channels) 
        output = output.permute(0,2,1) # (n_batch, out_channels, L)
        return output


'''Following arXiv:2001.03622. Embedding circuit with variational parameters. Embedds classical vectors in the quantum Hilbert space and takes their inner product'''
class MetricLearning(nn.Module):
    
    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer(), entanglement=1):
        super().__init__()
                
        '''Initializing module parameters'''
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        '''Initializing quantum circuit parameters (Not input to be embedded)'''
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        self.zz_params = nn.Parameter(torch.rand(1, n_qubits-1, n_layers, dtype=torch.float32)*entanglement).to(device)
        self.y_params = nn.Parameter(torch.rand(1, n_qubits, n_layers, dtype=torch.float32)*entanglement).to(device)
        
        '''Initializing simulator'''
        self.sim = ParallelQtreeSimulator(backend=ParallelTorchBackend())
        
        '''Tree optimization'''
        inputs1_filler = torch.rand(1, n_qubits, requires_grad=False).to(device)
        inputs2_filler = torch.rand(1, n_qubits, requires_grad=False).to(device)
        self.com = MetricLearningCircuitComposer(n_qubits)
        self.com.expectation_circuit(inputs1_filler, inputs2_filler, self.zz_params, self.y_params)
        tn, _, _ = ParallelQtreeTensorNet.from_qtree_gates(self.com.static_circuit)
        
        '''peo is the tensor network contraction order'''
        self.peo = circuit_optimization(n_qubits, n_layers, tn, optimizer, MetricLearningCircuitComposer)

    def forward(self, inputs1, inputs2):
        n_batch = inputs1.shape[0]
        zz_params = self.zz_params.expand(n_batch, -1, -1) # (n_batch, n_qubits-1, n_layers)
        y_params = self.y_params.expand(n_batch, -1, -1) # (n_batch, n_qubits, n_layers)
        self.com.expectation_circuit(inputs1, inputs2, zz_params, y_params)
        out = self.sim.simulate_batch(self.com.static_circuit, peo=self.peo) # (out_features*n_batch)
        return out
        

# Function for returning the contraction order. Searching if previously saved contraction orders exist.
def circuit_optimization(n_qubits, n_layers, tn, optimizer, composer):
    begin_dir = './Saved_Contraction_Orders/' + type(optimizer).__name__ + '/' + composer.name()
    end_dir = '/n_qubits_{}_n_layers_{}.pickle'.format(n_qubits, n_layers)
    # If optimizer is Tamaki, search all folders with the higher or equal wait_time for matching circuit descriptions (n_qubits, variational_layers). This is because higher wait_time gives better optimization results.
    if isinstance(optimizer, TamakiOptimizer):
        matched_max_wait_time = optimizer.wait_time
        matched_max_wait_time_dir = None
        if os.path.isdir(begin_dir):
            folders = os.listdir(begin_dir)
            for folder in folders:
                if len(folder) >= 11:
                    if folder[:10] == 'wait_time_':
                        folder_wait_time = int(folder[10:])
                        if folder_wait_time >= matched_max_wait_time:
                            potential_dir = begin_dir + '/' + folder + end_dir
                            if os.path.isfile(potential_dir):
                                matched_max_wait_time = folder_wait_time
                                matched_max_wait_time_dir = potential_dir
        if matched_max_wait_time_dir == None:
            peo, tn = optimizer.optimize(tn)
            new_peo_dir = begin_dir + '/wait_time_' + str(matched_max_wait_time)
            new_peo_file_dir = new_peo_dir + end_dir
            if not os.path.isdir(new_peo_dir):
                os.makedirs(new_peo_dir)
            with open(new_peo_file_dir, 'wb') as peo_dir:
                pickle.dump(peo, peo_dir)
                print('New contraction order, save at ', peo_dir)
        else:
            with open(matched_max_wait_time_dir, 'rb') as peo_dir:
                peo = pickle.load(peo_dir)
                print('Using previously saved contraction order at ', peo_dir)

    # Searching for other optimizer types
    else:
        target_file_dir = begin_dir + end_dir
        
        if os.path.isfile(target_file_dir):
            with open(target_file_dir, 'rb') as peo_dir:
                peo = pickle.load(peo_dir)
                print('Using previously saved contraction order at ', peo_dir)
        else:
            if not os.path.isdir(begin_dir):
                os.makedirs(begin_dir)
            peo, tn = optimizer.optimize(tn)
            with open(target_file_dir, 'wb') as peo_dir:
                pickle.dump(peo, peo_dir)
                print('New contraction order, save at ', peo_dir)

    return peo