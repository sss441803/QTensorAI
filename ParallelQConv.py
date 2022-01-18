import torch
from torch import nn

from ParallelQTensor import ParallelTorchQkernelComposer, ParallelSimulator
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import DefaultOptimizer

from functools import partial
from IPython.utils import io

        
    
class QConv1D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        
        '''Initializing keyword argument values'''
        self.kwargs = {'variational_layers':1, 'higher_order':False, 'Optimizer':DefaultOptimizer(), 'dilation':1, 'padding':0, 'stride':1}
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
                
        '''Initializing module parameters'''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = in_channels * kernel_size
        
        '''Defining unfold operation for convolution'''
        self.unfold = nn.Unfold(kernel_size=(kernel_size,1), dilation=(self.kwargs['dilation'],1), padding=(self.kwargs['padding'],0), stride=(self.kwargs['stride'],1))
        
        '''Initializing simulator'''
        self.sim = ParallelSimulator
        
        '''Tree optimization'''
        init_inputs = torch.zeros(1, self.n_qubits, requires_grad=False);
        init_params = torch.zeros(1, self.n_qubits, self.kwargs['variational_layers'], requires_grad=False)
        com = ParallelTorchQkernelComposer(self.n_qubits)
        com.higher_order = self.kwargs['higher_order']
        com.energy_expectation(init_inputs, init_params)
        tn = QtreeTensorNet.from_qtree_gates(com.circuit)
        opt = self.kwargs['Optimizer']
        self.peo, tn = opt.optimize(tn)
        
        '''self.weight are model weights'''
        self.weight = nn.Parameter(torch.randn(out_channels, self.n_qubits, self.kwargs['variational_layers'], dtype=torch.float32))
        
    '''This function runs the quantum simulation'''    
    def q_kernel(self, x):
        n_batch_times_L = x.size(0)
        x = x.repeat(self.out_channels, 1) # (out_channels*n_batch*L, kernel_size*in_channels=n_qubits)
        params = self.weight.unsqueeze(1) # (out_channels, 1, n_qubits, variational_layers)
        params = params.repeat(1, n_batch_times_L, 1, 1) # (out_channels, n_batch*L, kernel_size*in_channels=n_qubits, variational_layers)
        params = params.reshape(self.out_channels*n_batch_times_L, self.n_qubits, self.kwargs['variational_layers']) # (out_channels*n_batch*L, kernel_size*in_channels=n_qubits, variational_layers)
        com = ParallelTorchQkernelComposer(self.n_qubits)
        com.higher_order = self.kwargs['higher_order']
        com.energy_expectation(x, params)
        with io.capture_output() as captured:
            out = torch.real(self.sim.simulate_batch(com.circuit, peo=self.peo)) # (out_channels*n_batch*L)
        out = out.reshape(self.out_channels, n_batch_times_L) # (out_channels, n_batch*L)
        out = out.permute(0, 1) # (n_batch*L, out_channels)
        return out
    
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
        output = self.q_kernel(x) # (n_batch*L, out_channels)  
        output = output.reshape(n_batch, -1, self.out_channels) # (n_batch, L, out_channels) 
        output = output.permute(0,2,1) # (n_batch, out_channels, L)
        return output