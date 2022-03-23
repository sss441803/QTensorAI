import torch
from torch import nn

from qtensor_ai import HybridModule, DefaultOptimizer
from Custom_Circuit_Composers import QNNComposer, MetricLearningComposer

        
'''This is a drop-in replacement of linear layers.
The number of input features is the number of qubits.
Each output feature is computed by an independently parameterized circuit'''
class QNN(HybridModule):
    
    def __init__(self, in_features, out_features, variational_layers=1, higher_order=False, optimizer=DefaultOptimizer()):
                
        '''Initializing module parameters'''
        circuit_name = 'n_{}_l_{}'.format(in_features, variational_layers)
        self.higher_order = higher_order
        self.in_features = in_features
        self.out_features = out_features
        self.variational_layers = variational_layers
        self.higher_order = higher_order
        
        '''Define the circuit composer and initialize the hybrid module'''
        composer = QNNComposer(in_features, variational_layers, higher_order=higher_order)
        super(QNN, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

        '''self.weight are model weights. Weights must be defined after super().__init__()'''
        self.weight = nn.Parameter(torch.randn(out_features, in_features, variational_layers, dtype=torch.float32))


    def forward(self, x):
        n_batch = x.shape[0] # (n_batch, in_features)
        x = x.repeat(self.out_features, 1) # (out_features*n_batch, in_features)
        params = self.weight.unsqueeze(1) # (out_features, 1, in_features, variational_layers)
        params = params.expand(-1, n_batch, -1, -1) # (out_features, n_batch, in_features, variational_layers)
        params = params.reshape(self.out_features*n_batch, self.in_features, self.variational_layers) # (out_features*n_batch, in_features, variational_layers)

        '''The actual simulation must be run by calling the parent_forward method of the parent class. 
        The parameters should be the same parameters as those accepted by the circuit composer'''
        out = self.parent_forward(data=x, params=params) # (out_features*n_batch)

        out = torch.real(out) # (out_features*n_batch)
        out = out.reshape(self.out_features, n_batch) # (out_features, n_batch)
        out = out.permute(1, 0) # (n_batch, out_features)
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


'''Module for evaluating the inner product between quantum embeddings
learned by circuits proposed by Lloyd et. al. arXiv:2001.03622.'''
class MetricLearning(HybridModule):
    
    def __init__(self, n_qubits, n_layers, optimizer=DefaultOptimizer(), entanglement=2):
                
        '''Initializing module parameters'''
        circuit_name = 'n_{}_l_{}'.format(n_qubits, n_layers)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        '''Define the circuit composer and initialize the hybrid module'''
        composer = MetricLearningComposer(n_qubits, n_layers)
        super(MetricLearning, self).__init__(circuit_name=circuit_name, composer=composer, optimizer=optimizer)

        '''Initializing quantum circuit parameters (Not input to be embedded)'''
        '''self.weight are model weights. Weights must be defined after super().__init__()'''
        self.zz_params = nn.Parameter(torch.rand(1, n_qubits-1, n_layers, dtype=torch.float32)*entanglement)
        self.y_params = nn.Parameter(torch.rand(1, n_qubits, n_layers, dtype=torch.float32)*entanglement)

    def forward(self, inputs1, inputs2):

        """
        Parameters
        ----------
        inputs1: torch.tensor
                Classical rotation angles for the circuit encoding Rx gates for the first embedding.

        inputs2: torch.tensor
                Classical rotation angles for the circuit encoding Rx gates for the second embedding.
        """

        n_batch = inputs1.shape[0]
        zz_params = self.zz_params.expand(n_batch, -1, -1) # (n_batch, n_qubits-1, n_layers)
        y_params = self.y_params.expand(n_batch, -1, -1) # (n_batch, n_qubits, n_layers)
        '''The actual simulation must be run by calling the parent_forward method of the parent class. 
        The parameters should be the same parameters as those accepted by the circuit composer'''
        out = self.parent_forward(inputs1=inputs1, inputs2=inputs2, zz_params=zz_params, y_params=y_params)
        return out
        

