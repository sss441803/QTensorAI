import torch
from qtensor_ai import QConv1D, QNN, TamakiOptimizer
import time

in_channels =  1
out_channels = 1
kernel_size = 15

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

# Defining quantum neural network, quantum convolutioinal layer, and classical convolutional layer
optimizer=TamakiOptimizer(wait_time=5) # If you do not want to use this, remove the optimizer keyword below. wait_time=5 is default
qconv = QConv1D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, variational_layers=5, optimizer=optimizer).to(device)
qnn = QNN(kernel_size, out_channels).to(device)
conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size).to(device)


n_batch = 1
sequence_length = kernel_size + 1 # The longer the sequence, the more parallel simulations of quantum circuits there are.


# This code block tests a regular quantum neural network with a number of input and output features, a replacement for linear layers of classical neural networks
x = torch.rand(n_batch*1, kernel_size).to(device)
start = time.time()
qy = qnn(x)
stop = time.time()
print('time for qnn', stop-start)
x = torch.rand(n_batch, in_channels, sequence_length, requires_grad=True).to(device)

# This code block tests the quantum convolutional layer
start = time.time()
qy = qconv(x)
stop = time.time()
print('time for qconv ', stop-start)
print(qy.shape)
loss = qy.sum()
loss.backward()
#print(x.grad[0])

# This  code block tests classical convolution
x = torch.rand(n_batch, in_channels, sequence_length, requires_grad=True).to(device)
start = time.time()
cy = conv(x)
stop = time.time()
print('time for classical ', stop-start)
print(cy.shape)