import torch
import torch.nn as nn
import torch.optim as optim
import gc
import time

from Custom_Modules import QConv1D
from qtensor_ai import TamakiOptimizer

'''For more details on PyTorch CUDAGraph, see https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
The comments in this document are all relevant to our implementation only and we suggest not skipping them when reading.

For this benchmarking task, we are simulating circuits with 40 qubits and 5 variational layers (no higher order encoding).
Ths is a multichannel quantum convolution task.
For each batch, there are out_channels*(sequence_length-kernel_size+1)=5*(20-4)=80 circuits producing outputs.
We are using 10 batches. Therefore, 800 circuits are running in parallel.
Using CUDAGraph takes the simulation time of 100 epochs from 152 s to 113 s, a 35% gain in speed.
'''


def train(model, n_batch, in_channels, sequence_length, n_epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    model = model.cuda()
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        '''Warming up with some iterations
        We here use n_epochs+1 iterations to compare against the performance of CUDAGraph.
        In actual runs, we only need 3 or so iterations for warm up.'''
        for i in range(n_epochs + 1):
            if i == 1:
                start = time.time()
            optimizer.zero_grad(set_to_none=True)
            x = torch.rand(n_batch, in_channels, sequence_length).cuda()
            y = torch.rand(n_batch, 1).cuda()
            y_hat = model.forward(x)
            loss = loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()
        stop = time.time()
        print('Time taken for {} epochs during without CUDAGraph: {} seconds.'.format(int(n_epochs), stop-start))
        print('Capturing CUDA stream.')

    torch.cuda.current_stream().wait_stream(s)
    print('GPU memory allocated and reserved after capturing stream:')
    print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    gc.collect()
    torch.cuda.empty_cache()

    train_graph = torch.cuda.CUDAGraph()
    '''CUDAGraph will write all the data into the same memory address and we must use the same tensors.
    This is why we call our tensors static.'''
    static_x = torch.rand(n_batch, in_channels, sequence_length).cuda()
    static_y = torch.rand(n_batch, 1).cuda()

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(train_graph):
        static_y_hat = model.forward(static_x)
        static_loss = loss_fn(static_y, static_y_hat)
        static_loss.backward()
        optimizer.step()
    
    print('GPU memory allocated and reserved after tracing cuda graph:')
    print(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    '''Because CUDAGraph uses the same memory address, we must copy new tensor values to the static tensors used during tracing.
    Similarly, doing things with outputs must use cloned values.'''
    for i in range(n_epochs + 1):
        if i == 1:
            start = time.time()
        static_x.copy_(torch.rand(n_batch, in_channels, sequence_length).cuda())
        static_y.copy_(torch.rand(n_batch, 1).cuda())
        train_graph.replay()
        loss_clone = torch.clone(static_loss)
    stop = time.time()
    print('Time taken for {} epochs with CUDAGraph: {} seconds.'.format(int(n_epochs), stop-start))
    print('Epoch loss is: ', loss_clone)


def main():
    n_batch, in_channels, sequence_length, n_epochs = 10, 10, 20, 100
    out_channels, kernel_size, variational_layers, optimizer = 5, 4, 5, TamakiOptimizer(wait_time=30)
    model = nn.Sequential(QConv1D(in_channels, out_channels, kernel_size, variational_layers=variational_layers, optimizer=optimizer),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(out_channels*(sequence_length-kernel_size+1), 1),
                            nn.ReLU())
    train(model, n_batch, in_channels, sequence_length, n_epochs)


if __name__ == "__main__":
    main()
