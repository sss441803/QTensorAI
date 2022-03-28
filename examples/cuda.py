import torch
import torch.nn as nn
import torch.optim as optim
import time

from Custom_Modules import QConv1D
from qtensor_ai import TamakiOptimizer


def train(model, n_batch, in_channels, sequence_length, n_epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    model = model.cuda()

    for _ in range(n_epochs):
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(n_batch, in_channels, sequence_length).cuda()
        y = torch.rand(n_batch, 1).cuda()
        y_hat = model.forward(x)
        loss = loss_fn(y, y_hat)
        loss.backward()
        optimizer.step()
        stop = time.time()
        print('Time taken for one epoch: {} seconds.'.format(stop-start))


def main():
    n_batch, in_channels, sequence_length, n_epochs = 5, 5, 20, 10
    out_channels, kernel_size, variational_layers, optimizer = 5, 4, 7, TamakiOptimizer(wait_time=20)
    model = nn.Sequential(QConv1D(in_channels, out_channels, kernel_size, variational_layers=variational_layers, optimizer=optimizer),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(out_channels*(sequence_length-kernel_size+1), 1),
                            nn.ReLU())
    train(model, n_batch, in_channels, sequence_length, n_epochs)


if __name__ == "__main__":
    main()
