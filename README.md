# QTensorAI
A hybrid quantum-classical neural network simulation platform. Quantum simulation uses QTensor, a state-of-the-art tensor network-based simulator that usually has linear complexity in the number of qubits for shallow circuits, instead of exponential complexity. This opens up the possibility to simulate large hybrid models with many qubits. The hybrid model is a PyTorch model, batch-parallelized, GPU compatible and fully differentiable.
# Installation
Install QTensor first following the repository https://github.com/danlkv/QTensor.git. Be sure to install the Tamaki optimizer for efficient simulation of larger circuits. Then install PyTorch > 1.10, works better with pip in experience.
