# QTensorAI
A hybrid quantum-classical neural network simulation platform. Quantum simulation uses QTensor, a state-of-the-art tensor network-based simulator that usually has linear complexity in the number of qubits for shallow circuits, instead of exponential complexity. This opens up the possibility to simulate large hybrid models with many qubits. The hybrid model is a PyTorch model, batch-parallelized, GPU compatible and fully differentiable.
# Installation
Creat your environment with conda first. Make sure the python version is NOT 3.10, as it creates issues with `cirq` at this point.
Install QTensor first following the repository https://github.com/danlkv/QTensor.git. First, clone the repository:
```bash
git clone --recurse-submodules https://github.com/DaniloZZZ/QTensor
```
Cloning with `--recurse-submodules` is important for installing the submodule `qtree`. Further, the `master` branch works well for installation, but you can then switch to other branches to see new features. This is not necessary for just using our QTensorAI library. Then, install `qtree`:
```bash
cd QTensor
cd qtree
pip install .
```
Then, install the Tamaki optimizer. This is optional but recommended. If you do not install it, you will need to remove it's import from the script. More details for installation can be found at https://github.com/danlkv/QTensor.git.
```bash
> cd ~/QTensor/qtree/thirdparty/tamaki_treewidth
> make heuristic 
javac tw/heuristic/*.java
```
If `javac` is not already available, install `openjdk` with conda:
```bash
conda install -c anaconda openjdk
```
Then, install `qtensor`:
```bash
cd ~/QTensor
pip install .
```
You might need to separately install `pynauty`, `loguru`, `grpcio`, and `cryptography` using conda separately depending on your configuration.

Finally, install PyTorch > 1.10.
