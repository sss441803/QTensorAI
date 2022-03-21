# QTensorAI
A hybrid quantum-classical neural network simulation platform. Quantum simulation uses QTensor, a state-of-the-art tensor network-based simulator that usually has linear complexity in the number of qubits for shallow circuits, instead of exponential complexity. This opens up the possibility to simulate large hybrid models with many qubits. The hybrid model is a PyTorch model, batch-parallelized, GPU compatible and fully differentiable.

This librar is based on QTensor at https://github.com/danlkv/QTensor.git, which is built on qtree at https://github.com/Huawei-HiQ/qtree.git. However, these libraries have heavy dependencies, and we restructured the code base and removed unnecessary components to create a minimal implementation that is functional.
# Installation
Creat your `pytorch` environment with conda first:
```bash
conda create --name qtensor_ai pytorch cudatoolkit=10.2 -c pytorch
```
Clone the repository QTensorAI:
```bash
git clone https://github.com/sss441803/QTensorAI.git
```
Install the `qtensor_ai` library:
```bash
cd QTensorAI
python setup.py install
```
Then, install the Tamaki optimizer. This is optional but recommended. If you do not install it, you will need to remove it's import from the script. More details for installation can be found at https://github.com/danlkv/QTensor.git.
```bash
> cd qtensor_ai/qtensor/qtree/thirdparty/tamaki_treewidth
> make heuristic 
javac tw/heuristic/*.java
```
If `javac` is not already available, install `openjdk` with conda:
```bash
conda install -c anaconda openjdk
```
Finally, add the directory `QTensorAI/qtensor_ai/qtensor/qtree/thirdparty/tamaki_treewidth` to `$PATH`.