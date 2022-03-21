from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'numpy',
    'networkx>=2.3',
    'networkit',
    'loguru',
    'tqdm',
    'lazy-import',
    'matplotlib',
    'sarge'
]

setup(
    name='qtensor_ai',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    version='0.1.0',
    description='A hybrid quantum-classical neural network simulation platform. Quantum simulation uses QTensor, a state-of-the-art tensor network-based simulator that usually has linear complexity in the number of qubits for shallow circuits, instead of exponential complexity. This opens up the possibility to simulate large hybrid models with many qubits. The hybrid model is a PyTorch model, batch-parallelized, GPU compatible and fully differentiable.',
    author='Minzhao Liu',
    license='MIT',
)