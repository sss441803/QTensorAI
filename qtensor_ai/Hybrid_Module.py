import torch.nn as nn
import os
import pickle
from .qtensor.optimisation.Optimizer import DefaultOptimizer, TamakiOptimizer
from .Simulate import ParallelSimulator
from .Backend import ParallelTorchBackend
from .TensorNet import ParallelTensorNet


class HybridModule(nn.Module):
    """
    This defines the hybrid quantum classical nn.Modules replacements.
    Create custom hybrid modules inheriting from this class.

    Attributes
    ----------
    circuit: list
            List that contains all the gates for the circuit.

    Methods
    -------
    parent_forward(**parameters): torch.tensor
            Call this function in the forward method of the custom module
            for quantum simulation.
    """

    def __init__(self, circuit_name, composer, optimizer=DefaultOptimizer()):
        
        """
        Parameters
        ----------
        circuit_name: str
                The name of the circuit, which is used to save and find saved
                optimized contraction order. No need to ensure different names
                different circuit composers or different optimizers (or for 
                Tamaki optimizers with different wait_time).

        composer: qtensor_ai.ParallelComposer
                your custom circuit composer

        optimizer: qtensor_ai.qtensor.optimization.Optimizer
                Optimizer selected for optimizing the contraction order.
        """

        self.com = composer # Circuit composer used to create the circuit of the module
        self.sim = ParallelSimulator(backend=ParallelTorchBackend()) 
        self.optimizer = optimizer # Optimizer to optimize the contraction order
        self.peo = None # Tensor network contraction order
        self.circuit_name = circuit_name # Unique name for the circuit with this topology to help storing contraction order to file for later use
        super(HybridModule, self).__init__()

    def optimize_contraction_order(self):
        '''Tree optimization'''
        tn, _, _ = ParallelTensorNet.from_qtree_gates(self.com.final_circuit)
        self.peo = circuit_optimization(self.circuit_name, tn, self.optimizer, self.com)

    def parent_forward(self, **parameters):
        if self.peo == None: # If first time running the circuit, find the contraction order first
            self.com.produce_circuit(**parameters) # Update circuit variational parameters
            self.optimize_contraction_order()
        self.com.produce_circuit(**parameters) # Update circuit variational parameters
        out = self.sim.simulate_batch(self.com.final_circuit, peo=self.peo) # Actual simulation
        return out

    # When defining the forward method for custom hybrid modules, must call the self.parent_forward method to do the actual simulation
    def forward(self, **parameters):
        raise NotImplementedError


# Function for returning the contraction order. Searching if previously saved contraction orders exist.
def circuit_optimization(circuit_name, tn, optimizer, composer):
    begin_dir = os.path.expanduser('~') + '/Saved_Contraction_Orders/' + type(optimizer).__name__ + '/' + composer.name()
    end_dir = '/{}.pickle'.format(circuit_name)
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