########################################################################
#Defining parallel simulator.                                          #
########################################################################
from .qtensor.Simulate import QtreeSimulator
from .TensorNet import ParallelTensorNet
from .qtensor import qtree

class ParallelSimulator(QtreeSimulator):

    def __init__(self, **kwargs):
        self.measurement_circ = None
        self.measurement_op = None
        super().__init__(**kwargs)

    def _create_buckets(self):
        self.tn, self.measurement_circ, self.measurement_op = ParallelTensorNet.from_qtree_gates(self.all_gates, self.measurement_circ, self.measurement_op, backend=self.backend)
        self.tn.backend = self.backend
    
    def prepare_buckets(self, qc, batch_vars=0, peo=None):
        self._new_circuit(qc)
        self._create_buckets()
        # Collect free qubit variables
        if isinstance(batch_vars, int):
            free_final_qubits = list(range(batch_vars))
        else:
            free_final_qubits = batch_vars

        self._set_free_qubits(free_final_qubits)
        if peo is None:
            self._optimize_buckets()
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')
        else:
            self.peo = peo

        all_indices = sum([list(t.indices) for bucket in self.tn.buckets for t in bucket], [])
        identity_map = {v.name: v for v in all_indices}
        self.peo = [identity_map[i.name] for i in self.peo]

        self._reorder_buckets()
        slice_dict = self._get_slice_dict()
        
        sliced_buckets = self.tn.slice(slice_dict)
        self.buckets = sliced_buckets

    def simulate_batch(self, qc, batch_vars=0, peo=None):
        self.prepare_buckets(qc, batch_vars, peo)
        result = qtree.optimizer.bucket_elimination(
            self.buckets, self.backend.process_bucket,
            n_var_nosum=len(self.tn.free_vars)
        )
        return self.backend.get_result_data(result).flatten()