import functools
import itertools
import random

from .qtensor.qtree import operators as ops
from .qtensor.qtree.optimizer import Tensor
from .qtensor.optimisation.TensorNet import QtreeTensorNet
from .Optimizer import ParallelVar

random.seed(0)


class ParallelTensorNet(QtreeTensorNet):

    def __init__(self, buckets, data_dict, bra_vars, ket_vars, **kwargs):
        self.measurement_circ = None
        self.measurement_op = None
        super().__init__(buckets, data_dict, bra_vars, ket_vars, **kwargs)

    def slice(self, slice_dict):
        sliced_buckets = self.backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        self.buckets = sliced_buckets
        return self.buckets

    @classmethod
    def circ2buckets(cls, qubit_count, circuit, measurement_circ=None, measurement_op=None, pdict={}, max_depth=None):
        # import pdb
        # pdb.set_trace()
        n_batch = circuit[0][0].gen_tensor().shape[0]
        device = circuit[0][0].gen_tensor().device
        
        if max_depth is None:
            max_depth = len(circuit)

        data_dict = {}

        # Let's build buckets for bucket elimination algorithm.
        # The circuit is built from left to right, as it operates
        # on the ket ( |0> ) from the left. We thus first place
        # the bra ( <x| ) and then put gates in the reverse order

        # Fill the variable `frame`
        layer_variables = [ParallelVar(qubit, name=f'o_{qubit}')
                        for qubit in range(qubit_count)]
        current_var_idx = qubit_count

        # Save variables of the bra
        bra_variables = [var for var in layer_variables]

        # Initialize buckets
        for qubit in range(qubit_count):
            buckets = [[] for qubit in range(qubit_count)]
        # Place safeguard measurement circuits before and after
        # the circuit
        if measurement_circ == None:
            measurement_circ = [[ops.M(qubit, n_batch=n_batch, device=device, is_placeholder=False) for qubit in range(qubit_count)]]
        else:
            for op in measurement_circ[0]:
                op._parameters['n_batch'] = n_batch

        combined_circ = functools.reduce(
            lambda x, y: itertools.chain(x, y),
            [measurement_circ, reversed(circuit[:max_depth])])

        # Start building the graph in reverse order
        for layer in combined_circ:
            for op in layer:
                # CUSTOM
                # build the indices of the gate. If gate
                # changes the basis of a qubit, a new variable
                # has to be introduced and current_var_idx is increased.
                # The order of indices
                # is always (a_new, a, b_new, b, ...), as
                # this is how gate tensors are chosen to be stored
                variables = []
                current_var_idx_copy = current_var_idx
                min_var_idx = current_var_idx
                for qubit in op.qubits:
                    if qubit in op.changed_qubits:
                        variables.extend(
                            [layer_variables[qubit],
                            ParallelVar(current_var_idx_copy)])
                        current_var_idx_copy += 1
                    else:
                        variables.extend([layer_variables[qubit]])
                    min_var_idx = min(min_var_idx,
                                    int(layer_variables[qubit]))

                # fill placeholders in parameters if any
                for par, value in op.parameters.items():
                    if isinstance(value, ops.placeholder):
                        op._parameters[par] = pdict[value]

                data_key = (op.name,
                            hash((op.name, tuple(op.parameters.items()))))
                # Build a tensor
                t = Tensor(op.name, variables,
                        data_key=data_key)

                # Insert tensor data into data dict
                data_dict[data_key] = op.gen_tensor()

                # Append tensor to buckets
                # first_qubit_var = layer_variables[op.qubits[0]]
                buckets[min_var_idx].append(t)

                # Create new buckets and update current variable frame
                for qubit in op.changed_qubits:
                    layer_variables[qubit] = ParallelVar(current_var_idx)
                    buckets.append(
                        []
                    )
                    current_var_idx += 1

        # Finally go over the qubits, append measurement gates
        # and collect ket variables
        ket_variables = []

        if measurement_op==None:
            measurement_op = ops.M(0, n_batch=n_batch, device=device, is_placeholder=False)  # create a single measurement gate object
        else:
            measurement_op._parameters['n_batch'] = n_batch
        data_key = (measurement_op.name, hash((measurement_op.name, tuple(measurement_op.parameters.items()))))
        data_dict.update({data_key: measurement_op.gen_tensor()})

        for qubit in range(qubit_count):
            var = layer_variables[qubit]
            new_var = ParallelVar(current_var_idx, name=f'i_{qubit}', size=2)
            ket_variables.append(new_var)
            # update buckets and variable `frame`
            buckets[int(var)].append(
                Tensor(measurement_op.name,
                    indices=[var, new_var],
                    data_key=data_key)
            )
            buckets.append([])
            layer_variables[qubit] = new_var
            current_var_idx += 1

        return buckets, data_dict, bra_variables, ket_variables, measurement_circ, measurement_op

    @classmethod
    def from_qtree_gates(cls, qc, measurement_circ=None, measurement_op=None, **kwargs):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        qtree_circuit = [[g] for g in qc]
        buckets, data_dict, bra_vars, ket_vars, measurement_circ, measurement_op = cls.circ2buckets(n_qubits, qtree_circuit, measurement_circ, measurement_op)
        tn = cls(buckets, data_dict, bra_vars, ket_vars, **kwargs)
        return tn, measurement_circ, measurement_op