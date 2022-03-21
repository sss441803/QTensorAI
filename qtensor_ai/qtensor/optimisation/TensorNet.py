from .. import qtree
from ..contraction_backends import ContractionBackend
from .. import utils
from loguru import logger as log

class TensorNet:
    @property
    def tensors(self):
        return self._tensors

    def slice(self, slice_dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.tensors)

    def get_line_graph(self):
        raise NotImplementedError


class QtreeTensorNet(TensorNet):
    def __init__(self, buckets, data_dict
                 , bra_vars, ket_vars, free_vars=[]
                 , backend=ContractionBackend()):
        self.buckets = buckets
        self.data_dict = data_dict
        self.bra_vars = bra_vars
        self.ket_vars = ket_vars
        self.free_vars = free_vars
        self.backend = backend

    def set_free_qubits(self, free):
        self.free_vars = [self.bra_vars[i] for i in free]
        self.bra_vars = [var for var in self.bra_vars if var not in self.free_vars]

    def simulation_cost(self, peo):
        ignore_vars = self.bra_vars + self.ket_vars + self.free_vars
        peo = [int(x) for x in peo if x not in ignore_vars]
        g, _ = utils.reorder_graph(self.get_line_graph(), peo)
        mems, flops = qtree.graph_model.get_contraction_costs(g)
        return mems, flops

    @property
    def _tensors(self):
        return sum(self.buckets, [])

    def slice(self, slice_dict):
        sliced_buckets = self.backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        self.buckets = sliced_buckets
        return self.buckets

    def get_line_graph(self):
        ignored_vars = self.bra_vars + self.ket_vars
        graph =  qtree.graph_model.buckets2graph(self.buckets,
                                               ignore_variables=ignored_vars)
        log.debug('Line graph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        return graph