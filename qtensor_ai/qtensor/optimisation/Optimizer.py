from .. import qtree
import numpy as np
import networkx as nx
import copy


from .. import utils


class Optimizer:
    def optimize(self, tensor_net):
        raise NotImplementedError

class WithoutOptimizer(Optimizer):

    def optimize(self, tensor_net):
        line_graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net. bra_vars
        graph = line_graph


        peo = sorted([int(v) for v in graph.nodes()])
        # magic line
        peo = list(reversed(peo))
        _, path = utils.get_neighbors_path(graph, peo)
        self.treewidth = max(path)
        self.peo_ints = peo

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if free_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.graph = graph
        self.ignored_vars = ignored_vars
        return peo, tensor_net


class OrderingOptimizer(Optimizer):
    def _get_ordering_ints(self, graph, free_vars=[]):
        #mapping = {a:b for a,b in zip(graph.nodes(), reversed(list(graph.nodes())))}
        #graph = nx.relabel_nodes(graph, mapping)
        peo_ints, path = utils.get_neighbours_peo(graph)

        return peo_ints, path

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        # performing ordering inplace reduces time for ordering by 60%
        peo, path = utils.get_neighbours_peo_vars(graph, inplace=inplace)

        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        return peo, path

    def optimize(self, tensor_net):
        graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net.bra_vars

        if free_vars:
            # It's more efficient to find ordering in-place to avoid copying
            # We'll need the copy of a graph only if we have free_vars
            graph = qtree.graph_model.make_clique_on(graph, free_vars)
            graph_copy = copy.deepcopy(graph)
            self.graph = graph_copy

        peo, path = self._get_ordering(graph, inplace=True)
        self.treewidth = max(path)
        self.peo_ints = [int(x) for x in peo]

        if free_vars:
            peo = qtree.graph_model.get_equivalent_peo(self.graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.ignored_vars = ignored_vars
        return peo, tensor_net



class TamakiOptimizer(OrderingOptimizer):
    def __init__(self, *args, wait_time=5, **kwargs):

        """
        Parameters
        ----------
        wait_time: int, double, optional
                The time the optimize will spend on optimization.
                The more time it takes, the more likely it results in
                smaller tree widths and less memory consumption.
        """

        super().__init__(*args, **kwargs)
        self.wait_time = wait_time

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017_interactive(
                graph, method="tamaki", max_time=self.wait_time)


        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        self.treewidth = tw
        return peo, [tw]

class TamakiExactOptimizer(OrderingOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017_interactive(
                graph, method="tamaki_exact", max_time=np.inf)


        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        self.treewidth = tw
        return peo, [tw]


# an alias that makes sense


DefaultOptimizer = OrderingOptimizer
