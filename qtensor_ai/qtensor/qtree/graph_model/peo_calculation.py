"""
This module implements heuristics for approximate
tree decomposition
"""

import networkx as nx
import numpy as np
import random
import re
import os
import sys
import functools
import itertools

import time

from  .. import system_defs as defs
from .exporters import (generate_cnf_file, generate_gr_file)
from .importers import read_td_file, get_stats_from_td_file

from ..optimizer import Var
from .base import (
    eliminate_node, relabel_graph_nodes, get_simple_graph)


def get_treewidth_from_peo(old_graph, peo):
    """
    This function checks the treewidth of a given peo.
    The graph is simplified: all selfloops and parallel
    edges are removed.

    Parameters
    ----------
    old_graph : networkx.Graph or networkx.MultiGraph
            graph to use
    peo : list
            list of nodes in the perfect elimination order

    Returns
    -------
    treewidth : int
            treewidth corresponding to peo
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))

    # Copy graph and make it simple
    graph = get_simple_graph(old_graph)

    treewidth = 0
    for node in peo:
        # Get the size of the next clique - 1
        neighbors = list(graph[node])
        n_neighbors = len(neighbors)
        if len(neighbors) > 1:
            edges = itertools.combinations(neighbors, 2)
        else:
            edges = None

        # Treewidth is the size of the maximal clique - 1
        treewidth = max(n_neighbors, treewidth)

        graph.remove_node(node)

        # Make the next clique
        if edges is not None:
            graph.add_edges_from(edges)

    return treewidth


def get_node_min_fill_heuristic(graph, randomize=False):
    """
    Calculates the next node for the min-fill
    heuristic, as described in V. Gogate and R.  Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min fill node is selected at random among
                nodes with the same minimal fill
    Returns
    -------
    node : node-type
           node with minimal fill
    degree : int
           degree of the node
    """
    min_fill = np.inf

    min_fill_nodes = []
    for node in graph.nodes:
        neighbors_g = graph.subgraph(
            graph.neighbors(node))
        degree = neighbors_g.number_of_nodes()
        n_edges_filled = neighbors_g.number_of_edges()

        # All possible edges without selfloops
        n_edges_max = int(degree*(degree-1) // 2)
        fill = n_edges_max - n_edges_filled
        if fill == min_fill:
            min_fill_nodes.append((node, degree))
        elif fill < min_fill:
            min_fill_nodes = [(node, degree)]
            min_fill = fill
        else:
            continue
    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        min_fill_nodes_d = dict(min_fill_nodes)
        node = np.random.choice(min_fill_nodes_d)
        degree = min_fill_nodes_d[node]
    else:
        node, degree = min_fill_nodes[-1]
    return node, degree


def get_node_min_degree_heuristic(graph, randomize=False):
    """
    Calculates the next node for the min-degree
    heuristic, as described in V. Gogate and R.  Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min degree node is selected at random among
                nodes with the same minimal degree

    Returns
    -------
    node : node-type
           node with minimal degree
    degree : int
           degree of the node
    """
    nodes_by_degree = sorted(list(graph.degree()),
                             key=lambda pair: pair[1])
    min_degree = nodes_by_degree[0][1]

    min_degree_nodes = []
    for idx, (node, degree) in enumerate(nodes_by_degree):
        if degree > min_degree:
            break
        min_degree_nodes.append(node)

    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        node = np.random.choice(min_degree_nodes)
    else:
        node = min_degree_nodes[-1]

    return node, min_degree


def get_node_max_cardinality_heuristic(graph, randomize=False):
    """
    Calculates the next node for the maximal cardinality search
    heuristic

    Parameters
    ----------
    graph : networkx.Graph graph to estimate
    randomize : bool, default False
                if a min degree node is selected at random among
                nodes with the same minimal degree

    Returns
    -------
    node : node-type
           node with minimal degree
    degree : int
           degree of the node
    """
    max_cardinality = -1
    max_cardinality_nodes = []

    for node in graph.nodes:
        cardinality = graph.nodes[node].get('cardinality', 0)
        degree = graph.degree(node)
        if cardinality > max_cardinality:
            max_cardinality_nodes = [(node, degree)]
        elif cardinality == max_cardinality:
            max_cardinality_nodes.append((node, degree))
        else:
            continue
    # Either choose the node at random among equivalent or use
    # the last one
    if randomize:
        max_cardinality_nodes_d = dict(max_cardinality_nodes)
        node = np.random.choice(max_cardinality_nodes_d)
        degree = max_cardinality_nodes_d[node]
    else:
        node, degree = max_cardinality_nodes[-1]

    # update the graph to hold the cardinality information
    for neighbor in graph.neighbors(node):
        cardinality = graph.nodes[neighbor].get('cardinality', 0)
        graph.nodes[neighbor]['cardinality'] = cardinality + 1

    return node, degree


def get_upper_bound_peo_builtin(old_graph, method="min_fill"):
    """
    Calculates an upper bound on treewidth using one of the
    heuristics.

    Best among implemented here is min-fill,
    as described in V. Gogate and R. Dechter
    :url:`http://arxiv.org/abs/1207.4109`

    Parameters
    ----------
    graph : networkx.Graph
           graph to estimate
    method : str
           one of {"min_fill", "min_degree", "cardinality"}

    Returns
    -------
    peo : list
           list of nodes in perfect elimination order
    treewidth : int
           treewidth corresponding to peo
    """
    methods = {"min_fill": get_node_min_fill_heuristic,
               "min_degree": get_node_min_degree_heuristic,
               "cardinality": get_node_max_cardinality_heuristic}
    assert method in methods.keys()
    node_heuristic_fn = methods[method]

    # copy graph as we will destroy it here
    # and relabel to consequtive ints
    graph, inv_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(
            old_graph.nodes, range(1, old_graph.number_of_nodes()+1))))

    # Remove selfloops and parallel edges. Critical
    graph = get_simple_graph(graph)

    node, max_degree = node_heuristic_fn(graph)
    peo = [node]
    eliminate_node(graph, node, self_loops=False)

    for ii in range(graph.number_of_nodes()):
        node, degree = node_heuristic_fn(graph)
        peo.append(node)
        max_degree = max(max_degree, degree)
        eliminate_node(graph, node, self_loops=False)

    # relabel peo back
    peo = [inv_dict[pp] for pp in peo]
    return peo, max_degree  # this is clique size - 1


def get_upper_bound_peo_pace2017_interactive(
        old_graph, method="tamaki", max_time=60, max_width=None, print_stats=False):
    """
    Calculates a PEO and treewidth using one of the external solvers

    Parameters
    ----------
    graph : networkx.Graph
           graph to estimate
    method : str
           one of {"tamaki"}
    max_time : float
            Run until not reached time
    max_width : int
           Run until not reached width

    Returns
    -------
    peo : list

    treewidth : int
           treewidth
    """
    from .clique_trees import get_peo_from_tree
    from . import pace2017_solver_api as api
    method_args = {
        'tamaki':
        {'command': './tw-heuristic',
         'cwd': defs.TAMAKI_SOLVER_PATH,
         }
    }

    assert(method in method_args.keys())
    # ensure graph is labelad starting from 1 with integers
    graph, inv_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(old_graph.nodes,
                 range(1, old_graph.number_of_nodes()+1))))

    # Remove selfloops and parallel edges. Critical
    graph = get_simple_graph(graph)

    data = generate_gr_file(graph)
    start = time.time()
    def callback(line_info):
        ts, width = line_info
        print(f'Time={ts}, width={width}', file=sys.stderr)
        elapsed = time.time() - start
        if max_time:
            if elapsed > max_time:
                raise StopIteration('Timeout')
        if max_width and width:
            if width <= max_width:
                raise StopIteration('Solution is good enough')

    out_data = api.run_heuristic_solver_interactive(
        data, callback, **method_args[method]
    )
    try:
        stats = get_stats_from_td_file(out_data)
        if print_stats:
            print('stats', stats)
        tree, treewidth = read_td_file(out_data, as_data=True)
    except ValueError:
        print(out_data)
        raise
    peo = get_peo_from_tree(tree)

    # return to the original labelling
    peo = [inv_dict[pp] for pp in peo]

    return peo, treewidth

def get_upper_bound_peo_pace2017(
        old_graph, method="tamaki", wait_time=60, print_stats=False):
    """
    Calculates a PEO and treewidth using one of the external solvers

    Parameters
    ----------
    graph : networkx.Graph
           graph to estimate
    method : str
           one of {"tamaki"}
    wait_time : float
           allowed running time (in seconds)

    Returns
    -------
    peo : list

    treewidth : int
           treewidth
    """
    from .clique_trees import get_peo_from_tree
    from . import pace2017_solver_api as api
    method_args = {
        'tamaki':
        {'command': './tw-heuristic',
         'cwd': defs.TAMAKI_SOLVER_PATH,
         'wait_time': wait_time}
    }

    assert(method in method_args.keys())
    # ensure graph is labelad starting from 1 with integers
    graph, inv_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(old_graph.nodes,
                 range(1, old_graph.number_of_nodes()+1))))

    # Remove selfloops and parallel edges. Critical
    graph = get_simple_graph(graph)

    data = generate_gr_file(graph)
    out_data = api.run_heuristic_solver(data, **method_args[method])
    try:
        stats = get_stats_from_td_file(out_data)
        if print_stats:
            print('stats', stats)
        tree, treewidth = read_td_file(out_data, as_data=True)
    except ValueError:
        print(out_data)
        raise
    peo = get_peo_from_tree(tree)

    # return to the original labelling
    peo = [inv_dict[pp] for pp in peo]

    return peo, treew


def get_upper_bound_peo(graph, method='tamaki', **kwargs):
    """
    Run one of the heuristics to get PEO and treewidth
    Parameters:
    -----------
    graph: networkx.Graph
           graph to calculate PEO
    method: str, default 'tamaki'
           solver to use
    **kwargs: default {}
           optional keyword arguments to pass to the solver
    """
    builtin_heuristics = {"min_fill", "min_degree", "cardinality"}
    pace_heuristics = {"tamaki"}

    if method in pace_heuristics:
        peo, tw = get_upper_bound_peo_pace2017(graph, method, **kwargs)
    elif method in builtin_heuristics:
        peo, tw = get_upper_bound_peo_builtin(graph, method)
    elif method == "quickbb":
        peo, tw = get_upper_bound_peo_quickbb(graph, **kwargs)
    else:
        raise ValueError(f'Unknown method: {method}')

    peo_vars = [Var(var, size=graph.nodes[var]['size'],
                    name=graph.nodes[var]['name'])
                for var in peo]

    return peo_vars, tw


def get_peo(old_graph, method="tamaki"):
    """
    Calculates a perfect elimination order using one of the
    external methods.

    Parameters
    ----------
    graph : networkx.Graph
           graph to estimate
    method : str
           one of {"tamaki"}
    Returns
    -------
    peo : list
           list of nodes in perfect elimination order
    treewidth : int
           treewidth corresponding to peo
    """
    from .clique_trees import get_peo_from_tree
    from . import pace2017_solver_api as api
    method_args = {
        'tamaki':
        {'command': './tw-exact',
         'cwd': defs.TAMAKI_SOLVER_PATH}
    }

    assert(method in method_args.keys())
    # ensure graph is labeled starting from 1 with integers
    graph, inv_dict = relabel_graph_nodes(
        old_graph,
        dict(zip(old_graph.nodes, range(1, old_graph.number_of_nodes()+1))))

    # Remove selfloops and parallel edges. Critical
    graph = get_simple_graph(graph)

    data = generate_gr_file(graph)
    out_data = api.run_exact_solver(data, **method_args[method])
    tree, treewidth = read_td_file(out_data, as_data=True)
    peo = get_peo_from_tree(tree)
    peo = [inv_dict[pp] for pp in peo]

    try:
        peo_vars = [Var(var, size=old_graph.nodes[var]['size'],
                        name=old_graph.nodes[var]['name'])
                    for var in peo]
    except:
        peo_vars = peo

    return peo_vars, treewidth


def test_method(method=get_upper_bound_peo_builtin):
    """
    Tests minfill heuristic using quickbb algorithm
    """
    from .base import wrap_general_graph_for_qtree
    from .generators import generate_erdos_graph

    # Test 1: path graph with treewidth 1
    print('Test 1. Path graph')
    graph = wrap_general_graph_for_qtree(
        nx.path_graph(8))

    peo1, tw1 = method(
        graph)
    peo2, tw2 = get_peo(graph)
    print(f'treewidth: {tw1}, reference: {tw2}')
    print(f'      peo: {peo1}\nreference: {peo2}')

    # Test 2: complete graph with treewidth n-1
    print('Test 2. Complete graph')
    graph = wrap_general_graph_for_qtree(
        nx.complete_graph(8)
    )

    peo1, tw1 = method(
        graph)
    peo2, tw2 = get_peo(graph)
    print(f'treewidth: {tw1}, reference: {tw2}')
    print(f'      peo: {peo1}\nreference: {peo2}')

    # Test 3: complicated graphs with indefinite treewidth
    print('Test 3. Probabilistic graph')
    graph = wrap_general_graph_for_qtree(
        generate_erdos_graph(50, 0.5)
        )

    peo1, tw1 = method(
        graph)
    peo2, tw2 = get_peo(graph)
    print(f'treewidth: {tw1}, reference: {tw2}')
    print(f'      peo: {peo1}\nreference: {peo2}')


def test_get_treewidth_from_peo():
    from qtree.graph_model.generators import generate_erdos_graph
    graph = generate_erdos_graph(50, 0.5)

    peo, tw1 = get_peo(graph)
    tw2 = get_treewidth_from_peo(graph, peo)
    print(f'treewidth: {tw2}, reference: {tw1}')


if __name__ == "__main__":
    test_method(functools.partial(get_upper_bound_peo_builtin,
                                  method='min_fill'))
    test_method(functools.partial(get_upper_bound_peo_builtin,
                                  method='min_degree'))
    test_method(functools.partial(get_upper_bound_peo_builtin,
                                  method='cardinality'))
    test_method(functools.partial(get_upper_bound_peo,
                                  method='tamaki', wait_time=10))
    test_method(functools.partial(get_upper_bound_peo_quickbb))
    test_method(functools.partial(get_peo, method='tamaki'))
    test_get_treewidth_from_peo()
