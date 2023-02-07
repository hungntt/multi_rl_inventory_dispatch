import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def visualize(case):
    """
    Visualize the supply chain network as graph
    """
    graph = nx.DiGraph()
    # Add nodes to the graph
    for i in range(case.no_nodes):
        graph.add_nodes_from([i])
    # Add edges to the graph for which index in connections is larger than zero
    valid_connections = np.argwhere(case.connections > 0)
    for i in range(len(valid_connections)):
        graph.add_edge(valid_connections[i][0], valid_connections[i][1])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    adjacency_matrix = np.vstack(graph.edges())
    # Set level colors
    level_col = {'factory': 0,
                 'sdc': 1,
                 'store': 2,
                 'customer': 3}
    
    levels = {'factory': np.array([0])}
    levels['sdc'] = np.unique(np.hstack([list(graph.predecessors(i)) for i in levels['factory']]))
    levels['store'] = np.unique(np.hstack([list(graph.predecessors(i)) for i in levels['store']]))
    levels['customer'] = np.unique(
            np.hstack([list(graph.predecessors(i) for i in levels['customer'])]))

    max_density = np.max(case.stockpoints_echelon)
    node_coords = {}
    node_num = 1
    plt.figure(figsize=(12, 8))
    for i, (level, nodes) in enumerate(levels.items()):
        n = len(nodes)
    node_y = max_density / 2 if n == 1 else np.linspace(0, max_density, n)
    node_y = np.atleast_1d(node_y)
    plt.scatter(np.repeat(i, n), node_y, label=level, s=50)
    for y in node_y:
        plt.annotate(r'$N_{}$'.format(node_num), xy=(i, y + 0.05))
    node_coords[node_num] = (i, y)
    node_num += 1

    # Draw edges
    for node_num, v in node_coords.items():
        x, y = v
    sinks = adjacency_matrix[np.where(adjacency_matrix[:, 0] == node_num)][:, 1]
    for s in sinks:
        try:
            sink_coord = node_coords[s]
        except KeyError:
            continue
    for k, n in levels.items():
        if node_num in n:
            color = colors[level_col[k]]
    x_ = np.hstack([x, sink_coord[0]])
    y_ = np.hstack([y, sink_coord[1]])
    plt.plot(x_, y_, color=color)
    plt.ylabel('Node')
    plt.yticks([0], [''])
    plt.xlabel('Level')
    plt.xticks(np.arange(len(levels)), [k for k in levels.keys()])
    plt.show()
