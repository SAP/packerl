import networkx as nx


def network_graphs_equal(g1: nx.DiGraph, g2: nx.DiGraph,
                         global_attrs=None, node_attrs=None, edge_attrs=None) -> bool:
    """
    Check if two networkx graphs are equal in terms of topology and attributes.
    Return True if they are equal, False otherwise.
    """

    # Check for base topology equality
    if set(g1.nodes()) != set(g2.nodes()) or set(g1.edges()) != set(g2.edges()):
        return False

    # Check global attributes for equality
    if global_attrs is not None:
        for attr in global_attrs:
            if g1.graph.get(attr) != g2.graph.get(attr):
                return False

    # Check node attributes for equality
    if node_attrs is not None:
        for node in g1.nodes():
            for attr in node_attrs:
                if g1.nodes[node].get(attr) != g2.nodes[node].get(attr):
                    return False

    # Check edge attributes for equality
    if edge_attrs is not None:
        for (u, v) in g1.edges():
            for attr in edge_attrs:
                if g1[u][v].get(attr) != g2[u][v].get(attr):
                    return False
    return True
