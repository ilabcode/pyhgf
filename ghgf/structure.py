# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, Tuple


def structure_validation(node: Tuple):
    """Verify that the node structure is valid."""
    assert len(node) == 3
    assert isinstance(node[0], Dict)
    if node[1] is not None:
        assert isinstance(node[1], tuple)
    if node[2] is not None:
        assert isinstance(node[2], tuple)

    for n in [1, 2]:
        if node[n] is not None:
            assert isinstance(node[n], tuple)
            assert len(node[n]) > 0

            for i in range(len(node[n])):
                structure_validation(node[n][i])


def structure_as_dict(node_structure, node_id: int = 0, structure_dict: Dict = {}):
    """Transform a HGF node structure into a dictionary of nodes for analysis.

    Parameters
    ----------
    node_structure : tuple
        A node structure comparible with the HGF updates.
    node_id : int
        The identifier for the current (starting) node.
    structure_dict : dict
        The node dictionary. Defaults is an empty dictionary.

    Returns
    -------
    structure_dict : dict
        A dictionary where every key is a node.

    """
    structure_dict[f"node_{node_id}"] = node_structure[0]
    node_id += 1

    # for values and volatility parents
    for i in range(1, 3):
        if node_structure[i] is not None:
            # for each parent
            for n in node_structure[i]:
                structure_dict = structure_as_dict(n, node_id, structure_dict)
    return structure_dict
