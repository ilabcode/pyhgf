import numpy as np
from scipy.stats import multivariate_normal
# Import the function from math instead and adapt MI function

def get_coupling_strength(node_idx, parent_idx, attributes, edges):
    """
    Retrieve the coupling strength between a specified parent and child node.

    Parameters:
    - node_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - attributes: Node attributes dictionary for the network.
    - edges: Edge structure providing access to parent nodes.

    Returns:
    - Coupling strength for the specified parent-child relationship as a float, or None if not found.
    """
    try:
        parent_position = edges[node_idx].value_parents.index(parent_idx)
        # Return the single float coupling strength value for the specified parent
        return attributes[node_idx]['value_coupling_parents'][parent_position]
    except ValueError:
        return None

def set_coupling_strength(node_idx, parent_idx, new_strength, attributes, edges):
    """
    Set a new coupling strength between a specified parent and child node.

    Parameters:
    - node_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - new_strength: New coupling strength value to set.
    - attributes: Node attributes dictionary for the network.
    - edges: Edge structure providing access to parent nodes.
    """
    try:
        parent_position = edges[node_idx].value_parents.index(parent_idx)
        # Convert to list to modify
        coupling_list = list(attributes[node_idx]['value_coupling_parents'])
        coupling_list[parent_position] = new_strength
        # Set back as tuple
        attributes[node_idx]['value_coupling_parents'] = tuple(coupling_list)
    except ValueError:
        raise ValueError("Specified parent node not found for the given child node.")



# Information Theory Functions
def calculate_mutual_information(node_idx, attributes, edges, num_samples=1000, eps=1e-8):
    """
    Calculate mutual information for each parent-child, parent-parent, and self relationship
    in a probabilistic network by using entropy-based mutual information calculations with sampled covariance.

    Parameters:
    - node_idx: Index of the child node.
    - attributes: Node attributes, including expected_mean and expected_precision for each node.
    - edges: Edge structure providing access to parent nodes.
    - num_samples: Number of samples for covariance estimation.
    - eps: Small constant to avoid log(0) issues.

    Returns:
    - Dictionary of mutual information values: `parent_child`, `parent_parent`, `self`.
    """
    mutual_info_dict = {"parent_child": {}, "parent_parent": {}, "self": None}

    # Sample from the child node distribution
    node_mean = attributes[node_idx]["expected_mean"]
    node_precision = attributes[node_idx]["expected_precision"]
    child_samples = np.random.normal(node_mean, np.sqrt(1 / node_precision), num_samples)

    # Retrieve parent nodes
    parent_nodes = edges[node_idx].value_parents

    # Gather data for each parent node
    data = [
        np.random.normal(attributes[parent_idx]["expected_mean"],
                         np.sqrt(1 / attributes[parent_idx]["expected_precision"]),
                         num_samples)
        for parent_idx in parent_nodes
    ]
    data.append(child_samples)  # Add child samples as well

    # Calculate covariance matrix from samples
    data = np.vstack(data).T
    cov = np.cov(data, rowvar=False) + eps * np.eye(data.shape[1])

    # Calculate entropy values and mutual information
    child_entropy = 0.5 * np.log(2 * np.pi * np.e * cov[-1, -1])
    mutual_info_dict["self"] = {node_idx: child_entropy}

    for i, parent_idx in enumerate(parent_nodes):
        parent_entropy = 0.5 * np.log(2 * np.pi * np.e * cov[i, i])
        joint_entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, -1]][:, [i, -1]]))
        mutual_info_dict["parent_child"][(parent_idx, node_idx)] = max(
            parent_entropy + child_entropy - joint_entropy, 0
        )

    # Calculate mutual information between parent nodes if there is more than one parent
    if len(parent_nodes) > 1:
        for i in range(len(parent_nodes)):
            for j in range(i + 1, len(parent_nodes)):
                parent_i, parent_j = parent_nodes[i], parent_nodes[j]
                entropy_i = 0.5 * np.log(2 * np.pi * np.e * cov[i, i])
                entropy_j = 0.5 * np.log(2 * np.pi * np.e * cov[j, j])
                joint_entropy_ij = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, j]][:, [i, j]]))
                mutual_info_dict["parent_parent"][(parent_i, parent_j)] = max(
                    entropy_i + entropy_j - joint_entropy_ij, 0
                )

    return mutual_info_dict


def calculate_surd(mutual_info_dict, child_idx):
    """
    Calculate the SURD decomposition for each causal parent in relation to a given child node.
    Based on mutual information calculated in the `calculate_mutual_information` function.

    Parameters:
    - mutual_info_dict: Output from `calculate_mutual_information`, containing `parent_child`, `parent_parent`, and `self` values.
    - child_idx: Index of the child node.

    Returns: # Add more description here
    - Dictionary of SURD values for each parent:
      - `Redundant`: Information shared with other parents.
      - `Unique`: Information unique to the specified parent.
      - `Synergistic`: Information arising from joint effects.
      - `Leak`: Unobserved influences on the child.
    """
    surd_dict = {}
    child_self_info = mutual_info_dict["self"].get(child_idx, 0)
    all_parents = [p for (p, c) in mutual_info_dict["parent_child"] if c == child_idx]
    total_parent_influence = sum(mutual_info_dict["parent_child"].get((p, child_idx), 0) for p in all_parents)

    for parent_idx in all_parents:
        parent_child_mi = mutual_info_dict["parent_child"].get((parent_idx, child_idx), 0)
        redundant_info = sum(min(parent_child_mi, mutual_info_dict["parent_parent"].get((other_parent, parent_idx), 0))
                             for other_parent in all_parents if other_parent != parent_idx)
        unique_info = max(parent_child_mi - redundant_info, 0)
        synergistic_info = max(child_self_info - (total_parent_influence + unique_info + redundant_info), 0)
        leak_info = max(child_self_info - total_parent_influence, 0)

        surd_dict[parent_idx] = {
            "Redundant": redundant_info,
            "Unique": unique_info,
            "Synergistic": synergistic_info,
            "Leak": leak_info
        }

    return surd_dict

def update_coupling_strength_surd(node_idx, attributes, edges, learning_rate=0.01):
    """
    Update the causal coupling strength for each parent-child relationship based on SURD decomposition.

    Parameters:
    - node_idx: Index of the child node.
    - attributes: Node attributes.
    - edges: Edge structure providing access to parent nodes.
    - learning_rate: Learning rate for updating the coupling strength.

    Returns:
    - Updated attributes with updated coupling strengths for each parent.
    """
    mutual_info_dict = calculate_mutual_information(node_idx, attributes, edges)  # Updated call
    surd_values = calculate_surd(mutual_info_dict, node_idx)

    # Weights for the components
    weights = {"Unique": 2.0, "Redundant": -1.0, "Synergistic": 1.5, "Leak": 0}

    for parent_idx in edges[node_idx].value_parents:
        # Retrieve the current coupling strength
        current_strength = get_coupling_strength(node_idx, parent_idx, attributes, edges)

        # Extract the SURD values for the specified parent
        unique = surd_values[parent_idx]["Unique"]
        redundant = surd_values[parent_idx]["Redundant"]
        synergistic = surd_values[parent_idx]["Synergistic"]
        leak = surd_values[parent_idx]["Leak"]

        # Calculate score and adjustment
        score = (unique * weights["Unique"] + redundant * weights["Redundant"] +
                 synergistic * weights["Synergistic"] + leak * weights["Leak"])
        adjustment = learning_rate * (1 / (1 + np.exp(-(score - 0.5))))
        updated_strength = np.clip(current_strength + adjustment, 0, 1)

        # Update coupling strength
        set_coupling_strength(node_idx, parent_idx, updated_strength, attributes, edges)

    return attributes


def calculate_prediction_difference(node_idx, attributes, edges):
    """
    Calculate the difference in prediction error for each parent-child relationship, scaled by coupling weight.

    Parameters:
    - node_idx: Index of the child node.
    - attributes: Node attributes.
    - edges: Edge structure providing access to parent nodes.

    Returns:
    - Dictionary of prediction differences for each parent.
    """
    prediction_differences = {}
    child_prediction_error = attributes[node_idx]["temp"]["value_prediction_error"]

    for parent_idx in edges[node_idx].value_parents:
        # Retrieve the current coupling strength using the helper function
        current_strength = get_coupling_strength(node_idx, parent_idx, attributes, edges)
        parent_prediction_error = attributes[parent_idx]["temp"].get("value_prediction_error", 0.0)

        if not attributes[parent_idx].get("observed", True):
            parent_prediction_error = 0.0

        prediction_difference = child_prediction_error - (current_strength * parent_prediction_error)
        prediction_differences[parent_idx] = prediction_difference

    return prediction_differences


def update_coupling_strength_prediction_error(node_idx, attributes, edges, learning_rate=0.1):
    """
    Update the causal coupling strength for each parent-child relationship based on prediction error.

    Parameters:
    - node_idx: Index of the child node.
    - attributes: Node attributes.
    - edges: Edge structure providing access to parent nodes.
    - learning_rate: Learning rate for updating the coupling strength.

    Returns:
    - Updated attributes with updated coupling strengths for each parent.
    """
    child_prediction_error = attributes[node_idx]["temp"]["value_prediction_error"]

    for parent_idx in edges[node_idx].value_parents:
        # Retrieve the current coupling strength
        current_strength = get_coupling_strength(node_idx, parent_idx, attributes, edges)
        prediction_difference = calculate_prediction_difference(node_idx, attributes, edges)[parent_idx]
        reliability_adjustment = prediction_difference / (child_prediction_error + 1e-8)
        influence_adjustment = learning_rate * (1 / (1 + np.exp(-(reliability_adjustment - 1))))
        updated_strength = np.clip(current_strength + influence_adjustment, 0, 1)

        # Update the coupling strength
        set_coupling_strength(node_idx, parent_idx, updated_strength, attributes, edges)

    return attributes

