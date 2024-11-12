###################
import numpy as np
from scipy.stats import multivariate_normal

# Helper Functions
def get_coupling_strength(network, child_idx, parent_idx):
    """
    Retrieve the coupling strength between a specified parent and child node.

    Parameters:
    - network: The Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.

    Returns:
    - Coupling strength for the specified parent-child relationship as a float, or None if not found.
    """
    try:
        parent_position = network.edges[child_idx].value_parents.index(parent_idx)
        # Return the single float coupling strength value for the specified parent
        return network.attributes[child_idx]['value_coupling_parents'][parent_position]
    except ValueError:
        return None



def set_coupling_strength(network, child_idx, parent_idx, new_strength):
    """
    Set a new coupling strength between a specified parent and child node.


    Parameters:
    - network: The Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - new_strength: New coupling strength value to set.
    """
    try:
        parent_position = network.edges[child_idx].value_parents.index(parent_idx)
        # Convert to list to modify
        coupling_list = list(network.attributes[child_idx]['value_coupling_parents'])
        coupling_list[parent_position] = new_strength
        # Set back as tuple
        network.attributes[child_idx]['value_coupling_parents'] = tuple(coupling_list)
    except ValueError:
        raise ValueError("Specified parent node not found for the given child node.")


# Information Theory Functions
def calculate_mutual_information(network, child_idx, num_samples=1000, eps=1e-8):
    """
    Calculate mutual information for each parent-child, parent-parent, and self relationship
    in a Network by using entropy-based mutual information calculations with sampled covariance.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - num_samples: Number of samples for covariance estimation.
    - eps: Small constant to avoid log(0) issues.

    Returns:
    - Dictionary of mutual information values: `parent_child`, `parent_parent`, `self`.
    """
    mutual_info_dict = {"parent_child": {}, "parent_parent": {}, "self": None}

    # Sample from the child node distribution
    node_mean = network.attributes[child_idx]["expected_mean"]
    node_precision = network.attributes[child_idx]["expected_precision"]
    child_samples = np.random.normal(node_mean, np.sqrt(1 / node_precision), num_samples)

    # CGet data from parents
    value_parents_idxs = network.edges[child_idx].value_parents
    data = [np.random.normal(network.attributes[parent_idx]["expected_mean"],
                             np.sqrt(1 / network.attributes[parent_idx]["expected_precision"]),
                             num_samples) for parent_idx in value_parents_idxs]
    data.append(child_samples) # Add child samples as well

    # Calculate covariance matrix from samples
    data = np.vstack(data).T
    cov = np.cov(data, rowvar=False) + eps * np.eye(data.shape[1])

    # Calculate entropz values and mutual information
    child_entropy = 0.5 * np.log(2 * np.pi * np.e * cov[-1, -1])
    mutual_info_dict["self"] = {child_idx: child_entropy}

    for i, parent_idx in enumerate(value_parents_idxs):
        parent_entropy = 0.5 * np.log(2 * np.pi * np.e * cov[i, i])
        joint_entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, -1]][:, [i, -1]]))
        mutual_info_dict["parent_child"][(parent_idx, child_idx)] = max(
            parent_entropy + child_entropy - joint_entropy, 0
        )

    if len(value_parents_idxs) > 1:
        for i in range(len(value_parents_idxs)):
            for j in range(i + 1, len(value_parents_idxs)):
                parent_i, parent_j = value_parents_idxs[i], value_parents_idxs[j]
                entropy_i, entropy_j = 0.5 * np.log(2 * np.pi * np.e * cov[i, i]), 0.5 * np.log(2 * np.pi * np.e * cov[j, j])
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


# Update Functions
def update_coupling_strength_surd(network, child_idx, parent_idx, learning_rate=0.01):
    """
    Update the causal coupling strength between a specified parent-child relationship based on SURD decomposition.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - learning_rate: Learning rate for updating the coupling strength.

    Returns:
    - Updated coupling strength for the specified parent-child relationship.
    """
    mutual_info_dict = calculate_mutual_information(network, child_idx)
    surd_values = calculate_surd(mutual_info_dict, child_idx)

    # Weights for the components
    weights = {"Unique": 2.0, "Redundant": -1.0, "Synergistic": 1.5, "Leak": 0}

    # Retrieve the current coupling strength
    current_strength = get_coupling_strength(network, child_idx, parent_idx)

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

    # Update the coupling strength in the network
    set_coupling_strength(network, child_idx, parent_idx, updated_strength)

    return updated_strength



def calculate_prediction_difference(network, child_idx, parent_idx):
    """
    Calculate the difference in prediction error for a specific parent-child relationship, scaled by coupling weight.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.

    Returns:
    - Prediction difference for the specified parent.
    """
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]
    current_strength = get_coupling_strength(network, child_idx, parent_idx)

    parent_prediction_error = network.attributes[parent_idx]["temp"].get("value_prediction_error", 0.0)

    if not network.attributes[parent_idx].get("observed", True):
        parent_prediction_error = 0.0

    prediction_difference = child_prediction_error - (current_strength * parent_prediction_error)
    return prediction_difference


def update_coupling_strength_prediction_error(network, child_idx, parent_idx, learning_rate=0.1):
    """
    Update the causal coupling strength between a specified parent and child node based on prediction error.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - learning_rate: Learning rate for updating the coupling strength.

    Returns:
    - Updated coupling strength for the specified parent-child relationship.
    """
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]

    # Retrieve the current coupling strength as a float
    current_strength = get_coupling_strength(network, child_idx, parent_idx)

    # Calculate prediction difference for the specified parent
    prediction_difference = calculate_prediction_difference(network, child_idx, parent_idx)
    reliability_adjustment = prediction_difference / (child_prediction_error + 1e-8)
    influence_adjustment = learning_rate * (1 / (1 + np.exp(-(reliability_adjustment - 1))))
    updated_strength = np.clip(current_strength + influence_adjustment, 0, 1)

    # Update the coupling strength in the network
    set_coupling_strength(network, child_idx, parent_idx, updated_strength)

    return updated_strength



def edge_update(network, child_idx, parent_idx, method='surd', learning_rate=0.1):
    """
    Update coupling strength for a specific parent-child relationship using either the SURD or prediction error difference method.

    Parameters:
    - network: The Network.
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.
    - method: Either 'surd' or 'prediction_error' to select update method.
    - learning_rate: Learning rate for coupling strength updates.

    Returns:
    - Updated coupling strength for the specified parent-child relationship.
    """
    if method == 'surd':
        updated_strength = update_coupling_strength_surd(network, child_idx, parent_idx, learning_rate)
    elif method == 'prediction_error':
        updated_strength = update_coupling_strength_prediction_error(network, child_idx, parent_idx, learning_rate)
    else:
        raise ValueError("Invalid method. Choose 'surd' or 'prediction_error'.")

    return updated_strength

