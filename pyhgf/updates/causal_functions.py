import numpy as np
from scipy.stats import multivariate_normal
import itertools

def calculate_mutual_information(network, node_idx, num_samples=1000, eps=1e-8):
    """
    Calculate mutual information for each parent-child, parent-parent, and self relationship
    in a Network instance using entropy-based mutual information calculations with sampled covariance.

    Parameters:
    - network: A Network instance.
    - node_idx: Index of the child node for mutual information calculation.
    - num_samples: Number of samples used for covariance and mutual information estimation (default 1000).
    - eps: Small constant to avoid log(0) issues (default 1e-8).

    Returns:
    - Dictionary of mutual information values:
      - `parent_child`: mutual information for each parent-child relationship.
      - `parent_parent`: mutual information for each pair of parents.
      - `self`: self-entropy (entropy of the child node itself).
    """
    mutual_info_dict = {"parent_child": {}, "parent_parent": {}, "self": None}

    # Sample the child node distribution
    node_mean = network.attributes[node_idx]["expected_mean"]
    node_precision = network.attributes[node_idx]["expected_precision"]
    node_variance = 1 / node_precision
    child_samples = np.random.normal(node_mean, np.sqrt(node_variance), num_samples)

    # List the value parents
    value_parents_idxs = network.edges[node_idx].value_parents

    # Sample data for each parent
    data = []
    for parent_idx in value_parents_idxs:
        parent_mean = network.attributes[parent_idx]["expected_mean"]
        parent_precision = network.attributes[parent_idx]["expected_precision"]
        parent_variance = 1 / parent_precision
        parent_samples = np.random.normal(parent_mean, np.sqrt(parent_variance), num_samples)
        data.append(parent_samples)

    # Append child samples to data array
    data.append(child_samples)
    
    # Stack data columns and calculate the covariance matrix
    data = np.vstack(data).T
    cov = np.cov(data, rowvar=False) + eps * np.eye(data.shape[1])  # adjusting the diagonal of the matrix

    # Self-entropy for the child node 
    child_variance = cov[-1, -1]
    child_entropy = 0.5 * np.log(2 * np.pi * np.e * child_variance)
    mutual_info_dict["self"] = {node_idx: child_entropy}

    # Calculate MI for each parent-child pair using entropy method
    for i, parent_idx in enumerate(value_parents_idxs):
        # Entropies of individual distributions
        parent_variance = cov[i, i]
        parent_entropy = 0.5 * np.log(2 * np.pi * np.e * parent_variance)
        
        # calculating the joint entropy
        joint_entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, -1]][:, [i, -1]]))
        
        # MI(X; Y) = H(X) + H(Y) - H(X, Y)
        mutual_info_dict["parent_child"][(parent_idx, node_idx)] = max(
            parent_entropy + child_entropy - joint_entropy, 0  # Ensure non-negative MI
        )

    # Calculate MI for each parent-parent pair if there are multiple parents
    if len(value_parents_idxs) > 1:
        for i in range(len(value_parents_idxs)):
            for j in range(i + 1, len(value_parents_idxs)):
                parent_i = value_parents_idxs[i]
                parent_j = value_parents_idxs[j]
                
                # Entropies of individual distributions
                var_i, var_j = cov[i, i], cov[j, j]
                entropy_i = 0.5 * np.log(2 * np.pi * np.e * var_i)
                entropy_j = 0.5 * np.log(2 * np.pi * np.e * var_j)
                
                # Joint entropy of parent_i and parent_j
                joint_entropy_ij = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, j]][:, [i, j]]))

                # MI(X; Y) = H(X) + H(Y) - H(X, Y)
                mutual_info_dict["parent_parent"][(parent_i, parent_j)] = max(
                    entropy_i + entropy_j - joint_entropy_ij, 0  # Ensure non-negative MI
                )

    return mutual_info_dict




def calculate_surd(mutual_info_dict, child_idx, parent_idx):
    """
    Calculate the SURD decomposition for a specified
    causal parent in relation to a given child node.

    Parameters:
    - mutual_info_dict: Dictionary output from `calculate_mutual_information_sampled`, containing
      mutual information values for `parent_child`, `parent_parent`, and `self` relationships.
    - child_idx: Index of the child node
    - parent_idx: Index of the causal parent node in relation to the child node.

    Returns:
    - Dictionary of SURD values:
      - `Redundant`: Redundant causality shared with other parents.
      - `Unique`: Unique causality provided only by the specified parent.
      - `Synergistic`: Synergistic causality arising from the joint effect with other parents.
      - `Leak`: Causality leak representing unobserved influences on the child.
    """

    # Retrieve mutual information between the parent and child
    parent_child_mi = mutual_info_dict["parent_child"].get((parent_idx, child_idx), 0)
    
    # Retrieve self-information of the child node 
    child_self_info = mutual_info_dict["self"].get(child_idx, 0)
    
    # Initialize redundant, unique, synergistic, and leak information
    redundant_info = 0
    unique_info = parent_child_mi
    synergistic_info = 0
    leak_info = child_self_info - parent_child_mi  # Start with remaining uncertainty
    
    # Identify other parents of the child node
    other_parents = [p for (p, c) in mutual_info_dict["parent_child"] if c == child_idx and p != parent_idx]
    
    # Get other MIs if other parents exists
    if other_parents:
        redundant_mis = []
        for other_parent in other_parents:
            # Mutual information between the current parent and each other parent
            parent_parent_mi = mutual_info_dict["parent_parent"].get((other_parent, parent_idx), 0)
            redundant_mis.append(parent_parent_mi)
        
        # Calculate total redundant information as the average shared info, or should it be total?
        redundant_info = sum(redundant_mis) / len(redundant_mis) if redundant_mis else 0
        
        # Calculate unique information by subtracting redundancy from the parent-child MI
        unique_info = max(parent_child_mi - redundant_info, 0)
        
        # Synergistic information is any information gain when considering multiple parents together
        # Estimated as the extra information not accounted for by unique and redundant parts
        combined_parent_mi = sum(mutual_info_dict["parent_child"].get((p, child_idx), 0) for p in other_parents)
        synergistic_info = max(child_self_info - (combined_parent_mi + unique_info + redundant_info), 0)

    # Calculate causality leak as remaining uncertainty in the child not captured by any parent
    leak_info = max(child_self_info - (unique_info + redundant_info + synergistic_info), 0)
    
    return {
        "Redundant": redundant_info,
        "Unique": unique_info,
        "Synergistic": synergistic_info,
        "Leak": leak_info
    }


def approximate_counterfactual_prediction_error(network, parent_idx, child_idx):
    """
    Calculate the prediction error for a child node in a Network by
    removing the influence of a specified parent node (for now, this is based on the current coupling strength) 
    and assessing the child's prediction error without that influence.
    The function is based on the logic that the parent’s prediction error 
    influences the child’s prediction error in proportion to the coupling strength between them.

    Parameters:
    - network: A Network. 
    - parent_idx: Index of the causal parent node whose influence we want to exclude.
    - child_idx: Index of the child node for which to calculate the counterfactual prediction error.

    Returns:
    - Approximated counterfactual prediction error: An estimate of the child node's prediction 
      error if the parent's influence were excluded.
    """
    # retrieve the child’s prediction error
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]
    
    # Get the current strength
    # the coupling strenght is currently not appropriately accessed
    coupling_strength = network.attributes[parent_idx]["value_coupling_children"][child_idx]
    
    # retrieve the parent’s prediction error, if observed
    parent_prediction_error = network.attributes[parent_idx]["temp"].get("value_prediction_error", 0.0)
    if not network.attributes[parent_idx].get("observed", True):
        parent_prediction_error = 0.0
    
    # Calculate the counterfactual prediction error as if the parent had no influence
    counterfactual_prediction_error = child_prediction_error + coupling_strength * parent_prediction_error
    
    return counterfactual_prediction_error



def update_causal_coupling_strength(network, parent_idx, child_idx, learning_rate= 0.1):
    """
    Update the causal coupling strength between a specified parent and child node in a Network
    based on whether the parent's influence positively influenced the child's prediction. 

    Parameters:
    - network: A Network instance containing nodes and their attributes, edges, and observation flags.
    - parent_idx: Index of the causal parent node.
    - child_idx: Index of the child node.
    - learning_rate: Learning rate parameter (eta) controlling the update step size.

    Returns:
    - Updated causal coupling strength.
    """
    # Get the current causal coupling strength for the specific parent-child edge
    # Same issue as previously
    current_coupling_strength = network.attributes[child_idx]["value_coupling_children"][parent_idx]
    
    # GEt the child’s prediction error
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]

    # Get the parent's prediction error, ensuring it's only considered if observed
    parent_prediction_error = approximate_counterfactual_prediction_error(network, parent_idx, child_idx)
    
    # Estimate the counterfactual prediction error, using the coupling strengt as an approximation
    counterfactual_prediction_error = child_prediction_error + current_coupling_strength * parent_prediction_error
    
    # Calculate the influence effect 
    influence_effect = counterfactual_prediction_error - child_prediction_error
    
    # Update based on sigmoid transformation, learning rate and influence 
    updated_coupling_strength = current_coupling_strength + learning_rate * (1 / (1 + np.exp(-influence_effect)))
    
    # Store the updated coupling strength in attributes for the specific edge
    # network.attributes[child_idx]["value_coupling_children"][parent_idx] = updated_coupling_strength
    
    return updated_coupling_strength
