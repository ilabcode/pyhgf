import numpy as np
from scipy.stats import multivariate_normal
import itertools


def get_coupling_strength(network, child_idx, parent_idx):
    """
    Retrieve the coupling strength between a specified parent and child node in a network class.

    Parameters:
    - network: The Network
    - child_idx: Index of the child node.
    - parent_idx: Index of the parent node.

    Returns:
    - Coupling strength for the specified parent-child relationship, or None if the node is not found.
    """
    # Access the list of parents for the specified child node
    parents = network.edges[child_idx].value_parents
    if parent_idx in parents:
        # Get the index of the parent in the value_parents list
        parent_position = parents.index(parent_idx)
        # Use this index to find the corresponding coupling strength
        return network.attributes[child_idx]['value_coupling_parents'][parent_position]
    else:
        return None 



def calculate_mutual_information(network, node_idx, num_samples=1000, eps=1e-8):
    """
    Calculate mutual information for each parent-child, parent-parent, and self relationship
    in a Network by using entropy-based mutual information calculations with sampled covariance. 
    Samples are drawn from the nodes distributions, given the expected mean and precision. 

    Parameters:
    - network: A Network.
    - node_idx: Index of the child node.
    - num_samples: Number of samples used for estimating the covariance of the nodes.
    - eps: Small constant to avoid log(0) issues.

    Returns:
    - Dictionary of mutual information values:
      - `parent_child`: mutual information for each parent-child relationship.
      - `parent_parent`: mutual information for each pair of parents.
      - `self`: self-entropy (entropy of the child node itself).
    """
    mutual_info_dict = {"parent_child": {}, "parent_parent": {}, "self": None}

    # Sample from the child node distribution
    node_mean = network.attributes[node_idx]["expected_mean"]
    node_precision = network.attributes[node_idx]["expected_precision"]
    node_variance = 1 / node_precision
    child_samples = np.random.normal(node_mean, np.sqrt(node_variance), num_samples)

    # List the value parents
    value_parents_idxs = network.edges[node_idx].value_parents

    # Sample data for each parent from the parent's expected distributions
    data = [] # initialise data vector for child and parent's samples
    for parent_idx in value_parents_idxs:
        parent_mean = network.attributes[parent_idx]["expected_mean"]
        parent_precision = network.attributes[parent_idx]["expected_precision"]
        parent_variance = 1 / parent_precision
        parent_samples = np.random.normal(parent_mean, np.sqrt(parent_variance), num_samples)
        data.append(parent_samples)

    # Append child samples to data array
    data.append(child_samples)
    
    # Stack data columns and calculate the covariance matrix
    data = np.vstack(data).T # Transfrom to array where comlumns are nodes and rows are samples
    cov = np.cov(data, rowvar=False) + eps * np.eye(data.shape[1]) # matrix with variances between nodes and nodes themselves

    # Self-entropy for the child node 
    child_variance = cov[-1, -1] # get child variance from last cell since it was transposed 
    child_entropy = 0.5 * np.log(2 * np.pi * np.e * child_variance)
    mutual_info_dict["self"] = {node_idx: child_entropy} # store in dict

    # Calculate MI for each parent-child
    for i, parent_idx in enumerate(value_parents_idxs):
        parent_variance = cov[i, i]
        parent_entropy = 0.5 * np.log(2 * np.pi * np.e * parent_variance)
        joint_entropy = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, -1]][:, [i, -1]]))
        
        # MI(X; Y) = H(X) + H(Y) - H(X, Y), based on entropy 
        mutual_info_dict["parent_child"][(parent_idx, node_idx)] = max(
            parent_entropy + child_entropy - joint_entropy, 0
        )

    # Calculate MI for each parent-parent pair if there are multiple parents
    if len(value_parents_idxs) > 1:
        for i in range(len(value_parents_idxs)):
            for j in range(i + 1, len(value_parents_idxs)):
                parent_i = value_parents_idxs[i]
                parent_j = value_parents_idxs[j]
                
                var_i, var_j = cov[i, i], cov[j, j]
                entropy_i = 0.5 * np.log(2 * np.pi * np.e * var_i)
                entropy_j = 0.5 * np.log(2 * np.pi * np.e * var_j)
                joint_entropy_ij = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov[[i, j]][:, [i, j]]))

                mutual_info_dict["parent_parent"][(parent_i, parent_j)] = max(
                    entropy_i + entropy_j - joint_entropy_ij, 0
                )

    return mutual_info_dict



def calculate_surd(mutual_info_dict, child_idx):
    """
    Calculate the SURD decomposition for each causal parent in relation to a given child node. 
    Based on the mutual infromation calculated in the mutual information function. 

    Parameters:
    - mutual_info_dict: Dictionary output from `calculate_mutual_information`, containing
      mutual information values for `parent_child`, `parent_parent`, and `self` relationships.
    - child_idx: Index of the child node.

    Returns:
    - Dictionary of SURD values for each parent:
      - `Redundant`: Redundant causality shared with other parents, including shared information about the child.
      - `Unique`: Unique causality provided only by the specified parent.
        Unique is derived by subtracting the Redundant information from the total mutual information between child and parent. 
      - `Synergistic`: Synergistic causality arising from the joint effect with other parents.
        Synergistic is computed as the difference between the Self-Entropy of the child and the sum of redundant abnd unique information.
        It currently only considers synergistic form considering all parents together. 
      - `Leak`: Causality leak representing unobserved influences on the child.
    """

    surd_dict = {}
    
    # Get the self entropy of the child
    child_self_info = mutual_info_dict["self"].get(child_idx, 0)
    
    # List all parents of the current child from the dictionary 
    all_parents = [p for (p, c) in mutual_info_dict["parent_child"] if c == child_idx] 
    
    # Calculate total influence of all parents on the child as the sum of all parent-child MIs
    total_parent_influence = sum(
        mutual_info_dict["parent_child"].get((p, child_idx), 0) for p in all_parents
    )
    # For each parent, calculate SURD values
    for parent_idx in all_parents:
        # Get mutual information between the specific parent and the child
        parent_child_mi = mutual_info_dict["parent_child"].get((parent_idx, child_idx), 0)
        
        # Identify other parents, excluding the current parent
        other_parents = [p for p in all_parents if p != parent_idx]
        
        # This includes the information about the child that is shared by this parent and the others (not unique to the current parent).
        redundant_info = 0
        for other_parent in other_parents:
            # MI between this parent and each other parent
            parent_parent_mi = mutual_info_dict["parent_parent"].get((other_parent, parent_idx), 0)
            
            # takes the minimum value between the current parent and child and parent-parent
            # This should take the overlapping information given by another parent as well
            # Added together for all other parents
            redundant_info += min(parent_child_mi, parent_parent_mi)

        # Calculate unique information
        # Unique information is the part of the parent-child MI that is not given by redundancy with other parents.
        unique_info = max(parent_child_mi - redundant_info, 0)
        
        #Calculate synergistic information
        # Synergistic information is the additional information gained from considering all parents together.
        combined_parent_mi = total_parent_influence
        synergistic_info = max(child_self_info - (combined_parent_mi + unique_info + redundant_info), 0)

        # Calculate leak information
        # Leak is the remaining unexplained uncertainty in the child after accounting for all parent influences.
        leak_info = max(child_self_info - total_parent_influence, 0)

        # Store the calculated SURD values in the dictionary for the current parent
        surd_dict[parent_idx] = {
            "Redundant": redundant_info,
            "Unique": unique_info,
            "Synergistic": synergistic_info,
            "Leak": leak_info
        }

    return surd_dict


def update_coupling_strength_based_on_surd(network, child_idx, learning_rate=0.01):
    """
    Update the causal coupling strength for each parent-child relationship based on the SURD decomposition, 
    as in the MI and SURD functions.
    
    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - learning_rate: Learning rate parameter controlling the update step size.
    
    Returns:
    - Dictionary of updated causal coupling strengths for each parent-child pair.
    """
    updated_strengths = {}

    # Calculate mutual information and SURD values for the child node
    mutual_info_dict = calculate_mutual_information(network, child_idx)
    surd_values = calculate_surd(mutual_info_dict, child_idx)

    # Define weights for the components
    # Scaling the influence of the different components
    weights = {
        "Unique": 2.0,      
        "Redundant": -1.0,  
        "Synergistic": 1.5,
        "Leak": 0 #       
    }

    # Access list of value parents
    value_parents = network.edges[child_idx].value_parents

    for parent_idx in value_parents:
        # Retrieve the current coupling strength
        current_coupling_strength = get_coupling_strength(network, child_idx, parent_idx)

        # Get values for this parent
        unique_info = surd_values[parent_idx]["Unique"]
        redundant_info = surd_values[parent_idx]["Redundant"]
        synergistic_info = surd_values[parent_idx]["Synergistic"]
        leak_info = surd_values[parent_idx]["Leak"]

        # Calculate weighted score
        score = (
            (unique_info * weights["Unique"]) +
            (redundant_info * weights["Redundant"]) +
            (synergistic_info * weights["Synergistic"]) +
            (leak_info * weights["Leak"])
        )

        # Adjust influence based on score with a sigmoid transformation
        influence_adjustment = learning_rate * (1 / (1 + np.exp(-(score - 0.5))))

        # Update and clip coupling strength to [0, 1]
        updated_coupling_strength = np.clip(current_coupling_strength + influence_adjustment, 0, 1)

        # Store the updated strength
        updated_strengths[parent_idx] = updated_coupling_strength

    return updated_strengths



def calculate_prediction_difference(network, child_idx):
    """
    Calculate the difference in prediction error for a child node by removing the influence of each parent node.
    This difference helps assess the impact of each parent's prediction error on the child's prediction error.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.

    Returns:
    - Dictionary of prediction differences for each parent.
    """
    prediction_differences = {}
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]

    # Access list of value parents.
    value_parents = network.edges[child_idx].value_parents

    for parent_idx in value_parents:
        # Retrieve the coupling strength
        coupling_strength = get_coupling_strength(network, child_idx, parent_idx)

        # Retrieve the parent's prediction error
        parent_prediction_error = network.attributes[parent_idx]["temp"].get("value_prediction_error", 0.0)

        # Only consider observed parent prediction errors
        if not network.attributes[parent_idx].get("observed", True):
            parent_prediction_error = 0.0

        # Calculate the difference in prediction error for the child node 
        # removing the part of the child's PE that the parent also has scaled by coupling strnegth
        prediction_difference = child_prediction_error - (coupling_strength * parent_prediction_error)
        prediction_differences[parent_idx] = prediction_difference

    return prediction_differences

def update_causal_coupling_strength(network, child_idx, learning_rate=0.1):
    """
    Update the causal coupling strength for each parent-child relationship based on the reliability of each parent.

    Parameters:
    - network: A Network.
    - child_idx: Index of the child node.
    - learning_rate: Learning rate parameter controlling the update step size.

    Returns:
    - Dictionary of updated causal coupling strengths for each parent-child pair.
    """
    updated_strengths = {}
    child_prediction_error = network.attributes[child_idx]["temp"]["value_prediction_error"]

    # Access list of value parents.
    value_parents = network.edges[child_idx].value_parents

    for parent_idx in value_parents:
        # Retrieve the current coupling strength using the helper function
        current_coupling_strength = get_coupling_strength(network, child_idx, parent_idx)

        # Get the prediction difference for this parent (from the approximate_prediction_difference function)
        prediction_difference = calculate_prediction_difference(network, child_idx)[parent_idx]

        # Calculate the reliability adjustment based on whether the parent's influence was beneficial
        reliability_adjustment = prediction_difference / (child_prediction_error + 1e-8)

        # Apply a sigmoid transformation
        influence_adjustment = learning_rate * (1 / (1 + np.exp(-(reliability_adjustment - 1))))

        # Update coupling strength and clip to bound to range 
        updated_coupling_strength = np.clip(current_coupling_strength + influence_adjustment, 0, 1)

        # Store updated strengths
        updated_strengths[parent_idx] = updated_coupling_strength

    return updated_strengths
