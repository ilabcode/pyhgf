import numpy as np
from scipy.stats import multivariate_normal
import itertools



def calculate_mutual_information_sampled(attributes, edges, node_idx, num_samples=1000, eps=1e-10):
    """
    Calculate mutual information for each parent-child, parent-parent, and self relationship.
    Mutual information between two continuous Gaussian variables is estimated by sampling and 
    approximating the joint and marginal distributions.

    Mathematical::
    The mutual information \( I(X; Y) \) between two continuous Gaussian variables \( X \) and \( Y \) is defined as:

    \[
    I(X; Y) = \int \int p(x, y) \log \left( \ frac{p(x, y)}{p(x) p(y)} \ right) dx \, dy
    \]

    Using discrete samples, mutual information is approximated as:

    \[
    I(X; Y) \ approx \sum_{i,j} p(x_i, y_j) \left( \log(p(x_i, y_j) + \ text{eps}) - \log(p(x_i) + \ text{eps}) - \log(p(y_j) + \ text{eps}) \ right)
    \]

    Parameters:
    - `attributes`: dictionary with node attributes, such as expected precision and mean.
    - `edges`: tuple of adjacency lists for value and volatility parents and children.
    - `node_idx`: index of the child node for mutual information calculation.
    - `num_samples`: (default 1000) number of samples used for covariance and mutual information estimation.
    - `eps`: small constant to avoid log(0) issues (default 1e-10).

    Returns:
    - Dictionary of mutual information values:
      - `parent_child`: mutual information for each parent-child relationship.
      - `parent_parent`: mutual information for each pair of parents.
      - `self`: self-entropy (entropy of the child node itself).
    """
    mutual_info_dict = {"parent_child": {}, "parent_parent": {}, "self": None}

    # Sample the child node distribution
    node_mean = attributes[node_idx]["expected_mean"]
    node_precision = attributes[node_idx]["expected_precision"]
    node_variance = 1 / node_precision
    child_samples = np.random.normal(node_mean, np.sqrt(node_variance), num_samples)

    # List the value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Store sampled data for covariance and MI
    data = []
    for parent_idx in value_parents_idxs:
        parent_mean = attributes[parent_idx]["expected_mean"]
        parent_precision = attributes[parent_idx]["expected_precision"]
        parent_variance = 1 / parent_precision
        parent_samples = np.random.normal(parent_mean, np.sqrt(parent_variance), num_samples)
        data.append(parent_samples)
    data.append(child_samples)
    data = np.vstack(data).T

    # Means and variances from samples
    means = data.mean(axis=0)
    cov = np.cov(data.T)

    # Distributions, marginal and joint 
    for i, parent_idx in enumerate(value_parents_idxs):
        # Get marginal and joint distributions
        d_child = multivariate_normal(means[-1], cov[-1, -1])
        d_parent = multivariate_normal(means[i], cov[i, i])
        jd_parent_child = multivariate_normal(means[[i, -1]], cov[[i, -1]][:, [i, -1]])

        # Discretize samples and calculate MI
        child_vals = np.linspace(child_samples.min(), child_samples.max(), 100)
        parent_vals = np.linspace(data[:, i].min(), data[:, i].max(), 100)
        triplets = ((jd_parent_child.pdf(tup), d_parent.pdf(tup[0]), d_child.pdf(tup[1])) 
                    for tup in itertools.product(parent_vals, child_vals))
        
        # Compute MI (using `eps` to avoid log(0))
        mutual_information = np.sum([
            p_xy * (np.log(p_xy + eps) - np.log(p_x + eps) - np.log(p_y + eps)) 
            for p_xy, p_x, p_y in triplets
        ])
        
        # Store the calculated Mi for current pair
        mutual_info_dict["parent_child"][(parent_idx, node_idx)] = mutual_information

    # if more than one parent, compute pairs
    if len(value_parents_idxs) > 1:
        for i in range(len(value_parents_idxs)):
            for j in range(i + 1, len(value_parents_idxs)):
                parent_i = value_parents_idxs[i]
                parent_j = value_parents_idxs[j]
                
                # Marginal and joint dist
                d_i = multivariate_normal(means[i], cov[i, i])
                d_j = multivariate_normal(means[j], cov[j, j])
                jd_ij = multivariate_normal(means[[i, j]], cov[[i, j]][:, [i, j]])

                # discretize sampled data and calculate mutual information
                vals_i = np.linspace(data[:, i].min(), data[:, i].max(), 100)
                vals_j = np.linspace(data[:, j].min(), data[:, j].max(), 100)
                triplets = ((jd_ij.pdf(tup), d_i.pdf(tup[0]), d_j.pdf(tup[1])) 
                            for tup in itertools.product(vals_i, vals_j))
                
                # Gte MI for pair
                mutual_information = np.sum([
                    p_xy * (np.log(p_xy + eps) - np.log(p_x + eps) - np.log(p_y + eps)) 
                    for p_xy, p_x, p_y in triplets
                ])
                
                # Store mutual information for parent-parent relationship
                mutual_info_dict["parent_parent"][(parent_i, parent_j)] = mutual_information

    # self-entropy for the child node (entropy of a Gaussian variable)
    self_entropy = 0.5 * np.log(2 * np.pi * np.e * node_variance)
    mutual_info_dict["self"] = {node_idx: self_entropy}

    return mutual_info_dict


def calculate_surd(mutual_info_dict, child_idx, parent_idx):
    """
    Calculate the SURD for a specified causal parent in relation to a given child node.

    Parameters:
    - mutual_info_dict: Dictionary output from `calculate_mutual_information_sampled`, containing
      mutual information values for `parent_child`, `parent_parent`, and `self` relationships.
    - child_idx: Index of the child node for which to calculate SURD values.
    - parent_idx: Index of the causal parent node in relation to the child node.

    Returns:
    - Dictionary of SURD values:
      - `Unique`: Unique information provided by the causal parent to the child.
      - `Redundant`: Redundant information shared by the causal parent with other parents.
      - `Synergistic`: Synergistic information only provided when considering multiple parents together.
      - `Differential`: Differential information, representing the remaining uncertainty in the child.
    """

    # retrieve MI for parent child
    unique_info = mutual_info_dict["parent_child"].get((parent_idx, child_idx), 0)
    
    # get self info for child
    child_self_info = mutual_info_dict["self"].get(child_idx, 0)
    
    # Initiliase infos
    redundant_info = 0
    synergistic_info = 0
    
    # CHecking for other parents
    other_parents = [p for (p, c) in mutual_info_dict["parent_child"] if c == child_idx and p != parent_idx]
    
    # Calculate shared (reduntant) info
    if other_parents:
        # sum mutual information between parent of interest and other parents
        redundant_info = sum(
            mutual_info_dict["parent_parent"].get((p, parent_idx), 0) for p in other_parents
        ) / len(other_parents) if len(other_parents) > 1 else 0

        # Synergistic information would only be non-zero if more than one parent is contributing
        # Since we have a single parent, it's zero
        synergistic_info = 0

    # Differential information: 
    # remaining uncertainty in the child node after accounting for the other infos
    differential_info = child_self_info - unique_info - redundant_info - synergistic_info
    
    return {
        "Unique": unique_info,
        "Redundant": redundant_info,
        "Synergistic": synergistic_info,
        "Differential": max(differential_info, 0)  # Ensures differential information is non-negative
    }

