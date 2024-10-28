use crate::model::Network;
use crate::math::sufficient_statistics;

/// Updating an exponential family state node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn prediction_error_exponential_state_node(network: &mut Network, node_idx: usize) {

    if let Some(floats_attributes) = network.attributes.floats.get_mut(&node_idx) {
        if let Some(vectors_attributes) = network.attributes.vectors.get_mut(&node_idx) {
            let mean = floats_attributes.get("mean");
            let nus = floats_attributes.get("nus");
            let xis = vectors_attributes.get("xis");
            let new_xis = match (mean, nus, xis) {
                (Some(mean), Some(nus), Some(xis)) => {
                    let suf_stats = sufficient_statistics(mean);
                    let mut new_xis = xis.clone();
                    for i in 0..suf_stats.len() {
                        new_xis[i] = new_xis[i] + (1.0 / (1.0 + nus)) * (suf_stats[i] - xis[i]);
                    }
                    new_xis
                    }
                    _ => Vec::new(),
                };
                if let Some(xis) = vectors_attributes.get_mut("xis") {
                    *xis = new_xis; // Modify the value directly
                } 
            }
        }
    }