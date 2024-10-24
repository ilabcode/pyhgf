use crate::model::{Network, Node};
use crate::math::sufficient_statistics;

/// Updating an exponential family state node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The node index.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn posterior_update_exponential_state_node(network: &mut Network, node_idx: usize) {

    match network.nodes.get_mut(&node_idx) {
        Some(Node::Exponential(ref mut node)) => {
            let suf_stats = sufficient_statistics(&node.mean);
            for i in 0..suf_stats.len() {
                node.xis[i] = node.xis[i] + (1.0 / (1.0 + node.nus)) * (suf_stats[i] - node.xis[i]);
            }           
        }
        _ => (),
    }
}