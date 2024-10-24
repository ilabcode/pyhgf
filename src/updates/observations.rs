use crate::model::{Network, Node};


/// Inject new observations into an input node
/// 
/// # Arguments
/// * `network` - The main network containing the node.
/// * `node_idx` - The input node index.
/// * `observations` - The new observations.
/// 
/// # Returns
/// * `network` - The network after message passing.
pub fn observation_update(network: &mut Network, node_idx: usize, observations: f64) {

    match network.nodes.get_mut(&node_idx) {
        Some(Node::Exponential(ref mut node)) => {
            node.mean = observations;
            }
            _ => (),
        }
}