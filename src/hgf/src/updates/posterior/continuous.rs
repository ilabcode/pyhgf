use crate::network::ExponentialNode;
use crate::math::sufficient_statistics;

pub fn posterior_update_exponential(node: &mut ExponentialNode) {
    let suf_stats = sufficient_statistics(&node.observation);
    for i in 0..suf_stats.len() {
        node.xis[i] = node.xis[i] + (1.0 / (1.0 + node.nus)) * (suf_stats[i] - node.xis[i]);
    }
}