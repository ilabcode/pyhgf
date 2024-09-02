use crate::network::Network;


pub fn get_update_order(network: Network) -> Vec<usize> {
        
    let mut update_list = Vec::new();

    // list all nodes availables in the network
    let mut nodes_idxs: Vec<usize> = network.nodes.keys().cloned().collect();

    // remove the input nodes
    nodes_idxs.retain(|x| !network.inputs.contains(x));

    // start with the value parents of input nodes
    for input_idx in network.inputs {
        let value_parent_idxs = network.edges[input_idx].value_parents;
        match value_parent_idxs {
            Some(idx) => {
                // if this parent is still in the list, update it now
                if nodes_idxs.contains(&idx) {

                    // add the node in the update list
                    update_list.push(idx);

                    // remove the parent from the availables nodes list
                    nodes_idxs.retain(|&x| x != idx);

                }
            }
            None => println!("The value is None")
        }
    }
    nodes_idxs
}
