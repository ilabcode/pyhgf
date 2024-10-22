use crate::network::Network;


pub fn get_update_order(network: Network) -> Vec<usize> {
        
    let mut update_list = Vec::new();

    // list all nodes availables in the network
    let mut nodes_idxs: Vec<usize> = network.nodes.keys().cloned().collect();

    // remove the input nodes
    nodes_idxs.retain(|x| !network.inputs.contains(x));

    let mut remaining = nodes_idxs.len();

    while remaining > 0 {

        let mut has_update = false;
        // loop over all available
        for i in 0..nodes_idxs.len() {

            let idx = nodes_idxs[i];
            let value_children_idxs = network.edges[idx].value_children;
    
            // check if there is any element in value children 
            // that is found in the to-be-updated list of nodes
            let contains_common = value_children_idxs.iter().any(|&item| nodes_idxs.contains(&item));
    
            if !(contains_common) {
    
                // add the node in the update list
                update_list.push(idx);
    
                // remove the parent from the availables nodes list
                nodes_idxs.retain(|&x| x != idx);

                remaining -= 1;
                has_update = true;
                break;
            }
            }
        if !(has_update) {
            break;
        }
    }
    update_list
}


// Tests module for unit tests
#[cfg(test)] // Only compile and include this module when running tests
mod tests {
    use super::*; // Import the parent module's items to test them

    #[test]
    fn test_get_update_order() {
    
        // initialize network
        let mut network = Network::new();
    
        // create a network
        network.add_nodes(
            "continuous-state",
            Some(1),
            None,
            None,
            Some(2),
        );
        network.add_nodes(
            "continuous-state",
            None,
            Some(0),
            None,
            None,
        );
        network.add_nodes(
            "continuous-state",
            None,
            None,
            Some(0),
            None,
        );
        network.add_nodes(
            "exponential-node",
            None,
            None,
            None,
            None,
        );

        println!("Network: {:?}", network);
        println!("Update order: {:?}", get_update_order(network));
    
    }
}
