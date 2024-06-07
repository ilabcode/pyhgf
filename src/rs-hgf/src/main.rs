use std::collections::HashMap;

#[derive(Debug)]
struct AdjacencyLists{
    value_parents: Option<usize>,
    value_children: Option<usize>,
}
#[derive(Debug, Clone)]
struct GenericInputNode{
    observation: f64,
    time_step: f64,
}
#[derive(Debug, Clone)]
struct ExponentialNode {
    observation: f64,
    nus: f64,
    xis: [f64; 2],
}

#[derive(Debug, Clone)]
enum Node {
    Generic(GenericInputNode),
    Exponential(ExponentialNode),
}

#[derive(Debug)]
struct Network{
    nodes: HashMap<usize, Node>,
    edges: Vec<AdjacencyLists>,
    inputs: Vec<usize>,
}

fn sufficient_statistics(x: &f64) -> [f64; 2] {
    [*x, x.powf(2.0)]
}

impl Network {
    // Create a new graph
    fn new() -> Self {
        Network {
            nodes: HashMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
        }
    }

    // Add a node to the graph
    fn add_node(&mut self, kind: String, value_parents: Option<usize>, value_childrens: Option<usize>) {
        
        // the node ID is equal to the number of nodes already in the network
        let node_id: usize = self.nodes.len();
        
        let edges = AdjacencyLists{
            value_children: value_parents, 
            value_parents: value_childrens,
        };
        
        // add edges and attributes
        if kind == "generic-input" {
            let generic_input = GenericInputNode{observation: 0.0, time_step: 0.0};
            let node = Node::Generic(generic_input);
            self.nodes.insert(node_id, node);
            self.edges.push(edges);
            self.inputs.push(node_id);
        } else if kind == "exponential-node" {
            let exponential_node: ExponentialNode = ExponentialNode{observation: 0.0, nus: 0.0, xis: [0.0, 0.0]};
            let node = Node::Exponential(exponential_node);
            self.nodes.insert(node_id, node);
            self.edges.push(edges);
        } else {
            println!("Invalid type of node provided ({}).", kind);
        }
    }

    fn posterior_update(&mut self, node_idx: &usize, observation: f64) {

        match self.nodes.get_mut(node_idx) {
            Some(Node::Generic(ref mut node)) => {
                node.observation = observation
            }
            Some(Node::Exponential(ref mut node)) => {
                let suf_stats = sufficient_statistics(&node.observation);
                for i in 0..suf_stats.len() {
                    node.xis[i] = node.xis[i] + (1.0 / (1.0 + node.nus)) * (suf_stats[i] - node.xis[i]);
                }
            }
            None => println!("The value is None")
        }
    }

    fn prediction_error(&mut self, node_idx: usize) {

        // get the observation value
        let observation;
        match self.nodes[&node_idx] {
            Node::Generic(ref node) => {
                observation = node.observation;
            }
            Node::Exponential(ref node) => {
                observation = node.observation;
            }
        }

        let value_parent_idx = &self.edges[node_idx].value_parents;
        match value_parent_idx {
            Some(idx) => {
                match self.nodes.get_mut(idx)  {
                    Some(Node::Generic(ref mut parent)) => {
                        parent.observation = observation
                    }
                    Some(Node::Exponential(ref mut parent)) => {
                        parent.observation = observation
                    }
                    None => println!("The value is None"),
                }
            }
            None => println!("The value is None"),
        }
        }
    
    fn belief_propagation(&mut self, observations: Vec<f64>) {

        // 1. prediction propagation

        // 2. inject the observations into the input nodes
        for i in 0..observations.len() {

            let input_node_idx = self.inputs[i];
            self.posterior_update(&input_node_idx, observations[i]);
            self.prediction_error(input_node_idx);
        }

        // 3. posterior update - prediction errors propagation
    }

    fn input_data(&mut self, input_data: Vec<Vec<f64>>) {
        for observation in input_data {
            self.belief_propagation(observation);
        }
    }

    fn get_update_order(self) -> Vec<usize> {
        
        let mut update_list = Vec::new();

        // list all nodes availables in the network
        let mut nodes_idxs: Vec<usize> = self.nodes.keys().cloned().collect();

        // remove the input nodes
        nodes_idxs.retain(|x| !self.inputs.contains(x));

        // start with the value parents of input nodes
        for input_idx in self.inputs {
            let value_parent_idxs = self.edges[input_idx].value_parents;
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

    }


fn main() {

    // initialize network
    let mut network = Network::new();

    // create a network
    network.add_node(
        String::from("generic-input"),
        None,
        None,
    );
    network.add_node(
        String::from("generic-input"),
        None,
        None,
    );
    network.add_node(
        String::from("exponential-node"),
        None,
        Some(0),
    );
    network.add_node(
        String::from("exponential-node"),
        None,
        Some(1),
    );

    println!("Graph before belief propagation: {:?}", network);

    // belief propagation
    let input_data = vec![
        vec![1.1, 2.2],
        vec![1.2, 2.1],
        vec![1.0, 2.0],
        vec![1.3, 2.2],
        vec![1.1, 2.5],
        vec![1.0, 2.6],
    ];
    
    network.input_data(input_data);
    
    println!("Graph after belief propagation: {:?}", network);

}