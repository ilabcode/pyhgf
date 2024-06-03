
struct AdjacencyLists{
    value_parents: Option<usize>,
    value_children: Option<usize>,
}

struct GenericInputNode{
    observation: f64,
    time_step: f64,
    edges: AdjacencyLists,
}

struct ExponentialNode {
    observation: f64,
    nus: f64,
    xis: f64,
    edges: AdjacencyLists,
}
enum Nodes {
    Generic(GenericInputNode),
    Exponential(ExponentialNode),
}

struct Network{
    nodes: Vec<Nodes>,
    update_sequence: Vec<fn(Network) -> Network>,
}

pub trait NodeControls {
    fn prediction_error(&self, network: Network) -> Network;
    fn get_observation(&mut self, observation: f64);
}


impl GenericInputNode {

    fn get_observation(&mut self, observation: f64) {
        self.observation = observation
    }

    fn prediction_error(&mut self, observation: f64) {
        self.observation = observation
    }

}

impl ExponentialNode {

    fn get_observation(&mut self, observation: f64) {
        self.observation = observation
    }

    fn prediction_error(&mut self, network: Network) -> Network {
        self.observation = observation
    }

}


impl NodeControls for Nodes {

    fn get_observation(&mut self, observation: f64) {

        match self {
            Nodes::Generic(node) => node.get_observation(observation),
            Nodes::Exponential(node) => node.get_observation(observation),
        }
    }

    fn prediction_error(&self, network: Network) -> Network {

        match self {
            Nodes::Generic(node) => node.prediction_error(network),
            Nodes::Exponential(node) => node.prediction_error(network),
        }
    }  
}



















impl NodeControls for Nodes {

    fn prediction_error(&self, network: Network) -> Network {

        // get value parents
        let value_parent_idx = self.edges.value_parents;

        if let Some(idx) = value_parent_idx {
            network.nodes[idx].get_observation(self.observation);
        }
        else {
            println!("No index provided");
        }
        network
    }

    fn get_observation(&mut self, observation: f64) {
        self.observation = observation
    }   
}


impl NodeControls for ExponentialNode {

    fn get_observation(&mut self, observation: f64) {
        self.observation = observation
    }   

    fn prediction_error(&self, network: Network) -> Network {

        // get value parents
        let value_parent_idx = self.edges.value_parents;

        if let Some(idx) = value_parent_idx {
            network.nodes[idx].get_observation(self.observation);
        }
        else {
            println!("No index provided");
        }
        network
    }

}




































pub trait PredictionError {
    fn prediction_error(&self, network: Network) -> Network {
        network
    }
}

impl Nodes {
    fn prediction_error(&self, network: Network) -> Network {

        match network.nodes[idx] {
            Nodes::GenericInputNode | Nodes::ExponentialNode(ref mut observation) => {
                // get value parent
                let value_parent_idx = self.edges.value_parents;
                // store the input value in the value parent, if any
                if let Some(idx) = value_parent_idx {
                    *observation = self.observation;
                },
                _ => {
                    println!("This node cannot receive observations.")
                } else {
                    println!("No index provided");
                }
                }
            }

        }    
        network
}

fn main() {

    // create input data
    let values = [1.1, 1.2, 1.0, 4.0, 1.1];
    let time_steps = [1.0, 1.0, 1.0, 1.0, 1.0];

    // define the update sequence
    let update_sequence: [fn(); 2] = [generic_input_update, exponential_node_update];

    // define the attributes
    let mut generic_input: GenericInput = GenericInput{
        observation: 0.0,
        time_step: 0.0,
        edges: AdjacencyLists{
            value_children: None,
            value_parents: Some(1),
        }
    };

    let mut exponential_node: ExponentialNode = ExponentialNode{
        observation: 0.0,
        nus: 0.0,
        xis: 0.0,
        edges: AdjacencyLists{
            value_children: Some(0),
            value_parents: None,
        }
    };

    // define edges
    let edges: (AdjacencyLists, AdjacencyLists) = (edge_1, edge_2);

    // Iterate over the update sequence and call each function with node pointer
    for func in update_sequence.iter() {
        func();
    }
    
}