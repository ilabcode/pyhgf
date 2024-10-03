use std::collections::HashMap;
use crate::updates::posterior;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Debug)]
pub struct AdjacencyLists{
    pub value_parents: Option<usize>,
    pub value_children: Option<usize>,
}
#[derive(Debug, Clone)]
pub struct GenericInputNode{
    pub observation: f64,
    pub time_step: f64,
}
#[derive(Debug, Clone)]
pub struct ExponentialNode {
    pub observation: f64,
    pub nus: f64,
    pub xis: [f64; 2],
}

#[derive(Debug, Clone)]
pub enum Node {
    Generic(GenericInputNode),
    Exponential(ExponentialNode),
}

#[derive(Debug)]
#[pyclass]
pub struct Network{
    pub nodes: HashMap<usize, Node>,
    pub edges: Vec<AdjacencyLists>,
    pub inputs: Vec<usize>,
}

#[pymethods]
impl Network {

    // Create a new graph
    #[new]  // Define the constructor accessible from Python
    pub fn new() -> Self {
        Network {
            nodes: HashMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
        }
    }

    // Add a node to the graph
    #[pyo3(signature = (kind, value_parents=None, value_childrens=None))]
    pub fn add_node(&mut self, kind: String, value_parents: Option<usize>, value_childrens: Option<usize>) {
        
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

    pub fn prediction_error(&mut self, node_idx: usize) {

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
                    None => println!("No prediction error for this type of node."),
                }
            }
            None => println!("No value parent"),
        }
        }

    pub fn posterior_update(&mut self, node_idx: usize, observation: f64) {

        match self.nodes.get_mut(&node_idx) {
            Some(Node::Generic(ref mut node)) => {
                node.observation = observation
            }
            Some(Node::Exponential(ref mut node)) => {
                posterior::continuous::posterior_update_exponential(node)
            }
            None => println!("No posterior update for this type of node.")
        }
    }

    pub fn belief_propagation(&mut self, observations: Vec<f64>) {

        // 1. prediction propagation

        for i in 0..observations.len() {
            
            let input_node_idx = self.inputs[i];
            // 2. inject the observations into the input nodes
            self.posterior_update(input_node_idx, observations[i]);
            // 3. posterior update - prediction errors propagation
            self.prediction_error(input_node_idx);
        }

    }

    pub fn input_data(&mut self, input_data: Vec<Vec<f64>>) {
        for observation in input_data {
            self.belief_propagation(observation);
        }
    }
    }


// Create a module to expose the class to Python
#[pymodule]
fn my_rust_library(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Network>()?; // Add the class to the Python module
    Ok(())
}

// Tests module for unit tests
#[cfg(test)] // Only compile and include this module when running tests
mod tests {
    use super::*; // Import the parent module's items to test them


    #[test]
    fn test_exponential_family_gaussian() {
    
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
    
        // println!("Graph before belief propagation: {:?}", network);
    
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
        
        // println!("Graph after belief propagation: {:?}", network);
    
    }
}
