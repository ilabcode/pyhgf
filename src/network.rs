use std::collections::HashMap;
use crate::updates::posterior;
use pyo3::prelude::*;

#[derive(Debug)]
pub struct AdjacencyLists{
    pub value_parents: Option<usize>,
    pub value_children: Option<usize>,
    pub volatility_parents: Option<usize>,
    pub volatility_children: Option<usize>,
}
#[derive(Debug, Clone)]
pub struct ContinuousStateNode{
    pub mean: f64,
    pub expected_mean: f64,
    pub precision: f64,
    pub expected_precision: f64,
    pub tonic_volatility: f64,
    pub tonic_drift: f64,
    pub autoconnection_strength: f64,
}
#[derive(Debug, Clone)]
pub struct ExponentialFamiliyStateNode {
    pub mean: f64,
    pub expected_mean: f64,
    pub nus: f64,
    pub xis: [f64; 2],
}

#[derive(Debug, Clone)]
pub enum Node {
    Continuous(ContinuousStateNode),
    Exponential(ExponentialFamiliyStateNode),
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
    #[pyo3(signature = (kind, value_parents=None, value_children=None, volatility_children=None, volatility_parents=None))]
    pub fn add_node(&mut self, kind: String, value_parents: Option<usize>, value_children: Option<usize>, volatility_children: Option<usize>, volatility_parents: Option<usize>) {
        
        // the node ID is equal to the number of nodes already in the network
        let node_id: usize = self.nodes.len();
        
        let edges = AdjacencyLists{
            value_parents: value_children,
            value_children: value_parents, 
            volatility_parents: volatility_parents,
            volatility_children: volatility_children,
        };
        
        // add edges and attributes
        if kind == "continuous-state" {
            let continuous_state = ContinuousStateNode{
                mean: 0.0, expected_mean: 0.0, precision: 1.0, expected_precision: 1.0,
                tonic_drift: 0.0, tonic_volatility: -4.0, autoconnection_strength: 1.0
            };
            let node = Node::Continuous(continuous_state);
            self.nodes.insert(node_id, node);
            self.edges.push(edges);

            // if this node has no children, this is an input node
            if (value_children == None) & (volatility_children == None) {
                self.inputs.push(node_id);
            }
        } else if kind == "exponential-state" {
            let exponential_node: ExponentialFamiliyStateNode = ExponentialFamiliyStateNode{
                mean: 0.0, expected_mean: 0.0, nus: 0.0, xis: [0.0, 0.0]
            };
            let node = Node::Exponential(exponential_node);
            self.nodes.insert(node_id, node);
            self.edges.push(edges);
            
            // if this node has no children, this is an input node
            if (value_children == None) & (volatility_children == None) {
                self.inputs.push(node_id);
            }

        } else {
            println!("Invalid type of node provided ({}).", kind);
        }
    }

    pub fn prediction_error(&mut self, node_idx: usize) {

        // get the observation value
        let mean;
        match self.nodes[&node_idx] {
            Node::Continuous(ref node) => {
                mean = node.mean;
            }
            Node::Exponential(ref node) => {
                mean = node.mean;
            }
        }

        let value_parent_idx = &self.edges[node_idx].value_parents;
        match value_parent_idx {
            Some(idx) => {
                match self.nodes.get_mut(idx)  {
                    Some(Node::Continuous(ref mut parent)) => {
                        parent.mean = mean
                    }
                    Some(Node::Exponential(ref mut parent)) => {
                        parent.mean = mean
                    }
                    None => println!("No prediction error for this type of node."),
                }
            }
            None => println!("No value parent"),
        }
        }

    pub fn posterior_update(&mut self, node_idx: usize, observation: f64) {

        match self.nodes.get_mut(&node_idx) {
            Some(Node::Continuous(ref mut node)) => {
                node.mean = observation
            }
            Some(Node::Exponential(ref mut node)) => {
                posterior::exponential::posterior_update_exponential(node)
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
fn rshgf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Network>()?;
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
    
        // create a network with two exponential family state nodes
        network.add_node(
            String::from("exponential-state"),
            None,
            None,
            None,
            None
        );
        network.add_node(
            String::from("exponential-state"),
            None,
            None,
            None,
            None
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
