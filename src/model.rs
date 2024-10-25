use std::collections::HashMap;
use crate::{updates::observations::observation_update, utils::function_pointer::FnType};
use crate::utils::set_sequence::set_update_sequence;
use crate::utils::function_pointer::get_func_map;
use pyo3::types::PyTuple;
use pyo3::{prelude::*, types::{PyList, PyDict}};

#[derive(Debug)]
#[pyclass]
pub struct AdjacencyLists{
    #[pyo3(get, set)]
    pub value_parents: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub value_children: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub volatility_parents: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub volatility_children: Option<Vec<usize>>,
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
pub struct UpdateSequence {
    pub predictions: Vec<(usize, FnType)>,
    pub updates: Vec<(usize, FnType)>,
}

#[derive(Debug)]
#[pyclass]
pub struct Network{
    pub nodes: HashMap<usize, Node>,
    pub edges: Vec<AdjacencyLists>,
    pub inputs: Vec<usize>,
    pub update_sequence: UpdateSequence,
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
            update_sequence: UpdateSequence {predictions: Vec::new(), updates: Vec::new()}
        }
    }

    /// Add nodes to the network.
    /// 
    /// # Arguments
    /// * `kind` - The type of node that should be added.
    /// * `value_parents` - The indexes of the node's value parents.
    /// * `value_children` - The indexes of the node's value children.
    /// * `volatility_children` - The indexes of the node's volatility children.
    /// * `volatility_parents` - The indexes of the node's volatility parents.
    #[pyo3(signature = (kind="continuous-state", value_parents=None, value_children=None, volatility_parents=None, volatility_children=None,))]
    pub fn add_nodes(&mut self, kind: &str, value_parents: Option<Vec<usize>>, 
        value_children: Option<Vec<usize>>,
        volatility_parents: Option<Vec<usize>>, volatility_children: Option<Vec<usize>>, ) {

        // the node ID is equal to the number of nodes already in the network
        let node_id: usize = self.nodes.len();

        // if this node has no children, this is an input node
        if (value_children == None) & (volatility_children == None) {
            self.inputs.push(node_id);
        }
        
        let edges = AdjacencyLists{
            value_parents: value_parents,
            value_children: value_children, 
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

        } else if kind == "exponential-state" {
            let exponential_node: ExponentialFamiliyStateNode = ExponentialFamiliyStateNode{
                mean: 0.0, expected_mean: 0.0, nus: 0.0, xis: [0.0, 0.0]
            };
            let node = Node::Exponential(exponential_node);
            self.nodes.insert(node_id, node);
            self.edges.push(edges);

        } else {
            println!("Invalid type of node provided ({}).", kind);
        }
    }

    pub fn set_update_sequence(&mut self) {
        self.update_sequence = set_update_sequence(self);
    }

    /// Single time slice belief propagation.
    /// 
    /// # Arguments
    /// * `observations` - A vector of values, each value is one new observation associated
    /// with one node.
    pub fn belief_propagation(&mut self, observations_set: Vec<f64>) {

        let predictions = self.update_sequence.predictions.clone();
        let updates = self.update_sequence.updates.clone();

        // 1. prediction steps
        for (idx, step) in predictions.iter() {
            step(self, *idx);
        }
        
        // 2. observation steps
        for (i, observations) in observations_set.iter().enumerate() {
            let idx = self.inputs[i];
            observation_update(self, idx, *observations);
        } 

        // 3. update steps
        for (idx, step) in updates.iter() {
            step(self, *idx);
        }
    }

    /// Add a sequence of observations.
    /// 
    /// # Arguments
    /// * `input_data` - A vector of vectors. Each vector is a time series of observations
    /// associated with one node.
    pub fn input_data(&mut self, input_data: Vec<Vec<f64>>) {
        for observation in input_data {
            self.belief_propagation(observation);
        }
    }

    #[getter]
    pub fn get_inputs<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        let py_list = PyList::new(py, &self.inputs);  // Create a PyList from Vec<usize>
        Ok(py_list)
    }

    #[getter]
    pub fn get_edges<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
        // Create a new Python list
        let py_list = PyList::empty(py);

        // Convert each struct in the Vec to a Python object and add to PyList
        for s in &self.edges {
            // Create a new Python dictionary for each MyStruct
            let py_dict = PyDict::new(py);
            py_dict.set_item("value_parents", &s.value_parents)?;
            py_dict.set_item("value_children", &s.value_children)?;
            py_dict.set_item("volatility_parents", &s.volatility_parents)?;
            py_dict.set_item("volatility_children", &s.volatility_children)?;

            // Add the dictionary to the list
            py_list.append(py_dict)?;
        }
        Ok(py_list)
    }

    #[getter]
    pub fn get_update_sequence<'py>(&self, py: Python<'py>) -> PyResult<&'py PyList> {
    
    let func_map = get_func_map();
    let py_list = PyList::empty(py);
    // Iterate over the Rust vector and convert each tuple
    for &(num, func) in self.update_sequence.predictions.iter() {
        // Retrieve the function name from the map
        let func_name = func_map.get(&func).unwrap_or(&"unknown");

        // Convert the Rust tuple to a Python tuple with the function name as a string
        let py_tuple = PyTuple::new(py, &[num.into_py(py), (*func_name).into_py(py)]);

        // Append the Python tuple to the Python list
        py_list.append(py_tuple)?;
    }        
    for &(num, func) in self.update_sequence.updates.iter() {
        // Retrieve the function name from the map
        let func_name = func_map.get(&func).unwrap_or(&"unknown");

        // Convert the Rust tuple to a Python tuple with the function name as a string
        let py_tuple = PyTuple::new(py, &[num.into_py(py), (*func_name).into_py(py)]);

        // Append the Python tuple to the Python list
        py_list.append(py_tuple)?;
    }        
    Ok(py_list)
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
        network.add_nodes(
            "exponential-state",
            None,
            None,
            None,
            None
        );
        network.add_nodes(
            "exponential-state",
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
