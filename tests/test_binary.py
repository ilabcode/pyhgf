# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan

from pyhgf import load_data
from pyhgf.structure import structure_validation
from pyhgf.binary import (
    loop_binary_inputs,
    binary_input_update,
    binary_surprise,
    gaussian_density,
    sgm
)


class Testbinary(TestCase):

    def test_gaussian_density(self):
        surprise = gaussian_density(
            x=jnp.array([1.0, 1.0]),
            mu=jnp.array([0.0, 0.0]),
            pi=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 0.24197073))

    def test_sgm(self):
        assert jnp.all(jnp.isclose(sgm(jnp.array([.3, .3])), 0.5744425))

    def test_binary_surprise(self):
        surprise = binary_surprise(
            x=jnp.array([1.0]),
            muhat=jnp.array([0.2]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.609438))
        
    def test_update_binary_input_parents(self):

        # One binary node and one continuous parent
        input_node_parameters = {
            "pihat": jnp.inf,
            "eta0": 0.0,
            "eta1": 1.0,
        }

        x2_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        x2 = (x2_parameters, None, None),

        binary_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        binary_node = (binary_parameters, x2, None),

        surprise, output_node = binary_input_update(
            input_node=(input_node_parameters, binary_node, None),
            value=1.0,
            old_time=jnp.array(1.0),
            new_time=jnp.array(2.0),
        )

        assert surprise == 0.12692808

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert isinstance(output_node[1], tuple)
        assert output_node[2] is None

    def test_loop_binary_inputs(self):
        """Test the function that should be scanned"""

        # One binary node and one continuous parent
        input_node_parameters = {
            "pihat": 1e6,
            "eta0": 0.0,
            "eta1": 1.0,
        }

        x2_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        x2 = (x2_parameters, None, None),

        binary_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        binary_node = (binary_parameters, x2, None),
        
        input_node=(input_node_parameters, binary_node, None)

        el = jnp.array(1.0), jnp.array(1.0)  # value, new_time
        res_init = (
            input_node,
            {"time": jnp.array(0.0), "value": jnp.array(0.0), "surprise": 0.0},
        )

        res, _ = loop_binary_inputs(res=res_init, el=el)

        _, results = res

        assert results["time"] == 1.0
        assert results["value"] == 1.0
        assert jnp.isclose(results["surprise"], -12.769644)

    def test_scan_loop(self):

        ##############
        # Binary HGF #
        ##############

        timeserie = load_data("binary")

        # One binary node and one continuous parent
        input_node_parameters = {
            "pihat": jnp.inf,
            "eta0": 0.0,
            "eta1": 1.0,
        }

        x2_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        x2 = (x2_parameters, None, None),

        binary_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        binary_node = (binary_parameters, x2, None),
        
        input_node=(input_node_parameters, binary_node, None)

        res_init = (
            input_node,
            {
                "time": jnp.array(0.0),
                "value": jnp.array(0.0),
                "surprise": jnp.array(0.0),
            },
        )
        # Create the data (value and time vectors)
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        # Run the entire for loop
        last, final = scan(loop_binary_inputs, res_init, data)

        node_structure, results = final


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
