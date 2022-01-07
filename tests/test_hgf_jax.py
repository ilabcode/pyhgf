import os
import unittest
from unittest import TestCase
import jax.numpy as jnp
from jax.lax import scan
from numpy import loadtxt
from typing import List, Tuple
import pytest

from ghgf.hgf_jax import (
    update_parents,
    update_input_parents,
    gaussian_surprise,
    loop_inputs,
)


class Testsdt(TestCase):
    def test_update_parents(self):

        # No value parent - no volatility parents
        node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [1.0],
            "omega": 1.0,
            "rho": 1.0,
        }

        value_parents = [None]
        volatility_parents = [None]

        update_parents(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # No volatility parents
        node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [1.0],
            "omega": 1.0,
            "rho": 1.0,
        }
        parent_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [None],
            "omega": 1.0,
            "rho": 1.0,
        }

        value_parents = [[parent_parameters, [None], [None]]]

        volatility_parents = [None]

        update_parents(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # No value parents
        node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [1.0],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [None],
            "omega": 1.0,
            "rho": 1.0,
        }
        parent_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [None],
            "omega": 1.0,
            "rho": 1.0,
        }

        volatility_parents = [[parent_parameters, [None], [None]]]

        value_parents = [None]

        update_parents(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # Both value and volatility parents
        node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [1.0],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [1.0],
            "omega": 1.0,
            "rho": 1.0,
        }
        parent_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [None],
            "omega": 1.0,
            "rho": 1.0,
        }

        volatility_parents = [[parent_parameters, [None], [None]]]

        value_parents = [[parent_parameters, [None], [None]]]

        update_parents(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

    def test_gaussian_surprise(self):
        surprise = gaussian_surprise(
            x=jnp.array(1.0), muhat=jnp.array(0.0), pihat=jnp.array(1.0)
        )
        assert jnp.all(jnp.isclose(surprise, 1.4189385))

    def test_update_input_parents(self):

        # No value parent - no volatility parents
        input_node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [1.0],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [1.0],
            "omega": 1.0,
            "rho": 1.0,
        }

        parent_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": [None],
            "mu": 1.0,
            "nu": 1.0,
            "psis": [None],
            "omega": 1.0,
            "rho": 1.0,
        }

        volatility_parents = [[parent_parameters, [None], [None]]]
        value_parents = [[parent_parameters, [None], [None]]]

        update_input_parents(
            input_node=[input_node_parameters, value_parents, volatility_parents],
            value=0.2,
            old_time=1,
            new_time=2,
        )

    def test_loop_inputs(self):
        """Test the entire scan loop"""

        # No value parent - no volatility parents
        input_node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": [jnp.array(1.0)],
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": [jnp.array(1.0)],
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        parent_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": [jnp.array(1.0)],
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": [jnp.array(1.0)],
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        volatility_parents = [[parent_parameters, [None], [None]]]
        value_parents = [[parent_parameters, [None], [None]]]

        input_node = [input_node_parameters, value_parents, volatility_parents]

        el = jnp.array(5.0), jnp.array(1.0)  # value, new_time
        res_init = (
            input_node,
            {"time": jnp.array(0.0), "value": jnp.array(0.0), "surprise": 0.0},
        )

        res, _ = loop_inputs(res=res_init, el=el)

        node_structure, results = res

        assert results["time"] == 1.0
        assert results["value"] == 5.0
        assert results["surprise"] == 2.5279474

    def test_scan_loop(self):

        timeserie = loadtxt(f"/home/nicolas/git/ghgf/tests/data/usdchf.dat")

        # No value parent - no volatility parents
        input_node_parameters = {
            "mu": jnp.array(1.04),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1e4),
            "pihat": jnp.array(1.0),
            "kappas": [jnp.array(1.0)],
            "nu": jnp.array(1.0),
            "psis": [jnp.array(1.0)],
            "omega": jnp.log(1e-4),
            "rho": jnp.array(0.0),
        }

        parent_parameters = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": [jnp.array(1.0)],
            "nu": jnp.array(1.0),
            "psis": [jnp.array(1.0)],
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }

        volatility_parents = [[parent_parameters, [None], [None]]]
        value_parents = [[parent_parameters, [None], [None]]]

        input_node = [input_node_parameters, value_parents, volatility_parents]

        res_init = (
            input_node,
            {"time": jnp.array(0.0), "value": jnp.array(0.0), "surprise": 0.0},
        )
        # Create the data (value and time vectors)
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        # Run the entire for loop
        last, final = scan(loop_inputs, res_init, data)

        node_structure, results = final


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
