# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan

from ghgf import load_data
from ghgf.continuous import (
    gaussian_surprise,
    loop_continuous_inputs,
    continuous_input_update,
    continuous_node_update,
)
from ghgf.structure import structure_validation


class Testcontinuous(TestCase):

    def test_continuous_node_update(self):

        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        value_parents = None
        volatility_parents = None

        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=jnp.array(0.0),
            new_time=jnp.array(1.0),
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert output_node[1] is None
        assert output_node[2] is None

        #########################
        # No volatility parents #
        #########################
        node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }
        parent_parameters = {
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

        value_parents = ((parent_parameters, None, None),)

        volatility_parents = None

        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=jnp.array(0.0),
            new_time=jnp.array(1.0),
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert isinstance(output_node[1], tuple)
        assert output_node[2] is None

        ####################
        # No value parents #
        ####################
        node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": (jnp.array(1.0),),
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }
        parent_parameters = {
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

        volatility_parents = ((parent_parameters, None, None),)

        value_parents = None

        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=jnp.array(1),
            new_time=jnp.array(2),
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert output_node[1] is None
        assert isinstance(output_node[2], tuple)

        #####################################
        # Both value and volatility parents #
        #####################################
        node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": (1.0,),
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }
        parent_parameters = {
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

        volatility_parents = ((parent_parameters, None, None),)
        value_parents = ((parent_parameters, None, None),)

        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert isinstance(output_node[1], tuple)
        assert isinstance(output_node[2], tuple)

        ########################################################################
        # Both value and volatility parents with values and volatility parents #
        ########################################################################

        # Second level
        vo_pa_2 = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": None,
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }
        vo_pa_3 = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": None,
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }
        volatility_parent_2 = vo_pa_2, None, None
        volatility_parent_3 = vo_pa_3, None, None

        va_pa_2 = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": (jnp.array(1.0),),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }
        value_parent_2 = va_pa_2, None, (volatility_parent_3,)

        # First level
        vo_pa_1 = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": None,
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }
        va_pa = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": (jnp.array(1.0),),
            "nu": jnp.array(1.0),
            "psis": None,
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }
        volatility_parent_1 = vo_pa_1, (value_parent_2,), None
        value_parent_1 = va_pa, None, (volatility_parent_2,)

        node_parameters, value_parents, volatility_parents = volatility_parent_1
        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert isinstance(output_node[1], tuple)
        assert output_node[2] is None

        node_parameters, value_parents, volatility_parents = value_parent_1

        output_node = continuous_node_update(
            node_parameters=node_parameters,
            value_parents=value_parents,
            volatility_parents=volatility_parents,
            old_time=1,
            new_time=2,
        )

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert output_node[1] is None
        assert isinstance(output_node[2], tuple)

    def test_gaussian_surprise(self):
        surprise = gaussian_surprise(
            x=jnp.array([1.0, 1.0]),
            muhat=jnp.array([0.0, 0.0]),
            pihat=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.4189385))

    def test_update_continuous_input_parents(self):

        # No value parent - no volatility parents
        input_node_parameters = {
            "omega": jnp.array(1.0),
            "kappas": jnp.array(1.0),
        }

        parent_parameters = {
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

        volatility_parents = (parent_parameters, None, None),
        value_parents = (parent_parameters, None, None),

        surprise, output_node = continuous_input_update(
            input_node=(input_node_parameters, value_parents, volatility_parents),
            value=0.2,
            old_time=jnp.array(1.0),
            new_time=jnp.array(2.0),
        )

        assert surprise == 2.1381817

        # Verify node structure
        structure_validation(output_node)
        assert len(output_node) == 3
        assert isinstance(output_node[0], dict)
        assert isinstance(output_node[1], tuple)
        assert isinstance(output_node[2], tuple)

    def test_loop_continuous_inputs(self):
        """Test the function that should be scanned"""

        # No value parent - no volatility parents
        input_node_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": jnp.array(1.0),
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": jnp.array(1.0),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        parent_parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": (jnp.array(1.0),),
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
        }

        volatility_parents = (parent_parameters, None, None),
        value_parents = (parent_parameters, None, None),

        input_node = input_node_parameters, value_parents, volatility_parents

        el = jnp.array(5.0), jnp.array(1.0)  # value, new_time
        res_init = (
            input_node,
            {"time": jnp.array(0.0), "value": jnp.array(0.0), "surprise": 0.0},
        )

        res, _ = loop_continuous_inputs(res=res_init, el=el)

        _, results = res

        assert results["time"] == 1.0
        assert results["value"] == 5.0
        assert jnp.isclose(results["surprise"], 2.5279474)


    def test_scan_loop(self):

        ##################
        # Continuous HGF #
        ##################

        timeserie = load_data("continuous")

        # No value parent - no volatility parents
        input_node_parameters = {
            "mu": jnp.array(1.04),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1e4),
            "pihat": jnp.array(1.0),
            "kappas": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": jnp.array(1.0),
            "omega": jnp.log(1e-4),
            "rho": jnp.array(0.0),
        }

        parent_parameters = {
            "mu": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "pihat": jnp.array(1.0),
            "kappas": (jnp.array(1.0),),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(-2.0),
            "rho": jnp.array(0.0),
        }

        volatility_parents = (parent_parameters, None, None),
        value_parents = (parent_parameters, None, None),

        input_node = input_node_parameters, value_parents, volatility_parents

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
        last, final = scan(loop_continuous_inputs, res_init, data)

        node_structure, results = final


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
