# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

from pyhgf import load_data
from pyhgf.model import HGF
from pyhgf.structure import structure_as_dict, structure_validation

class TestStructure(TestCase):

    def test_structure_as_dict(self):

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        ).input_data(input_data=timeserie)
        hgf_dict = structure_as_dict(node_structure=hgf.node_trajectories)
        assert list(hgf_dict.keys()) == ['node_0', 'node_1', 'node_2']

    def test_structure_validation(self):
        
        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        structure_validation(hgf.node_structure)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
