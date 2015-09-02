import pytest
import numpy as np
import hgf


def test_node_setup():
    # Manually set up a simple HGF hierarchy
    x2 = hgf.StateNode(prior_mu=1,
                       prior_pi=np.inf,
                       omega=-2.0)
    x1 = hgf.StateNode(prior_mu=1.04,
                       prior_pi=np.inf,
                       omega=-12.0)
    xU = hgf.InputNode(omega=0.0)

    x1.add_volatility_parent(parent=x2, kappa=1)
    xU.set_value_parent(parent=x1)

    # Give an input
    xU.input(0.5)

    # Has update worked?
    assert x1.mus[1] == 1.0399909812322021
    assert x2.mus[1] == 0.99999925013985835


def test_standard_hgf():
    # Set up standard 2-level HGF for continuous inputs
    stdhgf = hgf.StandardHGF(prior_mu1=1.04,
                             prior_pi1=np.inf,
                             omega1=-12.0,
                             kappa1=1,
                             prior_mu2=1,
                             prior_pi2=np.inf,
                             omega2=-2,
                             omega_input=0.0)

    # Give some inputs
    for u in [0.5, 0.5, 0.5]:
        stdhgf.input(u)

    # Has update worked?
    assert stdhgf.x1.mus[1] == 1.0399909812322021
    assert stdhgf.x2.mus[1] == 0.99999925013985835

    # Check if NegativePosteriorPrecisionError exception is correctly raised
    with pytest.raises(hgf.NegativePosteriorPrecisionError):
        stdhgf.input(1e5)
