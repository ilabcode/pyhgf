import pytest
import pickle
import numpy as np
import hgf


def test_node_setup():
    # Manually set up a simple HGF hierarchy
    x2 = hgf.StateNode(prior_mu=1.0,
                       prior_pi=np.inf,
                       omega=-2.0)
    x1 = hgf.StateNode(prior_mu=1.04,
                       prior_pi=np.inf,
                       omega=-12.0)
    xU = hgf.InputNode(omega=0.0)

    x1.add_volatility_parent(parent=x2, kappa=1)
    xU.set_value_parent(parent=x1)

    # Give some inputs
    for u in [0.5, 0.5, 0.5]:
        xU.input(u)

    # Check if NegativePosteriorPrecisionError exception is correctly raised
    with pytest.raises(hgf.NegativePosteriorPrecisionError):
        xU.input(1e5)

    # Has update worked?
    assert x1.mus[1] == 1.0399909812322021
    assert x2.mus[1] == 0.99999925013985835


def test_binary_node_setup():
    # Manually set up a simple binary HGF hierarchy
    x3 = hgf.StateNode(prior_mu=1.0,
                       prior_pi=1.0,
                       omega=-6.0)
    x2 = hgf.StateNode(prior_mu=0.0,
                       prior_pi=1.0,
                       omega=-2.5)
    x1 = hgf.BinaryNode()
    xU = hgf.BinaryInputNode()

    x2.add_volatility_parent(parent=x3, kappa=1)
    x1.set_parent(parent=x2)
    xU.set_parent(parent=x1)

    # Give some inputs
    for u in [1, 0, 1]:
        xU.input(u)

    # Has update worked?
    assert x2.mus[3] == 0.37068231113996203
    assert x3.mus[3] == 0.99713458886147199


def test_standard_hgf():
    # Set up standard 2-level HGF for continuous inputs
    stdhgf = hgf.StandardHGF(prior_mu1=1.04,
                             prior_pi1=1e4,
                             omega1=-13.0,
                             kappa1=1,
                             prior_mu2=1,
                             prior_pi2=1e1,
                             omega2=-2,
                             omega_input=np.log(1e-4))

    # Read USD-CHF data
    usdchf = np.loadtxt('usdchf.dat')

    # Feed input
    stdhgf.input(usdchf)

    # Load benchmark
    with open('stdhgf.pickle', 'rb') as f:
        benchmark = pickle.load(f)

    # Compare to benchmark
    assert stdhgf.x1.mus == benchmark.x1.mus
    assert stdhgf.x1.pis == benchmark.x1.pis
    assert stdhgf.x2.mus == benchmark.x2.mus
    assert stdhgf.x2.pis == benchmark.x2.pis
