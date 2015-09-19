import pytest
import pickle
import numpy as np
import hgf


@pytest.fixture
def nodes():
    # Set up a collection of binary HGF nodes
    x3 = hgf.StateNode(initial_mu=1.0,
                       initial_pi=1.0,
                       omega=-6.0)
    x2 = hgf.StateNode(initial_mu=0.0,
                       initial_pi=1.0,
                       omega=-2.5)
    x1 = hgf.BinaryNode()
    xU = hgf.BinaryInputNode()

    # Collect nodes
    def n():
        pass

    n.xU = xU
    n.x1 = x1
    n.x2 = x2
    n.x3 = x3

    return n


@pytest.fixture
def cont_hier():
    # Set up a standard continuous HGF hierarchy
    x2 = hgf.StateNode(initial_mu=1.0,
                       initial_pi=np.inf,
                       omega=-2.0)
    x1 = hgf.StateNode(initial_mu=1.04,
                       initial_pi=np.inf,
                       omega=-12.0)
    xU = hgf.InputNode(omega=0.0)

    x1.add_volatility_parent(parent=x2, kappa=1)
    xU.set_value_parent(parent=x1)

    # Collect nodes
    def h():
        pass

    h.xU = xU
    h.x1 = x1
    h.x2 = x2

    return h


@pytest.fixture
def bin_hier():
    # Set up a simple binary HGF hierarchy
    x3 = hgf.StateNode(initial_mu=1.0,
                       initial_pi=1.0,
                       omega=-6.0)
    x2 = hgf.StateNode(initial_mu=0.0,
                       initial_pi=1.0,
                       omega=-2.5)
    x1 = hgf.BinaryNode()
    xU = hgf.BinaryInputNode()

    x2.add_volatility_parent(parent=x3, kappa=1)
    x1.set_parent(parent=x2)
    xU.set_parent(parent=x1)

    # Collect nodes
    def h():
        pass

    h.xU = xU
    h.x1 = x1
    h.x2 = x2
    h.x3 = x3

    return h


def test_model_config_error(bin_hier):
    m = hgf.Model()

    with pytest.raises(hgf.ModelConfigurationError):
        m.add_node(bin_hier.xU)


def test_model_setup(nodes):
    # Get nodes
    n = nodes
    # Set up model
    m = hgf.Model()

    # Check basic model setup
    assert m.nodes == []
    assert m.params == []
    assert m.var_params == []
    assert m.param_values == []
    assert m.param_trans_values == []
    assert m.surprise() == 0

    # Add nodes
    m.add_node(n.xU)
    m.add_node(n.x1)
    m.add_node(n.x2)
    m.add_node(n.x3)

    # Set up hierarchy
    n.x2.add_volatility_parent(parent=n.x3, kappa=1)
    n.x1.set_parent(parent=n.x2)
    n.xU.set_parent(parent=n.x1)

    # Read binary input from Iglesias et al. (2013)
    binary = np.loadtxt('binary_input.dat')

    # Feed input
    n.xU.input(binary)

    # Recalculate and check result
    mu3 = n.x3.mus
    m.recalculate()
    assert n.x3.mus == mu3

    return m


def test_continuous_hierarchy_setup():
    # Set up a simple HGF hierarchy
    x2 = hgf.StateNode(initial_mu=1.0,
                       initial_pi=np.inf,
                       omega=-2.0)
    x1 = hgf.StateNode(initial_mu=1.04,
                       initial_pi=np.inf,
                       omega=-12.0)
    xU = hgf.InputNode(omega=0.0)

    x1.add_volatility_parent(parent=x2, kappa=1)
    xU.set_value_parent(parent=x1)


def test_node_config_error():
    with pytest.raises(hgf.NodeConfigurationError):
        x1 = hgf.StateNode(initial_mu=1.04,
                           initial_pi=np.inf,
                           omega=-12.0,
                           rho=1,
                           phi=1)


def test_input_continuous(cont_hier):
    # Get the hierarchy
    h = cont_hier

    # Give some inputs
    for u in [0.5, 0.5, 0.5]:
        h.xU.input(u)

    # Check if NegativePosteriorPrecisionError exception is correctly raised
    with pytest.raises(hgf.NegativePosteriorPrecisionError):
        h.xU.input(1e5)

    # Has update worked?
    assert h.x1.mus[1] == 1.0399909812322021
    assert h.x2.mus[1] == 0.99999925013985835


def test_binary_node_setup():
    # Set up a simple binary HGF hierarchy
    x3 = hgf.StateNode(initial_mu=1.0,
                       initial_pi=1.0,
                       omega=-6.0)
    x2 = hgf.StateNode(initial_mu=0.0,
                       initial_pi=1.0,
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
    stdhgf = hgf.StandardHGF(initial_mu1=1.04,
                             initial_pi1=1e4,
                             omega1=-13.0,
                             kappa1=1,
                             initial_mu2=1,
                             initial_pi2=1e1,
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

    # Does resetting work?
    stdhgf.reset()

    assert len(stdhgf.xU.times) == 1
    assert len(stdhgf.xU.inputs) == 1
    assert len(stdhgf.x1.times) == 1
    assert len(stdhgf.x1.mus) == 1
    assert len(stdhgf.x1.pis) == 1
    assert len(stdhgf.x2.times) == 1
    assert len(stdhgf.x2.mus) == 1
    assert len(stdhgf.x2.pis) == 1


def test_binary_standard_hgf():
    # Set up standard 3-level HGF for binary inputs
    binstdhgf = hgf.StandardBinaryHGF(initial_mu2=0.0,
                                      initial_pi2=1.0,
                                      omega2=-2.5,
                                      kappa2=1.0,
                                      initial_mu3=1.0,
                                      initial_pi3=1.0,
                                      omega3=-6.0)

    # Read binary input from Iglesias et al. (2013)
    binary = np.loadtxt('binary_input.dat')

    # Feed input
    binstdhgf.input(binary)

    # Load benchmark
    with open('binstdhgf.pickle', 'rb') as f:
        benchmark = pickle.load(f)

    # Compare to benchmark
    assert binstdhgf.x2.mus == benchmark.x2.mus
    assert binstdhgf.x2.pis == benchmark.x2.pis
    assert binstdhgf.x3.mus == benchmark.x3.mus
    assert binstdhgf.x3.pis == benchmark.x3.pis

    # Does resetting work?
    binstdhgf.reset()

    assert len(binstdhgf.xU.times) == 1
    assert len(binstdhgf.xU.inputs) == 1
    assert len(binstdhgf.x1.times) == 1
    assert len(binstdhgf.x1.mus) == 1
    assert len(binstdhgf.x1.pis) == 1
    assert len(binstdhgf.x2.times) == 1
    assert len(binstdhgf.x2.mus) == 1
    assert len(binstdhgf.x2.pis) == 1
    assert len(binstdhgf.x3.times) == 1
    assert len(binstdhgf.x3.mus) == 1
    assert len(binstdhgf.x3.pis) == 1
