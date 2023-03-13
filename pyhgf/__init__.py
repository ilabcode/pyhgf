# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import pkg_resources  # type: ignore
from numpy import loadtxt

__version__ = "0.0.2"


def load_data(dataset: str):
    """Load dataset for continuous or binary HGF.

    Parameters
    ----------
    dataset : str
        The type of data to load. Can be `"continous"` or `"binary"`.

    Returns
    -------
    data : np.ndarray
        The data (a 1d timeseries).

    Notes
    -----
    The continuous time series is the standard USD-CHF conversion rates over time used
    in the Matlab examples. The binary dataset is from Iglesias et al. (2013) [#].

    References
    ----------
    .. [#] Iglesias, S., Mathys, C., Brodersen, K. H., Kasper, L., Piccirelli, M., den
    Ouden, H. E. M., & Stephan, K. E. (2013). Hierarchical Prediction Errors in Midbrain
    and Basal Forebrain during Sensory Learning. In Neuron (Vol. 80, Issue 2, pp.
    519–530). Elsevier BV. https://doi.org/10.1016/j.neuron.2013.09.009

    """
    if dataset == "continuous":
        data = loadtxt(pkg_resources.resource_filename("pyhgf", "/data/usdchf.dat"))
    elif dataset == "binary":
        data = loadtxt(
            pkg_resources.resource_filename("pyhgf", "/data/binary_input.dat")
        )
    else:
        raise ValueError("Invalid dataset argument. Should be 'continous' or 'binary'.")

    return data
