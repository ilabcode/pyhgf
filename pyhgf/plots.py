# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import itertools
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from graphviz.sources import Source

    from pyhgf.model import HGF


def plot_trajectories(
    hgf: "HGF",
    ci: bool = True,
    surprise: bool = True,
    figsize: Tuple[int, int] = (18, 9),
    axs: Optional[Union[List, Axes]] = None,
) -> Axes:
    r"""Plot the trajectories of the nodes' sufficient statistics and surprise.

    This function will plot :math:`\hat{mu}`, :math:`\¨hat{pi}` (converted into standard
    deviation) and the surprise at each level of the node structure.

    Parameters
    ----------
    hgf :
        Instance of the HGF model.
    ci :
        Show the uncertainty aroud the values estimates (standard deviation).
    surprise :
        If `True` plot each node's surprise together witt the sufficient statistics.
        If `False`, only the input node's surprise is depicted.
    figsize :
        The width and height of the figure. Defaults to `(18, 9)` for a 2-levels model,
        or to `(18, 12)` for a 3-levels model.
    axs :
        A list of Matplotlib axes instance where to draw the trajectories. This should
        correspond to the number of nodes in the structure. Default is `None` (create a
        new figure).

    Returns
    -------
    axs :
        The Matplotlib axes instances where to plot the trajectories.

    Examples
    --------
    Visualization of nodes' trajectories from a three-level continuous HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF

        # Set up standard 3-level HGF for continuous inputs
        hgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")

        # Feed input
        hgf.input_data(input_data=timeserie)

        # Plot
        hgf.plot_trajectories()

    Visualization of nodes' trajectories from a three-level binary HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF
        import jax.numpy as jnp

        # Read binary input
        timeserie = load_data("binary")

        three_levels_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mu={"1": .0, "2": .5, "3": 0.},
            initial_pi={"1": .0, "2": 1e4, "3": 1e1},
            omega={"1": None, "2": -6.0, "3": -2.0},
            rho={"1": None, "2": 0.0, "3": 0.0},
            kappas={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            pihat = jnp.inf,
        )

        # Feed input
        three_levels_hgf = three_levels_hgf.input_data(timeserie)

        # Plot
        three_levels_hgf.plot_trajectories()

    """
    trajectories_df = hgf.to_pandas()
    n_nodes = trajectories_df.columns.str.contains("_muhat").sum()
    palette = itertools.cycle(sns.color_palette())

    if axs is None:
        _, axs = plt.subplots(nrows=n_nodes + 1, figsize=figsize, sharex=True)

    # input node
    # ----------
    if hgf.model_type == "continuous":
        axs[n_nodes - 1].scatter(
            trajectories_df.time,
            trajectories_df.observation_input_0,
            s=3,
            label="Input",
            color="#2a2a2a",
            zorder=10,
            alpha=0.5,
        )
    elif hgf.model_type == "binary":
        axs[n_nodes - 1].scatter(
            x=trajectories_df.time,
            y=trajectories_df.observation_input_0,
            label="Input",
            color="#4c72b0",
            alpha=0.4,
            edgecolors="k",
            zorder=10,
        )

    # loop over the node idexes
    # -------------------------
    for i in range(1, n_nodes + 1):
        # use different colors for each nodes
        color = next(palette)

        # which ax instance to use
        ax_i = n_nodes - i

        # extract the sufficient statistics from the data frame
        mu = trajectories_df[f"x_{i}_muhat"]
        pi = trajectories_df[f"x_{i}_pihat"]

        # plotting mean
        axs[ax_i].plot(
            trajectories_df.time,
            mu,
            label=r"$\hat{\mu}$",
            color=color,
            linewidth=0.5,
            zorder=2,
        )

        # plotting standard deviation
        if ci is True:
            # if this is the first level of a binary model do not show CI
            if not (hgf.model_type == "binary") & (i == 1):
                sd = np.sqrt(1 / pi)
                axs[ax_i].fill_between(
                    x=trajectories_df.time,
                    y1=trajectories_df[f"x_{i}_muhat"] - sd,
                    y2=trajectories_df[f"x_{i}_muhat"] + sd,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )

        # plotting surprise
        if surprise:
            surprise_ax = axs[ax_i].twinx()
            surprise_ax.plot(
                trajectories_df.time,
                trajectories_df[f"x_{i}_surprise"],
                color="#2a2a2a",
                linewidth=0.5,
                linestyle="--",
                zorder=-1,
                label="Surprise",
            )
            surprise_ax.fill_between(
                x=trajectories_df.time,
                y1=trajectories_df[f"x_{i}_surprise"],
                y2=trajectories_df[f"x_{i}_surprise"].min(),
                color="#7f7f7f",
                alpha=0.1,
                zorder=-1,
            )
            sp = trajectories_df[f"x_{i}_surprise"].sum()
            surprise_ax.set_title(
                f"Node {i} - Surprise: {sp:.2f}",
                loc="left",
            )
            surprise_ax.set_ylabel("Surprise")
        axs[ax_i].legend()
        axs[ax_i].set_ylabel(rf"$\mu_{i}$")

    # global surprise
    # ---------------
    surprise_ax = axs[n_nodes].twinx()
    surprise_ax.fill_between(
        x=trajectories_df.time,
        y1=trajectories_df.surprise,
        y2=trajectories_df.surprise.min(),
        label="Surprise",
        color="#7f7f7f",
        alpha=0.2,
    )
    surprise_ax.plot(
        trajectories_df.time,
        trajectories_df.surprise,
        color="#2a2a2a",
        linewidth=0.5,
        linestyle="--",
        zorder=-1,
        label="Surprise",
    )
    sp = trajectories_df.surprise.sum()
    surprise_ax.set_title(f"Total surprise: {sp:.2f}", loc="left")
    surprise_ax.set_ylabel("Surprise")
    surprise_ax.set_xlabel("Time")

    return axs


def plot_correlations(hgf: "HGF") -> Axes:
    """Plot the heatmap correlation of the sufficient statistics trajectories.

    Parameters
    ----------
    hgf :
        Instance of the HGF model.

    Returns
    -------
    axs :
        The Matplotlib ax instance containing the heatmap of parameters trajectories
        correlation.

    """
    trajectories_df = hgf.to_pandas()
    trajectories_df = pd.concat(
        [
            trajectories_df[["time", "observation_input_0", "surprise"]],
            trajectories_df.filter(regex="hat"),
        ],
        axis=1,
    )

    # rename columns with LateX expressions
    trajectories_df.columns = [
        r"$\hat{\mu}_" + f"{c[5]}$" if "muhat" in c else c
        for c in trajectories_df.columns
    ]
    trajectories_df.columns = [
        r"$\hat{\pi}_" + f"{c[5]}$" if "pihat" in c else c
        for c in trajectories_df.columns
    ]

    correlation_mat = trajectories_df.corr()
    ax = sns.heatmap(
        correlation_mat,
        annot=True,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        linewidths=2,
        square=True,
    )
    ax.set_title("Correlations between the model trajectories")

    return ax


def plot_network(hgf: "HGF") -> "Source":
    """Visualization of node network using GraphViz.

    Parameters
    ----------
    hgf :
        Instance of the HGF model containing a node structure.

    Notes
    -----
    This function requires [Graphviz](https://github.com/xflr6/graphviz) to be
    installed to work correctly.

    """
    try:
        import graphviz
    except ImportError:
        print(
            (
                "Graphviz is required to plot the nodes structure. "
                "See https://pypi.org/project/graphviz/"
            )
        )

    graphviz_structure = graphviz.Digraph("hgf-nodes", comment="Nodes structure")

    graphviz_structure.attr("node", shape="circle")

    # set input nodes
    for idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        graphviz_structure.node(
            f"x_{idx}",
            label=f"{kind.capitalize()[0]}I - {idx}",
            style="filled",
            shape="octagon",
        )

    # create the rest of nodes
    for i in range(len(hgf.node_structure)):
        # only if node is not an input node
        if i not in hgf.input_nodes_idx.idx:
            graphviz_structure.node(f"x_{i}", label=str(i), shape="circle")

    # connect value parents
    for i, index in enumerate(hgf.node_structure):
        value_parents = index.value_parents

        if value_parents is not None:
            for value_parents_idx in value_parents:
                graphviz_structure.edge(
                    f"x_{value_parents_idx}",
                    f"x_{i}",
                )

    # connect volatility parents
    for i, index in enumerate(hgf.node_structure):
        volatility_parents = index.volatility_parents

        if volatility_parents is not None:
            for volatility_parents_idx in volatility_parents:
                graphviz_structure.edge(
                    f"x_{volatility_parents_idx}",
                    f"x_{i}",
                    color="gray",
                    style="dashed",
                    arrowhead="dot",
                )

    # unflat the structure to better handle large/uneven networks
    graphviz_structure = graphviz_structure.unflatten(stagger=3)

    return graphviz_structure
