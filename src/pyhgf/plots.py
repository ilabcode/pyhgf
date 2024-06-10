# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import itertools
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from pyhgf.typing import input_types

if TYPE_CHECKING:
    from graphviz.sources import Source

    from pyhgf.model import Network


def plot_trajectories(
    network: "Network",
    ci: bool = True,
    show_surprise: bool = True,
    show_current_state: bool = False,
    show_observations: bool = False,
    figsize: Tuple[int, int] = (18, 9),
    axs: Optional[Union[List, Axes]] = None,
) -> Axes:
    r"""Plot the trajectories of the nodes' sufficient statistics and surprise.

    This function will plot the expected mean and precision (converted into standard
    deviation) and the surprise at each level of the node structure.

    Parameters
    ----------
    network :
        An instance of the main Network class.
    ci :
        Show the uncertainty around the values estimates (standard deviation).
    show_surprise :
        If `True` plot each node's surprise together with sufficient statistics.
        If `False`, only the input node's surprise is depicted.
    show_current_state :
        If `True`, plot the current mean and precision on the top of expected mean and
        precision. Defaults to `False`.
    show_observations :
        If `True`, show the observations received from the child node(s). In the
        situation of value coupled nodes, plot the mean of the child node(s). This
        feature is not supported in the situation of volatility coupling. Defaults to
        `False`.
    figsize :
        The width and height of the figure. Defaults to `(18, 9)` for a two-level model,
        or to `(18, 12)` for a three-level model.
    axs :
        A list of Matplotlib axes instances where to draw the trajectories. This should
        correspond to the number of nodes in the structure. The default is `None`
        (create a new figure).

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
            initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
            tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
            tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")

        # Feed input
        hgf.input_data(input_data=timeserie)

        # Plot
        hgf.plot_trajectories();

    Visualization of nodes' trajectories from a three-level binary HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF
        import jax.numpy as jnp

        # Read binary input
        u, _ = load_data("binary")

        three_levels_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mean={"1": .0, "2": .5, "3": 0.},
            initial_precision={"1": .0, "2": 1e4, "3": 1e1},
            tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
            tonic_drift={"1": None, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            binary_precision = jnp.inf,
        )

        # Feed input
        three_levels_hgf = three_levels_hgf.input_data(u)

        # Plot
        three_levels_hgf.plot_trajectories();

    """
    trajectories_df = network.to_pandas()
    n_nodes = len(network.edges)
    palette = itertools.cycle(sns.color_palette())

    if axs is None:
        _, axs = plt.subplots(nrows=n_nodes + 1, figsize=figsize, sharex=True)

    # plot the input node(s)
    # ----------------------
    for i, input_idx in enumerate(network.inputs.idx):
        plot_nodes(
            network=network,
            node_idxs=input_idx,
            axs=axs[-2 - i],
            show_surprise=show_surprise,
            show_current_state=show_current_state,
            ci=ci,
            show_observations=show_observations,
        )

    # plot continuous and binary nodes
    # --------------------------------
    ax_i = n_nodes - len(network.inputs.idx) - 1
    for node_idx in range(0, n_nodes):
        if node_idx not in network.inputs.idx:
            # use different colors for each node
            color = next(palette)
            plot_nodes(
                network=network,
                node_idxs=node_idx,
                axs=axs[ax_i],
                color=color,
                show_surprise=show_surprise,
                show_current_state=show_current_state,
                ci=ci,
                show_observations=show_observations,
            )
            ax_i -= 1

    # plot the global surprise of the model
    # -------------------------------------
    surprise_ax = axs[n_nodes].twinx()
    surprise_ax.fill_between(
        x=trajectories_df.time,
        y1=trajectories_df.total_surprise,
        y2=trajectories_df.total_surprise.min(),
        label="Surprise",
        color="#7f7f7f",
        alpha=0.2,
    )
    surprise_ax.plot(
        trajectories_df.time,
        trajectories_df.total_surprise,
        color="#2a2a2a",
        linewidth=0.5,
        zorder=-1,
        label="Surprise",
    )
    sp = trajectories_df.total_surprise.sum()
    surprise_ax.set_title(f"Total surprise: {sp:.2f}", loc="right")
    surprise_ax.set_ylabel("Surprise")
    surprise_ax.set_xlabel("Time")

    return axs


def plot_correlations(network: "Network") -> Axes:
    """Plot the heatmap correlation of the sufficient statistics trajectories.

    Parameters
    ----------
    network :
        An instance of the HGF model.

    Returns
    -------
    axs :
        The Matplotlib axe instance containing the heatmap of parameters trajectories
        correlation.

    """
    trajectories_df = network.to_pandas()
    trajectories_df = pd.concat(
        [
            trajectories_df[
                ["time", "observation_input_0", "observation_input_0_surprise"]
            ],
            trajectories_df.filter(regex="expected"),
        ],
        axis=1,
    )

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


def plot_network(network: "Network") -> "Source":
    """Visualization of node network using GraphViz.

    Parameters
    ----------
    network :
        An instance of main Network class.

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
    for idx, kind in zip(network.inputs.idx, network.inputs.kind):
        if kind == 0:
            label, shape = f"Co-{idx}", "oval"
        elif kind == 1:
            label, shape = f"Bi-{idx}", "box"
        elif kind == 2:
            label, shape = f"Ca-{idx}", "diamond"
        elif kind == 3:
            label, shape = f"Ge-{idx}", "point"
        graphviz_structure.node(
            f"x_{idx}",
            label=label,
            style="filled",
            shape=shape,
        )

    # create the rest of nodes
    for i in range(len(network.edges)):
        # only if node is not an input node
        if i not in network.inputs.idx:
            graphviz_structure.node(f"x_{i}", label=str(i), shape="circle")

    # connect value parents
    for i, index in enumerate(network.edges):
        value_parents = index.value_parents

        if value_parents is not None:
            for value_parents_idx in value_parents:
                graphviz_structure.edge(
                    f"x_{value_parents_idx}",
                    f"x_{i}",
                )

    # connect volatility parents
    for i, index in enumerate(network.edges):
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


def plot_nodes(
    network: "Network",
    node_idxs: Union[int, List[int]],
    ci: bool = True,
    show_surprise: bool = True,
    show_observations: bool = False,
    show_current_state: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    color: Optional[Union[Tuple, str]] = None,
    axs: Optional[Union[List, Axes]] = None,
):
    r"""Plot the trajectory of expected sufficient statistics of a set of nodes.

    This function will plot the expected mean and precision (converted into standard
    deviation) before observation, and the Gaussian surprise after observation. If
    `children_inputs` is `True`, will also plot the children input (mean for value
    coupling and precision for volatility coupling).

    Parameters
    ----------
    network :
        An instance of main Network class.
    node_idxs :
        The index(es) of the probabilistic node(s) that should be plotted. If multiple
        indexes are provided, multiple rows will be appended to the figure, one for
        each node.
    ci :
        Whether to show the uncertainty around the values estimates (using the standard
        deviation :math:`\sqrt{\frac{1}{\hat{\pi}}}`).
    show_surprise :
        If `True` the surprise, defined as the negative log probability of the
        observation given the expectation, is plotted in the backgroud of the figure
        as grey shadded area.
    show_observations :
        If `True`, show the observations received from the child node(s). In the
        situation of value coupled nodes, plot the expected mean of the child
        node(s). This feature is not supported in the situation of volatility coupling.
        Defaults to `False`.
    show_current_state :
        If `True`, plot the current states (mean and precision) on the top of
        expected states (mean and precision). Defaults to `False`.
    figsize :
        The width and height of the figure. Defaults to `(18, 9)` for a two-level model,
        or to `(18, 12)` for a three-level model.
    color :
        The color of the main curve showing the beliefs trajectory.
    axs :
        A list of Matplotlib axes instances where to draw the trajectories. This should
        correspond to the number of nodes in the structure. The default is `None`
        (create a new figure).

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
            initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
            tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
            tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")

        # Feed input
        hgf.input_data(input_data=timeserie)

        # Plot
        hgf.plot_nodes(node_idxs=1)

    """
    if not isinstance(node_idxs, list):
        node_idxs = [node_idxs]
    trajectories_df = network.to_pandas()

    if axs is None:
        _, axs = plt.subplots(nrows=len(node_idxs), figsize=figsize, sharex=True)

    if isinstance(node_idxs, int) | len(node_idxs) == 1:
        axs = [axs]

    for i, node_idx in enumerate(node_idxs):
        # plotting an input node
        # ----------------------
        if node_idx in network.inputs.idx:
            input_type = network.inputs.kind[network.inputs.idx.index(node_idx)]
            if input_type == 0:
                axs[i].scatter(
                    x=trajectories_df.time,
                    y=trajectories_df[f"observation_input_{node_idx}"],
                    s=3,
                    label="Input",
                    color="#2a2a2a",
                    zorder=10,
                    alpha=0.5,
                )
            elif input_type == 1:
                axs[i].scatter(
                    x=trajectories_df.time,
                    y=trajectories_df[f"observation_input_{node_idx}"],
                    label="Input",
                    color="#4c72b0",
                    alpha=0.2,
                    edgecolors="k",
                    zorder=10,
                )

            # plotting standard deviation
            if ci is True:
                precision = trajectories_df[
                    f"observation_input_{node_idx}_expected_precision"
                ]
                sd = np.sqrt(1 / precision)
                y1 = trajectories_df[f"observation_input_{node_idx}"] - sd
                y2 = trajectories_df[f"observation_input_{node_idx}"] + sd

                axs[i].fill_between(
                    x=trajectories_df["time"],
                    y1=y1,
                    y2=y2,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )
            input_label = list(input_types.keys())[
                input_type
            ].capitalize()  # type: ignore

            axs[i].set_title(f"{input_label} Input Node {node_idx}", loc="left")
            axs[i].legend()
        else:
            # plotting state nodes
            # --------------------

            axs[i].set_title(
                f"State Node {node_idx}",
                loc="left",
            )

            # show the expected states
            # ------------------------

            # extract sufficient statistics from the data frame
            mean = trajectories_df[f"x_{node_idx}_expected_mean"]
            precision = trajectories_df[f"x_{node_idx}_expected_precision"]

            # plotting mean
            axs[i].plot(
                trajectories_df.time,
                mean,
                label="Expected mean",
                color=color,
                linewidth=1,
                zorder=2,
            )
            axs[i].set_ylabel(rf"$\mu_{{{node_idx}}}$")

            # plotting standard deviation
            if ci is True:
                sd = np.sqrt(1 / precision)
                y1 = trajectories_df[f"x_{node_idx}_expected_mean"] - sd
                y2 = trajectories_df[f"x_{node_idx}_expected_mean"] + sd

                # if this is the value parent of an input node
                # the CI should be treated diffeently
                if network.edges[node_idx].value_children is not None:
                    if np.any(
                        [
                            (
                                i  # type : ignore
                                in network.edges[  # type: ignore
                                    node_idx  # type: ignore
                                ].value_children  # type: ignore
                            )
                            and kind == 1
                            for i, kind in enumerate(network.inputs.kind)
                        ]
                    ):
                        # get parent node
                        parent_idx = network.edges[  # type: ignore
                            node_idx  # type: ignore
                        ].value_parents[
                            0
                        ]  # type: ignore

                        # compute  mu +/- sd at time t-1
                        # and use the sigmoid transform before plotting
                        mean_parent = trajectories_df[f"x_{parent_idx}_expected_mean"]
                        precision_parent = trajectories_df[
                            f"x_{parent_idx}_expected_precision"
                        ]
                        sd = np.sqrt(1 / precision_parent)
                        y1 = 1 / (1 + np.exp(-mean_parent + sd))
                        y2 = 1 / (1 + np.exp(-mean_parent - sd))

                axs[i].fill_between(
                    x=trajectories_df.time,
                    y1=y1,
                    y2=y2,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )

            axs[i].legend(loc="upper left")

            # show the current states
            # -----------------------
            if show_current_state:
                # extract sufficient statistics from the data frame
                mean = trajectories_df[f"x_{node_idx}_mean"]
                precision = trajectories_df[f"x_{node_idx}_precision"]

                # plotting mean
                axs[i].plot(
                    trajectories_df.time,
                    mean,
                    label="Mean",
                    color="gray",
                    linewidth=0.5,
                    zorder=2,
                    linestyle="--",
                )

                # plotting standard deviation
                if ci is True:
                    sd = np.sqrt(1 / precision)
                    axs[i].fill_between(
                        x=trajectories_df.time,
                        y1=trajectories_df[f"x_{node_idx}_mean"] - sd,
                        y2=trajectories_df[f"x_{node_idx}_mean"] + sd,
                        alpha=0.1,
                        color=color,
                        zorder=2,
                    )
                axs[i].legend(loc="lower left")

            # plot the inputs from child nodes
            # --------------------------------
            if show_observations:
                # value coupling
                if network.edges[node_idx].value_children is not None:
                    input_colors = plt.cm.cividis(
                        np.linspace(
                            0,
                            1,
                            len(network.edges[node_idx].value_children),  # type: ignore
                        )
                    )

                    for ii, child_idx in enumerate(
                        network.edges[node_idx].value_children  # type: ignore
                    ):
                        if child_idx not in network.inputs.idx:
                            axs[i].scatter(
                                trajectories_df.time,
                                trajectories_df[f"x_{child_idx}_mean"],
                                s=3,
                                label=f"Value child node - {ii}",
                                alpha=0.5,
                                color=input_colors[ii],
                                edgecolors="grey",
                            )
                            axs[i].plot(
                                trajectories_df.time,
                                trajectories_df[f"x_{child_idx}_mean"],
                                linewidth=0.5,
                                linestyle="--",
                                alpha=0.5,
                                color=input_colors[ii],
                            )
                        else:
                            child_idx = np.where(
                                np.array(network.inputs.idx) == child_idx
                            )[0][0]
                            axs[i].scatter(
                                trajectories_df.time,
                                trajectories_df[f"observation_input_{child_idx}"],
                                s=3,
                                label=f"Value child node - {ii}",
                                alpha=0.3,
                                color=input_colors[ii],
                                edgecolors="grey",
                            )
                            axs[i].plot(
                                trajectories_df.time,
                                trajectories_df[f"observation_input_{child_idx}"],
                                linewidth=0.5,
                                linestyle="--",
                                alpha=0.3,
                                color=input_colors[ii],
                            )
                    axs[i].legend(loc="lower right")

        # plotting surprise
        # -----------------
        if show_surprise:
            if node_idx in network.inputs.idx:
                node_surprise = trajectories_df[
                    f"observation_input_{node_idx}_surprise"
                ].to_numpy()
            else:
                node_surprise = trajectories_df[f"x_{node_idx}_surprise"].to_numpy()

            if not np.isnan(node_surprise).all():
                surprise_ax = axs[i].twinx()

                sp = node_surprise.sum()
                surprise_ax.set_title(
                    f"Surprise: {sp:.2f}",
                    loc="right",
                )
                surprise_ax.fill_between(
                    x=trajectories_df.time,
                    y1=node_surprise,
                    y2=node_surprise.min(),
                    where=network.node_trajectories[node_idx]["observed"],
                    color="#7f7f7f",
                    alpha=0.1,
                    zorder=-1,
                )

                # hide surprise if the input was not observed
                node_surprise[network.node_trajectories[node_idx]["observed"] == 0] = (
                    np.nan
                )
                surprise_ax.plot(
                    trajectories_df.time,
                    node_surprise,
                    color="#2a2a2a",
                    linewidth=0.5,
                    zorder=-1,
                    label="Surprise",
                )
                surprise_ax.set_ylabel("Surprise")
                surprise_ax.legend(loc="upper right")
    return axs
