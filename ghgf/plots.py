# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import itertools
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ghgf.model import HGF


def plot_trajectories(
    hgf: "HGF",
    ci: bool = True,
    surprise: bool = True,
    figsize: Tuple[int, int] = (18, 9),
    axs: Optional[Union[List, Axes]] = None,
) -> Axes:
    r"""Plot the trajectories of the nodes' sufficient statistics and surprise.

    This function will plot :math:`\\mu`, :math:`\\pi` (converted into standard
    deviation) and the surprise at each level of the node structure.

    Parameters
    ----------
    hgf : class:`ghgf.model.HGF`
        Instance of the HGF model.
    ci : bool
        Show the uncertainty aroud the values estimates (standard deviation).
    surprise : bool
        If `True` plot each node's surprise together witt the sufficient statistics.
        If `False`, only the input node's surprise is depicted.
    figsize : tuple
        The width and height of the figure. Defaults to `(18, 9)` for a 2-levels model,
        or to `(18, 12)` for a 3-levels model.
    axs : :class:`matplotlib.axes.Axes` | list | None
        A list of Matplotlib axes instance where to draw the trajectories. This should
        correspond to the number of nodes in the structure. Default is `None` (create a
        new figure).

    Returns
    -------
    axs : :class:`matplotlib.axes.Axes`
        The Matplotlib axes instances where to plot the trajectories.

    Notes
    -----
    The node structure can be a standard continuous or binary HGF, of a custom
    structure. The nodes are traversed and plotted from bottom to top using
    :py:func:`ghgf.structure.structure_as_dict`.

    Examples
    --------
    Visualization of nodes' trajectories from a three-level continuous HGF model.

    .. plot::

        from ghgf import load_data
        from ghgf.model import HGF

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

        from ghgf import load_data
        from ghgf.model import HGF

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
    # get the sufficient statistics and surprise for each node in the structure
    # we get a ValueError if the model cannot fit using the parameters
    try:
        trajectories_df = hgf.to_pandas()
    except ValueError:
        return
    n_nodes = trajectories_df.columns.str.contains("_muhat").sum()
    palette = itertools.cycle(sns.color_palette())

    if axs is None:
        _, axs = plt.subplots(nrows=n_nodes + 1, figsize=figsize, sharex=True)

    # input node
    # ----------
    if hgf.model_type == "continuous":
        axs[n_nodes - 1].scatter(
            trajectories_df.time,
            trajectories_df.observation,
            s=3,
            label="Input",
            color="#2a2a2a",
            zorder=10,
            alpha=0.5,
        )
    elif hgf.model_type == "binary":
        axs[n_nodes - 1].scatter(
            x=trajectories_df.time,
            y=trajectories_df.observation,
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
        mu = trajectories_df[f"node_{i}_muhat"]
        pi = trajectories_df[f"node_{i}_pihat"]

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
                    y1=trajectories_df[f"node_{i}_muhat"] - sd,
                    y2=trajectories_df[f"node_{i}_muhat"] + sd,
                    alpha=0.4,
                    color=color,
                    zorder=2,
                )

        # plotting surprise
        if surprise:
            surprise_ax = axs[ax_i].twinx()
            surprise_ax.plot(
                trajectories_df.time,
                trajectories_df[f"node_{i}_surprise"],
                color="#2a2a2a",
                linewidth=0.5,
                linestyle="--",
                zorder=-1,
                label="Surprise",
            )
            surprise_ax.fill_between(
                x=trajectories_df.time,
                y1=trajectories_df[f"node_{i}_surprise"],
                y2=trajectories_df[f"node_{i}_surprise"].min(),
                color="#7f7f7f",
                alpha=0.2,
                zorder=-1,
            )
            surprise_ax.set_ylabel("Surprise")
        axs[ax_i].legend()
        axs[ax_i].set_ylabel(rf"$\mu_{i}$")

    # global surprise
    # ---------------
    axs[n_nodes].fill_between(
        x=trajectories_df.time,
        y1=trajectories_df.surprise,
        y2=trajectories_df.surprise.min(),
        label="Surprise",
        color="#7f7f7f",
        alpha=0.2,
    )
    axs[n_nodes].plot(
        trajectories_df.time,
        trajectories_df.surprise,
        color="#2a2a2a",
        linewidth=0.5,
        linestyle="--",
        zorder=-1,
        label="Surprise",
    )
    axs[n_nodes].set_ylabel("Surprise")
    axs[n_nodes].set_xlabel("Time")

    return axs


def plot_correlations(hgf: "HGF") -> Axes:
    """Plot the heatmap correlation of beliefs trajectories.

    Parameters
    ----------
    hgf : :py:class`ghgf.model.HGF`
        Instance of the HGF model.

    Returns
    -------
    axs : :class:`matplotlib.axes.Axes`
        The Matplotlib ax instance containing the heatmap of parameters trajectories
        correlation.

    """
    # Level 1
    mu_1 = hgf.node_trajectories[1][0][0]["mu"]
    pi_1 = hgf.node_trajectories[1][0][0]["pi"]
    pihat_1 = hgf.node_trajectories[1][0][0]["pihat"]
    muhat_1 = hgf.node_trajectories[1][0][0]["muhat"]
    nu_1 = hgf.node_trajectories[1][0][0]["nu"]

    # Level 2
    mu_2 = (
        hgf.node_trajectories[1][0][2][0][0]["mu"]
        if hgf.model_type == "continuous"
        else hgf.node_trajectories[1][0][1][0][0]["mu"]
    )
    pi_2 = (
        hgf.node_trajectories[1][0][2][0][0]["pi"]
        if hgf.model_type == "continuous"
        else hgf.node_trajectories[1][0][1][0][0]["pi"]
    )
    pihat_2 = (
        hgf.node_trajectories[1][0][2][0][0]["pihat"]
        if hgf.model_type == "continuous"
        else hgf.node_trajectories[1][0][1][0][0]["pihat"]
    )
    muhat_2 = (
        hgf.node_trajectories[1][0][2][0][0]["muhat"]
        if hgf.model_type == "continuous"
        else hgf.node_trajectories[1][0][1][0][0]["muhat"]
    )

    # Time series of the model beliefs
    df = pd.DataFrame(
        {
            r"$\nu_{1}$": nu_1,
            r"$\mu_{1}$": mu_1,
            r"$\pi_{1}$": pi_1,
            r"$\mu_{2}$": mu_2,
            r"$\pi_{2}$": pi_2,
            r"$\hat{\mu}_{1}$": muhat_1,
            r"$\hat{\pi}_{1}$": pihat_1,
            r"$\hat{\mu}_{2}$": muhat_2,
            r"$\hat{\pi}_{2}$": pihat_2,
        }
    )

    if hgf.n_levels == 3:
        if hgf.model_type == "continuous":
            df[r"$\mu_{3}$"] = hgf.node_trajectories[1][0][2][0][2][0][0]["mu"]
            df[r"$\pi_{3}$"] = hgf.node_trajectories[1][0][2][0][2][0][0]["pi"]
            df[r"$\hat{\mu}_{3}$"] = hgf.node_trajectories[1][0][2][0][2][0][0]["muhat"]
            df[r"$\hat{\pi}_{3}$"] = hgf.node_trajectories[1][0][2][0][2][0][0]["pihat"]
        elif hgf.model_type == "binary":
            df[r"$\mu_{3}$"] = hgf.node_trajectories[1][0][1][0][2][0][0]["mu"]
            df[r"$\pi_{3}$"] = hgf.node_trajectories[1][0][1][0][2][0][0]["pi"]
            df[r"$\hat{\mu}_{3}$"] = hgf.node_trajectories[1][0][1][0][2][0][0]["muhat"]
            df[r"$\hat{\pi}_{3}$"] = hgf.node_trajectories[1][0][1][0][2][0][0]["pihat"]

    correlation_mat = df.corr()
    ax = sns.heatmap(
        correlation_mat,
        annot=True,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        linewidths=2,
        square=True,
    )
    ax.set_title("Correlation between the model beliefs")

    return ax
