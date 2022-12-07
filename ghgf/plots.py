# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_trajectories(hgf, ci: bool = True, figsize: Tuple[int, int] = (18, 9)) -> Axes:
    """Plot perceptual HGF parameters time series.

    Parameters
    ----------
    hgf : py:class`ghgf.model.HGF`
        Instance of the HGF model.
    ci : bool
        Show the uncertainty aroud the values estimates (standard deviation).
    figsize : tuple
        The width and height of the figure.

    Returns
    -------
    axs : :class:`matplotlib.axes.Axes`
        The Matplotlib ax instance containing the parameters trajectories.

    """

    nrows = hgf.n_levels + 1
    time = hgf.hgf_results["final"][1]["time"]
    node, results = hgf.hgf_results["final"]

    _, axs = plt.subplots(nrows=nrows, figsize=figsize, sharex=True)

    # Level 3
    #########
    if hgf.n_levels == 3:
        if hgf.model_type == "continuous":
            mu = node[1][0][2][0][2][0][0]["mu"]
        elif hgf.model_type == "binary":
            mu = node[1][0][1][0][2][0][0]["mu"]
        axs[0].plot(time, mu, label=r"$\mu_3$", color="#55a868")
        if ci is True:
            if hgf.model_type == "continuous":
                pi = node[1][0][2][0][2][0][0]["pi"]
            elif hgf.model_type == "binary":
                pi = node[1][0][1][0][2][0][0]["pi"]
            sd = np.sqrt(1 / pi)
            axs[0].fill_between(
                x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#55a868"
            )
        axs[0].legend()

    # Level 2
    #########
    if hgf.model_type == "continuous":
        mu = node[1][0][2][0][0]["mu"]
    elif hgf.model_type == "binary":
        mu = node[1][0][1][0][0]["mu"]

    axs[hgf.n_levels - 2].plot(time, mu, label=r"$\mu_2$", color="#c44e52")
    if ci is True:
        if hgf.model_type == "continuous":
            pi = node[1][0][2][0][0]["pi"]
        elif hgf.model_type == "binary":
            pi = node[1][0][1][0][0]["pi"]
        sd = np.sqrt(1 / pi)
        axs[hgf.n_levels - 2].fill_between(
            x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#c44e52"
        )
    axs[hgf.n_levels - 2].legend()

    # Level 1
    #########
    if hgf.model_type == "continuous":
        mu = node[1][0][0]["mu"]
        axs[hgf.n_levels - 1].plot(time, mu, label=r"$\mu_1$", color="#55a868")
        if ci is True:
            pi = node[1][0][0]["pi"]
            sd = np.sqrt(1 / pi)
            axs[hgf.n_levels - 1].fill_between(
                x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#55a868"
            )

            axs[hgf.n_levels - 1].plot(
                time,
                hgf.hgf_results["final"][1]["value"],
                linewidth=2,
                linestyle="dotted",
                label="Input",
                color="#4c72b0",
            )
    elif hgf.model_type == "binary":
        mu = node[1][0][0]["muhat"]
        axs[hgf.n_levels - 1].plot(time, mu, label=r"$\mu_1$", color="#55a868")
        
        axs[hgf.n_levels - 1].scatter(
            time,
            hgf.hgf_results["final"][1]["value"],
            label="Input",
            color="#4c72b0",
        )
    axs[hgf.n_levels - 1].legend()

    # Surprise
    ##########
    axs[hgf.n_levels].plot(time, results["surprise"], label="Surprise", color="#7f7f7f")
    axs[hgf.n_levels].set_xlabel("Time")
    axs[hgf.n_levels].legend()

    return axs


def plot_correlations(hgf) -> Axes:
    """Plot the heatmap correlation of beliefs trajectories.

    Parameters
    ----------
    hgf : py:class`ghgf.model.HGF`
        Instance of the HGF model.

    Returns
    -------
    axs : :class:`matplotlib.axes.Axes`
        The Matplotlib ax instance containing the heatmap of parameters trajectories
        correlation.

    """

    node, results = hgf.hgf_results["final"]

    results["surprise"]

    # Level 1
    mu_1 = node[1][0][0]["mu"]
    pi_1 = node[1][0][0]["pi"]
    pihat_1 = node[1][0][0]["pihat"]
    muhat_1 = node[1][0][0]["muhat"]
    nu_1 = node[1][0][0]["nu"]

    # Level 2
    mu_2 = node[1][0][2][0][0]["mu"] if hgf.model_type == "continuous" else node[1][0][1][0][0]["mu"]
    pi_2 = node[1][0][2][0][0]["pi"] if hgf.model_type == "continuous" else node[1][0][1][0][0]["pi"]
    pihat_2 = node[1][0][2][0][0]["pihat"] if hgf.model_type == "continuous" else node[1][0][1][0][0]["pihat"]
    muhat_2 = node[1][0][2][0][0]["muhat"] if hgf.model_type == "continuous" else node[1][0][1][0][0]["muhat"]

    # Time series of the model beliefs
    df = pd.DataFrame(
        {
            r"$\nu_{1}$": nu_1,
            r"$\mu_{1}$": mu_1,
            r"$\pi_{1}$": pi_1,
            r"$\mu_{2}$": mu_2,
            r"$\pi_{2}$": pi_2,
            r"$\hat{mu}_{1}$": muhat_1,
            r"$\hat{pi}_{1}$": pihat_1,
            r"$\hat{mu}_{2}$": muhat_2,
            r"$\hat{pi}_{2}$": pihat_2,
        }
    )
    
    if hgf.n_levels == 3:
        if hgf.model_type == "continuous":
            df[r"$\mu_{3}$"] = node[1][0][2][0][2][0][0]["mu"]
            df[r"$\pi_{3}$"] = node[1][0][2][0][2][0][0]["pi"]
            df[r"$\hat{mu}_{3}$"] = node[1][0][2][0][2][0][0]["muhat"]
            df[r"$\hat{pi}_{3}$"] = node[1][0][2][0][2][0][0]["pihat"]
        elif hgf.model_type == "binary":
            df[r"$\mu_{3}$"] = node[1][0][1][0][2][0][0]["mu"]
            df[r"$\pi_{3}$"] = node[1][0][1][0][2][0][0]["pi"]
            df[r"$\hat{mu}_{3}$"] = node[1][0][1][0][2][0][0]["muhat"]
            df[r"$\hat{pi}_{3}$"] = node[1][0][1][0][2][0][0]["pihat"]

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
