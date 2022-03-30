import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def plot_trajectories(
    model, ci: bool = True, figsize: Optional[int] = None, backend="matplotlib"
) -> Figure:
    """Plot perceptual HGF parameters time series

    Parameters
    ----------
    model : py:class`ghgf.model.HGF`
        Instance of the HGF model.
    ci : bool
        Show the uncertainty aroud the values estimates (standard deviation).
    figsize : int
        The height of the figures.
    backend : str
        The plotting backend (`"matplotlib"` or `"bokeh"`).

    Returns
    -------
    axs : :class:`matplotlib.axes.Axes` | :class:`bokeh.plotting.figure.Figure`
        The figure (Matplotlib or Bokeh) containing the parameters trajectories.

    Raises
    ------
    ValueError:
        If an invalid backend is provided

    """

    if backend not in ["matplotlib", "bokeh"]:
        raise ValueError("Invalid backend provided. Should be `matplotlib` or `bokeh`")

    nrows = model.n_levels + 1
    time = model.final[1]["time"]
    node, results = model.final

    if backend == "matplotlib":

        _, axs = plt.subplots(nrows=nrows, figsize=(18, 9), sharex=True)

        # Level 3
        #########
        if model.n_levels == 3:
            mu = node[1][2][0][2][0][0]["mu"]
            axs[0].plot(time, mu, label=r"$\mu_3$", color="#55a868")
            if ci is True:
                pi = node[1][2][0][2][0][0]["pi"]
                sd = np.sqrt(1 / pi)
                axs[0].fill_between(
                    x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#55a868"
                )
            axs[0].legend()

        # Level 2
        #########
        mu = node[1][2][0][0]["mu"]
        axs[model.n_levels - 2].plot(time, mu, label=r"$\mu_2$", color="#c44e52")
        if ci is True:
            pi = node[1][2][0][0]["pi"]
            sd = np.sqrt(1 / pi)
            axs[model.n_levels - 2].fill_between(
                x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#c44e52"
            )
        axs[model.n_levels - 2].legend()

        # Level 1
        #########
        mu = node[1][0]["mu"]
        axs[model.n_levels - 1].plot(time, mu, label=r"$\mu_1$", color="#55a868")
        if ci is True:
            pi = node[1][0]["pi"]
            sd = np.sqrt(1 / pi)
            axs[model.n_levels - 1].fill_between(
                x=time, y1=mu - sd, y2=mu + sd, alpha=0.2, color="#55a868"
            )

        axs[model.n_levels - 1].plot(
            time,
            model.final[1]["value"],
            linewidth=2,
            linestyle="dotted",
            label="Input",
            color="#4c72b0",
        )
        axs[model.n_levels - 1].legend()

        # Surprise
        ##########
        axs[model.n_levels].plot(
            time, results["surprise"], label="Surprise", color="#7f7f7f"
        )
        axs[model.n_levels].set_xlabel("Time")
        axs[model.n_levels].legend()

        return axs

    elif backend == "bokeh":

        cols = ()

        if figsize is None:
            figsize = int(600 / nrows)
        else:
            figsize = int(figsize / nrows)

        x_axis_type = "auto"
        x_axis_label = "Observations"

        data = {"time": time, "input": np.array(model.final[1]["value"])}
        data["μ_1"] = np.array(node[1][0]["mu"])
        data["μ_2"] = np.array(node[1][2][0][0]["mu"])

        pi = np.array(node[1][0]["pi"])
        sd = np.sqrt(1 / pi)
        data["π_1_high"] = data["μ_1"] + sd
        data["π_1_low"] = data["μ_1"] - sd

        pi = np.array(node[1][2][0][0]["pi"])
        sd = np.sqrt(1 / pi)
        data["π_2_high"] = data["μ_2"] + sd
        data["π_2_low"] = data["μ_2"] - sd

        if model.n_levels == 3:
            data["μ_3"] = np.array(node[1][2][0][2][0][0]["mu"])
            pi = np.array(node[1][2][0][2][0][0]["pi"])
            sd = np.sqrt(1 / pi)
            data["π_3_high"] = data["μ_3"] + sd
            data["π_3_low"] = data["μ_3"] - sd

        data["surprise"] = np.array(results["surprise"])

        source = ColumnDataSource(data=data)

        # create a color iterator
        colors = itertools.cycle(palette)
        hgf = None
        for i, col in zip(reversed(range(1, nrows)), colors):

            if hgf:
                x_range = hgf.x_range
            else:
                x_range = (float(time[0]), float(time[-1]))

            hgf = figure(
                title=f"Level {i}",
                x_axis_type=x_axis_type,
                plot_height=figsize,
                x_axis_label=x_axis_label,
                y_axis_label="μ",
                output_backend="webgl",
                x_range=x_range,
            )

            if ci is True:
                hgf.varea(
                    x="time",
                    y1=f"π_{i}_low",
                    y2=f"π_{i}_high",
                    alpha=0.2,
                    legend_label="π",
                    color=col,
                    source=source,
                )

            hgf.line(
                x="time", y=f"μ_{i}", line_color=col, legend_label="μ", source=source
            )
            if i == 1:
                hgf.line(
                    x="time",
                    y="input",
                    line_color="navy",
                    line_dash="dashed",
                    legend_label="Input",
                    source=source,
                )

            cols += (hgf,)  # type: ignore

        # Plot input data
        input_timeseries = figure(
            title="Surprise",
            x_axis_type=x_axis_type,
            plot_height=figsize,
            x_axis_label=x_axis_label,
            y_axis_label="Surprise",
            output_backend="webgl",
            x_range=hgf.x_range,  # type: ignore
        )

        input_timeseries.line(
            x="time",
            y="surprise",
            line_color="#c02942",
            legend_label="Inputs",
            source=source,
        )
        cols += (input_timeseries,)  # type: ignore
        fig = column(*cols, sizing_mode="stretch_width")

        return fig
