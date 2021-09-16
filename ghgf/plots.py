import itertools
from typing import Optional

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure


def plot_trajectories(
    model, ci: bool = False, time=None, figsize: Optional[int] = None
) -> Figure:
    """Plot perceptual HGF parameters time series

    Parameters
    ----------
    model : py:class`ghgf.hgf.Model`
        Instance of the erceptual HGF model.
    ci : bool
        Show the uncertainty aroud the values estimates (standard deviation).
    time : np.ndarry, pd.core.indexes.datetimes.DatetimeIndex or None
        The time vector. If time is `None`, the x axis will label the number of
        observations.
    figsize : int
        The height of the figures.

    Returns
    -------
    fig : :class:`bokeh.plotting.figure.Figure`
        The figure containing the parameters trajectories.

    """

    n_subplots = model.n_levels + 1
    cols = ()

    if figsize is None:
        figsize = int(600 / n_subplots)
    else:
        figsize = int(figsize / n_subplots)

    if time is None:
        x_axis_type = "auto"
        x_axis_label = "Observations"
        time = np.array(model.xU.times[1:])
    else:
        x_axis_type = "datetime"
        x_axis_label = "Time"

    data = {"time": time, "input": np.array(model.xU.inputs[1:])}
    for i in range(model.n_levels):
        data[f"μ_{i+1}"] = getattr(model, f"x{i+1}").mus[1:]
        std = np.sqrt(1 / np.array(getattr(model, f"x{i+1}").pis[1:]))
        # Transform the precision (pi) into standard deviation (sqrt(1/pi))
        data[f"π_{i+1}_high"] = data[f"μ_{i+1}"] + std
        data[f"π_{i+1}_low"] = data[f"μ_{i+1}"] - std

    source = ColumnDataSource(data=data)

    # create a color iterator
    colors = itertools.cycle(palette)
    hgf = None
    for i, col in zip(reversed(range(1, n_subplots)), colors):

        if hgf:
            x_range = hgf.x_range
        else:
            x_range = (time[0], time[-1])

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

        hgf.line(x="time", y=f"μ_{i}", line_color=col, legend_label="μ", source=source)

        cols += (hgf,)

    # Plot input data
    input_timeseries = figure(
        title="Input time series",
        x_axis_type=x_axis_type,
        plot_height=figsize,
        x_axis_label=x_axis_label,
        y_axis_label="Input",
        output_backend="webgl",
        x_range=hgf.x_range,
    )

    input_timeseries.line(
        x="time", y="input", line_color="#c02942", legend_label="Inputs", source=source
    )
    cols += (input_timeseries,)
    fig = column(*cols, sizing_mode="stretch_width")

    return fig
