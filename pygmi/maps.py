# -----------------------------------------------------------------------------
# Name:        maps.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2026 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""A collection of functions for maps."""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, MaxNLocator
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar


def tick_formatter(x, pos):
    """
    Format thousands separator in ticks for plots.

    Parameters
    ----------
    x : float/int
        Number to be formatted.
    pos : int
        Position of tick.

    Returns
    -------
    newx : str
        Formatted coordinate.

    """
    if np.ma.is_masked(x):
        return "--"

    newx = f"{x:,.5f}".rstrip("0").rstrip(".")

    return newx


frm = FuncFormatter(tick_formatter)


def get_neat_intervals(start_dd, end_dd, num_intervals, islon=True):
    """
    Divides a decimal degree range into neat minute/degree intervals.
    """
    total_range = end_dd - start_dd
    interval_size = total_range / num_intervals

    # Round step down to the nearest degree or minute fraction
    minutes_step = interval_size * 60
    neat_steps = [1, 5, 10, 15, 30, 60]
    neat_minute = min(neat_steps, key=lambda x: abs(x - minutes_step))

    # Recalculate range with the neat step to find the adjusted end point
    neat_step_dd = neat_minute / 60

    intervals = np.arange(int(start_dd) - 1 + neat_step_dd, end_dd, neat_step_dd)
    intervalsm = np.arange(
        (int(start_dd) - 1) * 60 + neat_minute,
        (end_dd * 60),
        neat_minute,
    )
    intervalsm = intervalsm[intervals > start_dd]
    intervals = intervals[intervals > start_dd]

    degs = abs(intervalsm) // 60
    mins = abs(intervalsm) % 60

    if islon is True:
        sign = {-1: "W", 1: "E"}
    else:
        sign = {-1: "S", 1: "N"}

    txt = [
        f"{int(d):d}°{int(m):02d}'{sign[si]}"
        for d, m, si in zip(degs, mins, np.sign(intervals))
    ]

    return intervals, txt


def set_axes(ax, crs):
    """Setup the axes."""
    ax.ticklabel_format(style="plain", axis="both")
    ax.yaxis.tick_right()
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", labelrotation=90)
    ax.tick_params(axis="both", labelsize=8, labelcolor="#0070ff")

    for label in ax.yaxis.get_majorticklabels():
        label.set_horizontalalignment("left")
        label.set_verticalalignment("center")

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    transformeri = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xmind, ymind = transformer.transform(xmin, ymin)
    xmaxd, ymaxd = transformer.transform(xmax, ymax)

    deg_ticks, deg_txt = get_neat_intervals(xmind, xmaxd, 3)
    meter_ticks = [transformeri.transform(x, ymaxd)[0] for x in deg_ticks]

    secax = ax.secondary_xaxis("top")
    secax.xaxis.set_major_locator(FixedLocator(meter_ticks))
    secax.xaxis.set_major_formatter(FixedFormatter(deg_txt))
    secax.tick_params(axis="x", labelsize=9)

    deg_ticks, deg_txt = get_neat_intervals(ymind, ymaxd, 3, False)
    meter_ticks = [transformeri.transform(xmind, y)[1] for y in deg_ticks]

    secax1 = ax.secondary_yaxis("left")
    secax1.yaxis.set_major_locator(FixedLocator(meter_ticks))
    secax1.yaxis.set_major_formatter(FixedFormatter(deg_txt))
    secax1.tick_params(axis="y", labelrotation=90, labelsize=9)

    for label in secax1.yaxis.get_majorticklabels():
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")

    # north_arrow(
    #     ax,
    #     scale=0.2,
    #     location="upper left",
    #     label={"fontsize": 10},
    #     aob={"bbox_to_anchor": (0.05, -0.05), "bbox_transform": ax.transAxes},
    # )
    na = NorthArrow(
        ax,
        scale=0.2,
        location="upper left",
        label={"fontsize": 10},
        aob={"bbox_to_anchor": (0.05, -0.05), "bbox_transform": ax.transAxes},
    )

    na.set_in_layout(True)
    ax.add_artist(na)

    scale_bar(
        ax,
        location="upper right",
        style="ticks",
        bar={"projection": crs},
        text={"fontsize": 7},
        aob={"bbox_to_anchor": (0.5, -0.05), "bbox_transform": ax.transAxes, "pad": 1},
    )


def main():

    sfile = r"C:\Work\minerals\MinRan_Structures_forML.shp"

    gdfs = gpd.read_file(sfile)

    crs = gdfs.crs

    plt.figure(figsize=(8, 6))

    ax = plt.gca()
    ax.set_aspect("equal")

    gdfs.plot(ax=ax)

    set_axes(ax, crs)
    # north_arrow(
    #     ax,
    #     scale=0.2,
    #     location="upper left",
    #     label={"fontsize": 10},
    #     aob={"bbox_to_anchor": (0.05, -0.05), "bbox_transform": ax.transAxes},
    # )

    # scale_bar(
    #     ax,
    #     location="upper right",
    #     style="ticks",
    #     bar={"projection": crs},
    #     text={"fontsize": 7},
    #     aob={"bbox_to_anchor": (0.5, -0.05), "bbox_transform": ax.transAxes, "pad": 1},
    # )

    cx.add_basemap(
        ax,
        crs=crs,
        source=cx.providers.Esri.WorldImagery,
        attribution="",
    )

    ax.plot(
        [],
        [],
        color="blue",
        label="Structures",
        marker=r"_",
        linestyle="None",
        markersize=8,
    )

    ax.legend(loc="lower left", fontsize="small")

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
