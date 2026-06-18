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

# import contextily as cx
import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, MaxNLocator
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
from PySide6 import QtCore, QtWidgets

rcParams["savefig.dpi"] = 300


class CanvasModule(FigureCanvasQTAgg):
    """Canvas Module."""

    def __init__(self):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._delayed_resize)
        self.custom_resize = False

        self._pending_size = None

    def resizeEvent(self, event):
        """Overrides Qt's default resize event to suppress immediate rendering."""

        # QtWidgets.QWidget.resizeEvent(self, event)
        if self.custom_resize is True:
            QtWidgets.QWidget.resizeEvent(self, event)
            self._pending_size = event.size()
            self.resize_timer.start(200)
        else:
            super().resizeEvent(event)

    def _delayed_resize(self):
        """Triggers the expensive Matplotlib layout calculations and draw exactly once."""
        if self._pending_size is not None:
            w = self._pending_size.width()
            h = self._pending_size.height()

            self.figure.set_size_inches(
                w / self.figure.dpi, h / self.figure.dpi, forward=True
            )

            self.draw_idle()
            self._pending_size = None


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
    neat_steps = [1, 2, 5, 10, 15, 30, 60]
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
    ax.set_aspect("equal")
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


def set_northscale(ax, crs):
    """Sets the north arrow and the scale bar."""

    north_arrow(
        ax,
        scale=0.2,
        location="upper right",
        label={"fontsize": 10},
        aob={
            "bbox_to_anchor": (0.05, -0.05),
            "bbox_transform": ax.transAxes,
            "pad": 0.0,
        },
        shadow=False,
    )

    if not crs.is_geographic:
        scale_bar(
            ax,
            location="upper left",
            style="ticks",
            bar={"projection": crs},
            text={"fontsize": 7},
            aob={"bbox_to_anchor": (0.05, -0.05), "bbox_transform": ax.transAxes},
        )

    tmp = patches.Rectangle(
        (0, -0.25),
        0.1,
        0.1,
        transform=ax.transAxes,
        color="white",
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(tmp)

    # fig = ax.figure
    # fig.canvas.draw()
    # renderer = fig.canvas.get_renderer()
    # bbox = sb.get_window_extent(renderer=renderer)
    # tight_bbox = sb.get_tightbbox(fig.canvas.get_renderer())

    # ax_bbox = ax.get_position()
    # fig_w, fig_h = fig.get_size_inches()

    # ax_height_in = ax_bbox.height * fig_h
    # pct_height = 100 * na.scale / ax_height_in
    # clip_box = TransformedBbox(na.clipbox, ax.transAxes)
    # naclip = na.clipbox
    # sbclip = sb.clipbox
    # aa = TransformedBbox(Bbox([[0, 0], [1, 1]]), ax.transAxes)
    # pass


def main():

    sfile = r"D:\workdata\PyGMI Test Data\Vector\Rose\2329AC_lin_wgs84sutm35.shp"

    gdfs = gpd.read_file(sfile)

    crs = gdfs.crs

    fig, ax = plt.subplots()
    # plt.figure(figsize=(8, 6))

    ax = plt.gca()
    ax.set_aspect("equal")

    gdfs.plot(ax=ax)

    # fig.canvas.draw()
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

    # cx.add_basemap(
    #     ax,
    #     crs=crs,
    #     source=cx.providers.Esri.WorldImagery,
    #     attribution="",
    # )

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

    set_axes(ax, crs)
    set_northscale(ax, crs)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
