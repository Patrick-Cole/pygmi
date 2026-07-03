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
from matplotlib_map_utils.validation.scale_bar import units_standard
from PySide6 import QtCore, QtWidgets

rcParams["savefig.dpi"] = 300


class CanvasModule(FigureCanvasQTAgg):
    """Canvas Module."""

    def __init__(self, parent=None):

        if parent is None:
            # self.stdout_redirect = sys.stdout
            self.showlog = print
            # self.pbar = ProgressBarText()
            # self.process_is_active = lambda *args, **kwargs: None
        else:
            # self.stdout_redirect = EmittingStream(parent.showlog)
            self.showlog = parent.showlog
            # self.pbar = parent.pbar
            # if hasattr(parent, "process_is_active"):
            #     self.process_is_active = parent.process_is_active
            # else:
            #     self.process_is_active = lambda *args, **kwargs: None

        fig = Figure(layout="tight")
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._delayed_resize)
        self.custom_resize = False

        self._pending_size = None

    def resizeEvent(self, event):
        """
        Overrides Qt's default resize event to suppress immediate rendering.

        Parameters
        ----------
        event : event
            Event variable, used to estimate size.

        Returns
        -------
        None.

        """

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

    Parameters
    ----------
    start_dd : float
        Minimum coordinate in decimal degrees.
    end_dd : float
        Maximum coordinate in decimal degrees.
    num_intervals : int
        Number of intervals
    islon : bool, optional
        Coordinates are longitudes, by default True

    Returns
    -------
    intervals : list
        Tick coordinates.
    txt : list of str
        List of coordinates in degrees and minutes.
    """
    total_range = end_dd - start_dd
    interval_size = total_range / num_intervals

    # Round step down to the nearest degree or minute fraction
    minutes_step = interval_size * 60
    neat_steps = [1, 2, 5, 10, 15, 30, 60, 120, 240, 480]
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

    if not crs.is_geographic:
        ax.tick_params(axis="both", labelsize=8, labelcolor="#0070ff")

    for label in ax.yaxis.get_majorticklabels():
        label.set_horizontalalignment("left")
        label.set_verticalalignment("center")

    if crs.is_engineering:
        return

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

    if crs.is_geographic:
        ax.xaxis.set_major_locator(FixedLocator(meter_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(deg_txt))
        ax.tick_params(axis="x", labelsize=9)

    deg_ticks, deg_txt = get_neat_intervals(ymind, ymaxd, 3, False)
    meter_ticks = [transformeri.transform(xmind, y)[1] for y in deg_ticks]

    secax1 = ax.secondary_yaxis("left")
    secax1.yaxis.set_major_locator(FixedLocator(meter_ticks))
    secax1.yaxis.set_major_formatter(FixedFormatter(deg_txt))
    secax1.tick_params(axis="y", labelrotation=90, labelsize=9)

    for label in secax1.yaxis.get_majorticklabels():
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")

    if crs.is_geographic:
        ax.yaxis.set_major_locator(FixedLocator(meter_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(deg_txt))
        ax.tick_params(axis="y", labelrotation=90, labelsize=9)


def set_northscale(ax, crs, showlog=print):
    """
    Sets the north arrow and the scale bar.

    Parameters
    ----------
    ax : Matplotlib axes
        Primary Matplotlib axes.
    crs : CRS
        rasterio crs of data
    showlog : function, optional
        _Show information using a function, by default print
    """

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

    if crs.axis_info[0].unit_name not in units_standard:
        showlog("Problem with projection unit. Try redefining the projection.")
        return

    if not crs.is_geographic:
        scale_bar(
            ax,
            location="upper left",
            style="ticks",
            bar={"projection": crs},
            text={"fontsize": 7},
            aob={"bbox_to_anchor": (0.05, -0.05), "bbox_transform": ax.transAxes},
        )


def _testfn():
    """Test function"""

    sfile = r"D:\workdata\PyGMI Test Data\Vector\Rose\2329AC_lin_wgs84sutm35.shp"
    # ifile = r"D:\workdata\PyGMI Test Data\Raster\ER Mapper\magmicrolevel.PD.ers"

    # dataset = get_raster(ifile)

    gdfs = gpd.read_file(sfile)

    gdfs = gdfs.to_crs(4326)

    crs = gdfs.crs

    ax = plt.gca()

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
    #     attribution=False,
    # )
    # atxt = cx.providers.Esri.WorldImagery.attribution
    # cx.add_attribution(ax, atxt, font_size=6)

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
    _testfn()
