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

import os
from collections.abc import Callable
from itertools import pairwise
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from matplotlib import patches, rcParams
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LinearLocator, MaxNLocator
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
from matplotlib_map_utils.validation.scale_bar import units_standard
from numpy.typing import NDArray
from pyproj.crs import CRS
from PySide6 import QtCore, QtGui, QtWidgets

from pygmi.misc import BasicModule

rcParams["savefig.dpi"] = 300


class CustomToolbar(NavigationToolbar2QT):
    """Custom toolbar."""

    # Control which standard buttons appear and their order.
    # Each name must match an entry in NavigationToolbar2QT.toolitems.
    wanted: ClassVar = [
        "Home",
        "Back",
        "Forward",
        "Pan",
        "Zoom",
        "Save",
    ]

    toolitems: ClassVar = [
        item
        for name in wanted
        for item in NavigationToolbar2QT.toolitems
        if item[0] == name
    ]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.showlog = parent.showlog
        self.pbar = parent.pbar

        # Icons only, without text labels.
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.dlg = MapSettings(self)

    def add_map_btn(self, update: Callable, crs: CRS):
        """
        Add map button.

        Parameters
        ----------
        update
            Function which will update the map.
        crs
            PyProj crs of data
        """
        ipth = os.path.dirname(__file__) + r"/images/"

        self.custom_action = QtGui.QAction(
            QtGui.QIcon(ipth + "map.png"),
            "Change map parameters",
            self,
        )
        self.custom_action.setStatusTip("Change map parameters")
        self.custom_action.triggered.connect(self.change_map)

        existing_actions = self.actions()

        if existing_actions:
            # Matplotlib adds a stretching spacer widget or a separator near the end.
            # Inserting immediately after the last standard action element:
            last_standard_action = existing_actions[-1]
            self.insertAction(last_standard_action, self.custom_action)
        else:
            # Fallback if the actions list hasn't populated yet
            self.addAction(self.custom_action)

        self.cupdate = update
        self.crs = crs

    def change_map(self):
        """Refresh plot."""
        self.dlg.crs = self.crs
        success = self.dlg.settings()

        if not success:
            return

        self.canvas.show_north_arrow = self.dlg.cb_northarrow.isChecked()
        self.canvas.show_scalebar = self.dlg.cb_scalebar.isChecked()
        self.canvas.show_degrees = self.dlg.cb_degrees.isChecked()
        self.canvas.show_metres = self.dlg.cb_metres.isChecked()

        self.canvas.xticks = 10
        self.canvas.yticks = 10
        if self.dlg.cb_xticks.isChecked():
            self.canvas.xticks = self.dlg.sb_xticks.value()
        if self.dlg.cb_yticks.isChecked():
            self.canvas.yticks = self.dlg.sb_yticks.value()

        self.cupdate()
        self.canvas.figure.gca().autoscale()


class CanvasModule(FigureCanvasQTAgg):
    """Canvas Module."""

    def __init__(
        self,
        parent=None,
    ):

        if parent is None:
            self.showlog = print
        else:
            self.showlog = parent.showlog

        fig = Figure(layout="compressed")
        super().__init__(fig)

        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._delayed_resize)
        self.custom_resize = False

        self._pending_size = None
        self.show_north_arrow = False
        self.show_scalebar = False
        self.show_degrees = True
        self.show_metres = True
        self.xticks = 10
        self.yticks = 10

    def resizeEvent(
        self,
        event,
    ):
        """
        Override Qt's default resize event to suppress immediate rendering.

        Parameters
        ----------
        event
            Event variable, used to estimate size.
        """
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

    def set_map_axes(self, axes: Axes, crs: CRS):
        """
        Set the map axes and north arrow.

        Parameters
        ----------
        axes
            Primary Matplotlib axes.
        crs
            PyProj crs of data.
        """
        set_axes(
            axes, crs, self.show_degrees, self.show_metres, self.xticks, self.yticks
        )

        if self.show_north_arrow is True:
            set_north(axes, crs, self.showlog)
        if self.show_scalebar is True:
            set_scale(axes, crs, self.showlog)


class MapSettings(BasicModule):
    """
    Data Layer Stack GUI.

    This class merges datasets which have different rows and columns. It
    resamples them so that they have the same rows and columns.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.crs = CRS.from_string(
            'LOCAL_CS["Arbitrary",UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )

        self.cb_northarrow = QtWidgets.QCheckBox("Display north arrow")
        self.cb_scalebar = QtWidgets.QCheckBox("Display scale bar")
        self.cb_degrees = QtWidgets.QCheckBox("Use degrees on axis")
        self.cb_metres = QtWidgets.QCheckBox("Use metres on axis")
        self.cb_xticks = QtWidgets.QCheckBox("Maximum number of x-ticks:")
        self.cb_yticks = QtWidgets.QCheckBox("Maximum number of y-ticks:")
        self.sb_xticks = QtWidgets.QSpinBox()
        self.sb_yticks = QtWidgets.QSpinBox()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gl_main = QtWidgets.QGridLayout(self)

        self.sb_xticks.setMinimum(2)
        self.sb_xticks.setValue(3)
        self.sb_yticks.setMinimum(2)
        self.sb_yticks.setValue(3)

        # self.buttonbox.htmlfile = "raster.dm.layerstack"
        self.buttonbox.helpdocs.hide()

        self.cb_northarrow.setChecked(False)
        self.cb_scalebar.setChecked(False)
        self.cb_degrees.setChecked(True)
        self.cb_metres.setChecked(True)

        self.setWindowTitle("Map Properties")

        gl_main.addWidget(self.cb_northarrow, 1, 0, 1, 2)
        gl_main.addWidget(self.cb_scalebar, 2, 0, 1, 2)
        gl_main.addWidget(self.cb_degrees, 3, 0, 1, 2)
        gl_main.addWidget(self.cb_metres, 4, 0, 1, 2)
        gl_main.addWidget(self.cb_xticks, 5, 0, 1, 1)
        gl_main.addWidget(self.sb_xticks, 5, 1, 1, 1)
        gl_main.addWidget(self.cb_yticks, 6, 0, 1, 1)
        gl_main.addWidget(self.sb_yticks, 6, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 7, 0, 1, 2)

    def settings(
        self,
        nodialog: bool = False,
    ) -> bool:
        """
        Entry point into item.

        Parameters
        ----------
        nodialog
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if self.crs.is_geographic:
            self.cb_degrees.setChecked(True)
            self.cb_degrees.setDisabled(True)
            self.cb_metres.setChecked(False)
            self.cb_metres.setDisabled(True)
            self.cb_scalebar.setChecked(False)
            self.cb_scalebar.setDisabled(True)
        if self.crs.is_engineering:
            self.cb_degrees.setChecked(False)
            self.cb_degrees.setDisabled(True)
            self.cb_metres.setChecked(True)
            self.cb_metres.setDisabled(True)
            self.cb_scalebar.setDisabled(False)
        tmp = self.exec()

        return tmp == 1


def dd2txt(
    dd: float,
    islon: bool = True,
) -> str:
    """
    Divides a decimal degree range into neat minute/degree intervals.

    Parameters
    ----------
    start_dd
        Minimum coordinate in decimal degrees.
    end_dd
        Maximum coordinate in decimal degrees.
    num_intervals
        Number of intervals
    islon
        Coordinates are longitudes, by default True

    Returns
    -------
    intervals : ndarray
        Tick coordinates.
    txt : list of str
        List of coordinates in degrees and minutes.
    """
    dd = float(dd)
    dd *= 3600
    degs = abs(dd) // 3600
    mins = abs(dd) % 3600 // 60
    secs = abs(dd) % 60

    if islon is True:
        sign = {-1: "W", 1: "E", 0: ""}
    else:
        sign = {-1: "S", 1: "N", 0: ""}

    txt = f"{int(degs):d}°{int(mins):02d}'"
    if secs > 0:
        txt += f'{int(secs):02d}"'

    txt += f"{sign[np.sign(dd)]}"

    return txt


def discrete_colorbar(
    axes: Axes,
    csp,
    cdat: np.ma.MaskedArray | NDArray,
    lbls: list[str] | None = None,
):
    """
    Plot colour bar using discrete colours for a small range of values.

    Parameters
    ----------
    axes
        Current axes.
    csp
        Handle to Matplotlib plotting routine.
    cdat
        Array of values.
    lbls
        y tick labels, by default None
    """
    vals = np.unique(cdat)
    if np.ma.isMaskedArray(vals):
        vals = vals.compressed()
    vals = vals[~np.isnan(vals)]

    if len(vals) < 2:
        print("Too few discrete values")
        return
    # bnds = (vals - 0.5).tolist() + [vals.max() + .5]

    if hasattr(csp.norm, "boundaries"):
        bnds = csp.norm.boundaries
        ticks = np.diff(bnds) / 2 + vals
        cbar = axes.figure.colorbar(csp, ticks=ticks)
    else:
        bnds = vals.tolist() + [vals.max() + 1]
        ticks = np.diff(bnds) / 2 + vals
        cbar = axes.figure.colorbar(csp, boundaries=bnds, values=vals, ticks=ticks)

    if lbls is not None:
        cbar.ax.set_yticklabels(lbls)
    else:
        cbar.ax.set_yticklabels(vals)


def get_neat_intervals(
    start_dd: float,
    end_dd: float,
    num_intervals: int,
    islon: bool = True,
) -> tuple[NDArray, list[str]]:
    """
    Divides a decimal degree range into neat minute/degree intervals.

    Parameters
    ----------
    start_dd
        Minimum coordinate in decimal degrees.
    end_dd
        Maximum coordinate in decimal degrees.
    num_intervals
        Maximum number of intervals
    islon
        Coordinates are longitudes, by default True

    Returns
    -------
    intervals : ndarray
        Tick coordinates.
    txt : list of str
        List of coordinates in degrees and minutes.
    """
    total_range = end_dd - start_dd
    interval_size = total_range / num_intervals

    # Round step down to the nearest degree or minute fraction
    minutes_step = interval_size * 60
    neat_steps = np.array(
        [
            0.25,
            0.5,
            1,
            2,
            5,
            10,
            15,
            30,
            60,
            120,
            240,
            480,
            960,
            1920,
            3840,
            7680,
        ]
    )

    neat_minute = neat_steps[np.argmax((neat_steps - minutes_step) > 0)]
    # Recalculate range with the neat step to find the adjusted end point
    neat_step_dd = neat_minute / 60

    intervals = np.arange(int(start_dd) - 1 + neat_step_dd, end_dd, neat_step_dd)
    intervals = intervals[intervals > start_dd]

    itmp = np.rint(abs(intervals) * 3600)
    degs = itmp / 3600
    mins = itmp % 3600 / 60
    secs = itmp % 60

    if islon is True:
        sign = {-1: "W", 1: "E", 0: ""}
    else:
        sign = {-1: "S", 1: "N", 0: ""}

    txt = [
        f"{int(d):d}°{int(m):d}'{int(s):d}\"{sign[np.sign(si)]}"
        for d, m, s, si in zip(degs, mins, secs, intervals)
    ]

    txt = [i.replace("°0'0\"", "°") for i in txt]
    txt = [i.replace("'0\"", "'") for i in txt]
    return intervals, txt


def set_axes(
    ax: Axes,
    crs: CRS | None,
    showdegrees: bool = True,
    showmetres: bool = True,
    xticks: int = 10,
    yticks: int = 10,
):
    """
    Set the axes.

    Parameters
    ----------
    ax
        Primary Matplotlib axes.
    crs
        PyProj crs of data.
    showdegrees
        Show degrees on the map border.
    showmetres
        Show metres on the map border.
    xticks
        Maximum number of x-ticks, by default 10.
    yticks
        Maximum number of y-ticks, by default 10.
    """
    if crs is None:
        crs = CRS.from_string(
            'LOCAL_CS["Arbitrary",UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )
    else:
        crs = CRS.from_wkt(crs.to_wkt())

    if crs.is_engineering:
        showmetres = True
        showdegrees = False
    if crs.is_geographic:
        showmetres = False
        showdegrees = True

    if showmetres is False and showdegrees is False:
        showmetres = True

    ax.set_aspect("equal")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelrotation=90)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(frm)
    ax.yaxis.set_major_formatter(frm)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if showmetres is True:
        ax.tick_params(axis="both", labelsize=8, labelcolor="#0070ff")
        xlabels_m(ax, num=xticks)
        ylabels_m(ax, num=yticks)

    if showmetres is True and showdegrees is False:
        ax.tick_params(
            top=True,
            labeltop=True,  # Enable top ticks and labels
            right=True,
            labelright=True,  # Enable right ticks and labels
            bottom=True,
            labelbottom=True,  # Ensure bottom ticks and labels stay on
            left=True,
            labelleft=True,  # Ensure left ticks and labels stay on
        )

    if showdegrees is True and showmetres is False:
        ax.tick_params(axis="both", labelsize=9, labelcolor="k")
        xlabels_dd(ax, crs, ymin, num=xticks)
        ylabels_dd(ax, crs, xmax, num=yticks)

    for label in ax.yaxis.get_majorticklabels():
        if showdegrees:
            label.set_horizontalalignment("left")
        label.set_verticalalignment("center")

    for label in ax.xaxis.get_majorticklabels():
        if showdegrees:
            label.set_verticalalignment("top")

    if showdegrees is False:
        return

    # Axes for degrees below.
    # x-axis
    secax = ax.secondary_xaxis("top")
    secax.set_xlim(xmin, xmax)
    secax.tick_params(axis="x", labelsize=9)

    xlabels_dd(secax, crs, ymax, num=xticks)

    # y-axis
    secax1 = ax.secondary_yaxis("left")
    secax1.set_ylim(ymin, ymax)
    secax1.tick_params(axis="y", labelrotation=90, labelsize=9)

    ylabels_dd(secax1, crs, xmin, num=yticks)


def xlabels_dd(ax, crs, y, pad=15, num=10):
    """Generate labels for x axis in degrees."""
    fig = ax.get_figure()

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    transformeri = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xmin, xmax = ax.get_xlim()

    xmind, _ = transformer.transform(xmin, y)
    xmaxd, ymaxd = transformer.transform(xmax, y)

    while True:
        deg_ticks, labels = get_neat_intervals(xmind, xmaxd, num)
        num = len(deg_ticks) - 1
        ticks = [transformeri.transform(x, ymaxd)[0] for x in deg_ticks]

        ax.set_xticks(ticks, labels=labels)

        fig.draw_without_rendering()

        text_boxes = [label.get_window_extent() for label in ax.get_xticklabels()]
        overlap = any(b1.x1 + pad > b2.x0 for b1, b2 in pairwise(text_boxes))

        if not overlap:
            break

        num -= 1


def xlabels_m(ax, pad=15, num=10):
    """Generate labels for x axis in metres."""
    fig = ax.get_figure()

    while True:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=num))
        fig.draw_without_rendering()
        text_boxes = [label.get_window_extent() for label in ax.get_xticklabels()]
        overlap = any(b1.x1 + pad > b2.x0 for b1, b2 in pairwise(text_boxes))

        if not overlap:
            break

        num -= 1


def ylabels_dd(ax, crs, x, pad=20, num=10):
    """Generate labels for y axis in degrees."""
    fig = ax.get_figure()

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    transformeri = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ymin, ymax = ax.get_ylim()

    xd, ymind = transformer.transform(x, ymin)
    _, ymaxd = transformer.transform(x, ymax)

    while True:
        deg_ticks, labels = get_neat_intervals(ymind, ymaxd, num, False)
        num = len(deg_ticks) - 1
        ticks = [transformeri.transform(xd, y)[1] for y in deg_ticks]

        ax.set_yticks(ticks, labels=labels, va="center")
        fig.draw_without_rendering()

        text_boxes = [label.get_window_extent() for label in ax.get_yticklabels()]
        overlap = any(b1.y1 + pad > b2.y0 for b1, b2 in pairwise(text_boxes))

        if not overlap:
            break

        num -= 1


def ylabels_m(ax, pad=15, num=10):
    """Generate labels for y axis in metres."""
    fig = ax.get_figure()

    while True:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num))
        fig.draw_without_rendering()
        text_boxes = [label.get_window_extent() for label in ax.get_yticklabels()]

        overlap = any(b1.y1 + pad > b2.y0 for b1, b2 in pairwise(text_boxes))

        if not overlap:
            break

        num -= 1


def set_northscale(
    ax: Axes,
    crs: CRS | None,
    showlog: Callable[..., None] = print,
):
    """
    Set the north arrow and the scale bar.

    Parameters
    ----------
    ax
        Primary Matplotlib axes.
    crs
        PyProj crs of data
    showlog
        Show information using a function, by default print
    """
    if crs is None:
        crs = CRS.from_string(
            'LOCAL_CS["Arbitrary",UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )
    else:
        crs = CRS.from_wkt(crs.to_wkt())
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

    if crs.axis_info[0].unit_name not in units_standard and not crs.is_geographic:
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


def set_north(
    ax: Axes,
    crs: CRS | None,
    showlog: Callable[..., None] = print,
):
    """
    Set the north arrow and the scale bar.

    Parameters
    ----------
    ax
        Primary Matplotlib axes.
    crs
        PyProj crs of data
    showlog
        Show information using a function, by default print
    """
    if crs is None:
        crs = CRS.from_string(
            'LOCAL_CS["Arbitrary",UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )
    else:
        crs = CRS.from_wkt(crs.to_wkt())
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


def set_scale(
    ax: Axes,
    crs: CRS | None,
    showlog: Callable[..., None] = print,
):
    """
    Set the north arrow and the scale bar.

    Parameters
    ----------
    ax
        Primary Matplotlib axes.
    crs
        PyProj crs of data
    showlog
        Show information using a function, by default print
    """
    if crs is None:
        crs = CRS.from_string(
            'LOCAL_CS["Arbitrary",UNIT["metre",1,'
            'AUTHORITY["EPSG","9001"]],'
            'AXIS["Easting",EAST],'
            'AXIS["Northing",NORTH]]'
        )
    else:
        crs = CRS.from_wkt(crs.to_wkt())

    if crs.axis_info[0].unit_name not in units_standard and not crs.is_geographic:
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


def tick_formatter(
    x: float,
    pos: int,
) -> str:
    """
    Format thousands separator in ticks for plots.

    Parameters
    ----------
    x
        Number to be formatted.
    pos
        Position of tick.

    Returns
    -------
    str
        Formatted coordinate.

    """
    if np.ma.is_masked(x):
        return "--"

    newx = f"{x:,.5f}".rstrip("0").rstrip(".")

    return newx


# This function is the tick formatter.
frm = FuncFormatter(tick_formatter)


def _testfn():
    """Test function."""
    import geopandas as gpd

    # from pygmi.raster.iodefs import get_raster

    sfile = r"D:\workdata\PyGMI Test Data\Vector\Rose\2329AC_lin_wgs84sutm35.shp"
    # ifile = r"D:\Workdata\PyGMI Test Data\Raster\testdata.tif"

    plt.figure(layout="compressed")
    ax = plt.gca()

    gdfs = gpd.read_file(sfile)
    # gdfs = gdfs.to_crs(4326)
    crs = gdfs.crs
    gdfs.plot(ax=ax)

    # data = get_raster(ifile)
    # crs = data[0].crs
    # plt.imshow(data[0].data, extent=data[0].extent)
    # plt.colorbar()

    set_axes(ax, crs)
    set_northscale(ax, crs)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.show()


if __name__ == "__main__":
    _testfn()
