# -----------------------------------------------------------------------------
# Name:        misc.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2015 Council for Geoscience
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
"""Misc is a collection of routines which can be used in PyGMI in general."""

import os
import sys
import textwrap
import time
import types
import webbrowser
from collections.abc import Generator, Iterable

import geopandas as gpd
import numpy as np
import psutil
import requests
from matplotlib import cm, colors
from matplotlib.axes import Axes
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator

from pygmi import __version__
from pygmi.raster.reproj import GroupProj

# if os.name == 'nt':
#     import win32api
#     import win32job

PBAR_STYLE = """
QProgressBar{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0.5, y1: 0, x2: 0.5, y2: 1, stop: 0 green, stop: 1 white);
    width: 10px;
}
"""

PTIME = None


class EmittingStream(QtCore.QObject):
    """
    Class to intercept stdout for later use in a textbox.

    Parameters
    ----------
    textwritten
        Text written to stdout.

    """

    def __init__(self, textWritten):
        self.textWritten = textWritten

    def write(self, text: str):
        """
        Write text.

        Parameters
        ----------
        text
            Text to write.
        """
        self.textWritten(str(text))

    def flush(self):
        """Flush."""

    def fileno(self) -> int:
        """
        File number.

        Returns
        -------
        int
            Returns -1.

        """
        return -1


class BasicModule(QtWidgets.QDialog):
    """
    Basic Module.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    parent
        Reference to the parent routine
    indata
        Dictionary of input datasets
    outdata
        Dictionary of output datasets
    ifile
        Input file, used in IO routines and to pass filename back to main.py
    piter
        Reference to a progress bar iterator.
    pbar
        Reference to a progress bar.
    showlog
        Reference to a way to view messages, normally stdout or a Qt text box.
    is_import
        Used to indicate whether a routine contains an import within.
    projdata
        Project data.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        if parent is None:
            self.stdout_redirect = sys.stdout
            self.showlog = print
            self.pbar = ProgressBarText()
            self.process_is_active = lambda *args, **kwargs: None
        else:
            self.stdout_redirect = EmittingStream(parent.showlog)
            self.showlog = parent.showlog
            self.pbar = parent.pbar
            if hasattr(parent, "process_is_active"):
                self.process_is_active = parent.process_is_active
            else:
                self.process_is_active = lambda *args, **kwargs: None

        self.piter = self.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.projdata = {}
        self.parent = parent
        self.is_import = False
        self.ifile = ""

        regex_pattern = r"^$|^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
        self.qval = QRegularExpressionValidator(QRegularExpression(regex_pattern))

        ipth = os.path.dirname(__file__) + r"/images/"
        self.setWindowIcon(QtGui.QIcon(ipth + "logo256.ico"))

        self.buttonbox = PButtonBox(self)
        self.buttonbox.buttonbox.accepted.connect(self.accept)
        self.buttonbox.buttonbox.rejected.connect(self.reject)

    def settings(self, nodialog: bool = False) -> bool:
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
        return True

    def check_validation(self):
        """
        Check a widget's validation.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        for value in vars(self).values():
            if isinstance(value, QtWidgets.QLineEdit) and (
                not value.hasAcceptableInput()
            ):
                self.showlog("One of your inputs is incorrect. Please check.")
                return False

        return True

    def cmb_update(self, obj: QtWidgets.QComboBox, txtlist: list[str]):
        """
        Update combo box.

        Parameters
        ----------
        obj
            Combo box to add data to.
        txtlist
            List of strings to add to combo box.
        """
        obj.blockSignals(True)
        txt = str(obj.currentText())

        obj.clear()
        obj.addItems(txtlist)
        if txt != "":
            obj.setCurrentText(txt)

        obj.blockSignals(False)

    def data_init(self):
        """
        Initialise Data.

        Entry point into routine. This entry point exists for the case where data must
        be initialised before entering at the standard 'settings' sub module.
        """

    def loadproj(self, projdata: dict) -> bool:
        """
        Load project data into class.

        Parameters
        ----------
        projdata
            Project data loaded from JSON project file.

        Returns
        -------
        bool
            A check to see if settings was successfully run.
        """
        self.projdata = projdata

        for otxt in projdata:
            if otxt not in vars(self):
                self.showlog(
                    "Cannot load project, you may be using an old project format."
                )
                return False

        for otxt, pdata in projdata.items():
            obj = vars(self)[otxt]

            if obj is None:
                vars(self)[otxt] = pdata

            if isinstance(obj, (float, int, bool, list, np.ndarray, tuple, str, dict)):
                vars(self)[otxt] = pdata

            if isinstance(obj, gpd.GeoDataFrame):
                vars(self)[otxt] = gpd.read_file(pdata, driver="GeoJSON")

            if isinstance(obj, QtWidgets.QComboBox):
                obj.blockSignals(True)
                if obj.count() == 0:
                    obj.addItem(pdata)
                obj.setCurrentText(pdata)
                obj.blockSignals(False)

            if isinstance(obj, (QtWidgets.QLineEdit, QtWidgets.QTextEdit)):
                obj.blockSignals(True)
                obj.setText(pdata)
                obj.blockSignals(False)

            if isinstance(
                obj, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QSlider)
            ):
                obj.blockSignals(True)
                obj.setValue(pdata)
                obj.blockSignals(False)

            if isinstance(
                obj, (QtWidgets.QRadioButton, QtWidgets.QCheckBox, QtWidgets.QGroupBox)
            ):
                obj.blockSignals(True)
                obj.setChecked(pdata)
                obj.blockSignals(False)

            if isinstance(obj, QtWidgets.QDateEdit):
                obj.blockSignals(True)
                date = obj.date().fromString(pdata)
                obj.setDate(date)
                obj.blockSignals(False)

            if isinstance(obj, QtWidgets.QListWidget):
                obj.blockSignals(True)
                if obj.count() == 0:
                    obj.addItems(self.projdata[otxt]["all"])

                for i in range(obj.count()):
                    if obj.item(i).text() in self.projdata[otxt]["selected"]:
                        obj.item(i).setSelected(True)

                obj.blockSignals(False)

            if isinstance(obj, GroupProj):
                obj.cmb_datum.blockSignals(True)
                obj.cmb_datum.setCurrentText(pdata["datum"])
                obj.cmb_datum.blockSignals(False)
                obj.combo_datum_change()

                obj.cmb_proj.blockSignals(True)
                obj.cmb_proj.setCurrentText(pdata["proj"])
                obj.cmb_proj.blockSignals(False)
                obj.combo_change()

        if self.is_import is True:
            chk = self.settings(True)
        else:
            chk = False

        return chk

    def saveproj(self):
        """Save project data from class."""

    def saveobj(self, obj: object):
        """
        Save an object to a dictionary.

        This is a convenience function for saving project information.

        Parameters
        ----------
        obj
            A variable to be saved.
        """
        otxt = None
        for name in vars(self):
            if id(vars(self)[name]) == id(obj):
                otxt = name

        if otxt is None:
            return

        if isinstance(obj, (float, int, bool, list, np.ndarray, tuple, str, dict)):
            self.projdata[otxt] = obj

        if isinstance(obj, gpd.GeoDataFrame):
            self.projdata[otxt] = obj.to_json()

        if isinstance(obj, QtWidgets.QComboBox):
            self.projdata[otxt] = obj.currentText()

        if isinstance(obj, QtWidgets.QLineEdit):
            self.projdata[otxt] = obj.text()

        if isinstance(obj, QtWidgets.QTextEdit):
            self.projdata[otxt] = obj.toPlainText()

        if isinstance(
            obj, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QSlider)
        ):
            self.projdata[otxt] = obj.value()

        if isinstance(
            obj, (QtWidgets.QRadioButton, QtWidgets.QCheckBox, QtWidgets.QGroupBox)
        ):
            self.projdata[otxt] = obj.isChecked()

        if isinstance(obj, QtWidgets.QDateEdit):
            self.projdata[otxt] = obj.date().toString()

        if isinstance(obj, QtWidgets.QListWidget):
            self.projdata[otxt] = {"all": [], "selected": []}

            tmp = [i.text() for i in obj.selectedItems()]
            self.projdata[otxt]["selected"] = tmp
            tmp = [obj.item(i).text() for i in range(obj.count())]
            self.projdata[otxt]["all"] = tmp

        if isinstance(obj, GroupProj):
            self.projdata[otxt] = {
                "datum": obj.cmb_datum.currentText(),
                "proj": obj.cmb_proj.currentText(),
            }


class ContextModule(QtWidgets.QDialog):
    """
    Context Module.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    parent
        Reference to the parent routine
    indata
        Dictionary of input datasets
    outdata
        Dictionary of output datasets
    piter
        Reference to a progress bar iterator.
    pbar
        Reference to a progress bar.
    showlog
        Reference to a way to view messages, normally stdout or a Qt text box.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.stdout_redirect = sys.stdout
            self.showlog = print
            self.pbar = ProgressBarText()
            self.process_is_active = lambda *args, **kwargs: None
        else:
            self.stdout_redirect = EmittingStream(parent.showlog)
            self.showlog = parent.showlog
            self.pbar = parent.pbar
            if hasattr(parent, "process_is_active"):
                self.process_is_active = parent.process_is_active
            else:
                self.process_is_active = lambda *args, **kwargs: None

        self.piter = self.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        regex_pattern = r"^$|^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
        self.qval = QRegularExpressionValidator(QRegularExpression(regex_pattern))

        ipth = os.path.dirname(__file__) + r"/images/"
        self.setWindowIcon(QtGui.QIcon(ipth + "logo256.ico"))

        self.buttonbox = PButtonBox(self)
        self.buttonbox.buttonbox.accepted.connect(self.accept)
        self.buttonbox.buttonbox.rejected.connect(self.reject)

    def check_validation(self) -> bool:
        """
        Check a widget's validation.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        for value in vars(self).values():
            if isinstance(value, QtWidgets.QLineEdit) and (
                not value.hasAcceptableInput()
            ):
                self.showlog("One of your inputs is incorrect. Please check.")
                return False

        return True

    def cmb_update(
        self, obj: QtWidgets.QComboBox, txtlist: list[str], curindex: int = 0
    ):
        """
        Update combo box.

        Parameters
        ----------
        obj
            Combo box to add data to.
        txtlist
            List of strings to add to combo box.
        curindex
            Current index.
        """
        obj.blockSignals(True)
        obj.clear()
        obj.addItems(txtlist)
        obj.setCurrentIndex(curindex)
        obj.blockSignals(False)

    def run(self):
        """Run context menu item."""


class PButtonBox(QtWidgets.QWidget):
    """
    Custom buttonbox with help.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = QtWidgets.QPushButton()

        self.htmlfile = None

        helpdocs.setMinimumHeight(32)
        helpdocs.setMinimumWidth(52)

        ipth = os.path.dirname(__file__) + r"/images/"

        helpdocs.setIcon(QtGui.QIcon(ipth + "help.png"))
        helpdocs.setIconSize(helpdocs.minimumSize())
        helpdocs.clicked.connect(self.help_docs)
        helpdocs.setFlat(True)

        buttonbox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(
            buttonbox.StandardButton.Cancel | buttonbox.StandardButton.Ok
        )

        hbl = QtWidgets.QHBoxLayout()

        hbl.addWidget(helpdocs, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        hbl.addWidget(buttonbox, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        self.setLayout(hbl)
        self.buttonbox = buttonbox
        self.helpdocs = helpdocs

    def help_docs(self):
        """Help Routine."""
        if self.htmlfile is not None:
            ipth = os.path.dirname(__file__) + r"/helpdocs/html"
            if ".html" not in self.htmlfile:
                self.htmlfile = self.htmlfile + ".html"
            hfile = os.path.join(ipth, self.htmlfile)
            webbrowser.open("file://" + hfile)


class QVStack2Layout(QtWidgets.QGridLayout):
    """
    QVStack2Layout custom Qt QGridLayot.

    This works like VBoxLayout, except each row takes two widgets.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.indx = 0

    def addWidget(self, widget1: str | QtWidgets.QWidget, widget2: QtWidgets.QWidget):
        """
        Add two widgets on a row, widget can also be text.

        Parameters
        ----------
        widget1
            First Widget or Label on the row.
        widget2
            Last Widget.

        """
        if isinstance(widget1, str):
            widget1 = QtWidgets.QLabel(widget1)

        if isinstance(widget2, str):
            widget2 = QtWidgets.QLabel(widget2)

        self.addWidgetOld(widget1, self.indx, 0)
        self.addWidgetOld(widget2, self.indx, 1)
        self.indx += 1

    def addWidgetOld(self, *args, **kwargs):
        """Original Add Widget."""
        super().addWidget(*args, **kwargs)


class PTime:
    """
    PTime class.

    Main class in the ptimer module. Once activated, this class keeps track
    of all time since activation. Times are stored whenever its methods are
    called.

    Attributes
    ----------
    tchk
        List of times generated by the time.perf_counter routine.
    """

    def __init__(self):
        self.tchk = [time.perf_counter()]

    def since_first_call(
        self, msg: str = "since first call", show: bool = True
    ) -> float:
        """
        Time lapsed since first call.

        This function prints out a message and lets you know the time
        passed since the first call.

        Parameters
        ----------
        msg
            Optional message, by default "since first call"
        show
            Show output, by default True.

        Returns
        -------
        float
            Time difference.
        """
        self.tchk.append(time.perf_counter())
        tdiff = self.tchk[-1] - self.tchk[0]
        if show:
            if tdiff < 60:
                print(msg, "time (s):", tdiff)
            else:
                mins = int(tdiff / 60)
                secs = tdiff - mins * 60
                print(msg, "time (s): ", mins, " minutes ", secs, " seconds")
        return tdiff

    def since_last_call(self, msg: str = "since last call", show: bool = True) -> float:
        """
        Time lapsed since last call.

        This function prints out a message and lets you know the time
        passed since the last call.

        Parameters
        ----------
        msg
            Optional message
        show
            Show output, by default True.

        Returns
        -------
        float
            Time difference.
        """
        self.tchk.append(time.perf_counter())
        tdiff = self.tchk[-1] - self.tchk[-2]
        if show:
            print(msg, "time(s):", tdiff, "since last call")
        return tdiff


class ProgressBar(QtWidgets.QProgressBar):
    """
    Qt custom progress bar.

    Progress Bar routine which expands the QProgressBar class slightly so that
    there is a time function as well as a convenient of calling it via an
    iterable.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    Attributes
    ----------
    otime
        This is the original time recorded when the progress bar starts.
    total
        Maximum progress bar value. The default is 100.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimum(0)
        self.setValue(0)
        self.otime = 0
        self.setStyleSheet(PBAR_STYLE)
        self.total = 100

    def iter(self, iterable: Iterable) -> Generator[object, None, None]:
        """
        Iterate Routine.

        Parameters
        ----------
        iterable
            Iterable for progress bar to track.

        Yields
        ------
        object
            Object in iterable.
        """
        if not isinstance(iterable, types.GeneratorType):
            self.total = len(iterable)

        self.setMaximum(self.total)
        self.setMinimum(0)
        self.setValue(0)

        self.otime = time.perf_counter()
        time1 = self.otime
        time2 = self.otime

        for i, obj in enumerate(iterable):
            yield obj

            time2 = time.perf_counter()
            if time2 - time1 > 1:
                self.setValue(i)
                tleft = (self.total - i) * (time2 - self.otime) / (i + 1)
                if tleft > 60:
                    tleft = int(tleft // 60)
                    self.setFormat("%p% " + str(tleft) + "min left ")
                else:
                    tleft = int(tleft)
                    self.setFormat("%p% " + str(tleft) + "s left   ")
                QtWidgets.QApplication.processEvents()
                time1 = time2

        self.setFormat("%p%")
        self.setValue(self.total)

    def to_max(self):
        """Set the progress to maximum."""
        self.setMaximum(self.total)
        self.setMinimum(0)
        self.setValue(self.total)
        QtWidgets.QApplication.processEvents()


class ProgressBarText:
    """
    Text Progress bar.

    Attributes
    ----------
    otime
        This is the original time recorded when the progress bar starts.
    total
        Maximum progress bar value. The default is 100.
    """

    def __init__(self):
        self.otime = 0
        self.total = 100
        self.decimals = 1
        self.length = 40
        self.fill = "#"
        self.prefix = "Progress:"

    def iter(self, iterable: Iterable) -> Generator[object, None, None]:
        """
        Iterate Routine.

        Parameters
        ----------
        iterable
            Iterable for progress bar to track.

        Yields
        ------
        object
            Object in iterable.
        """
        if not isinstance(iterable, types.GeneratorType):
            self.total = len(iterable)

        if self.total == 0:
            self.total = 1

        self.otime = time.perf_counter()

        oldperc = 0
        gottototal = False
        for i, obj in enumerate(iterable):
            yield obj

            time2 = time.perf_counter()
            curperc = int(i * 100 / self.total)
            if curperc > oldperc or oldperc == 0:
                oldperc = curperc

                tleft = (self.total - i) * (time2 - self.otime) / (i + 1)
                if tleft > 60:
                    timestr = f" {tleft // 60:.0f} min left "
                else:
                    timestr = f" {tleft:.1f} sec left "
                timestr += f" {time2 - self.otime:.1f} sec total      "

                self.printprogressbar(i, suffix=timestr)
                if i == self.total:
                    gottototal = True

        if not gottototal:
            self.printprogressbar(self.total)

    def printprogressbar(self, iteration: int, suffix: str = ""):
        """
        Call in a loop to create terminal progress bar.

        Code by Alexander Veysov. (https://gist.github.com/snakers4).

        Parameters
        ----------
        iteration
            Current iteration
        suffix
            Suffix string. The default is ''.
        """
        perc = 100 * (iteration / float(self.total))
        percent = f"{perc:.{self.decimals}f}"
        filledlength = int(self.length * iteration // self.total)
        pbar = self.fill * filledlength + "-" * (self.length - filledlength)
        pbar = f"\r{self.prefix} |{pbar}| {percent}% {suffix}"
        print(pbar, end="\r")
        # Print New Line on Complete
        if iteration == self.total:
            print()

    def setMaximum(self, val: int):
        """
        Set the maximum value.

        Parameters
        ----------
        val
            Maximum value of progressbar.
        """
        self.total = int(val)

    def setValue(self, val: int):
        """
        Set the progressbar value.

        Parameters
        ----------
        val
            Value of progressbar.
        """
        self.printprogressbar(int(val))

    def to_max(self):
        """Set the progress to maximum."""
        self.printprogressbar(self.total)


def check_for_updates() -> str:
    """
    Check GitHub for updates.

    Returns
    -------
    str
        Version path, embedded in an html string.
    """
    # GitHub API endpoint for the latest release
    url = "https://api.github.com/repos/Patrick-Cole/pygmi/releases/latest"

    verpath = ""

    try:
        response = requests.get(url, timeout=10)

        # Handle successful response
        if response.status_code == 200:
            release_data = response.json()
            # Extract version tag (e.g., "v2.1.0") and strip 'v' if present
            latest_version_str = release_data["tag_name"].lstrip("v")

            versions = [__version__, latest_version_str]

            versions.sort()

            if versions[-1] != __version__:
                verpath = f"<a href={release_data['html_url']}>v{versions[-1]}</a>"

        elif response.status_code == 404:
            print("❌ Repository or release not found.")
        else:
            print(f"⚠️ Failed to check updates. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"📡 Network error occurred: {e}")

    return verpath


def discrete_colorbar(
    axes: Axes, csp, cdat: np.ma.MaskedArray | NDArray, lbls: list[str] | None = None
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


def getinfo(txt: str | float | None = None, reset: bool = False, hide: bool = False):
    """
    Get time and memory info.

    Parameters
    ----------
    txt
        Descriptor used for headings, by default None
    reset
        Flag used to reset the time difference to zero, by default False
    hide
        Hide the output text. Useful if you don't want to show the initialising reading, by default False
    """
    global PTIME

    timebefore = PTIME
    PTIME = time.perf_counter()

    if timebefore is None or reset is True:
        tdiff = 0.0
    else:
        tdiff = PTIME - timebefore

    if txt is not None:
        heading = "===== " + str(txt) + ": "
    else:
        heading = "===== Info: "

    mem = psutil.virtual_memory()
    memtxt = f"RAM memory used: {mem.used:,.1f} B ({mem.percent}%)"

    if hide is False:
        print(heading + memtxt + f" Time(s): {tdiff:.3f}")


# def limit_memory(memory_limit):
#     """
#     Limit memory in Windows.

#     Based on https://stackoverflow.com/questions/54949110/limit-python-script-ram-usage-in-windows

#     Parameters
#     ----------
#     memory_limit : int
#         Memory limit in GB.

#     Returns
#     -------
#     None.

#     """
#     memory_limit = int(memory_limit * 1024**3)

#     hjob = win32job.CreateJobObject(None, '')
#     hprocess = win32api.GetCurrentProcess()
#     win32job.AssignProcessToJobObject(hjob, hprocess)
#     info = win32job.QueryInformationJobObject(
#         hjob, win32job.JobObjectExtendedLimitInformation)
#     info['ProcessMemoryLimit'] = memory_limit
#     info['BasicLimitInformation']['LimitFlags'] |= (
#         win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
#     win32job.SetInformationJobObject(
#         hjob, win32job.JobObjectExtendedLimitInformation, info)


def textwrap2(
    text: str, width: int, placeholder: str = "...", max_lines: int | None = None
) -> str:
    """
    Provide slightly different placeholder functionality to textwrap.

    Placeholders will be a part of last line, instead of replacing it.

    Parameters
    ----------
    text
        Text to wrap.
    width
        Maximum line length.
    placeholder
        Placeholder when lines exceed max_lines. The default is '...'.
    max_lines
        Maximum number of lines. The default is None.

    Returns
    -------
    str
        Output wrapped text.

    """
    text2 = textwrap.wrap(text, width=width)

    if max_lines is not None and text2:
        text2 = text2[:max_lines]
        if len(text2[-1]) == width:
            text2[-1] = text2[-1][: -len(placeholder)] + placeholder

    text2 = "\n".join(text2)

    return text2


def _testfn():
    """Test function."""
    # _ = QtWidgets.QApplication(sys.argv)

    # tmp = BasicModule()
    # tmp.ifile = QtWidgets.QLineEdit('test')
    # tmp.saveobj(tmp.ifile)

    # print(tmp.projdata)

    import matplotlib.pyplot as plt

    data = [[0, 45, 50], [0, 45, 50], [0, 44, 50]]

    lbls = ["a", "b", "c", "d"]

    vals = np.unique(data)
    if np.ma.isMaskedArray(vals):
        vals = vals.compressed()
    vals = vals[~np.isnan(vals)]

    bnds = vals.tolist() + [vals.max() + 1]

    cmap = cm.viridis
    norm = colors.BoundaryNorm(bnds, cmap.N)

    fig = plt.figure(dpi=200)
    ax = fig.gca()
    cax = ax.imshow(data, norm=norm)

    discrete_colorbar(ax, cax, data, lbls)
    plt.show()


if __name__ == "__main__":
    # _testfn()
    check_for_updates()
