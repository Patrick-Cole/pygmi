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
"""
Misc is a collection of routines which can be used in PyGMI in general.
"""

import os
import sys
import types
import time
import psutil
import numpy as np
from matplotlib import ticker
from PyQt5 import QtWidgets, QtCore, QtGui


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
    """Class to intercept stdout for later use in a textbox."""

    def __init__(self, textWritten):
        self.textWritten = textWritten

    def write(self, text):
        """
        Write text.

        Parameters
        ----------
        text : str
            Text to write.

        Returns
        -------
        None.

        """
        self.textWritten(str(text))

    def flush(self):
        """
        Flush.

        Returns
        -------
        None.

        """

    def fileno(self):
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

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file, used in IO routines and to pass filename back to main.py
    piter : iter
        reference to a progress bar iterator.
    pbar : progressbar
        reference to a progress bar.
    showlog: stdout or alternative
        reference to a way to view messages, normally stdout or a Qt text box.
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
            self.process_is_active = parent.process_is_active

        self.piter = self.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.ifile = ''
        self.ipth = os.path.dirname(__file__)+r'/images/'
        self.setWindowIcon(QtGui.QIcon(self.ipth+'logo256.ico'))

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        return True

    def data_init(self):
        """
        Initialise Data.

        Entry point into routine. This entry point exists for
        the case  where data must be initialised before entering at the
        standard 'settings' sub module.

        Returns
        -------
        None.

        """

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

        return projdata


class ContextModule(QtWidgets.QDialog):
    """
    Context Module.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    piter : iter
        reference to a progress bar iterator.
    pbar : progressbar
        reference to a progress bar.
    showlog: stdout or alternative
        reference to a way to view messages, normally stdout or a Qt text box.
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
            self.process_is_active = parent.process_is_active

        self.piter = self.pbar.iter

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.ipth = os.path.dirname(__file__)+r'/images/'
        self.setWindowIcon(QtGui.QIcon(self.ipth+'logo256.ico'))

    def run(self):
        """
        Run context menu item.

        Returns
        -------
        None.

        """


class QLabelVStack:
    """QLabelVStack."""

    def __init__(self, parent=None):
        self.layout = QtWidgets.QGridLayout(parent)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.indx = 0

    def addWidget(self, widget1, widget2):
        """
        Add two widgets on a row, widget1 can also be text.

        Parameters
        ----------
        widget1 : str or QWidget
            First Widget or Label on the row.
        widget2 : QWidget
            Last Widget.

        Returns
        -------
        None.

        """
        if isinstance(widget1, str):
            widget1 = QtWidgets.QLabel(widget1)

        self.layout.addWidget(widget1, self.indx, 0)
        self.layout.addWidget(widget2, self.indx, 1)
        self.indx += 1


class PTime():
    """
    PTime class.

    Main class in the ptimer module. Once activated, this class keeps track
    of all time since activation. Times are stored whenever its methods are
    called.

    Attributes
    ----------
    tchk : list
        List of times generated by the time.perf_counter routine.
    """

    def __init__(self):
        self.tchk = [time.perf_counter()]

    def since_first_call(self, msg='since first call', show=True):
        """
        Time lapsed since first call.

        This function prints out a message and lets you know the time
        passed since the first call.

        Parameters
        ----------
        msg : str
            Optional message
        """
        self.tchk.append(time.perf_counter())
        tdiff = self.tchk[-1] - self.tchk[0]
        if show:
            if tdiff < 60:
                print(msg, 'time (s):', tdiff)
            else:
                mins = int(tdiff/60)
                secs = tdiff-mins*60
                print(msg, 'time (s): ', mins, ' minutes ', secs, ' seconds')
        return tdiff

    def since_last_call(self, msg='since last call', show=True):
        """
        Time lapsed since last call.

        This function prints out a message and lets you know the time
        passed since the last call.

        Parameters
        ----------
        msg : str
            Optional message
        """
        self.tchk.append(time.perf_counter())
        tdiff = self.tchk[-1] - self.tchk[-2]
        if show:
            print(msg, 'time(s):', tdiff, 'since last call')
        return tdiff


class ProgressBar(QtWidgets.QProgressBar):
    """
    Progress bar.

    Progress Bar routine which expands the QProgressBar class slightly so that
    there is a time function as well as a convenient of calling it via an
    iterable.

    Attributes
    ----------
    otime : integer
        This is the original time recorded when the progress bar starts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimum(0)
        self.setValue(0)
        self.otime = 0
        self.setStyleSheet(PBAR_STYLE)
        self.total = 100

    def iter(self, iterable):
        """Iterate Routine."""
        if not isinstance(iterable, types.GeneratorType):
            self.total = len(iterable)

        self.setMaximum(self.total)
        self.setMinimum(0)
        self.setValue(0)

        self.otime = time.perf_counter()
        time1 = self.otime
        time2 = self.otime

        i = 0
        for obj in iterable:
            yield obj
            i += 1

            time2 = time.perf_counter()
            if time2-time1 > 1:
                self.setValue(i)
                tleft = (self.total-i)*(time2-self.otime)/i
                if tleft > 60:
                    tleft = int(tleft // 60)
                    self.setFormat('%p% '+str(tleft)+'min left ')
                else:
                    tleft = int(tleft)
                    self.setFormat('%p% '+str(tleft)+'s left   ')
                QtWidgets.QApplication.processEvents()
                time1 = time2

        self.setFormat('%p%')
        self.setValue(self.total)

    def to_max(self):
        """Set the progress to maximum."""
        self.setMaximum(self.total)
        self.setMinimum(0)
        self.setValue(self.total)
        QtWidgets.QApplication.processEvents()


class ProgressBarText():
    """Text Progress bar."""

    def __init__(self):
        self.otime = 0
        self.total = 100
        self.decimals = 1
        self.length = 40
        self.fill = '#'
        self.prefix = 'Progress:'

    def iter(self, iterable):
        """Iterate Routine."""
        if not isinstance(iterable, types.GeneratorType):
            self.total = len(iterable)

        if self.total == 0:
            self.total = 1

        self.otime = time.perf_counter()
        time1 = self.otime
        time2 = self.otime

        i = 0
        oldval = 0
        gottototal = False
        for obj in iterable:
            yield obj
            i += 1

            time2 = time.perf_counter()
            if time2-time1 > 1 and int(i*100/self.total) > oldval:
                oldval = int(i*100/self.total)

                tleft = (self.total-i)*(time2-self.otime)/i
                if tleft > 60:
                    timestr = f' {tleft // 60:.0f} min left '
                else:
                    timestr = f' {tleft:.1f} sec left '
                timestr += f' {time2-self.otime:.1f} sec total      '

                self.printprogressbar(i, suffix=timestr)
                time1 = time2
                if i == self.total:
                    gottototal = True

        if not gottototal:
            self.printprogressbar(self.total)

    def printprogressbar(self, iteration, suffix=''):
        """
        Call in a loop to create terminal progress bar.

        Code by Alexander Veysov. (https://gist.github.com/snakers4).

        Parameters
        ----------
        iteration : int
            current iteration
        suffix : str, optional
            Suffix string. The default is ''.

        Returns
        -------
        None.

        """
        perc = 100*(iteration/float(self.total))
        percent = f'{perc:.{self.decimals}f}'
        filledlength = int(self.length*iteration//self.total)
        pbar = self.fill*filledlength + '-'*(self.length - filledlength)
        pbar = f'\r{self.prefix} |{pbar}| {percent}% {suffix}'
        print(pbar, end='\r')
        # Print New Line on Complete
        if iteration == self.total:
            print()

    def to_max(self):
        """Set the progress to maximum."""
        self.printprogressbar(self.total)


def getinfo(txt=None, reset=False):
    """
    Get time and memory info.

    Parameters
    ----------
    txt : str/int/float, optional
        Descriptor used for headings. The default is None.
    reset : bool
        Flag used to reset the time difference to zero.

    Returns
    -------
    None.

    """
    global PTIME

    timebefore = PTIME
    PTIME = time.perf_counter()

    if timebefore is None or reset is True:
        tdiff = 0.
    else:
        tdiff = PTIME - timebefore

    if txt is not None:
        heading = '===== '+str(txt)+': '
    else:
        heading = '===== Info: '

    mem = psutil.virtual_memory()
    memtxt = f'RAM memory used: {mem.used:,.1f} B ({mem.percent}%)'

    print(heading+memtxt+f' Time(s): {tdiff:.3f}')


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
        return '--'

    newx = f'{x:,.5f}'.rstrip('0').rstrip('.')

    return newx


frm = ticker.FuncFormatter(tick_formatter)
