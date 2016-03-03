# -*- coding: utf-8 -*-
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
""" Misc is a collection of routines which can be used in PyGMI in general.

ptimer is utility module used to simplify checking how much time has passed
in a program. It also outputs a message at the point when called. """

from PyQt4 import QtGui
import time


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

#    background: qlineargradient(x1: 0.5, y1: 0, x2: 0.5, y2: 1, stop: 0 green, stop: 1 white);
#    background-color: #05B8CC;

class PTime(object):
    """ Main class in the ptimer module. Once activated, this class keeps track
    of all time since activation. Times are stored whenever its methods are
    called.

    Attributes
    ----------
    tchk : list
        List of times generated by the time.clock routine.
    """
    def __init__(self):
        self.tchk = [time.clock()]

    def since_first_call(self, msg='since first call', show=True):
        """ This function prints out a message and lets you know the time
        passed since the first call.

        Parameters
        ----------
        msg : str
            Optional message
        """
        self.tchk.append(time.clock())
        tdiff = self.tchk[-1] - self.tchk[0]
        if show:
            print(msg, 'at time (s):', tdiff)
        return tdiff

    def since_last_call(self, msg='since last call', show=True):
        """ This function prints out a message and lets you know the time
        passed since the last call.

        Parameters
        ----------
        msg : str
            Optional message"""

        self.tchk.append(time.clock())
        tdiff = self.tchk[-1] - self.tchk[-2]
        if show:
            print(msg, 'time(s):', tdiff, 'since last call')
        return tdiff


class ProgressBar(QtGui.QProgressBar):
    """
    Progress Bar routine which expands the QProgressBar class slightly so that
    there is a time function as well as a convenient of calling it via an
    iterable.

    Attributes
    ----------
    otime : integer
        This is the original time recorded when the progress bar starts.
    """
    def __init__(self, parent=None):
        QtGui.QProgressBar.__init__(self, parent)
        self.setMinimum(0)
        self.setValue(0)
        self.otime = 0
        self.setStyleSheet(PBAR_STYLE)

    def iter(self, iterable):
        """
        Iterator Routine
        """
        total = len(iterable)
        self.setMaximum(total)
        self.setMinimum(0)
        self.setValue(0)

        self.otime = time.clock()
        time1 = self.otime
        time2 = self.otime

        n = 0
        for obj in iterable:
            yield obj
            n += 1

            time2 = time.clock()
            if time2-time1 > 1:
                self.setValue(n)
                tleft = (total-n)*(time2-self.otime)/n
                if tleft > 60:
                    tleft = int(tleft // 60)
                    self.setFormat('%p% '+str(tleft)+'min left')
                else:
                    tleft = int(tleft)
                    self.setFormat('%p% '+str(tleft)+'s left')
                QtGui.QApplication.processEvents()
                time1 = time2

        self.setFormat('%p%')
        self.setValue(total)

    def to_max(self):
        """ Sets the progress to maximum """
        self.setMaximum(1)
        self.setMinimum(0)
        self.setValue(1)
        QtGui.QApplication.processEvents()
