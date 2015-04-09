# -----------------------------------------------------------------------------
# Name:        misc.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
""" These are miscellaneous functions for the program """

from PyQt4 import QtGui, QtCore
import pygmi.menu_default as menu_default
import time


def update_lith_lw(lmod, lwidget):
    """ Updates the lithology list widget """
    lwidget.clear()
    for i in lmod.lith_list.keys():
        lwidget.addItem(i)

    for i in range(lwidget.count()):
        tmp = lwidget.item(i)
        tindex = lmod.lith_list[str(tmp.text())].lith_index
        tcol = lmod.mlut[tindex]
        tmp.setBackground(QtGui.QColor(tcol[0], tcol[1], tcol[2], 255))


class RangedCopy(QtGui.QDialog):
    """ Class to call up a dialog for ranged copying """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)

        self.sb_master = QtGui.QSpinBox()
        self.sb_start = QtGui.QSpinBox()
        self.lw_lithdel = QtGui.QListWidget()
        self.lw_lithcopy = QtGui.QListWidget()
        self.sb_end = QtGui.QSpinBox()

        self.setupui()

    def setupui(self):
        """ Setup UI """
        gridlayout = QtGui.QGridLayout(self)
        buttonbox = QtGui.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.pfmod.misc.rangedcopy')

        label = QtGui.QLabel()
        label_2 = QtGui.QLabel()
        label_3 = QtGui.QLabel()
        label_4 = QtGui.QLabel()
        label_5 = QtGui.QLabel()

        self.sb_master.setMaximum(999999999)
        self.sb_start.setMaximum(999999999)
        self.lw_lithcopy.setSelectionMode(self.lw_lithcopy.MultiSelection)
        self.lw_lithdel.setSelectionMode(self.lw_lithdel.MultiSelection)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)
        self.sb_end.setMaximum(999999999)

        self.setWindowTitle("Ranged Copy")
        label.setText("Range Start")
        label_2.setText("Master Profile")
        label_3.setText("Lithologies To Copy")
        label_4.setText("Lithologies To Overwrite")
        label_5.setText("Range End")

        gridlayout.addWidget(label_2, 0, 0, 1, 1)
        gridlayout.addWidget(self.sb_master, 0, 1, 1, 1)
        gridlayout.addWidget(label, 1, 0, 1, 1)
        gridlayout.addWidget(self.sb_start, 1, 1, 1, 1)
        gridlayout.addWidget(label_5, 2, 0, 1, 1)
        gridlayout.addWidget(self.sb_end, 2, 1, 1, 1)
        gridlayout.addWidget(label_3, 3, 0, 1, 1)
        gridlayout.addWidget(self.lw_lithcopy, 3, 1, 1, 1)
        gridlayout.addWidget(label_4, 4, 0, 1, 1)
        gridlayout.addWidget(self.lw_lithdel, 4, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout.addWidget(buttonbox, 5, 1, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)


def rcopy_dialog(lmod1, islayer=True, is_ew=True):
    """ Main routine to perform actual ranged copy """
    rcopy = RangedCopy()
    for i in lmod1.lith_list.keys():
        rcopy.lw_lithcopy.addItem(i)
        rcopy.lw_lithdel.addItem(i)

    if islayer is True:
        rmax = lmod1.numz-1
    elif is_ew is True:
        rmax = lmod1.numy-1
    else:
        rmax = lmod1.numx-1

    rcopy.sb_start.setMaximum(rmax)
    rcopy.sb_end.setMaximum(rmax)
    rcopy.sb_end.setValue(rmax)

    if islayer is True:
        rcopy.sb_master.setValue(lmod1.curlayer)
    else:
        rcopy.sb_master.setValue(lmod1.curprof)

    tmp = rcopy.exec_()
    if tmp == 0:
        return

    lithcopy = rcopy.lw_lithcopy.selectedItems()
    lithdel = rcopy.lw_lithdel.selectedItems()
    lstart = rcopy.sb_start.value()
    lend = rcopy.sb_end.value()
    lmaster = rcopy.sb_master.value()

    if lstart > lend:
        lstart, lend = lend, lstart

    if islayer is True:
        mtmp = lmod1.lith_index[:, :, lmaster]
        if lend > lmod1.numz:
            lend = lmod1.numz
    else:
        if is_ew is True:
            mtmp = lmod1.lith_index[:, lmaster, ::-1]
            if lend > lmod1.numy:
                lend = lmod1.numy
        else:
            mtmp = lmod1.lith_index[lmaster, :, ::-1]
            if lend > lmod1.numx:
                lend = lmod1.numx

    mslice = mtmp * 0
    for i in lithcopy:
        mslice[mtmp == lmod1.lith_list[i.text()].lith_index] = 1

    for i in range(lstart, lend+1):
        if islayer is True:
            ltmp = lmod1.lith_index[:, :, i]
        else:
            if is_ew is True:
                ltmp = lmod1.lith_index[:, i, ::-1]
            else:
                ltmp = lmod1.lith_index[i, :, ::-1]

        lslice = ltmp * 0
        for j in lithdel:
            lslice[ltmp == lmod1.lith_list[j.text()].lith_index] = 1
        mlslice = mslice + lslice
        mlslice[mlslice == 1] = 0
        mlslice[mlslice > 0] = 1
        mtmp2 = mtmp.copy()
        mtmp2[mlslice == 0] = 0
        ltmp[mlslice == 1] = 0
        ltmp += mtmp2


class ProgressBar(object):
    """ Wrapper for a progress bar """
    def __init__(self, pbar, pbarmain):
        self.pbar = pbar
        self.pbarmain = pbarmain
        self.max = 1
        self.value = 0
        self.mmax = 1
        self.mvalue = 0
        self.otime = None
        self.mtime = None
        self.resetall()

    def incr(self):
        """ increases value by one """
        if self.value < self.max:
            self.value += 1
            if self.value == self.max and self.mvalue < self.mmax:
                self.mvalue += 1
                self.value = 0
                self.pbarmain.setValue(self.mvalue)
            if self.mvalue == self.mmax:
                self.value = self.max
            self.pbar.setValue(self.value)

        QtCore.QCoreApplication.processEvents()

    def iter(self, iterable):
        """
        Iterator Routine
        """
        total = len(iterable)
        self.max = total
        self.pbar.setMaximum(total)
        self.pbar.setMinimum(0)
        self.pbar.setValue(0)

        self.otime = time.clock()
        time1 = self.otime
        time2 = self.otime

        n = 0
        for obj in iterable:
            yield obj
            n += 1

            time2 = time.clock()
            if time2-time1 > 1:
                self.pbar.setValue(n)
                tleft = (total-n)*(time2-self.otime)/n
                if tleft > 60:
                    tleft = int(tleft // 60)
                    self.pbar.setFormat('%p% '+str(tleft)+'min left')
                else:
                    tleft = int(tleft)
                    self.pbar.setFormat('%p% '+str(tleft)+'s left')
                QtGui.QApplication.processEvents()
                time1 = time2

        self.pbar.setFormat('%p%')
        self.pbar.setValue(total)

        self.incrmain()
        self.value = 0
        QtGui.QApplication.processEvents()

    def incrmain(self, i=1):
        """ increases value by one """
        self.mvalue += i
        self.pbarmain.setValue(self.mvalue)

        n = self.mvalue
        total = self.mmax
        tleft = (total-n)*(time.clock()-self.mtime)/n
        if tleft > 60:
            tleft = int(tleft // 60)
            self.pbarmain.setFormat('%p% '+str(tleft)+'min left')
        else:
            tleft = int(tleft)
            self.pbarmain.setFormat('%p% '+str(tleft)+'s left')
        QtGui.QApplication.processEvents()

    def maxall(self):
        """ Sets all progress bars to maximum value """
        self.mvalue = self.mmax
        self.value = self.max
        self.pbarmain.setValue(self.mvalue)
        self.pbar.setValue(self.value)
        self.pbar.setFormat('%p%')
        self.pbarmain.setFormat('%p%')

    def resetall(self, maximum=1, mmax=1):
        """ Sets min and max and resets all bars to 0 """

        self.pbar.setFormat('%p%')
        self.pbarmain.setFormat('%p%')
        self.mtime = time.clock()

        self.max = maximum
        self.value = 0
        self.mmax = mmax
        self.mvalue = 0
        self.pbar.setMinimum(self.value)
        self.pbar.setMaximum(self.max)
        self.pbar.setValue(self.value)
        self.pbarmain.setMinimum(self.mvalue)
        self.pbarmain.setMaximum(self.mmax)
        self.pbarmain.setValue(self.mvalue)

    def resetsub(self, maximum=1):
        """ Sets min and max and resets sub bar to 0 """

        self.pbar.setFormat('%p%')
        self.max = maximum
        self.value = 0
        self.pbar.setMinimum(self.value)
        self.pbar.setMaximum(self.max)
        self.pbar.setValue(self.value)

    def busysub(self):
        """ Busy """
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(0)
        self.pbar.setValue(-1)
        QtCore.QCoreApplication.processEvents()
