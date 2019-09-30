# -----------------------------------------------------------------------------
# Name:        dataprep.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
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
"""A set of Data Preparation routines."""

from __future__ import print_function

from PyQt5 import QtWidgets, QtCore
import numpy as np
import numpy.lib.recfunctions as nplrf
import pygmi.menu_default as menu_default
from pygmi.vector.datatypes import LData


class ProcessData(QtWidgets.QDialog):
    """
    Process Gravity Data.

    This class grids point data using a nearest neighbourhood technique.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar

        self.basethres = 10000

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.dsb_null = QtWidgets.QDoubleSpinBox()
        self.dataid = QtWidgets.QComboBox()
        self.checkbox = QtWidgets.QCheckBox('Apply to all stations:')
        self.density = QtWidgets.QLineEdit('2670')
        self.absbase = QtWidgets.QLineEdit('978032.67715')
        self.basethres = QtWidgets.QLineEdit('10000')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.grav.dataprep.processdata')
        label_density = QtWidgets.QLabel('Background Density (kg/m3):')
        label_absbase = QtWidgets.QLabel('Base Station Absolute Gravity (mGal):')
        label_bthres = QtWidgets.QLabel('Minimum Base Station Number:')

#        self.dsb_null.setMaximum(np.finfo(np.double).max)
#        self.dsb_null.setMinimum(np.finfo(np.double).min)
#        self.dsb_dxy.setMaximum(9999999999.0)
#        self.dsb_dxy.setMinimum(0.00001)
#        self.dsb_dxy.setDecimals(5)
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Gravity Data Processing')

        gridlayout_main.addWidget(label_density, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.density, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_absbase, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.absbase, 1, 1, 1, 1)
        gridlayout_main.addWidget(label_bthres, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.basethres, 2, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def settings(self):
        """Settings."""
        tmp = []
        if 'Line' not in self.indata:
            self.parent.showprocesslog('No Line Data')
            return False
        if self.indata['Line'].dataid != 'Gravity':
            self.parent.showprocesslog('Not Gravity Data')
            return False

        tmp = self.exec_()

        try:
            float(self.density.text())
            float(self.absbase.text())
            float(self.basethres.text())
        except ValueError:
            self.parent.showprocesslog('Value Error')
            return False

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """Accept."""
        newdat = self.indata['Line']

        dat = self.indata['Line'].data

        basethres = float(self.basethres.text())

# Convert multiple lines to single dataset.
        pdat = None
        for i in dat:
            if pdat is None:
                pdat = dat[i]
                continue
            pdat = np.append(pdat, dat[i])

# Drift Correction, to abs base value
        driftdat = pdat[pdat['STATION'] > basethres]

        pdat = pdat[pdat['STATION'] < basethres]

        x = pdat['DECTIMEDATE']
        xp = driftdat['DECTIMEDATE']
        fp = driftdat['GRAV']
        fp = fp - fp[0]

        dcor = np.interp(x, xp, fp)

        gobs = pdat['GRAV'] - dcor + float(self.absbase.text())

# Theoretical Gravity
        lat = np.deg2rad(pdat[newdat.ychannel])

        gT = 978032.67715*((1 + 0.001931851353 * np.sin(lat)**2) /
                           np.sqrt(1 - 0.0066943800229*np.sin(lat)**2))


# Atmospheric Correction
        h = pdat[newdat.zchannel]  # This is the ellipsoidal height

        gATM = 0.874-9.9*1e-5*h+3.56*1e-9*h**2

# Height Correction

        gHC = -(0.308769109-0.000439773*np.sin(lat)**2)*h+7.21252*1e-8*h**2


# Spherical Bouguer
        S = 166735  # Bullard B radius
        R0 = 6371000  # Mean radius of the earth
        a = 6378137
        b = 6356752.314245
        G = 6.67834*1e-11
        dens = float(self.density.text())

        alpha = S/R0

        R = np.sqrt(((a**2 * np.cos(lat))**2 + (b**2 * np.sin(lat))**2) /
                    ((a * np.cos(lat))**2 + (b * np.sin(lat))**2))

        delta = R0/R
        eta = h/R
        d = 3*np.cos(alpha)**2-2
        f = np.cos(alpha)
        k = np.sin(alpha)**2
        p = -6*np.cos(alpha)**2*np.sin(alpha/2) + 4*np.sin(alpha/2)**3
        m = -3*np.sin(alpha)**2*np.cos(alpha)
        n = 2*(np.sin(alpha/2)-np.sin(alpha/2)**2)  # is this abs?
        mu = 1 + eta**2/3-eta

        fdk = np.sqrt((f-delta)**2+k)
        t1 = (d+f*delta+delta**2)*fdk
        t2 = m*np.log(n/(f-delta+fdk))

        lamda = (t1+p+t2)/3

        gSB = 2*np.pi*G*dens*(mu*h - lamda*R)*1e5

# Bouguer Anomaly
        gba = gobs - gT + gATM - gHC - gSB  # add or subtract atm


#        gba = gobs - ((gT - gATM) + gHC + gSB)  # add or subtract atm
#       gba = gobs - gT + gATM - gHC - gSB

        pdat = nplrf.append_fields(pdat, 'BOUGUER', gba, usemask=False)

        dat2a = {}
        for i in dat:
            dat2a[i] = pdat[pdat['LINE'] == float(i)]

        dat2 = LData()

        dat2.xchannel = newdat.xchannel
        dat2.ychannel = newdat.ychannel
        dat2.zchannel = newdat.zchannel
        dat2.data = dat2a
        dat2.dataid = 'Gravity'

        self.outdata['Line'] = dat2
