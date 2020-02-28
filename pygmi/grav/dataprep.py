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
        super().__init__(parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
#        self.pbar = parent.pbar

        self.basethres = 10000

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.dsb_null = QtWidgets.QDoubleSpinBox()
        self.dataid = QtWidgets.QComboBox()
        self.checkbox = QtWidgets.QCheckBox('Apply to all stations:')
        self.density = QtWidgets.QLineEdit('2670')
        self.knownstat = QtWidgets.QLineEdit('88888.')
        self.knownbase = QtWidgets.QLineEdit('978000.0')
        self.absbase = QtWidgets.QLineEdit('978032.67715')
        self.basethres = QtWidgets.QLineEdit('10000')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.grav.dataprep.processdata')
        label_density = QtWidgets.QLabel('Background Density (kg/m3):')
        label_absbase = QtWidgets.QLabel('Base Station Absolute Gravity (mGal):')
        label_bthres = QtWidgets.QLabel('Minimum Base Station Number:')
        label_kstat = QtWidgets.QLabel('Known Base Station Number:')
        label_kbase = QtWidgets.QLabel('Known Base Station Absolute Gravity (mGal):')
        pb_calcbase = QtWidgets.QPushButton('Calculate local base value')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle('Gravity Data Processing')

        gridlayout_main.addWidget(label_kstat, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.knownstat, 0, 1, 1, 1)
        gridlayout_main.addWidget(label_kbase, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.knownbase, 1, 1, 1, 1)

        gridlayout_main.addWidget(pb_calcbase, 2, 0, 1, 2)

        gridlayout_main.addWidget(label_density, 3, 0, 1, 1)
        gridlayout_main.addWidget(self.density, 3, 1, 1, 1)
        gridlayout_main.addWidget(label_absbase, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.absbase, 4, 1, 1, 1)
        gridlayout_main.addWidget(label_bthres, 5, 0, 1, 1)
        gridlayout_main.addWidget(self.basethres, 5, 1, 1, 1)
        gridlayout_main.addWidget(helpdocs, 6, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 6, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_calcbase.pressed.connect(self.calcbase)

    def settings(self, test=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Line' not in self.indata:
            print('No Line Data')
            return False
        if self.indata['Line'].dataid != 'Gravity':
            print('Not Gravity Data')
            return False

        if not test:
            tmp = self.exec_()
        else:
            tmp = 1

        try:
            float(self.density.text())
            float(self.absbase.text())
            float(self.basethres.text())
            float(self.knownbase.text())
            float(self.knownstat.text())
        except ValueError:
            print('Value Error')
            return False

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        newdat = self.indata['Line']

        dat = self.indata['Line'].data

        basethres = float(self.basethres.text())
        kstat = float(self.knownstat.text())

# Convert multiple lines to single dataset.
        pdat = None
        for i in dat:
            if pdat is None:
                pdat = dat[i]
                continue
            pdat = np.append(pdat, dat[i])

# Make sure there are no local base stations before the known base
        if kstat in pdat['STATION']:
            tmp = (pdat['STATION'] == kstat)
            itmp = np.nonzero(tmp)[0][0]
            pdat = pdat[itmp:]

# Drift Correction, to abs base value
        tmp = pdat[pdat['STATION'] > basethres]

        driftdat = tmp[tmp['STATION'] != kstat]
        pdat = pdat[pdat['STATION'] < basethres]

        x = pdat['DECTIMEDATE']
        xp = driftdat['DECTIMEDATE']
        fp = driftdat['GRAV']

        dcor = np.interp(x, xp, fp)

        drifttime = (x[-1]-x[0])*24*60
        driftrate = (dcor[-1]-dcor[0])/drifttime  # per day
        print('QC - survey time (mins)', drifttime)
        print('QC - survey drift (mGal/min):', driftrate)

        gobs = pdat['GRAV'] - dcor + float(self.absbase.text())

# Variables used
        lat = np.deg2rad(pdat[newdat.ychannel])
        h = pdat[newdat.zchannel]  # This is the ellipsoidal (gps) height
        dens = float(self.density.text())

# Corrections
        gT = theoretical_gravity(lat)
        gATM = atmospheric_correction(h)
        gHC = height_correction(lat, h)
        gSB = spherical_bouguer(h, dens)

# Bouguer Anomaly
        gba = gobs - gT + gATM - gHC - gSB  # add or subtract atm

#        pdat = nplrf.append_fields(pdat, 'DRIFT', dcor, usemask=False)
        pdat = nplrf.append_fields(pdat, 'gobs(drift)', gobs, usemask=False)
        pdat = nplrf.append_fields(pdat, 'gT', gT, usemask=False)
        pdat = nplrf.append_fields(pdat, 'gATM', gATM, usemask=False)
        pdat = nplrf.append_fields(pdat, 'gHC', gHC, usemask=False)
        pdat = nplrf.append_fields(pdat, 'gSB', gSB, usemask=False)
        pdat = nplrf.append_fields(pdat, 'BOUGUER', gba, usemask=False)

        dat2a = {}
        for i in dat:
            tmp = pdat[pdat['LINE'] == float(i)]
            if tmp.size > 0:
                dat2a[i] = pdat[pdat['LINE'] == float(i)]

        dat2 = LData()

        dat2.xchannel = newdat.xchannel
        dat2.ychannel = newdat.ychannel
        dat2.zchannel = newdat.zchannel
        dat2.data = dat2a
        dat2.dataid = 'Gravity'

        self.outdata['Line'] = dat2

    def calcbase(self):
        """
        Calculate local base station value.

        Ties in the local base station to a known absolute base station.

        Returns
        -------
        None.


        """
        dat = self.indata['Line'].data
        basethres = float(self.basethres.text())
        kstat = float(self.knownstat.text())

# Convert multiple lines to single dataset.
        pdat = None
        for i in dat:
            if pdat is None:
                pdat = dat[i]
                continue
            pdat = np.append(pdat, dat[i])

        if kstat not in pdat['STATION']:
            txt = ('Invalid base station number.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          txt, QtWidgets.QMessageBox.Ok)
            return

# Drift Correction, to abs base value
        tmp = pdat[pdat['STATION'] > basethres]
        kbasevals = tmp[tmp['STATION'] == kstat]
        abasevals = tmp[tmp['STATION'] != kstat]

        x = abasevals['DECTIMEDATE']
        grv = abasevals['GRAV']
        xp = kbasevals['DECTIMEDATE']
        fp = kbasevals['GRAV']

        filt = np.logical_and(x >= xp.min(), x <= xp.max())
        grv = grv[filt]
        x = x[filt]

        if x.size == 0:
            txt = ('Your known base values need to be before and after at '
                   'least one local base station value.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          txt, QtWidgets.QMessageBox.Ok)
            return

        absbase = grv-np.interp(x, xp, fp) + float(self.knownbase.text())
        self.absbase.setText(str(absbase[0]))


def geocentric_radius(lat):
    """
    Calculate the distance from the Earth's center to a point on the spheroid
    surface at a specified geodetic latitude.

    Parameters
    ----------
    lat : numpy array
        Latitude in radians

    Returns
    -------
    R : Numpy array
        Array of radii.

    """
    a = 6378137
    b = 6356752.314245

    R = np.sqrt(((a**2 * np.cos(lat))**2 + (b**2 * np.sin(lat))**2) /
                ((a * np.cos(lat))**2 + (b * np.sin(lat))**2))

    return R


def theoretical_gravity(lat):
    """
    Calculate the theoretical gravity.

    Parameters
    ----------
    lat : numpy array
        Latitude in radians

    Returns
    -------
    gT : numpy array
        Array of theoretrical gravity values.

    """

    gT = 978032.67715*((1 + 0.001931851353 * np.sin(lat)**2) /
                       np.sqrt(1 - 0.0066943800229*np.sin(lat)**2))

    return gT


def atmospheric_correction(h):
    """
    Calculate the atmospheric correction.

    Parameters
    ----------
    h : numpy array
        Heights relative to elipsoid (GPS heights).

    Returns
    -------
    gATM : numpy array.
        Atmospheric correction

    """

    gATM = 0.874-9.9*1e-5*h+3.56*1e-9*h**2

    return gATM


def height_correction(lat, h):
    """
    Calculate height correction.

    Parameters
    ----------
    lat : numpy array
        Latitude in radians.
    h : numpy array
        Heights relative to elipsoid (GPS heights).

    Returns
    -------
    gHC : numpy array
        Height corrections

    """

    gHC = -(0.308769109-0.000439773*np.sin(lat)**2)*h+7.2125*1e-8*h**2

    return gHC


def spherical_bouguer(h, dens):
    """


    Parameters
    ----------
    h : numpy array
        Heights relative to elipsoid (GPS heights).
    dens : float
        Density.

    Returns
    -------
    gSB : numpy array
        Spherical Bouguer correction.

    """
    S = 166700  # Bullard B radius
    R0 = 6371000  # Mean radius of the earth
    G = 6.67384*1e-11

    alpha = S/R0
    R = R0 + h

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

    return gSB
