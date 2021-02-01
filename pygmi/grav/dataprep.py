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

import sys
from PyQt5 import QtWidgets, QtCore
import numpy as np
import matplotlib.pyplot as plt
import pygmi.menu_default as menu_default
import pygmi.grav.iodefs as iodefs


class ProcessData(QtWidgets.QDialog):
    """
    Process Gravity Data.

    This class processes gravity data.

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
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        self.dsb_dxy = QtWidgets.QDoubleSpinBox()
        self.dsb_null = QtWidgets.QDoubleSpinBox()
        self.dataid = QtWidgets.QComboBox()
        self.density = QtWidgets.QLineEdit('2670')
        self.knownstat = QtWidgets.QLineEdit('None')
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
        label_absbase = QtWidgets.QLabel('Base Station Absolute Gravity '
                                         '(mGal):')
        label_bthres = QtWidgets.QLabel('Minimum Base Station Number:')
        label_kstat = QtWidgets.QLabel('Known Base Station Number:')
        label_kbase = QtWidgets.QLabel('Known Base Station Absolute Gravity '
                                       '(mGal):')
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

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if 'Line' not in self.indata:
            self.showprocesslog('No Line Data')
            return False
        if 'Gravity' not in self.indata['Line']:
            self.showprocesslog('Not Gravity Data')
            return False

        if not nodialog:
            tmp = self.exec_()
        else:
            tmp = 1

        try:
            float(self.density.text())
            float(self.absbase.text())
            float(self.basethres.text())
            float(self.knownbase.text())
        except ValueError:
            self.showprocesslog('Value Error')
            return False

        if self.knownstat.text() != 'None':
            try:
                float(self.knownstat.text())
            except ValueError:
                self.showprocesslog('Value Error')
                return False

        if tmp != 1:
            return False

        self.acceptall(nodialog)

        return True

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

        self.dsb_dxy.setValue(projdata['dxy'])
        self.dsb_null.setValue(projdata['null'])
        self.dataid.setCurrentText(projdata['dataid'])
        self.density.setText(projdata['density'])
        self.knownstat.setText(projdata['knownstat'])
        self.knownbase.setText(projdata['knownbase'])
        self.absbase.setText(projdata['absbase'])
        self.basethres.setText(projdata['basethres'])

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

        projdata['dxy'] = self.dsb_dxy.value()
        projdata['null'] = self.dsb_null.value()
        projdata['dataid'] = self.dataid.currentText()
        projdata['density'] = self.density.text()
        projdata['knownstat'] = self.knownstat.text()
        projdata['knownbase'] = self.knownbase.text()
        projdata['absbase'] = self.absbase.text()
        projdata['basethres'] = self.basethres.text()

        return projdata

    def acceptall(self, nodialog):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        pdat = self.indata['Line']['Gravity']
#        pdat.sort_values(by=['DECTIMEDATE'], inplace=True)

        basethres = float(self.basethres.text())
        kstat = self.knownstat.text()
        if kstat == 'None':
            kstat = -1.0
        else:
            kstat = float(kstat)

# Make sure there are no local base stations before the known base
        if kstat in pdat['STATION']:
            tmp = (pdat['STATION'] == kstat)
            itmp = np.nonzero(tmp)[0][0]
            pdat = pdat[itmp:]

# Drift Correction, to abs base value
        tmp = pdat[pdat['STATION'] >= basethres]

        driftdat = tmp[tmp['STATION'] != kstat]
        pdat = pdat[pdat['STATION'] < basethres]

#        x = pdat['DECTIMEDATE'].values
#        xp1 = driftdat['DECTIMEDATE'].values
        xp1 = driftdat['TIME'].apply(time_convert)

        fp = driftdat['GRAV'].values

        x = pdat.index.values
        xp = driftdat.index.values

        dcor = np.interp(x, xp, fp)

        self.showprocesslog('Quality Control')
        self.showprocesslog('---------------')
        tmp = driftdat['DECTIMEDATE'].values.astype(int)
        tmp2 = []
        ix = []
        tcnt = 0
        for i, val in enumerate(tmp[:-1]):
            tmp2.append(tcnt)
            if tmp[i+1] != val:
                ix += tmp2
                tmp2 = []
                tcnt += 1
        tmp2.append(tcnt)
        ix += tmp2

        drate = []
        dtime = []
        dday = []
        for iday in np.unique(ix):
            filt = (ix == iday)
            x2 = xp1[filt].values/60.
            dcor2 = fp[filt]
            drifttime = (x2[-1]-x2[0])
            driftrate = (dcor2[-1]-dcor2[0])/drifttime
            self.showprocesslog(f'Day {iday+1} drift: {driftrate:.3e} '
                                f'mGal/min over {drifttime:.3f} minutes.')
            dday.append(iday+1+x2[-1]/1440)
            drate.append(driftrate)
            dtime.append(drifttime)

        xp2 = xp1/86400 + ix+1

        if not nodialog:
            plt.figure('QC: Gravimeter Drift')
            plt.subplot(2, 1, 1)
            plt.xlabel('Decimal Days')
            plt.ylabel('mGal')
            plt.grid(True)
            plt.plot(xp2, fp, '.-')
            plt.xticks(range(1, ix[-1]+2, 1))
            ax = plt.gca()

            plt.subplot(2, 1, 2, sharex=ax)
            plt.xlabel('Decimal Days')
            plt.ylabel('mGal/min')
            plt.grid(True)
            plt.plot(dday, drate, '.-')
            # plt.xticks(range(1, ix[-1]+2, 1))
            plt.tight_layout()

            plt.get_current_fig_manager().window.setWindowIcon(self.parent.windowIcon())
            plt.show()

        gobs = pdat['GRAV'] - dcor + float(self.absbase.text())
###################################################################

# Variables used
        lat = np.deg2rad(pdat.latitude)
        h = pdat['elevation']  # This is the ellipsoidal (gps) height
        dens = float(self.density.text())

# Corrections
        gT = theoretical_gravity(lat)
        gATM = atmospheric_correction(h)
        gHC = height_correction(lat, h)
        gSB = spherical_bouguer(h, dens)

# Bouguer Anomaly
        gba = gobs - gT + gATM - gHC - gSB  # add or subtract atm

        pdat = pdat.assign(dcor=dcor)
        pdat = pdat.assign(gobs_drift=gobs)
        pdat = pdat.assign(gT=gT)
        pdat = pdat.assign(gATM=gATM)
        pdat = pdat.assign(gHC=gHC)
        pdat = pdat.assign(gSB=gSB)
        pdat = pdat.assign(BOUGUER=gba)

        pdat.sort_values(by=['LINE', 'STATION'], inplace=True)

        self.outdata['Line'] = {'Gravity': pdat}

    def calcbase(self):
        """
        Calculate local base station value.

        Ties in the local base station to a known absolute base station.

        Returns
        -------
        None.


        """
        pdat = self.indata['Line']['Gravity']

        basethres = float(self.basethres.text())

        if self.knownstat.text() == 'None':
            txt = ('Invalid base station number.')
            QtWidgets.QMessageBox.warning(self.parent, 'Error',
                                          txt, QtWidgets.QMessageBox.Ok)
            return

        kstat = float(self.knownstat.text())
        if kstat not in pdat['STATION'].values:
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
        self.absbase.setText(str(absbase.iloc[0]))


def geocentric_radius(lat):
    """
    Geocentric radius calculation.

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
    Calculate spherical bouguer.

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


def time_convert(x):
    """
    Convert hh:mm:ss to seconds.

    Parameters
    ----------
    x : str
        Time in hh:mm:ss.

    Returns
    -------
    float
        Time in seconds.

    """
    h, m, s = map(int, x.decode().split(':'))
    return (h*60+m)*60+s


def test():
    """Test routine."""
    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    grvfile = r'C:\Work\Workdata\gravity\skeifontein 2018.txt'
    gpsfile = r'C:\Work\Workdata\gravity\skei_dgps.csv'

    # grvfile = r'C:\Work\Workdata\gravity\Laxeygarvity until2511.txt'
    # gpsfile = r'C:\Work\Workdata\gravity\laxey.dgps.csv'

# Import Data
    IO = iodefs.ImportCG5(None)
    IO.get_cg5(grvfile)
    IO.get_gps(gpsfile)
    IO.settings(True)

# Process Data
    PD = ProcessData()
    PD.indata = IO.outdata
    PD.knownstat.setText('88888.0')
    PD.knownbase.setText('978864.74')
    PD.calcbase()

    PD.settings(False)

    datout = PD.outdata['Line']

    gdf = datout['Gravity']

    gdf = gdf[(gdf.STATION >4470) & (gdf.STATION<4472)]
#    gdf = gdf[(gdf.STATION >2213) & (gdf.STATION<2900)]

    plt.plot(gdf.longitude, gdf.latitude, '.')
    plt.show()


if __name__ == "__main__":
    test()

    print('Finished!')
