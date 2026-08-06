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
"""A set of data processing routines for gravity."""

import sys

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from pygmi.maggrv import iodefs
from pygmi.misc import BasicModule, ContextModule


class MyMplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for the actual plot."""

    def __init__(self):
        fig = Figure(layout="tight")
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

    def update_raster(self, drift):
        """
        Update the raster plot.

        Parameters
        ----------
        drift : dict
            Dictionary containing information for drift plots.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.figure.suptitle("QC: Gravimeter Drift")

        axes = self.figure.add_subplot(211)
        axes.set_xlabel("Decimal Days")
        axes.set_ylabel("mGal")
        axes.grid(True)
        axes.plot(drift["xp2"], drift["fp"], ".-")
        axes.set_xticks(range(1, drift["ix"][-1] + 2, 1))
        ax = axes

        axes = self.figure.add_subplot(212, sharex=ax)
        axes.set_xlabel("Decimal Days")
        axes.set_ylabel("mGal/min")
        axes.grid(True)
        axes.plot(drift["dday"], drift["drate"], ".-")

        self.figure.canvas.draw()


class PlotDrift(ContextModule):
    """
    Plot Raster Class.

    Parameters
    ----------
    parent : ProcessData, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None, data=None):
        super().__init__(parent)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Drift Plot")

        vbl = QtWidgets.QVBoxLayout(self)
        self.mmc = MyMplCanvas()
        mpl_toolbar = NavigationToolbar2QT(self.mmc)

        self.buttonbox.hide()
        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)

        self.setFocus()

        self.show()
        if data is not None:
            self.mmc.update_raster(data)


class ProcessData(BasicModule):
    """
    Process Gravity Data.

    This class processes gravity data.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.le_density = QtWidgets.QLineEdit("2670")
        self.le_knownstat = QtWidgets.QLineEdit()
        self.le_knownstat.setPlaceholderText("None")
        self.le_knownbase = QtWidgets.QLineEdit("978000.0")
        self.le_absbase = QtWidgets.QLineEdit("978032.67715")
        self.le_basethres = QtWidgets.QLineEdit("10000")

        self.le_density.setValidator(QtGui.QDoubleValidator(self))
        self.le_knownstat.setValidator(self.qval)
        self.le_knownbase.setValidator(QtGui.QDoubleValidator(self))
        self.le_absbase.setValidator(QtGui.QDoubleValidator(self))
        self.le_basethres.setValidator(QtGui.QDoubleValidator(self))

        self.gdata = None

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gl_main = QtWidgets.QGridLayout(self)
        self.buttonbox.htmlfile = "maggrv.dm.processgravity"

        lbl_density = QtWidgets.QLabel("Background Density (kg/m<sup>3</sup>):")
        lbl_absbase = QtWidgets.QLabel("Local Base Station Absolute Gravity (mGal):")
        lbl_bthres = QtWidgets.QLabel("Minimum Base Station Number:")
        lbl_kstat = QtWidgets.QLabel("Known Base Station Number:")
        lbl_kbase = QtWidgets.QLabel("Known Base Station Absolute Gravity (mGal):")
        pb_calcbase = QtWidgets.QPushButton("Calculate local base value")

        self.setWindowTitle("Gravity Data Processing")

        gl_main.addWidget(lbl_kstat, 0, 0, 1, 1)
        gl_main.addWidget(self.le_knownstat, 0, 1, 1, 1)
        gl_main.addWidget(lbl_kbase, 1, 0, 1, 1)
        gl_main.addWidget(self.le_knownbase, 1, 1, 1, 1)

        gl_main.addWidget(pb_calcbase, 2, 0, 1, 2)

        gl_main.addWidget(lbl_density, 3, 0, 1, 1)
        gl_main.addWidget(self.le_density, 3, 1, 1, 1)
        gl_main.addWidget(lbl_absbase, 4, 0, 1, 1)
        gl_main.addWidget(self.le_absbase, 4, 1, 1, 1)
        gl_main.addWidget(lbl_bthres, 5, 0, 1, 1)
        gl_main.addWidget(self.le_basethres, 5, 1, 1, 1)
        gl_main.addWidget(self.buttonbox, 6, 0, 1, 4)

        pb_calcbase.pressed.connect(self.calcbase)

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
        self.gdata = None
        tmp = []
        if "Vector" not in self.indata:
            self.showlog("No Line Data")
            return False

        for i in self.indata["Vector"]:
            if "Gravity" in i.attrs:
                self.gdata = i
                break

        if self.gdata is None:
            self.showlog("Not Gravity Data")
            return False

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        if not self.check_validation():
            return False

        self.acceptall(nodialog)

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.le_density)
        self.saveobj(self.le_knownstat)
        self.saveobj(self.le_knownbase)
        self.saveobj(self.le_absbase)
        self.saveobj(self.le_basethres)

    def acceptall(self, nodialog):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        pdat = self.gdata
        basethres = float(self.le_basethres.text())
        kstat = self.le_knownstat.text()
        absbase = float(self.le_absbase.text())
        dens = float(self.le_density.text())

        pdat, drift = gravcor(pdat, basethres, kstat, absbase, dens, self.showlog)
        self.outdata["Vector"] = [pdat]

        if not nodialog:
            ptest = PlotDrift(data=drift)
            ptest.exec()

    def calcbase(self):
        """
        Calculate local base station value.

        Ties in the local base station to a known absolute base station.

        Returns
        -------
        None.


        """
        pdat = self.gdata

        basethres = float(self.le_basethres.text())

        if self.le_knownstat.text() == "None":
            txt = "Invalid base station number."
            QtWidgets.QMessageBox.warning(
                self.parent, "Error", txt, QtWidgets.QMessageBox.StandardButton.Ok
            )
            return

        kstat = float(self.le_knownstat.text())
        if kstat not in pdat["STATION"].values:
            txt = "Invalid base station number."
            QtWidgets.QMessageBox.warning(
                self.parent, "Error", txt, QtWidgets.QMessageBox.StandardButton.Ok
            )
            return

        # Drift Correction, to abs base value
        tmp = pdat[pdat["STATION"] > basethres]
        kbasevals = tmp[tmp["STATION"] == kstat]
        abasevals = tmp[tmp["STATION"] != kstat]

        if tmp.STATION.unique()[0] == kstat and tmp.STATION.unique().size == 1:
            abasevals = kbasevals

        x = abasevals["DECTIMEDATE"]
        grv = abasevals["GRAV"]
        xp = kbasevals["DECTIMEDATE"]
        fp = kbasevals["GRAV"]

        filt = np.logical_and(x >= xp.min(), x <= xp.max())
        grv = grv[filt]
        x = x[filt]

        if x.size == 0:
            txt = (
                "Your known base values need to be before and after at "
                "least one local base station value."
            )
            QtWidgets.QMessageBox.warning(
                self.parent, "Error", txt, QtWidgets.QMessageBox.StandardButton.Ok
            )
            return

        absbase = grv - np.interp(x, xp, fp) + float(self.le_knownbase.text())
        self.le_absbase.setText(str(absbase.iloc[0]))


def gravcor(pdat, basethres, kstat="", absbase=978032.67715, dens=2670, showlog=print):
    """
    Gravity corrections.

    Parameters
    ----------
    pdat : Pandas DataFrame
        Gravity data.
    basethres : float
       Threshold for base station numbers.
    kstat : float
        Known base station number, by default 'None'
    absbase : float
        Known base station absolute gravity, by default 978032.67715
    dens : float
        Background Density (kg/m3), by default 2670
    showlog : function, optional
        Display information. The default is print.

    Returns
    -------
    pdat : Pandas DataFrame
        Gravity data.
    drift : dict
        Dictionary containing information for drift plots.

    """

    pdat.sort_values(by=["DECTIMEDATE"], inplace=True)

    if kstat == "":
        kstat = -1.0
    else:
        kstat = float(kstat)

    # Make sure there are no local base stations before the known base
    if kstat in pdat["STATION"]:
        tmp = pdat["STATION"] == kstat
        itmp = np.nonzero(tmp)[0][0]
        pdat = pdat[itmp:]

    # Drift Correction, to abs base value
    tmp = pdat[pdat["STATION"] >= basethres]

    driftdat = tmp[tmp["STATION"] != kstat]
    pdat = pdat[pdat["STATION"] < basethres]

    if tmp.STATION.unique()[0] == kstat and tmp.STATION.unique().size == 1:
        driftdat = tmp[tmp["STATION"] == kstat]

    xp1 = driftdat["TIME"].apply(time_convert)

    fp = driftdat["GRAV"].values

    x = pdat["DECTIMEDATE"].values
    xp = driftdat["DECTIMEDATE"].values

    dcor = np.interp(x, xp, fp)

    showlog("Quality Control")
    showlog("---------------")
    tmp = driftdat["DECTIMEDATE"].values.astype(int)
    tmp2 = []
    ix = []
    tcnt = 0
    for i, val in enumerate(tmp[:-1]):
        tmp2.append(tcnt)
        if tmp[i + 1] != val:
            ix += tmp2
            tmp2 = []
            tcnt += 1
    tmp2.append(tcnt)
    ix += tmp2

    drate = []
    dtime = []
    dday = []
    for iday in np.unique(ix):
        filt = ix == iday
        x2 = xp1[filt].values / 60.0
        dcor2 = fp[filt]
        drifttime = x2[-1] - x2[0]
        if drifttime == 0.0:
            showlog(
                f"Day {iday + 1} drift: Only one reading, no drift result possible."
            )
            driftrate = np.nan

        else:
            driftrate = (dcor2[-1] - dcor2[0]) / drifttime
            showlog(
                f"Day {iday + 1} drift: {driftrate:.3e} "
                f"mGal/min over {drifttime:.3f} minutes."
            )
        dday.append(iday + 1 + x2[-1] / 1440)
        drate.append(driftrate)
        dtime.append(drifttime)

    xp2 = xp1 / 86400 + ix + 1

    gobs = pdat["GRAV"] - dcor + float(absbase)

    # Variables used
    lat = np.deg2rad(pdat.latitude)
    h = pdat["elevation"]  # This is the ellipsoidal (gps) height

    # Corrections
    gT = theoretical_gravity(lat)
    gATM = atmospheric_correction(h)
    gHC = height_correction(lat, h)
    gSB = spherical_bouguer(h, dens)

    # Bouguer Anomaly
    gba = gobs - gT + gATM - gHC - gSB

    pdat = pdat.assign(dcor=dcor)
    pdat = pdat.assign(gobs_drift=gobs)
    pdat = pdat.assign(gT=gT)
    pdat = pdat.assign(gATM=gATM)
    pdat = pdat.assign(gHC=gHC)
    pdat = pdat.assign(gSB=gSB)
    pdat = pdat.assign(BOUGUER=gba)

    pdat.sort_values(by=["LINE", "STATION"], inplace=True)

    drift = {"xp2": xp2, "fp": fp, "ix": ix, "dday": dday, "drate": drate}

    return pdat, drift


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

    R = np.sqrt(
        ((a**2 * np.cos(lat)) ** 2 + (b**2 * np.sin(lat)) ** 2)
        / ((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2)
    )

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
        Array of theoretical gravity values.

    """
    gT = 978032.67715 * (
        (1 + 0.001931851353 * np.sin(lat) ** 2)
        / np.sqrt(1 - 0.0066943800229 * np.sin(lat) ** 2)
    )

    return gT


def atmospheric_correction(h):
    """
    Calculate the atmospheric correction.

    Parameters
    ----------
    h : numpy array
        Heights relative to ellipsoid (GPS heights).

    Returns
    -------
    gATM : numpy array.
        Atmospheric correction

    """
    gATM = 0.874 - 9.9 * 1e-5 * h + 3.56 * 1e-9 * h**2

    return gATM


def height_correction(lat, h):
    """
    Calculate height correction.

    Parameters
    ----------
    lat : numpy array
        Latitude in radians.
    h : numpy array
        Heights relative to ellipsoid (GPS heights).

    Returns
    -------
    gHC : numpy array
        Height corrections

    """
    gHC = -(0.308769109 - 0.000439773 * np.sin(lat) ** 2) * h + 7.2125 * 1e-8 * h**2

    return gHC


def spherical_bouguer(h, dens):
    """
    Calculate spherical Bouguer.

    Parameters
    ----------
    h : numpy array
        Heights relative to ellipsoid (GPS heights).
    dens : float
        Density.

    Returns
    -------
    gSB : numpy array
        Spherical Bouguer correction.

    """
    S = 166700  # Bullard B radius
    R0 = 6371000  # Mean radius of the earth
    G = 6.67384 * 1e-11

    alpha = S / R0
    R = R0 + h

    delta = R0 / R
    eta = h / R
    d = 3 * np.cos(alpha) ** 2 - 2
    f = np.cos(alpha)
    k = np.sin(alpha) ** 2
    p = -6 * np.cos(alpha) ** 2 * np.sin(alpha / 2) + 4 * np.sin(alpha / 2) ** 3
    m = -3 * np.sin(alpha) ** 2 * np.cos(alpha)
    n = 2 * (np.sin(alpha / 2) - np.sin(alpha / 2) ** 2)  # is this abs?
    mu = 1 + eta**2 / 3 - eta

    fdk = np.sqrt((f - delta) ** 2 + k)
    t1 = (d + f * delta + delta**2) * fdk
    t2 = m * np.log(n / (f - delta + fdk))

    lamda = (t1 + p + t2) / 3

    gSB = 2 * np.pi * G * dens * (mu * h - lamda * R) * 1e5

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
    try:
        h, m, s = map(int, x.decode().split(":"))
    except AttributeError:
        h, m, s = map(int, x.split(":"))
    return (h * 60 + m) * 60 + s


def _testfn():
    """Test routine."""
    import matplotlib.pyplot as plt

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    grvfile = r"D:\workdata\PyGMI Test Data\Gravity\Skeifontein 2018.txt"
    gpsfile = r"D:\workdata\PyGMI Test Data\Gravity\Skei_DGPS.csv"
    # grvfile = r"D:\workdata\PyGMI Test Data\Gravity\CG-6_Gravity Data for Dr Cole.txt"
    # gpsfile = r"D:\workdata\PyGMI Test Data\Gravity\Kuruman DGPS_for Dr Cole.csv"
    kbase = "88888"
    bthres = "10000"

    # Import Data
    IO = iodefs.ImportCG5(None)
    IO.le_basethres.setText(bthres)
    IO.get_cg5(grvfile)
    IO.get_gps(gpsfile)
    IO.settings()

    # Process Data
    PD = ProcessData()
    PD.indata = IO.outdata
    PD.le_basethres.setText(bthres)
    PD.le_knownstat.setText(kbase)
    PD.le_knownbase.setText("978794.53")

    PD.settings()

    out = PD.outdata["Vector"][0]

    out.plot("BOUGUER")
    plt.show()

    pass


def _test_lacoste():
    """Test for lacoste data"""
    import numpy as np
    import pandas as pd

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    gfile = r"D:\Steph\Preprocessed_Gravity.csv"
    cfile = r"D:\Steph\Calibration Table.csv"
    ofile = r"D:\Steph\out.csv"

    dfg = pd.read_csv(gfile)
    dfc = pd.read_csv(cfile)

    xp = dfc["Counter Reading"]
    fp = dfc["Value in Milligals"]
    x = dfg["counter"]

    dfg["GRAV"] = np.interp(x, xp, fp)

    dfg["STATION"] = dfg["station"].replace("Base", "88888")
    del dfg["station"]
    dfg["STATION"] = dfg["STATION"].astype(int)
    dfg["LINE"] = dfg["traverse"].astype(int)
    dfg["STATION"] = dfg["STATION"] + 1000 * dfg["LINE"]

    dfg["TIME"] = dfg["time"].apply(lambda x: x.encode("utf-8"))

    dfg["datetime"] = pd.to_datetime(dfg["datetime"])
    dfg["DECTIMEDATE"] = dfg["datetime"].astype(int) // 10**9
    dfg["DECTIMEDATE"] = dfg["DECTIMEDATE"] / (24 * 3600)

    # kbase = '88888'
    bthres = 10000

    out, _ = gravcor(dfg, bthres)

    out.to_csv(ofile, index=False)


if __name__ == "__main__":
    _testfn()
    # _test_lacoste()
