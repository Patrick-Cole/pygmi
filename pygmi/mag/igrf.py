# -----------------------------------------------------------------------------
# Name:        igrf.py (part of PyGMI)
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
"""
IGRF calculations.

This code is based on the Geomag software, with information given below. It was
translated into Python from the Geomag code.

| This program, originally written in FORTRAN, was developed using subroutines
| written by   : A. Zunde
|                USGS, MS 964, Box 25046 Federal Center, Denver, Co.  80225
|                and
|                S.R.C. Malin & D.R. Barraclough
|                Institute of Geological Sciences, United Kingdom.

| Translated
| into C by    : Craig H. Shaffer
|                29 July, 1988

| Rewritten by : David Owens
|                For Susan McLean

| Maintained by: Stefan Maus
| Contact      : stefan.maus@noaa.gov
|                National Geophysical Data Center
|                World Data Center-A for Solid Earth Geophysics
|                NOAA, E/GC1, 325 Broadway,
|                Boulder, CO  80303
"""

import os
from math import sin
from math import cos
from math import sqrt
from math import atan2
import copy

from numba import jit
from PyQt5 import QtWidgets, QtCore
import numpy as np
from osgeo import osr

import pygmi.raster.dataprep as dp
import pygmi.menu_default as menu_default
from pygmi.misc import ProgressBarText


class IGRF(QtWidgets.QDialog):
    """
    IGRF field calculation.

    This produces two datasets. The first is an IGRF dataset for the area of
    interest, defined by some input magnetic dataset. The second is the IGRF
    corrected form of that input magnetic dataset.

    To do this, the input dataset must be reprojected from its local projection
    to degrees, where the IGRF correction will take place. This is done within
    this class.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets

    Parameters
    ----------
    altmin : Double
        Minimum height of selected model.
    altmax : Double array
        array of MAXMOD Maximum height of model.
    maxalt : Double
        Maximum height of selected model.
    d : float
        Declination of the field from the geographic north (deg).
    sdate : float
        start date inputted
    ddot : float
        annual rate of change of decl. (arc-min/yr)
    alt : float
        altitude above WGS84 Ellipsoid
    epoch : list
        list of MAXMOD epoch of model.
    latitude : float
        Latitude.
    longitude : float
        Longitude.
    gh : numpy array
        Schmidt quasi-normal internal spherical harmonic coeff.
        Schmidt quasi-normal internal spherical harmonic coeff.
        Coefficients of resulting model.
        Coefficients of rate of change model.
    i : float
        Inclination (deg).
    idot : float
        Rate of change of i (arc-min/yr).
    igdgc : int
        Flag for geodetic or geocentric coordinate choice.
    irec_pos : int array
        array of MAXMOD Record counter for header
    fileline : int
        Current line in file (for errors)
    max1 : list, int
        array of MAXMOD Main field coefficient.
    max2 : list, int
        array of MAXMOD Secular variation coefficient.
    max3 : list, int
        array of MAXMOD Acceleration coefficient.
    minyr : float
        Min year of all models
    maxyr : float
        Max year of all models
    yrmax : list, float
        array of MAXMOD  Max year of model.
    yrmin : list, float
        array of MAXMOD  Min year of model.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.wkt = None
        if parent is None:
            self.piter = ProgressBarText().iter
        else:
            self.piter = self.parent.pbar.iter

        self.dsb_alt = QtWidgets.QDoubleSpinBox()
        self.dateedit = QtWidgets.QDateEdit()
        self.combobox_dtm = QtWidgets.QComboBox()
        self.combobox_mag = QtWidgets.QComboBox()
        self.proj = dp.GroupProj('Input Projection')

        self.setupui()

        self.ctrans = None

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.raster.igrf')

        label_0 = QtWidgets.QLabel('Sensor clearance above ground')
        label_1 = QtWidgets.QLabel('Date')
        label_2 = QtWidgets.QLabel('Digital Elevation Model')
        label_3 = QtWidgets.QLabel('Magnetic Data')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.dsb_alt.setMaximum(99999.9)

        self.setWindowTitle('IGRF')

        gridlayout.addWidget(self.proj, 0, 0, 1, 2)
        gridlayout.addWidget(label_0, 2, 0, 1, 1)
        gridlayout.addWidget(self.dsb_alt, 2, 1, 1, 1)
        gridlayout.addWidget(label_1, 3, 0, 1, 1)
        gridlayout.addWidget(self.dateedit, 3, 1, 1, 1)
        gridlayout.addWidget(label_2, 4, 0, 1, 1)
        gridlayout.addWidget(self.combobox_dtm, 4, 1, 1, 1)
        gridlayout.addWidget(label_3, 5, 0, 1, 1)
        gridlayout.addWidget(self.combobox_mag, 5, 1, 1, 1)
        gridlayout.addWidget(buttonbox, 6, 1, 1, 1)
        gridlayout.addWidget(helpdocs, 6, 0, 1, 1)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

    def acceptall(self):
        """
        Accept option.

        Returns
        -------
        None.

        """
        orig_wkt = self.proj.wkt

        orig = osr.SpatialReference()
        orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        if orig_wkt == '':
            orig.SetWellKnownGeogCS('WGS84')
        else:
            orig.ImportFromWkt(orig_wkt)

        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')
        targ.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        self.ctrans = osr.CoordinateTransformation(orig, targ)

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
        if self.wkt is None:
            self.wkt = self.indata['Raster'][0].wkt

        self.proj.set_current(self.wkt)

        # data = dp.merge(self.indata['Raster'])
        data = self.indata['Raster']

        self.combobox_dtm.clear()
        self.combobox_mag.clear()
        for i in data:
            self.combobox_dtm.addItem(i.dataid)
            self.combobox_mag.addItem(i.dataid)

        if len(data) > 1:
            self.combobox_dtm.setCurrentIndex(1)
            # nodialog = False
            # QtWidgets.QMessageBox.warning(self.parent, 'Warning',
            #                               'Please confirm raster bands.',
            #                               QtWidgets.QMessageBox.Ok)

        if not nodialog:
            tmp = self.exec_()

            if tmp == 0:
                return False

        self.acceptall()

        i = self.combobox_mag.currentIndex()

        for i in data:
            if i.dataid == self.combobox_mag.currentText():
                dxy = min(i.xdim, i.ydim)

        data = dp.merge(data, dxy=dxy, piter=self.piter,
                        pprint=self.showprocesslog)

        for i in data:
            if i.dataid == self.combobox_mag.currentText():
                maggrid = i
            if i.dataid == self.combobox_dtm.currentText():
                data = i
                wkt = i.wkt

        sdate = self.dateedit.date()
        sdate = sdate.year()+sdate.dayOfYear()/sdate.daysInYear()
        alt = self.dsb_alt.value()

        odata, fmean, imean, dmean = calc_igrf(data, sdate, alt=alt, wkt=wkt,
                                               igrfonly=False,
                                               piter=self.piter,
                                               pprint=self.showprocesslog)
        bname = 'Magnetic Data: IGRF Corrected '
        bname = bname + f'F:{fmean:.2f} I:{imean:.2f} D:{dmean:.2f}'

        self.outdata['Raster'] = odata  # copy.deepcopy(self.indata['Raster'])

        self.outdata['Raster'].append(copy.deepcopy(maggrid))
        self.outdata['Raster'][-1].data -= odata[0].data
        self.outdata['Raster'][-1].dataid = bname
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
        self.wkt = projdata['wkt']
        self.dsb_alt.setValue(projdata['alt'])
        date = self.dateedit.date().fromString(projdata['date'])
        self.dateedit.setDate(date)

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

        projdata['wkt'] = self.proj.wkt
        projdata['alt'] = self.dsb_alt.value()
        projdata['date'] = self.dateedit.date().toString()
        # projdata['dtm'] = self.combobox_dtm.currentText()
        # projdata['mag'] = self.combobox_mag.currentText()

        return projdata


def calc_igrf(data, sdate, alt=100, wkt=None, igrfonly=True, piter=iter,
              pprint=print):
    """
    Calculate IGRF.

    Parameters
    ----------
    nodialog : bool, optional
        Run settings without a dialog. The default is False.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """

    MAXDEG = 13
    MAXCOEFF = (MAXDEG*(MAXDEG+2)+1)

    gh = np.zeros([4, MAXCOEFF])
    # decl = 0
    # fint = 0
    # hint = 0
    # incl = 0
    # x = 0
    # y = 0
    # z = 0

    # if wkt is None:
    #     wkt = data.wkt

    if wkt is not None:
        orig_wkt = wkt

        orig = osr.SpatialReference()
        orig.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        if orig_wkt == '':
            orig.SetWellKnownGeogCS('WGS84')
        else:
            orig.ImportFromWkt(orig_wkt)

        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')
        targ.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ctrans = osr.CoordinateTransformation(orig, targ)

    with open(os.path.join(os.path.dirname(__file__),
                           'IGRF13.cof')) as mdf:
        modbuff = mdf.readlines()
    fileline = -1                            # First line will be 1
    model = []
    epoch = []
    max1 = []
    max2 = []
    max3 = []
    yrmin = []
    yrmax = []
    altmin = []
    altmax = []
    irec_pos = []
# First model will be 0
    for i in modbuff:
        fileline += 1  # On new line
        if i[:3] == '   ':
            i2 = i.split()
            model.append(i2[0])
            epoch.append(float(i2[1]))
            max1.append(int(i2[2]))
            max2.append(int(i2[3]))
            max3.append(int(i2[4]))
            yrmin.append(float(i2[5]))
            yrmax.append(float(i2[6]))
            altmin.append(float(i2[7]))
            altmax.append(float(i2[8]))
            irec_pos.append(fileline)

    # dxy = min(data.xdim, data.ydim)
    altgrid = data.data.flatten() * 0.001  # in km

    maxyr = max(yrmax)

    drows, dcols = data.data.shape
    dtlx = data.extent[0]
    dtly = data.extent[-1]
    xrange = dtlx+data.xdim/2.+np.arange(dcols)*data.xdim
    yrange = dtly-data.ydim/2.-np.arange(drows)*data.ydim

    xdat, ydat = np.meshgrid(xrange, yrange)
    xdat = xdat.flatten()
    ydat = ydat.flatten()

    igrf_F = altgrid * 0
    igrf_I = altgrid * 0
    igrf_D = altgrid * 0
    # Pick model
    yrmax = np.array(yrmax)
    modelI = sum(yrmax < sdate)
    igdgc = 1

    if maxyr < sdate < maxyr+1:
        pprint('Warning: The date ' + str(sdate) + ' is out of range,')
        pprint('but still within one year of model expiration date.')
        pprint('An updated model file is available before 1.1.' + str(maxyr))

    if max2[modelI] == 0:
        gh = getshc(modbuff, 1, irec_pos[modelI], max1[modelI], 0, gh)
        gh = getshc(modbuff, 1, irec_pos[modelI+1], max1[modelI+1], 1, gh)
        nmax, gh = interpsh(sdate, yrmin[modelI], max1[modelI],
                            yrmin[modelI+1], max1[modelI+1], 2, gh)
        nmax, gh = interpsh(sdate+1, yrmin[modelI], max1[modelI],
                            yrmin[modelI+1], max1[modelI+1], 3, gh)
    else:
        gh = getshc(modbuff, 1, irec_pos[modelI], max1[modelI], 0, gh)
        gh = getshc(modbuff, 0, irec_pos[modelI], max2[modelI], 1, gh)
        nmax, gh = extrapsh(sdate, epoch[modelI], max1[modelI],
                            max2[modelI], 2, gh)
        nmax, gh = extrapsh(sdate+1, epoch[modelI], max1[modelI],
                            max2[modelI], 3, gh)

    for i in piter(range(xdat.size)):
        if igrf_F.mask[i]:
            continue

        if wkt is not None:
            longitude, latitude, _ = ctrans.TransformPoint(xdat[i], ydat[i])
        else:
            longitude, latitude = xdat[i], ydat[i]

        alt = altgrid[i]

# Do the first calculations
        x, y, z = shval3(igdgc, latitude, longitude, alt, nmax, 3, gh)
        _, fint, incl, decl = dihf(x, y, z)

        igrf_F[i] = fint
        igrf_I[i] = np.rad2deg(incl)
        igrf_D[i] = np.rad2deg(decl)

    igrf_F = np.ma.array(igrf_F)
    igrf_F.shape = data.data.shape
    igrf_F.mask = np.ma.getmaskarray(data.data)

    igrf_I = np.ma.array(igrf_I)
    igrf_I.shape = data.data.shape
    igrf_I.mask = np.ma.getmaskarray(data.data)

    igrf_D = np.ma.array(igrf_D)
    igrf_D.shape = data.data.shape
    igrf_D.mask = np.ma.getmaskarray(data.data)

    fmean = igrf_F.mean()
    imean = igrf_I.mean()
    dmean = igrf_D.mean()

    pprint('Mean Values in Calculation')
    pprint('=============================')
    pprint(f'Total Intensity: {fmean:.2f}')
    pprint(f'Inclination: {imean:.2f}')
    pprint(f'Declination: {dmean:.2f}')

    outdata = []
    outdata.append(copy.deepcopy(data))
    outdata[-1].data = igrf_F
    outdata[-1].dataid = 'IGRF'

    if not igrfonly:
        outdata.append(copy.deepcopy(data))
        outdata[-1].data = igrf_I
        outdata[-1].dataid = 'Inclination'

        outdata.append(copy.deepcopy(data))
        outdata[-1].data = igrf_D
        outdata[-1].dataid = 'Declination'

    return outdata, fmean, imean, dmean


def getshc(file, iflag, strec, nmax_of_gh, igh, gh):
    """
    Read spherical harmonic coefficients from the specified model.

    Reads spherical harmonic coefficients from the specified model into an
    array (Schmidt quasi-normal internal spherical harmonic coefficients).

    | FORTRAN: Bill Flanagan, NOAA CORPS, DESDIS, NGDC, 325 Broadway,
    | Boulder CO.  80301
    | C: C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

    Parameters
    ----------
    file : file
        reference to a file object
    iflag :
        Flag for SV equal to 1 or not equal to 1 for designated read
        statements
    strec : int
        Starting record number to read from model
    nmax_of_gh : int
        Maximum degree and order of model
    igh : int
        Index for gh.

    Returns
    -------
    None.
    """
    ii = -1
    cnt = 0

    for nn in range(1, nmax_of_gh+1):
        for _ in range(nn+1):
            cnt += 1
            tmp = file[strec+cnt]
            tmp = tmp.split()
            m = int(tmp[1])

            if iflag == 1:
                g = float(tmp[2])
                hh = float(tmp[3])
            else:
                g = float(tmp[4])
                hh = float(tmp[5])

            ii = ii + 1
            gh[igh][ii] = g

            if m != 0:
                ii = ii + 1
                gh[igh][ii] = hh

    return gh


@jit(nopython=True)
def extrapsh(date, dte1, nmax1, nmax2, igh, gh):
    """
    Extrapolate a spherical harmonic model.

    Extrapolates linearly a spherical harmonic model with a rate-of-change
    model. Updates self.gh (Schmidt quasi-normal internal spherical
    harmonic coefficients).

    | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
    | CO. 80225
    | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

    Parameters
    ----------
    date : float
        date of resulting model (in decimal year)
    dte1 : float
        date of base model
    nmax1 : int
        maximum degree and order of base model
    nmax2 : int
        maximum degree and order of rate-of-change model
    gh  : numpy array
        Schmidt quasi-normal internal spherical harmonic coefficients of
        base model and rate-of-change model

    Returns
    -------
    nmax : int
        maximum degree and order of resulting model
    """
    factor = date - dte1
    if nmax1 == nmax2:
        k = nmax1 * (nmax1 + 2)
        nmax = nmax1
    else:
        if nmax1 > nmax2:
            k = nmax2 * (nmax2 + 2)
            l = nmax1 * (nmax1 + 2)
            for ii in range(k, l):
                gh[igh][ii] = gh[0][ii]

            nmax = nmax1
        else:
            k = nmax1 * (nmax1 + 2)
            l = nmax2 * (nmax2 + 2)
            for ii in range(k, l):
                gh[igh][ii] = factor * gh[1][ii]

            nmax = nmax2

    for ii in range(k):
        gh[igh][ii] = gh[0][ii] + factor * gh[1][ii]

    return nmax, gh


@jit(nopython=True)
def interpsh(date, dte1, nmax1, dte2, nmax2, igh, gh):
    """
    Temporal Interpolation between two spherical harmonic models.

    Interpolates linearly, in time, between two spherical harmonic
    models.

    Updates self.gh (Schmidt quasi-normal internal spherical harmonic
    coefficients).

    | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
    | CO. 80225
    | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

    Parameters
    ----------
    date : float
        date of resulting model (in decimal year)
    dte1 : float
        date of earlier model
    nmax1 : int
        maximum degree and order of earlier model
    dte2 : float
        date of later model
    nmax2 : int
        maximum degree and order of later model
    gh : numpy array
        Schmidt quasi-normal internal spherical harmonic coefficients of
        earlier model and internal model

    Returns
    -------
    nmax : int
        maximum degree and order of resulting model
    """
    factor = (date - dte1) / (dte2 - dte1)
    if nmax1 == nmax2:
        k = nmax1 * (nmax1 + 2)
        nmax = nmax1
    else:
        if nmax1 > nmax2:
            k = nmax2 * (nmax2 + 2)
            l = nmax1 * (nmax1 + 2)
            for ii in range(k, l):
                gh[igh][ii] = gh[0][ii] + factor*(-gh[0][ii])
            nmax = nmax1
        else:
            k = nmax1 * (nmax1 + 2)
            l = nmax2 * (nmax2 + 2)
            for ii in range(k, l):
                gh[igh][ii] = factor * gh[1][ii]

            nmax = nmax2

    for ii in range(k):
        gh[igh][ii] = gh[0][ii] + factor*(gh[1][ii] - gh[0][ii])

    return nmax, gh


@jit(nopython=True)
def shval3(igdgc, flat, flon, elev, nmax, igh, gh):
    """
    Calculate field components from spherical harmonic (sh) models.

    This routine updates self.x, self.y, self.z (Northward, Eastward and
    vertically downward components respectively NED)

    Based on subroutine 'igrf' by D. R. Barraclough and S. R. C. Malin,
    report no. 71/1, institute of geological sciences, U.K.

    | FORTRAN : Norman W. Peddie, USGS, MS 964, box 25046 Federal Center,
    | Denver, CO. 80225
    | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

    Parameters
    ----------
    igdgc : int
        indicates coordinate system used set equal to 1 if geodetic, 2 if
        geocentric
    flat : float
        north latitude, in degrees
    flon : float
        east longitude, in degrees
    elev : float
        WGS84 altitude above ellipsoid (igdgc=1), or radial distance from
        earth's center (igdgc=2)
    nmax : int
        maximum degree and order of coefficients
    gh : numpy array
        Schmidt quasi-normal internal spherical harmonic coefficients of
        earlier model and internal model

    Returns
    -------
    None.
    """
    sl = np.zeros(14)
    cl = np.zeros(14)
    p = np.zeros(119)
    q = np.zeros(119)
    earths_radius = 6371.2
    dtr = np.pi/180.0

    # a2,b2 are squares of semi-major and semi-minor axes of the reference
    # spheroid used for transforming between geodetic and geocentric
    # coordinates or components

    a2 = 40680631.59            # WGS84
    b2 = 40408299.98            # WGS84
    r = elev
    argument = flat * dtr
    slat = sin(argument)
    if (90.0 - flat) < 0.001:
        aa = 89.999            # 300 ft. from North pole
    elif (90.0 + flat) < 0.001:
        aa = -89.999        # 300 ft. from South pole
    else:
        aa = flat

    argument = aa * dtr
    clat = cos(argument)
    argument = flon * dtr
    sl[1] = sin(argument)
    cl[1] = cos(argument)

    x = 0
    y = 0
    z = 0

    sd = 0.0
    cd = 1.0
    l = 0
    n = 0
    m = 1
    npq = int((nmax * (nmax + 3)) / 2)
    if igdgc == 1:
        aa = a2 * clat * clat
        bb = b2 * slat * slat
        cc = aa + bb
        argument = cc
        dd = sqrt(argument)
        argument = elev * (elev + 2.0 * dd) + (a2 * aa + b2 * bb) / cc
        r = sqrt(argument)
        cd = (elev + dd) / r
        sd = (a2 - b2) / dd * slat * clat / r
        aa = slat
        slat = slat * cd - clat * sd
        clat = clat * cd + aa * sd

    ratio = earths_radius / r
    argument = 3.0
    aa = sqrt(argument)
    p[1] = 2.0 * slat
    p[2] = 2.0 * clat
    p[3] = 4.5 * slat * slat - 1.5
    p[4] = 3.0 * aa * clat * slat
    q[1] = -clat
    q[2] = slat
    q[3] = -3.0 * clat * slat
    q[4] = aa * (slat * slat - clat * clat)
    for k in range(1, npq+1):
        if n < m:
            m = 0
            n = n + 1
            argument = ratio
            power = n + 2
            rr = pow(argument, power)
            fn = n

        fm = m
        if k >= 5:
            if m == n:
                argument = (1.0 - 0.5/fm)
                aa = sqrt(argument)
                j = k - n - 1
                p[k] = (1.0 + 1.0/fm) * aa * clat * p[j]
                q[k] = aa * (clat * q[j] + slat/fm * p[j])
                sl[m] = sl[m-1] * cl[1] + cl[m-1] * sl[1]
                cl[m] = cl[m-1] * cl[1] - sl[m-1] * sl[1]
            else:
                argument = fn*fn - fm*fm
                aa = sqrt(argument)
                argument = ((fn - 1.0)*(fn-1.0)) - (fm * fm)
                bb = sqrt(argument)/aa
                cc = (2.0 * fn - 1.0)/aa
                ii = k - n
                j = k - 2 * n + 1
                p[k] = (fn + 1.0) * (cc * slat/fn * p[ii]-bb/(fn-1.0)*p[j])
                q[k] = cc * (slat * q[ii] - clat/fn * p[ii]) - bb * q[j]

        aa = rr * gh[igh -1][l]

        if m == 0:
            x = x + aa * q[k]
            z = z - aa * p[k]

            l = l + 1
        else:
            bb = rr * gh[igh-1][l+1]
            cc = aa * cl[m] + bb * sl[m]
            x = x + cc * q[k]
            z = z - cc * p[k]
            if clat > 0:
                y = (y + (aa*sl[m] - bb*cl[m])*fm*p[k]/((fn + 1.0)*clat))
            else:
                y = y + (aa*sl[m] - bb*cl[m])*q[k]*slat
            l = l + 2

        m = m + 1

    aa = x
    x = x * cd + z * sd
    z = z * cd - aa * sd

    return x, y, z


@jit(nopython=True)
def dihf(x, y, z):
    """
    Compute the geomagnetic d, i, h, and f from x, y, and z.

    This updates self.d, self.i, self.h and self.f (declination,
    inclination, horizontal intensity and total intensity).

    | FORTRAN : A. Zunde, USGS, MS 964, box 25046 Federal Center, Denver,
    | CO. 80225
    | C : C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA

    Parameters
    ----------
    x : float
        northward component
    y : float
        eastward component
    z : float
        vertically-downward component

    Returns
    -------
    None.

    """
    sn = 0.0001

    for _ in range(2):
        h2 = x*x + y*y
        argument = h2
        h = sqrt(argument)       # calculate horizontal intensity
        argument = h2 + z*z
        f = sqrt(argument)       # calculate total intensity
        if f < sn:
            d = np.nan           # If d and i cannot be determined,
            i = np.nan           # set equal to NaN
        else:
            argument = z
            argument2 = h
            i = atan2(argument, argument2)
            if h < sn:
                d = np.nan
            else:
                hpx = h + x
                if hpx < sn:
                    d = np.pi
                else:
                    argument = y
                    argument2 = hpx
                    d = 2.0 * atan2(argument, argument2)

    return h, f, i, d
