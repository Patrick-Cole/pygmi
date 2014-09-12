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
This code is based on the Geomag software, with information given below. It was
translated into Python from the Geomag code.

| This program, originally written in FORTRAN, was developed using subroutines
| written by    : A. Zunde
|               USGS, MS 964, Box 25046 Federal Center, Denver, Co.  80225
|               and
|               S.R.C. Malin & D.R. Barraclough
|               Institute of Geological Sciences, United Kingdom.

| Translated
| into C by    : Craig H. Shaffer
|               29 July, 1988

| Rewritten by : David Owens
|                For Susan McLean

| Maintained by: Stefan Maus
| Contact      : stefan.maus@noaa.gov
|                National Geophysical Data Center
|                World Data Center-A for Solid Earth Geophysics
|                NOAA, E/GC1, 325 Broadway,
|                Boulder, CO  80303
"""
# *************************************************************************
#
#      Some variables used in this program
#
#    Name         Type                    Usage
# ------------------------------------------------------------------------
#
#   a2,b2      Scalar Double          Squares of semi-major and semi-minor
#                                     axes of the reference spheroid used
#                                     for transforming between geodetic or
#                                     geocentric coordinates.
#
#   minalt     Double array of MAXMOD Minimum height of model.
#
#   altmin     Double                 Minimum height of selected model.
#
#   altmax     Double array of MAXMOD Maximum height of model.
#
#   maxalt     Double                 Maximum height of selected model.
#
#   d          Scalar Double          Declination of the field from the
#                                     geographic north (deg).
#
#   sdate      Scalar Double          start date inputted
#
#   ddot       Scalar Double          annual rate of change of decl.
#                                     (arc-min/yr)
#
#   alt        Scalar Double          altitude above WGS84 Ellipsoid
#
#   epoch      Double array of MAXMOD epoch of model.
#
#   ext        Scalar Double          Three 1st-degree external coeff.
#
#   latitude   Scalar Double          Latitude.
#
#   longitude  Scalar Double          Longitude.
#
#   gh1        Double array           Schmidt quasi-normal internal
#                                     spherical harmonic coeff.
#
#   gh2        Double array           Schmidt quasi-normal internal
#                                     spherical harmonic coeff.
#
#   gha        Double array           Coefficients of resulting model.
#
#   ghb        Double array           Coefficients of rate of change model.
#
#   i          Scalar Double          Inclination (deg).
#
#   idot       Scalar Double          Rate of change of i (arc-min/yr).
#
#   igdgc      Integer                Flag for geodetic or geocentric
#                                     coordinate choice.
#
#   inbuff     Char a of MAXINBUF     Input buffer.
#
#   irec_pos   Integer array of MAXMOD Record counter for header
#
#   stream  Integer                   File handles for an opened file.
#
#   fileline   Integer                Current line in file (for errors)
#
#   max1       Integer array of MAXMOD Main field coefficient.
#
#   max2       Integer array of MAXMOD Secular variation coefficient.
#
#   max3       Integer array of MAXMOD Acceleration coefficient.
#
#   mdfile     Character array of PATH  Model file name.
#
#   minyr      Double                  Min year of all models
#
#   maxyr      Double                  Max year of all models
#
#   yrmax      Double array of MAXMOD  Max year of model.
#
#   yrmin      Double array of MAXMOD  Min year of model.
#
# *************************************************************************

from PyQt4 import QtGui, QtCore
import numpy as np
from math import sin
from math import cos
from math import sqrt
from math import atan2
from osgeo import osr
import copy
from . import dataprep as dp


class IGRF(QtGui.QDialog):
    """ IGRF field calculation

    This produces two datasets. The first is an IGRF dataset for the area of
    interest, defined by some input magnetic dataset. The second is the IGRF
    corrected form of that input magnetic dataset.

    To do this, the input dataset must be reprojected from its local projection
    to degrees, where the IGRF correction will take place. This is done within
    this class.
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent=None)

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.reportback = self.parent.showprocesslog

        MAXDEG = 13
        MAXCOEFF = (MAXDEG*(MAXDEG+2)+1)

        self.gh = np.zeros([4, MAXCOEFF])
        self.d = 0
        self.f = 0
        self.h = 0
        self.i = 0
        self.dtemp = 0
        self.ftemp = 0
        self.htemp = 0
        self.itemp = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.xtemp = 0
        self.ytemp = 0
        self.ztemp = 0
        self.xdot = 0
        self.ydot = 0
        self.zdot = 0
        self.fdot = 0
        self.hdot = 0
        self.idot = 0
        self.ddot = 0

        self.gridlayout_2 = QtGui.QGridLayout(self)
        self.groupbox = QtGui.QGroupBox(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.gridlayout = QtGui.QGridLayout(self.groupbox)
        self.combobox_kmz_datum = QtGui.QComboBox(self.groupbox)
        self.combobox_kmz_proj = QtGui.QComboBox(self.groupbox)
        self.dsb_kmz_latorigin = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_cm = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_scalefactor = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_fnorthing = QtGui.QDoubleSpinBox(self.groupbox)
        self.dsb_kmz_feasting = QtGui.QDoubleSpinBox(self.groupbox)
        self.sb_kmz_zone = QtGui.QSpinBox(self.groupbox)
        self.dsb_alt = QtGui.QDoubleSpinBox(self)
        self.dateedit = QtGui.QDateEdit(self)
        self.combobox_dtm = QtGui.QComboBox(self)
        self.combobox_mag = QtGui.QComboBox(self)

        self.setupui()

        self.combobox_kmz_datum.addItem('WGS84')
        self.combobox_kmz_datum.addItem('Cape (Clarke1880)')
        self.combobox_kmz_proj.addItem('UTM (South)')
        self.combobox_kmz_proj.addItem('UTM (North)')
        self.combobox_kmz_proj.addItem('Transverse Mercator')
        self.zone(35)

        self.combobox_kmz_proj.currentIndexChanged.connect(self.proj)
        self.sb_kmz_zone.valueChanged.connect(self.zone)
        self.buttonbox.accepted.connect(self.acceptall)

        self.datum = {}
        self.datum['Cape'] = (
            'GEOGCS["Cape",' +
            'DATUM["D_Cape",' +
            'SPHEROID["Clarke_1880_Arc",6378249.145,293.4663077]],' +
            'PRIMEM["Greenwich",0],' +
            'UNIT["Degree",0.017453292519943295]]')
        self.datum['Hartebeesthoek94'] = (
            'GEOGCS["Hartebeesthoek94",' +
            'DATUM["D_Hartebeesthoek_1994",' +
            'SPHEROID["WGS_1984",6378137,298.257223563]],' +
            'PRIMEM["Greenwich",0],' +
            'UNIT["Degree",0.017453292519943295]]')

        self.ctrans = None

    def setupui(self):
        """ Setup UI """

        label_3 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_3, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.combobox_kmz_datum, 0, 1, 1, 1)

        label_4 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_4, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.combobox_kmz_proj, 1, 1, 1, 1)

        label_6 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_6, 3, 0, 1, 1)
        self.dsb_kmz_latorigin.setMinimum(-90.0)
        self.dsb_kmz_latorigin.setMaximum(90.0)
        self.gridlayout.addWidget(self.dsb_kmz_latorigin, 3, 1, 1, 1)

        label_7 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_7, 4, 0, 1, 1)
        self.dsb_kmz_cm.setMinimum(-180.0)
        self.dsb_kmz_cm.setMaximum(180.0)
        self.dsb_kmz_cm.setProperty("value", 27.0)
        self.gridlayout.addWidget(self.dsb_kmz_cm, 4, 1, 1, 1)

        label_8 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_8, 5, 0, 1, 1)
        self.dsb_kmz_scalefactor.setDecimals(4)
        self.dsb_kmz_scalefactor.setProperty("value", 0.9996)
        self.gridlayout.addWidget(self.dsb_kmz_scalefactor, 5, 1, 1, 1)

        label_10 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_10, 7, 0, 1, 1)
        self.dsb_kmz_fnorthing.setMaximum(1000000000.0)
        self.dsb_kmz_fnorthing.setProperty("value", 10000000.0)
        self.gridlayout.addWidget(self.dsb_kmz_fnorthing, 7, 1, 1, 1)

        label_9 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_9, 6, 0, 1, 1)
        self.dsb_kmz_feasting.setMaximum(1000000000.0)
        self.dsb_kmz_feasting.setProperty("value", 500000.0)
        self.gridlayout.addWidget(self.dsb_kmz_feasting, 6, 1, 1, 1)

        label_11 = QtGui.QLabel(self.groupbox)
        self.gridlayout.addWidget(label_11, 2, 0, 1, 1)
        self.sb_kmz_zone.setMaximum(60)
        self.sb_kmz_zone.setProperty("value", 35)
        self.gridlayout.addWidget(self.sb_kmz_zone, 2, 1, 1, 1)

        self.gridlayout_2.addWidget(self.groupbox, 0, 0, 1, 2)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout_2.addWidget(self.buttonbox, 6, 0, 1, 2)

        label_20 = QtGui.QLabel(self)
        self.gridlayout_2.addWidget(label_20, 2, 0, 1, 1)
        self.gridlayout_2.addWidget(self.dsb_alt, 2, 1, 1, 1)
        self.dsb_alt.setMaximum(99999.9)

        label_21 = QtGui.QLabel(self)
        self.gridlayout_2.addWidget(label_21, 3, 0, 1, 1)
        self.gridlayout_2.addWidget(self.dateedit, 3, 1, 1, 1)

        label_22 = QtGui.QLabel(self)
        self.gridlayout_2.addWidget(label_22, 4, 0, 1, 1)
        self.gridlayout_2.addWidget(self.combobox_dtm, 4, 1, 1, 1)

        label_23 = QtGui.QLabel(self)
        self.gridlayout_2.addWidget(label_23, 5, 0, 1, 1)
        self.gridlayout_2.addWidget(self.combobox_mag, 5, 1, 1, 1)

        self.setWindowTitle("IGRF")
        self.groupbox.setTitle("Input Projection")
        label_3.setText("Datum")
        label_4.setText("Projection")
        label_6.setText("Latitude of Origin")
        label_7.setText("Central Meridian")
        label_8.setText("Scale Factor")
        label_9.setText("False Easting")
        label_10.setText("False Northing")
        label_11.setText("UTM Zone")
        label_20.setText("Sensor clearance above ground")
        label_21.setText("Date")
        label_22.setText("Digital Elevation Model")
        label_23.setText("Magnetic Data")

        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def acceptall(self):
        """ accept button"""
        orig = osr.SpatialReference()
        orig.SetWellKnownGeogCS('WGS84')
        targ = osr.SpatialReference()
        targ.SetWellKnownGeogCS('WGS84')

        indx = self.combobox_kmz_datum.currentIndex()
        txt = self.combobox_kmz_datum.itemText(indx)

        if 'Cape' in txt:
            orig.ImportFromWkt(self.datum['Cape'])

        indx = self.combobox_kmz_proj.currentIndex()
        txt = self.combobox_kmz_proj.itemText(indx)

        if 'UTM' in txt:
            utmzone = self.sb_kmz_zone.value()
            if 'North' in txt:
                orig.SetUTM(utmzone, True)
            else:
                orig.SetUTM(utmzone, False)

        if 'Transverse Mercator' in txt:
            clat = self.dsb_kmz_latorigin.value()
            clong = self.dsb_kmz_cm.value()
            scale = self.dsb_kmz_scalefactor.value()
            f_e = self.dsb_kmz_feasting.value()
            f_n = self.dsb_kmz_fnorthing.value()
            orig.SetTM(clat, clong, scale, f_e, f_n)

        self.ctrans = osr.CoordinateTransformation(orig, targ)

        self.accept()

    def proj(self, indx):
        """ used for choosing the projection """
        txt = self.combobox_kmz_proj.itemText(indx)
        if 'UTM' in txt:
            self.sb_kmz_zone.setEnabled(True)
            self.zone(self.sb_kmz_zone.value())
        else:
            self.sb_kmz_zone.setEnabled(False)

        if txt == 'Transverse Mercator':
            self.dsb_kmz_feasting.setValue(0.)
            self.dsb_kmz_fnorthing.setValue(0.)
            self.dsb_kmz_scalefactor.setValue(1.0)

    def zone(self, val):
        """ used for changing UTM zone """
        c_m = -180.+(val-1)*6+3
        self.dsb_kmz_cm.setValue(c_m)
        self.dsb_kmz_latorigin.setValue(0.)
        self.dsb_kmz_feasting.setValue(500000.)
        self.dsb_kmz_fnorthing.setValue(0.)
        self.dsb_kmz_scalefactor.setValue(0.9996)

        indx = self.combobox_kmz_proj.currentIndex()
        txt = self.combobox_kmz_proj.itemText(indx)

        if txt == 'UTM (South)':
            self.dsb_kmz_fnorthing.setValue(10000000.)

    def settings(self):
        """ Settings Dialog"""
# Variable declaration
# Control variables
        data = dp.merge(self.indata['Raster'])
        self.combobox_dtm.clear()
        self.combobox_mag.clear()
        for i in data:
            self.combobox_dtm.addItem(i.bandid)
            self.combobox_mag.addItem(i.bandid)

        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True
        else:
            return False

#        again = 1
        mdf = open(__file__.rpartition('\\')[0]+'\\IGRF11.cof')
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

        i = self.combobox_mag.currentIndex()
        maggrid = data[i]

        i = self.combobox_dtm.currentIndex()
        data = data[i]
        altgrid = data.data.flatten() * 0.001  # in km

        maxyr = max(yrmax)
        sdate = self.dateedit.date()
        sdate = sdate.year()+sdate.dayOfYear()/sdate.daysInYear()
        alt = self.dsb_alt.value()
        xrange = data.tlx+data.xdim/2.+np.arange(data.cols)*data.xdim
        yrange = data.tly-data.ydim/2.-np.arange(data.rows)*data.ydim
        xdat, ydat = np.meshgrid(xrange, yrange)
        xdat = xdat.flatten()
        ydat = ydat.flatten()

        igrf_F = altgrid * 0
        # Pick model
        yrmax = np.array(yrmax)
        modelI = sum(yrmax < sdate)
        igdgc = 1

        if (sdate > maxyr) and (sdate < maxyr+1):
            print("\nWarning: The date %4.2f is out of range,\n", sdate)
            print("but still within one year of model expiration date.\n")
            print("An updated model file is available before 1.1.%4.0f\n",
                  maxyr)

        if max2[modelI] == 0:
            self.getshc(modbuff, 1, irec_pos[modelI], max1[modelI], 0)
            self.getshc(modbuff, 1, irec_pos[modelI+1], max1[modelI+1], 1)
            nmax = self.interpsh(sdate, yrmin[modelI], max1[modelI],
                                 yrmin[modelI+1], max1[modelI+1], 2)
            nmax = self.interpsh(sdate+1, yrmin[modelI], max1[modelI],
                                 yrmin[modelI+1], max1[modelI+1], 3)
        else:
            self.getshc(modbuff, 1, irec_pos[modelI], max1[modelI], 0)
            self.getshc(modbuff, 0, irec_pos[modelI], max2[modelI], 1)
            nmax = self.extrapsh(sdate, epoch[modelI], max1[modelI],
                                 max2[modelI], 2)
            nmax = self.extrapsh(sdate+1, epoch[modelI], max1[modelI],
                                 max2[modelI], 3)

        progress = 0
        maxlen = xdat.size

        for i in range(maxlen):
            if igrf_F.mask[i] == True:
                continue

            tmp = int(i*100/maxlen)
            if tmp > progress:
                progress = tmp
                self.reportback('Calculation: ' + str(progress) + '%', True)

            longitude, latitude, _ = self.ctrans.TransformPoint(xdat[i],
                                                                ydat[i])
            alt = altgrid[i]

# Do the first calculations
            self.shval3(igdgc, latitude, longitude, alt, nmax, 3)
            self.dihf(3)
#            self.shval3(igdgc, latitude, longitude, alt, nmax, 4)
#            self.dihf(4)
#
#            RAD2DEG = (180.0/np.pi)
#
#            self.ddot = ((self.dtemp - self.d)*RAD2DEG)
#            if self.ddot > 180.0:
#                self.ddot -= 360.0
#            if self.ddot <= -180.0:
#                self.ddot += 360.0
#            self.ddot *= 60.0
#
#            self.idot = ((self.itemp - self.i)*RAD2DEG)*60
#            self.d = self.d*(RAD2DEG)
#            self.i = self.i*(RAD2DEG)
#            self.hdot = self.htemp - self.h
#            self.xdot = self.xtemp - self.x
#            self.ydot = self.ytemp - self.y
#            self.zdot = self.ztemp - self.z
#            self.fdot = self.ftemp - self.f
#
#          # deal with geographic and magnetic poles
#
#            if self.h < 100.0:  # at magnetic poles
#                self.d = np.nan
#                self.ddot = np.nan
#              # while rest is ok
#
#            if 90.0-abs(latitude) <= 0.001:  # at geographic poles
#                self.x = np.nan
#                self.y = np.nan
#                self.d = np.nan
#                self.xdot = np.nan
#                self.ydot = np.nan
#                self.ddot = np.nan

#            print('Test Data')
#            print('==========')
#
#            print('# Date 2014.5', sdate)
#            print('# Coord-System D')
#            print('# Altitude K100', alt)
#            print('# Latitude 70.3', latitude)
#            print('# Longitude 30.8', longitude)
#            print('# D_deg 13d', int(self.d))
#            print('# D_min 51m ', int((self.d-int(self.d))*60))
#            print('# I_deg 78d', int(self.i))
#            print('# I_min 55m', int((self.i-int(self.i))*60))
#            print('# H_nT 9987.9 {0:.1f}'.format(self.h))
#            print('# X_nT 9697.4 {0:.1f}'.format(self.x))
#            print('# Y_nT 2391.4 {0:.1f}'.format(self.y))
#            print('# Z_nT 51022.3 {0:.1f}'.format(self.z))
#            print('# F_nT 51990.7 {0:.1f}'.format(self.f))
#            print('# dD_min 10.9 {0:.1f}'.format(self.ddot))
#            print('# dI_min 1.0 {0:.1f}'.format(self.idot))
#            print('# dH_nT -10.4 {0:.1f}'.format(self.hdot))
#            print('# dX_nT -17.7 {0:.1f}'.format(self.xdot))
#            print('# dY_nT 28.1 {0:.1f}'.format(self.ydot))
#            print('# dZ_nT 29.0 {0:.1f}'.format(self.zdot))
#            print('# dF_nT 26.5 {0:.1f}'.format(self.fdot))
            igrf_F[i] = self.f

        self.outdata['Raster'] = copy.deepcopy(self.indata['Raster'])
        igrf_F = np.array(igrf_F)
        igrf_F.shape = data.data.shape
        self.outdata['Raster'].append(copy.deepcopy(data))
        self.outdata['Raster'][-1].data = igrf_F
        self.outdata['Raster'][-1].bandid = 'IGRF'
        self.outdata['Raster'].append(copy.deepcopy(maggrid))
        self.outdata['Raster'][-1].data -= igrf_F
        self.outdata['Raster'][-1].bandid = 'Magnetic Data: IGRF Corrected'

        self.reportback('Calculation: Completed', True)

        return True

    def getshc(self, file, iflag, strec, nmax_of_gh, gh):
        """ Reads spherical harmonic coefficients from the specified
            model into an array.

        Args:
            stream: Logical unit number
            iflag: Flag for SV equal to ) or not equal to 0 for designated
                read statements
            strec: Starting record number to read from model
            nmax_of_gh: Maximum degree and order of model

        Return:
            gh1 or gh2 - Schmidt quasi-normal internal spherical harmonic
                coefficients

        References:
            FORTRAN: Bill Flanagan, NOAA CORPS, DESDIS, NGDC, 325 Broadway,
            Boulder CO.  80301

            C: C. H. Shaffer, Lockheed Missiles and Space Company, Sunnyvale CA
        """
        ii = -1
        cnt = 0

        for nn in range(1, nmax_of_gh+1):
            for _ in range(nn+1):
                cnt += 1
                tmp = file[strec+cnt]
                tmp = tmp.split()
#                n = int(tmp[0])
                m = int(tmp[1])

                if iflag == 1:
                    g = float(tmp[2])
                    hh = float(tmp[3])
                else:
                    g = float(tmp[4])
                    hh = float(tmp[5])

                ii = ii + 1
                self.gh[gh][ii] = g

                if m != 0:
                    ii = ii + 1
                    self.gh[gh][ii] = hh

        return

# **************************************************************************
#
#                           Subroutine extrapsh
#
# **************************************************************************
#
#     Extrapolates linearly a spherical harmonic model with a
#     rate-of-change model.
#
#     Input:
#           date     - date of resulting model (in decimal year)
#           dte1     - date of base model
#           nmax1    - maximum degree and order of base model
#           gh1      - Schmidt quasi-normal internal spherical
#                      harmonic coefficients of base model
#           nmax2    - maximum degree and order of rate-of-change model
#           gh2      - Schmidt quasi-normal internal spherical
#                      harmonic coefficients of rate-of-change model
#
#     Output:
#           gha or b - Schmidt quasi-normal internal spherical
#                    harmonic coefficients
#           nmax   - maximum degree and order of resulting model
#
#     FORTRAN
#           A. Zunde
#           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225
#
#     C
#           C. H. Shaffer
#           Lockheed Missiles and Space Company, Sunnyvale CA
#           August 16, 1988
#
# **************************************************************************

    def extrapsh(self, date, dte1, nmax1, nmax2, gh):
        """  Extrapolates linearly a spherical harmonic model with a
             rate-of-change model. """
        factor = date - dte1
        if nmax1 == nmax2:
            k = nmax1 * (nmax1 + 2)
            nmax = nmax1
        else:
            if nmax1 > nmax2:
                k = nmax2 * (nmax2 + 2)
                l = nmax1 * (nmax1 + 2)
                for ii in range(k, l):
                    self.gh[gh][ii] = self.gh[0][ii]

                nmax = nmax1
            else:
                k = nmax1 * (nmax1 + 2)
                l = nmax2 * (nmax2 + 2)
                for ii in range(k, l):
                    self.gh[gh][ii] = factor * self.gh[1][ii]

                nmax = nmax2

        for ii in range(k):
            self.gh[gh][ii] = self.gh[0][ii] + factor * self.gh[1][ii]

        return nmax

# **************************************************************************
#
#                           Subroutine interpsh
#
# **************************************************************************
#
#     Interpolates linearly, in time, between two spherical harmonic
#     models.
#
#     Input:
#           date     - date of resulting model (in decimal year)
#           dte1     - date of earlier model
#           nmax1    - maximum degree and order of earlier model
#           gh1      - Schmidt quasi-normal internal spherical
#                      harmonic coefficients of earlier model
#           dte2     - date of later model
#           nmax2    - maximum degree and order of later model
#           gh2      - Schmidt quasi-normal internal spherical
#                      harmonic coefficients of internal model
#
#     Output:
#           gha or b - coefficients of resulting model
#           nmax     - maximum degree and order of resulting model
#
#     FORTRAN
#           A. Zunde
#           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225
#
#     C
#           C. H. Shaffer
#           Lockheed Missiles and Space Company, Sunnyvale CA
#           August 17, 1988
#
# **************************************************************************

    def interpsh(self, date, dte1, nmax1, dte2, nmax2, gh):
        """ Interpolates linearly, in time, between two spherical harmonic
            models. """
        factor = (date - dte1) / (dte2 - dte1)
        if nmax1 == nmax2:
            k = nmax1 * (nmax1 + 2)
            nmax = nmax1
        else:
            if nmax1 > nmax2:
                k = nmax2 * (nmax2 + 2)
                l = nmax1 * (nmax1 + 2)
                for ii in range(k, l):
                    self.gh[gh][ii] = self.gh[0][ii] + factor*(-self.gh[0][ii])
                nmax = nmax1
            else:
                k = nmax1 * (nmax1 + 2)
                l = nmax2 * (nmax2 + 2)
                for ii in range(k, l):
                    self.gh[gh][ii] = factor * self.gh[1][ii]

                nmax = nmax2

        for ii in range(k):
            self.gh[gh][ii] = self.gh[0][ii] + factor*(self.gh[1][ii] -
                                                       self.gh[0][ii])

        return nmax

# **************************************************************************
#
#                           Subroutine shval3
#
# **************************************************************************
#
#     Calculates field components from spherical harmonic (sh)
#     models.
#
#     Input:
#           igdgc     - indicates coordinate system used set equal
#                       to 1 if geodetic, 2 if geocentric
#           latitude  - north latitude, in degrees
#           longitude - east longitude, in degrees
#           elev      - WGS84 altitude above ellipsoid (igdgc=1), or
#                       radial distance from earth's center (igdgc=2)
#           a2,b2     - squares of semi-major and semi-minor axes of
#                       the reference spheroid used for transforming
#                       between geodetic and geocentric coordinates
#                       or components
#           nmax      - maximum degree and order of coefficients
#           iext      - external coefficients flag (=0 if none)
#           ext1,2,3  - the three 1st-degree external coefficients
#                       (not used if iext = 0)
#
#     Output:
#           x         - northward component
#           y         - eastward component
#           z         - vertically-downward component
#
#     based on subroutine 'igrf' by D. R. Barraclough and S. R. C. Malin,
#     report no. 71/1, institute of geological sciences, U.K.
#
#     FORTRAN
#           Norman W. Peddie
#           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225
#
#     C
#           C. H. Shaffer
#           Lockheed Missiles and Space Company, Sunnyvale CA
#           August 17, 1988
#
# **************************************************************************

    def shval3(self, igdgc, flat, flon, elev, nmax, gh):
        """Calculates field components from spherical harmonic (sh) models."""

        sl = np.zeros(14)
        cl = np.zeros(14)
        p = np.zeros(119)
        q = np.zeros(119)
        earths_radius = 6371.2
        dtr = np.pi/180.0
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

        if gh == 3:
            self.x = 0
            self.y = 0
            self.z = 0
        elif gh == 4:
            self.xtemp = 0
            self.ytemp = 0
            self.ztemp = 0

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
                    p[k] = (fn + 1.0) * (cc * slat/fn * p[ii] - bb/(fn - 1.0)
                                         * p[j])
                    q[k] = cc * (slat * q[ii] - clat/fn * p[ii]) - bb * q[j]

            if gh == 3:
                aa = rr * self.gh[2][l]
            elif gh == 4:
                aa = rr * self.gh[3][l]

            if m == 0:
                if gh == 3:
                    self.x = self.x + aa * q[k]
                    self.z = self.z - aa * p[k]
                elif gh == 4:
                    self.xtemp = self.xtemp + aa * q[k]
                    self.ztemp = self.ztemp - aa * p[k]
                else:
                    print("\nError in subroutine shval3")

                l = l + 1
            else:
                if gh == 3:
                    bb = rr * self.gh[2][l+1]
                    cc = aa * cl[m] + bb * sl[m]
                    self.x = self.x + cc * q[k]
                    self.z = self.z - cc * p[k]
                    if clat > 0:
                        self.y = (self.y + (aa*sl[m] - bb*cl[m])*fm *
                                  p[k]/((fn + 1.0)*clat))
                    else:
                        self.y = self.y + (aa*sl[m] - bb*cl[m])*q[k]*slat
                    l = l + 2
                elif gh == 4:
                    bb = rr * self.gh[3][l+1]
                    cc = aa * cl[m] + bb * sl[m]
                    self.xtemp = self.xtemp + cc * q[k]
                    self.ztemp = self.ztemp - cc * p[k]
                    if clat > 0:
                        self.ytemp = (self.ytemp + (aa*sl[m] - bb*cl[m])*fm *
                                      p[k]/((fn + 1.0) * clat))
                    else:
                        self.ytemp = (self.ytemp + (aa*sl[m] - bb*cl[m]) *
                                      q[k]*slat)
                    l = l + 2

            m = m + 1

        if gh == 3:
            aa = self.x
            self.x = self.x * cd + self.z * sd
            self.z = self.z * cd - aa * sd
        elif gh == 4:
            aa = self.xtemp
            self.xtemp = self.xtemp * cd + self.ztemp * sd
            self.ztemp = self.ztemp * cd - aa * sd

# **************************************************************************
#
#                           Subroutine dihf
#
# **************************************************************************
#
#     Computes the geomagnetic d, i, h, and f from x, y, and z.
#
#     Input:
#           x  - northward component
#           y  - eastward component
#           z  - vertically-downward component
#
#     Output:
#           d  - declination
#           i  - inclination
#           h  - horizontal intensity
#           f  - total intensity
#
#     FORTRAN
#           A. Zunde
#           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225
#
#     C
#           C. H. Shaffer
#           Lockheed Missiles and Space Company, Sunnyvale CA
#           August 22, 1988
#
# **************************************************************************

    def dihf(self, gh):
        """ Computes the geomagnetic d, i, h, and f from x, y, and z. """
        sn = 0.0001

        if gh == 3:
            x = self.x
            y = self.y
            z = self.z
            h = self.h
            f = self.f
            i = self.i
            d = self.d
        else:
            x = self.xtemp
            y = self.ytemp
            z = self.ztemp
            h = self.htemp
            f = self.ftemp
            i = self.itemp
            d = self.dtemp

        for _ in range(2):
            h2 = x*x + y*y
            argument = h2
            h = sqrt(argument)       # calculate horizontal intensity
            argument = h2 + z*z
            f = sqrt(argument)      # calculate total intensity
            if f < sn:
                d = np.nan        # If d and i cannot be determined,
                i = np.nan        # set equal to NaN
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

        if gh == 3:
            self.h = h
            self.f = f
            self.i = i
            self.d = d
        else:
            self.htemp = h
            self.ftemp = f
            self.itemp = i
            self.dtemp = d


def get_igrf(dat):
    """ Merges datasets found in a single PyGMI data object. The aim is to
    ensure that all datasets have the same number of rows and columns. """
    needsmerge = False
    for i in dat:
        if i.rows != dat[0].rows or i.cols != dat[0].cols:
            needsmerge = True

    if needsmerge is False:
        return dat

#    mrg = DataMerge()
#    mrg.indata['Raster'] = dat
#    data = dat[0]
#    dxy0 = min(data.xdim, data.ydim)
#    for data in dat:
#        dxy = min(dxy0, data.xdim, data.ydim)
#
#    mrg.dsb_dxy.setValue(dxy)
#    mrg.acceptall()
#    return mrg.outdata['Raster']
