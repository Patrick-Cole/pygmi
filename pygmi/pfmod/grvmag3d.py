# -----------------------------------------------------------------------------
# Name:        grvmag3d.py (part of PyGMI)
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
Gravity and magnetic field calculations.

This uses the following algorithms:

References
----------
Singh, B., Guptasarma, D., 2001. New method for fast computation of gravity
and magnetic anomalies from arbitrary polyhedral. Geophysics 66, 521-526.

Blakely, R.J., 1996. Potential Theory in Gravity and Magnetic Applications,
1st edn. Cambridge University Press, Cambridge, UK, 441 pp. 200-201
"""

import copy
import tempfile
from math import sqrt
from PyQt5 import QtWidgets, QtCore

import numpy as np
from scipy.linalg import norm
from osgeo import gdal
from numba import jit, prange
from matplotlib import cm
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from pygmi.raster.dataprep import gdal_to_dat
from pygmi.raster.dataprep import data_to_gdal_mem
from pygmi.pfmod.datatypes import LithModel
from pygmi.misc import PTime


class GravMag():
    """
    The GravMag class holds generic magnetic and gravity modelling routines.

    Routine that will calculate the final versions of the field. Other,
    related code is here as well, such as the inversion routines.
    """

    def __init__(self, parent):

        self.parent = parent
        self.lmod2 = LithModel()
        self.lmod1 = parent.lmod1
        self.lmod = self.lmod1
        self.showtext = parent.showtext
        if hasattr(parent, 'pbars'):
            self.pbars = parent.pbars
        else:
            self.pbars = None
        self.oldlithindex = None
        self.mfname = self.parent.modelfilename
        self.tmpfiles = {}

        self.actionregionaltest = QtWidgets.QAction('Regional\nTest')
        self.actioncalculate = QtWidgets.QAction('Calculate\nGravity\n(All)')
        self.actioncalculate2 = QtWidgets.QAction('Calculate\nMagnetics\n(All)')
        self.actioncalculate3 = QtWidgets.QAction('Calculate\nGravity\n(Changes Only)')
        self.actioncalculate4 = QtWidgets.QAction('Calculate\nMagnetics\n(Changes Only)')
        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        self.parent.toolbardock.addSeparator()
        self.parent.toolbardock.addAction(self.actionregionaltest)
        self.parent.toolbardock.addSeparator()
        self.parent.toolbardock.addAction(self.actioncalculate)
        self.parent.toolbardock.addAction(self.actioncalculate2)
        self.parent.toolbardock.addAction(self.actioncalculate3)
        self.parent.toolbardock.addAction(self.actioncalculate4)
        self.parent.toolbardock.addSeparator()

        self.actionregionaltest.triggered.connect(self.test_pattern)
        self.actioncalculate.triggered.connect(self.calc_field_grav)
        self.actioncalculate2.triggered.connect(self.calc_field_mag)
        self.actioncalculate3.triggered.connect(self.calc_field_grav_changes)
        self.actioncalculate4.triggered.connect(self.calc_field_mag_changes)
        self.actioncalculate3.setEnabled(False)
        self.actioncalculate4.setEnabled(False)

    def calc_field_mag(self):
        """
        Pre field-calculation routine.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = True

        self.lmod.lith_index_mag_old[:] = -1

        # now do the calculations
        self.calc_field2(True, True)
        self.parent.profile.update_plot()

        self.actioncalculate4.setEnabled(True)

    def calc_field_grav(self):
        """
        Pre field-calculation routine.

        Returns
        -------
        None.

        """
        # Update this
        self.lmod1 = self.parent.lmod1
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False

        self.lmod.lith_index_grv_old[:] = -1

        # now do the calculations
        self.calc_field2(True)
        self.parent.profile.update_plot()

        self.actioncalculate3.setEnabled(True)

    def calc_field_mag_changes(self):
        """
        Calculate only magnetic field changes.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = True

        # now do the calculations
        self.calc_field2(True, True)
        self.parent.profile.update_plot()

    def calc_field_grav_changes(self):
        """
        Calculate only gravity field changes.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False

        # now do the calculations
        self.calc_field2(True)
        self.parent.profile.update_plot()

    def calc_field2(self, showreports=False, magcalc=False):
        """
        Calculate magnetic and gravity field.

        Parameters
        ----------
        showreports : bool, optional
            Flag for showing reports. The default is False.
        magcalc : bool, optional
            Flac for choosing the magnetic calculation. The default is False.

        Returns
        -------
        None.

        """
        calc_field(self.lmod, pbars=self.pbars, showtext=self.showtext,
                   parent=self.parent, showreports=showreports,
                   magcalc=magcalc)

    def calc_regional(self):
        """
        Calculate magnetic and gravity regional.

        Calculates a gravity and magnetic regional value based on a single
        solid lithology model. This gets used in tab_param. The principle is
        that the maximum value for a solid model with fixed extents and depth,
        using the most COMMON lithology, would be the MAXIMUM AVERAGE value for
        any model which we would do. Therefore the regional is simply:
        REGIONAL = OBS GRAVITY MEAN - CALC GRAVITY MAX
        This routine calculates the last term.

        Returns
        -------
        None.

        """
        ltmp = list(self.lmod1.lith_list.keys())
        ltmp.pop(ltmp.index('Background'))

        text, okay = QtWidgets.QInputDialog.getItem(
            self.parent, 'Regional Test',
            'Please choose the lithology to use:',
            ltmp)

        if not okay:
            return

        lmod1 = self.lmod1
        self.lmod2 = LithModel()
        self.lmod2.lith_list.clear()

        numlayers = lmod1.numz
        layerthickness = lmod1.d_z

        self.lmod2.update(lmod1.numx, lmod1.numy, numlayers, lmod1.xrange[0],
                          lmod1.yrange[1], lmod1.zrange[1], lmod1.dxy,
                          layerthickness, lmod1.mht, lmod1.ght)

        self.lmod2.lith_index = self.lmod1.lith_index.copy()
        self.lmod2.lith_index[self.lmod2.lith_index != -1] = 1

        self.lmod2.lith_list['Background'] = GeoData(
            self.parent, lmod1.numx, lmod1.numy, self.lmod2.numz, lmod1.dxy,
            self.lmod2.d_z, lmod1.mht, lmod1.ght)

        self.lmod2.lith_list['Regional'] = GeoData(
            self.parent, lmod1.numx, lmod1.numy, self.lmod2.numz, lmod1.dxy,
            self.lmod2.d_z, lmod1.mht, lmod1.ght)

        lithn = self.lmod2.lith_list['Regional']
        litho = self.lmod1.lith_list[text]
        lithn.hintn = litho.hintn
        lithn.finc = litho.finc
        lithn.fdec = litho.fdec
        lithn.zobsm = litho.zobsm
        lithn.susc = litho.susc
        lithn.mstrength = litho.mstrength
        lithn.qratio = litho.qratio
        lithn.minc = litho.minc
        lithn.mdec = litho.mdec
        lithn.density = litho.density
        lithn.bdensity = litho.bdensity
        lithn.zobsg = litho.zobsg
        lithn.lith_index = 1

        self.lmod = self.lmod2
        self.calc_field2(False, False)
        self.calc_field2(False, True)
        self.lmod = self.lmod1

    def test_pattern(self):
        """
        Displays a test pattern of the data.

        This is an indication of the edge of model field decay. It gives an
        idea about how reliable the calculated field on the edge of the model
        is.

        Returns
        -------
        None.

        """
        self.lmod1 = self.parent.lmod1
        self.lmod = self.lmod1

        self.calc_regional()

        magtmp = self.lmod2.griddata['Calculated Magnetics'].data
        grvtmp = self.lmod2.griddata['Calculated Gravity'].data

        regplt = plt.figure()
        axes = plt.subplot(121)
        etmp = dat_extent(self.lmod2.griddata['Calculated Magnetics'], axes)
        plt.title('Magnetic Data')
        ims = plt.imshow(magtmp, extent=etmp)
        mmin = magtmp.mean()-2*magtmp.std()
        mmax = magtmp.mean()+2*magtmp.std()
        mint = (magtmp.std()*4)/10.
        if magtmp.ptp() > 0:
            csrange = np.arange(mmin, mmax, mint)
#            cns = plt.contour(magtmp, levels=csrange, colors='b', extent=etmp)
            plt.contour(magtmp, levels=csrange, colors='b', extent=etmp)
#            plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('nT')

        axes = plt.subplot(122)
        etmp = dat_extent(self.lmod2.griddata['Calculated Gravity'], axes)
        plt.title('Gravity Data')
        ims = plt.imshow(grvtmp, extent=etmp)
        mmin = grvtmp.mean()-2*grvtmp.std()
        mmax = grvtmp.mean()+2*grvtmp.std()
        mint = (grvtmp.std()*4)/10.

        if grvtmp.ptp() > 0:
            csrange = np.arange(mmin, mmax, mint)
            plt.contour(grvtmp, levels=csrange, colors='y', extent=etmp)
#            cns = plt.contour(grvtmp, levels=csrange, colors='y', extent=etmp)
#            plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('mGal')
        plt.tight_layout()

        plt.get_current_fig_manager().window.setWindowIcon(self.parent.windowIcon())

        regplt.show()

    def update_graph(self, grvval, magval, modind):
        """
        Update the graph.

        Parameters
        ----------
        grvval : numpy array
            Array of gravity values.
        magval : numpy array
            Array of magnetic values.
        modind : numpy array
            Model indices.

        Returns
        -------
        None.

        """
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        self.lmod.lith_index = modind.copy()
        self.lmod.griddata['Calculated Gravity'].data = grvval.T.copy()
        self.lmod.griddata['Calculated Magnetics'].data = magval.T.copy()

        if tlabel == 'Layer Editor':
            self.parent.layer.combo()
        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot(slide=True)


class GeoData():
    """
    Data layer class.

    This class defines each geological type and calculates the field
    for one cube from the standard definitions.

    The is a class which contains the geophysical information for a single
    lithology. This includes the final calculated field for that lithology
    only.
    """

    def __init__(self, parent, ncols=10, nrows=10, numz=10, dxy=10.,
                 d_z=10., mht=80., ght=0.):
        self.lithcode = 0
        self.lithnotes = ''

        self.hintn = 30000.
        self.susc = 0.01
        self.mstrength = 0.
        self.finc = -63.
        self.fdec = -17.
        self.minc = -63.
        self.mdec = -17.
        self.theta = 90.
        self.bdensity = 2.67
        self.density = 2.85
        self.qratio = 0.0
        self.lith_index = 0
        self.parent = parent
        if hasattr(parent, 'pbars'):
            self.pbars = parent.pbars
        else:
            self.pbars = None

        if hasattr(parent, 'showtext'):
            self.showtext = parent.showtext
        else:
            self.showtext = print

    # ncols and nrows are the smaller dimension of the original grid.
    # numx, numy, numz are the dimensions of the larger grid to be used as a
    # template.

        self.modified = True
        self.g_cols = None
        self.g_rows = None
        self.g_dxy = None
        self.numz = None
        self.dxy = None
        self.d_z = None
        self.zobsm = None
        self.zobsg = None

        self.mlayers = None
        self.mtmp = None
        self.glayers = None

        self.x12 = None
        self.y12 = None
        self.z12 = None

        self.set_xyz(ncols, nrows, numz, dxy, mht, ght, d_z)

    def calc_origin_grav(self, hcor=None):
        """
        Calculate the field values for the lithologies.

        Parameters
        ----------
        hcor : numpy array or None, optional
            Height corrections. The default is None.

        Returns
        -------
        None.

        """

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -1*self.g_dxy/2,
                              -1*self.g_dxy, dtype=float)

            if hcor is None:
                hcor2 = 0
            else:
                hcor2 = int(self.numz-hcor.max())

            self.showtext('   Calculate gravity origin field')
            self.gboxmain(xdist, ydist, self.zobsg, hcor2)

            self.modified = False

    def calc_origin_mag(self, hcor=None):
        """
        Calculate the field values for the lithologies.

        Parameters
        ----------
        hcor : numpy array or None, optional
            Height corrections. The default is None.

        Returns
        -------
        None.

        """

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -1*self.g_dxy/2,
                              -1*self.g_dxy, dtype=float)

            self.showtext('   Calculate magnetic origin field')

            if hcor is None:
                hcor2 = 0
            else:
                hcor2 = int(self.numz-hcor.max())

            self.mboxmain(xdist, ydist, self.zobsm, hcor2)

            self.modified = False

    def rho(self):
        """
        Return the density contrast.

        Returns
        -------
        float
            Density contrast.

        """
        return self.density - self.bdensity

    def set_xyz(self, ncols, nrows, numz, g_dxy, mht, ght, d_z, dxy=None,
                modified=True):
        """
        Sets/updates xyz parameters.

        Parameters
        ----------
        ncols : int
            Number of columns.
        nrows : int
            Number of rows.
        numz : int
            Number of layers.
        g_dxy : float
            Grid spacing in x and y direction.
        mht : float
            Magnetic sensor height.
        ght : float
            Gravity sensor height.
        d_z : float
            Model spacing in z direction.
        dxy : float, optional
            Model spacing in x and y direction. The default is None.
        modified : bool, optional
            Whether the model was modified. The default is True.

        Returns
        -------
        None.

        """
        self.modified = modified
        self.g_cols = ncols*2+1
        self.g_rows = nrows*2+1
        self.numz = numz
        self.g_dxy = g_dxy
        self.d_z = d_z
        self.zobsm = -mht
        self.zobsg = -ght

        if dxy is None:
            self.dxy = g_dxy  # This must be a multiple of g_dxy or equal to it
        else:
            self.dxy = dxy  # This must be a multiple of g_dxy or equal to it.

        self.set_xyz12()

    def set_xyz12(self):
        """
        Set x12, y12, z12.

        This is the limits of the cubes for the model

        Returns
        -------
        None.

        """
        numx = self.g_cols*self.g_dxy
        numy = self.g_rows*self.g_dxy
        numz = self.numz*self.d_z
        dxy = self.dxy
        d_z = self.d_z

        self.x12 = np.array([numx/2-dxy/2, numx/2+dxy/2])
        self.y12 = np.array([numy/2-dxy/2, numy/2+dxy/2])
        self.z12 = np.arange(-numz, numz+d_z, d_z)



    def gboxmain(self, xobs, yobs, zobs, hcor):
        """
        Gbox routine by Blakely.

        Note: xobs, yobs and zobs must be floats or there will be problems
        later.

        Subroutine GBOX computes the vertical attraction of a
        rectangular prism.  Sides of prism are parallel to x,y,z axes,
        and z axis is vertical down.

        Input parameters:
        |    Observation point is (x0,y0,z0).  The prism extends from x1
        |    to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
        |    directions, respectively.  Density of prism is rho.  All
        |    distance parameters in units of m;

        Output parameters:
        |    Vertical attraction of gravity, g, in mGal/rho.
        |    Must still be multiplied by rho outside routine.
        |    Done this way for speed.

        Parameters
        ----------
        xobs : numpy array
            Observation X coordinates.
        yobs : numpy array
            Observation Y coordinates.
        zobs : numpy array
            Observation Z coordinates.
        hcor : numpy array
            Height corrections.

        Returns
        -------
        None.

        """
        glayers = []
        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter

        z1122 = self.z12.copy()
        x_1 = float(self.x12[0])
        y_1 = float(self.y12[0])
        x_2 = float(self.x12[1])
        y_2 = float(self.y12[1])
        z_0 = float(zobs)
        numx = int(self.g_cols)
        numy = int(self.g_rows)

        if zobs == 0:
            zobs = -0.01

        for z1 in piter(z1122[:-1]):
            if z1 < z1122[hcor]:
                glayers.append(np.zeros((self.g_cols, self.g_rows)))
                continue

            z2 = z1 + self.d_z

            gval = np.zeros([self.g_cols, self.g_rows])

            gval = _gbox(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z1,
                         x_2, y_2, z2, np.ones(2), np.ones(2), np.ones(2),
                         np.array([-1, 1]))

            gval *= 6.6732e-3
            glayers.append(gval)

        self.glayers = np.array(glayers)

    def mboxmain(self, xobs, yobs, zobs, hcor):
        """
        Mbox routine by Blakely

        Note: xobs, yobs and zobs must be floats or there will be problems
        later.

        Subroutine MBOX computes the total field anomaly of an infinitely
        extended rectangular prism.  Sides of prism are parallel to x,y,z
        axes, and z is vertical down.  Bottom of prism extends to infinity.
        Two calls to mbox can provide the anomaly of a prism with finite
        thickness; e.g.,

        |    call mbox(x0,y0,z0,x1,y1,z1,x2,y2,mi,md,fi,fd,m,theta,t1)
        |    call mbox(x0,y0,z0,x1,y1,z2,x2,y2,mi,md,fi,fd,m,theta,t2)
        |    t=t1-t2

        Requires subroutine DIRCOS.  Method from Bhattacharyya (1964).

        Input parameters:
        |    Observation point is (x0,y0,z0).  Prism extends from x1 to
        |    x2, y1 to y2, and z1 to infinity in x, y, and z directions,
        |    respectively.  Magnetization defined by inclination mi,
        |    declination md, intensity m.  Ambient field defined by
        |    inclination fi and declination fd.  X axis has declination
        |    theta. Distance units are irrelevant but must be consistent.
        |    Angles are in degrees, with inclinations positive below
        |    horizontal and declinations positive east of true north.
        |    Magnetization in A/m.

        Output paramters:
        |    Total field anomaly t, in nT.

        Parameters
        ----------
        xobs : numpy array
            Observation X coordinates.
        yobs : numpy array
            Observation Y coordinates.
        zobs : numpy array
            Observation Z coordinates.
        hcor : numpy array
            Height corrections.

        Returns
        -------
        None.

        """
        mlayers = []
        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter

        z1122 = self.z12.copy()
        z1122 = z1122.astype(float)
        x1 = float(self.x12[0])
        y1 = float(self.y12[0])
        x2 = float(self.x12[1])
        y2 = float(self.y12[1])
        z0 = float(zobs)
        numx = int(self.g_cols)
        numy = int(self.g_rows)

        ma, mb, mc = dircos(self.minc, self.mdec, self.theta)
        fa, fb, fc = dircos(self.finc, self.fdec, self.theta)

        mr = self.mstrength * np.array([ma, mb, mc]) * 100
        mi = self.susc*self.hintn*np.array([fa, fb, fc]) / (4*np.pi)
        m3 = mr+mi

        mt = np.sqrt(m3 @ m3)
        if mt > 0:
            m3 /= mt

        ma, mb, mc = m3

        fm1 = ma*fb + mb*fa
        fm2 = ma*fc + mc*fa
        fm3 = mb*fc + mc*fb
        fm4 = ma*fa
        fm5 = mb*fb
        fm6 = mc*fc

        if zobs == 0:
            zobs = -0.01

        z1122 = np.append(z1122, [2*z1122[-1]-z1122[-2]])

        for z1 in piter(z1122):
            if z1 < z1122[hcor]:
                mlayers.append(np.zeros((self.g_cols, self.g_rows)))
                continue

            mval = np.zeros([self.g_cols, self.g_rows])

            mval = _mbox(mval, xobs, yobs, numx, numy, z0, x1, y1, z1, x2, y2,
                         fm1, fm2, fm3, fm4, fm5, fm6, np.ones(2), np.ones(2))

            mlayers.append(mval)

        self.mlayers = np.array(mlayers) * mt
        self.mlayers = self.mlayers[:-1]-self.mlayers[1:]


def save_layer(mlist):
    """
    Routine to save the mlayer and glayer to a file.

    Parameters
    ----------
    mlist : list
        List with 2 elements - lithology name and LithModel.

    Returns
    -------
    outfile : TemporaryFile
        Link to a temporary file.

    """
    outfile = tempfile.TemporaryFile()

    outdict = {}

    outdict['mlayers'] = mlist[1].mlayers
    outdict['glayers'] = mlist[1].glayers

    np.savez(outfile, **outdict)
    outfile.seek(0)

    mlist[1].mlayers = None
    mlist[1].glayers = None

    return outfile


def gridmatch(lmod, ctxt, rtxt):
    """
    Matches the rows and columns of the second grid to the first
    grid.

    Parameters
    ----------
    lmod : LithModel
        Lithology Model.
    ctxt : str
        First grid text label.
    rtxt : str
        Second grid text label.

    Returns
    -------
    dat : numpy array
        Numpy array of data.

    """
    rgrv = lmod.griddata[rtxt]
    cgrv = lmod.griddata[ctxt]

    data = rgrv
    data2 = cgrv
    orig_wkt = data.wkt
    orig_wkt2 = data2.wkt

    doffset = 0.0
    if data.data.min() <= 0:
        doffset = data.data.min()-1.
        data.data = data.data - doffset

    rows, cols = data.data.shape
    rows2, cols2 = data2.data.shape

    gtr0 = data.get_gtr()
    gtr = data2.get_gtr()
    src = data_to_gdal_mem(data, gtr0, orig_wkt, cols, rows)
    dest = data_to_gdal_mem(data, gtr, orig_wkt2, cols2, rows2, True)

    gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt2, gdal.GRA_Bilinear)

    dat = gdal_to_dat(dest, data.dataid)

    if doffset != 0.0:
        dat.data = dat.data + doffset
        data.data = data.data + doffset

    return dat.data


def calc_field(lmod, pbars=None, showtext=None, parent=None,
               showreports=False, magcalc=False):
    """
    Calculate magnetic and gravity field.

    This function calculates the magnetic and gravity field. It has two
    different modes of operation, by using the magcalc switch. If magcalc=True
    then magnetic fields are calculated, otherwise only gravity is calculated.

    Parameters
    ----------
    lmod : LithModel
        PyGMI lithological model
    pbars : module
        progress bar routine if available. (internal use)
    showtext : module
        showtext routine if available. (internal use)
    showreports : bool
        show extra reports
    magcalc : bool
        if True, calculates magnetic data, otherwise only gravity.

    Returns
    -------
    lmod.griddata : dictionary
        dictionary of items of type Data.
    """

    if showtext is None:
        showtext = print
    if pbars is not None:
        pbars.resetall(mmax=2*(len(lmod.lith_list)-1)+1)
        piter = pbars.iter
    else:
        piter = iter
    if np.max(lmod.lith_index) == -1:
        showtext('Error: Create a model first')
        return None

    ttt = PTime()
    # Init some variables for convenience
    lmod.update_lithlist()

    numx = int(lmod.numx)
    numy = int(lmod.numy)
    numz = int(lmod.numz)

    tmpfiles = {}

# model index
    modind = lmod.lith_index.copy()
    if magcalc:
        modindcheck = lmod.lith_index_mag_old.copy()
    else:
        modindcheck = lmod.lith_index_grv_old.copy()

    tmp = (modind == modindcheck)
# If modind and modindcheck have different shapes, then tmp == False. The next
# line checks for that.
    if not isinstance(tmp, bool):
        modind[tmp] = -1
        modindcheck[tmp] = -1

    modindmax = modind.max()
    modindcheckmax = modindcheck.max()
#    if np.unique(modind).size == 1:
#        showtext('No changes to model!')
#        return None

    try:
        if False not in tmp:
            showtext('No changes to model!')
            return None
    except:
        breakpoint()

# get height corrections
    tmp = np.copy(lmod.lith_index)
    tmp[tmp > -1] = 0
    hcor = np.abs(tmp.sum(2))

#    if np.unique(modindcheck).size == 1 and np.unique(modindcheck)[0] == -1:
    for mlist in lmod.lith_list.items():
        mijk = mlist[1].lith_index
        if mijk not in modind and mijk not in modindcheck:
            continue
        if mlist[0] != 'Background':
            mlist[1].modified = True
            showtext(mlist[0]+':')
            if parent is not None:
                mlist[1].parent = parent
                mlist[1].pbars = parent.pbars
                mlist[1].showtext = parent.showtext
            if magcalc:
                mlist[1].calc_origin_mag(hcor)
            else:
                mlist[1].calc_origin_grav()
            tmpfiles[mlist[0]] = save_layer(mlist)
        lmod.tmpfiles = tmpfiles

    if showreports is True:
        showtext('Summing data')

    QtCore.QCoreApplication.processEvents()

# Get mlayers and glayers with correct rho and netmagn

    if pbars is not None:
        pbars.resetsub(maximum=(len(lmod.lith_list)-1))
        piter = pbars.iter

    mgvalin = np.zeros(numx*numy)
    mgval = np.zeros(numx*numy)

    hcorflat = numz-hcor.flatten()
    aaa = np.reshape(np.mgrid[0:numx, 0:numy], [2, numx*numy])

    for mlist in piter(lmod.lith_list.items()):
        if mlist[0] == 'Background':
            continue
        mijk = mlist[1].lith_index
        if mijk not in modind and mijk not in modindcheck:
            continue
        lmod.tmpfiles[mlist[0]].seek(0)

        mfile = np.load(lmod.tmpfiles[mlist[0]])

        if magcalc:
            mglayers = mfile['mlayers']
        else:
            mglayers = mfile['glayers']*mlist[1].rho()

        showtext('Summing '+mlist[0]+' (PyGMI may become non-responsive' +
                 ' during this calculation)')

        if modindmax > -1 and mijk in modind:
            QtWidgets.QApplication.processEvents()
            _, _, k = np.nonzero(modind == mijk)
            kuni = np.array(np.unique(k), dtype=np.int32)

            for k in kuni:
                baba = sum_fields(k, mgval, numx, numy, modind, aaa[0], aaa[1],
                                  mglayers, hcorflat, mijk)
                mgvalin += baba

        if modindcheckmax > -1 and mijk in modindcheck:
            QtWidgets.QApplication.processEvents()
            _, _, k = np.nonzero(modindcheck == mijk)
            kuni = np.array(np.unique(k), dtype=np.int32)

            for k in kuni:
                baba = sum_fields(k, mgval, numx, numy, modindcheck, aaa[0],
                                  aaa[1], mglayers, hcorflat, mijk)
                mgvalin -= baba

        showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtWidgets.QApplication.processEvents()

    for i in lmod.tmpfiles:
        lmod.tmpfiles[i].close()

    mgvalin.resize([numx, numy])
    mgvalin = mgvalin.T
    mgvalin = mgvalin[::-1]
    mgvalin = np.ma.array(mgvalin)

#    if np.unique(modindcheck).size > 1:
    if modindcheckmax > -1:
        if magcalc:
            mgvalin += lmod.griddata['Calculated Magnetics'].data
        else:
            mgvalin += lmod.griddata['Calculated Gravity'].data

    if magcalc:
        lmod.griddata['Calculated Magnetics'].data = mgvalin
    else:
        lmod.griddata['Calculated Gravity'].data = mgvalin

    if ('Gravity Regional' in lmod.griddata and not magcalc and
            np.unique(modindcheck).size == 1):
        zfin = gridmatch(lmod, 'Calculated Gravity', 'Gravity Regional')
        lmod.griddata['Calculated Gravity'].data += zfin

    if lmod.lith_index.max() <= 0:
        lmod.griddata['Calculated Magnetics'].data *= 0.
        lmod.griddata['Calculated Gravity'].data *= 0.

    if 'Magnetic Dataset' in lmod.griddata:
        ztmp = gridmatch(lmod, 'Magnetic Dataset', 'Calculated Magnetics')
        lmod.griddata['Magnetic Residual'] = copy.deepcopy(
            lmod.griddata['Magnetic Dataset'])
        lmod.griddata['Magnetic Residual'].data = (
            lmod.griddata['Magnetic Dataset'].data - ztmp)
        lmod.griddata['Magnetic Residual'].dataid = 'Magnetic Residual'

    if 'Gravity Dataset' in lmod.griddata:
        ztmp = gridmatch(lmod, 'Gravity Dataset', 'Calculated Gravity')
        lmod.griddata['Gravity Residual'] = copy.deepcopy(
            lmod.griddata['Gravity Dataset'])
        lmod.griddata['Gravity Residual'].data = (
            lmod.griddata['Gravity Dataset'].data - ztmp - lmod.gregional)
        lmod.griddata['Gravity Residual'].dataid = 'Gravity Residual'

    if parent is not None:
        tmp = [i for i in set(lmod.griddata.values())]
        parent.outdata['Raster'] = tmp
    showtext('Calculation Finished')
    if pbars is not None:
        pbars.maxall()

    tdiff = ttt.since_last_call(show=False)
    mins = int(tdiff/60)
    secs = tdiff-mins*60

    if magcalc:
        lmod.lith_index_mag_old = np.copy(lmod.lith_index)
    else:
        lmod.lith_index_grv_old = np.copy(lmod.lith_index)

    showtext('Total Time: '+str(mins)+' minutes and '+str(secs)+' seconds')

    return lmod.griddata


@jit(nopython=True, parallel=True)
def sum_fields(k, mgval, numx, numy, modind, aaa0, aaa1, mlayers, hcorflat,
               mijk):
    """
    Sum magnetic and gravity field datasets to produce final model field.

    Parameters
    ----------
    k : int
        k index.
    mgval : numpy array
        DESCRIPTION.
    numx : int
        Number of x elements.
    numy : int
        Number of y elements.
    modind : numpy array
        model with indices representing lithologies.
    aaa0 : numpy array
        x indices for offsets.
    aaa1 : numpy array
        y indices for offsets.
    mlayers : numpy array
        Layer fields for summation.
    hcorflat : numpy array
        Height correction.
    mijk : int
        Current lithology index.

    Returns
    -------
    mgval : numpy array
        Output summed data.

    """
    b = numx*numy
    for j in range(b):
        mgval[j] = 0.

    for i in range(numx):
        xoff = numx-i
        for j in range(numy):
            yoff = numy-j
            if (modind[i, j, k] != mijk):
                continue
            for ijk in prange(b):
                xoff2 = xoff + aaa0[ijk]
                yoff2 = aaa1[ijk]+yoff
                hcor2 = hcorflat[ijk]+k
                mgval[ijk] += mlayers[hcor2, xoff2, yoff2]

    return mgval


def quick_model(numx=50, numy=40, numz=5, dxy=100., d_z=100.,
                tlx=0., tly=0., tlz=0., mht=100., ght=0., finc=-67, fdec=-17,
                inputliths=None, susc=None, dens=None, minc=None, mdec=None,
                mstrength=None, hintn=30000.):
    """
    Quick model function.

    Parameters
    ----------
    numx : int, optional
        Number of x elements. The default is 50.
    numy : int, optional
        Number of y elements. The default is 40.
    numz : TYPE, optional
        number of z elements (layers). The default is 5.
    dxy : float, optional
        Cell size in x and y direction. The default is 100..
    d_z : float, optional
        Layer thickness. The default is 100..
    tlx : float, optional
        Top left x coordinate. The default is 0..
    tly : float, optional
        Top left y coordinate. The default is 0..
    tlz : float, optional
        Top left z coordinate. The default is 0..
    mht : float, optional
        Magnetic sensor height. The default is 100..
    ght : float, optional
        Gravity sensor height. The default is 0..
    finc : float, optional
        Magnetic field inclination (degrees). The default is -67.
    fdec : TYPE, optional
        Magnetic field declination (degrees). The default is -17.
    inputliths : list or None, optional
        List of input lithologies. The default is None.
    susc : list or None, optional
        List of susceptibilities. The default is None.
    dens : list or None, optional
        List of densities. The default is None.
    minc : list or None, optional
        List of remanent inclinations (degrees). The default is None.
    mdec : list or None, optional
        List of remanent declinations (degrees). The default is None.
    mstrength : list or None, optional
        List of remanent magnetisations (A/m). The default is None.
    hintn : float, optional
        Magnetic field strength (nT). The default is 30000.

    Returns
    -------
    lmod : LithModel
        Output model.

    """
    if inputliths is None:
        inputliths = ['Generic']
    if susc is None:
        susc = [0.01]
    if dens is None:
        dens = [3.0]

    lmod = LithModel()
    lmod.update(numx, numy, numz, tlx, tly, tlz, dxy, d_z, mht, ght)

    lmod.lith_list['Background'] = GeoData(None, numx, numy, numz, dxy, d_z,
                                           mht, ght)
    lmod.lith_list['Background'].susc = 0
    lmod.lith_list['Background'].density = 2.67
    lmod.lith_list['Background'].finc = finc
    lmod.lith_list['Background'].fdec = fdec
    lmod.lith_list['Background'].minc = finc
    lmod.lith_list['Background'].mdec = fdec
    lmod.lith_list['Background'].hintn = hintn

    j = 0
    if len(inputliths) == 1:
        clrtmp = np.array([0])
    else:
        clrtmp = np.arange(len(inputliths))/(len(inputliths)-1)
    clrtmp = cm.jet(clrtmp)[:, :-1]
    clrtmp *= 255
    clrtmp = clrtmp.astype(int)

    for i in inputliths:
        j += 1
        lmod.mlut[j] = clrtmp[j-1]
        lmod.lith_list[i] = GeoData(None, numx, numy, numz, dxy, d_z, mht, ght)

        lmod.lith_list[i].susc = susc[j-1]
        lmod.lith_list[i].density = dens[j-1]
        lmod.lith_list[i].lith_index = j
        lmod.lith_list[i].finc = finc
        lmod.lith_list[i].fdec = fdec
        lmod.lith_list[i].hintn = hintn
        if mstrength is not None:
            lmod.lith_list[i].minc = minc[j-1]
            lmod.lith_list[i].mdec = mdec[j-1]
            lmod.lith_list[i].mstrength = mstrength[j-1]

    return lmod


@jit(nopython=True, parallel=False)
def _mbox(mval, xobs, yobs, numx, numy, z0, x1, y1, z1, x2, y2, fm1, fm2, fm3,
          fm4, fm5, fm6, alpha, beta):
    """
    Mbox routine by Blakely, continued from Geodata.mboxmain. It exists
    in a separate function for JIT purposes.

    Note: xobs, yobs and zobs must be floats or there will be problems
    later.

    Subroutine MBOX computes the total field anomaly of an infinitely
    extended rectangular prism.  Sides of prism are parallel to x,y,z
    axes, and z is vertical down.  Bottom of prism extends to infinity.
    Two calls to mbox can provide the anomaly of a prism with finite
    thickness; e.g.,

    |    call mbox(x0,y0,z0,x1,y1,z1,x2,y2,mi,md,fi,fd,m,theta,t1)
    |    call mbox(x0,y0,z0,x1,y1,z2,x2,y2,mi,md,fi,fd,m,theta,t2)
    |    t=t1-t2

    Requires subroutine DIRCOS.  Method from Bhattacharyya (1964).

    Input parameters:
    |    Observation point is (x0,y0,z0).  Prism extends from x1 to
    |    x2, y1 to y2, and z1 to infinity in x, y, and z directions,
    |    respectively.  Magnetization defined by inclination mi,
    |    declination md, intensity m.  Ambient field defined by
    |    inclination fi and declination fd.  X axis has declination
    |    theta. Distance units are irrelevant but must be consistent.
    |    Angles are in degrees, with inclinations positive below
    |    horizontal and declinations positive east of true north.
    |    Magnetization in A/m.

    Output paramters:
    |    Total field anomaly t, in nT.


    Parameters
    ----------
    mval : numpy array
        DESCRIPTION.
    xobs : numpy array
        Observation X coordinates.
    yobs : numpy array
        Observation Y coordinates.
    numx : int
        Number of x elements.
    numy : int
        Number of y elements.
    z0 : float
        Observation height.
    x1 : float
        Prism coordinate.
    y1 : float
        Prism coordinate.
    z1 : float
        Prism coordinate.
    x2 : float
        Prism coordinate.
    y2 : float
        Prism coordinate.
    fm1 : float
        Calculation value passed from mboxmain.
    fm2 : float
        Calculation value passed from mboxmain.
    fm3 : float
        Calculation value passed from mboxmain.
    fm4 : float
        Calculation value passed from mboxmain.
    fm5 : float
        Calculation value passed from mboxmain.
    fm6 : float
        Calculation value passed from mboxmain.
    alpha : numpy array
        Calculation value passed from mboxmain.
    beta : numpy array
        Calculation value passed from mboxmain.

    Returns
    -------
    mval : numpy array
        Calculated magnetic values.

    """
    h = z1-z0
    hsq = h**2

    for ii in range(numx):
        alpha[0] = x1-xobs[ii]
        alpha[1] = x2-xobs[ii]
        for jj in range(numy):
            beta[0] = y1-yobs[jj]
            beta[1] = y2-yobs[jj]
            t = 0.

            for i in range(2):
                alphasq = alpha[i]**2
                for j in range(2):
                    sign = 1.
                    if i != j:
                        sign = -1.
                    r0sq = alphasq+beta[j]**2+hsq
                    r0 = np.sqrt(r0sq)
                    r0h = r0*h
                    alphabeta = alpha[i]*beta[j]
                    arg1 = (r0-alpha[i])/(r0+alpha[i])
                    arg2 = (r0-beta[j])/(r0+beta[j])
                    arg3 = alphasq+r0h+hsq
                    arg4 = r0sq+r0h-alphasq
                    tlog = (fm3*np.log(arg1)/2.+fm2*np.log(arg2)/2. -
                            fm1*np.log(r0+h))
                    tatan = (-fm4*np.arctan2(alphabeta, arg3) -
                             fm5*np.arctan2(alphabeta, arg4) +
                             fm6*np.arctan2(alphabeta, r0h))

                    t = t+sign*(tlog+tatan)
            mval[ii, jj] = t

    return mval


@jit(nopython=True, parallel=False)
def _gbox(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1, x_2, y_2, z_2,
          x, y, z, isign):
    """
    Gbox routine by Blakely, continued from Geodata.gboxmain. It exists
    in a separate function for JIT purposes.

    Note: xobs, yobs and zobs must be floats or there will be problems
    later.

    Subroutine GBOX computes the vertical attraction of a
    rectangular prism.  Sides of prism are parallel to x,y,z axes,
    and z axis is vertical down.

    Input parameters:
    |    Observation point is (x0,y0,z0).  The prism extends from x1
    |    to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
    |    directions, respectively.  Density of prism is rho.  All
    |    distance parameters in units of m;

    Output parameters:
    |    Vertical attraction of gravity, g, in mGal/rho.
    |    Must still be multiplied by rho outside routine.
    |    Done this way for speed.

    Parameters
    ----------
    gval : numpy array
        DESCRIPTION.
    xobs : numpy array
        Observation X coordinates.
    yobs : numpy array
        Observation Y coordinates.
    numx : TYPE
        DESCRIPTION.
    numy : TYPE
        DESCRIPTION.
    z_0 : float
        Observation height.
    x_1 : float
        Prism coordinate.
    y_1 : float
        Prism coordinate.
    z_1 : float
        Prism coordinate.
    x_2 : float
        Prism coordinate.
    y_2 : float
        Prism coordinate.
    z_2 : float
        Prism coordinate.
    x : numpy array
        Calculation value passed from gboxmain.
    y : numpy array
        Calculation value passed from gboxmain.
    z : numpy array
        Calculation value passed from gboxmain.
    isign : numpy array
        Calculation value passed from gboxmain.

    Returns
    -------
    gval : numpy array
        Calculated gravity values.

    """
    z[0] = z_0-z_1
    z[1] = z_0-z_2

    for ii in range(numx):
        x[0] = xobs[ii]-x_1
        x[1] = xobs[ii]-x_2
        for jj in range(numy):
            y[0] = yobs[jj]-y_1
            y[1] = yobs[jj]-y_2
            sumi = 0.
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        rijk = np.sqrt(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
                        ijk = isign[i]*isign[j]*isign[k]
                        arg1 = np.arctan2(x[i]*y[j], z[k]*rijk)

                        if arg1 < 0.:
                            arg1 = arg1 + 2 * np.pi
                        arg2 = rijk+y[j]
                        arg3 = rijk+x[i]
                        arg2 = np.log(arg2)
                        arg3 = np.log(arg3)
                        sumi += ijk*(z[k]*arg1-x[i]*arg2-y[j]*arg3)
            gval[ii, jj] = sumi

    return gval


def dircos(incl, decl, azim):
    """
    Subroutine DIRCOS computes direction cosines from inclination
    and declination.

    Parameters
    ----------
    incl : float
        inclination in degrees positive below horizontal.
    decl : float
        declination in degrees positive east of true north.
    azim : float
        azimuth of x axis in degrees positive east of north.

    Returns
    -------
    aaa : float
        First direction cosine.
    bbb : float
        Second direction cosine.
    ccc : float
        Third direction cosine.

    """
    d2rad = np.pi/180.
    xincl = incl*d2rad
    xdecl = decl*d2rad
    xazim = azim*d2rad
    aaa = np.cos(xincl)*np.cos(xdecl-xazim)
    bbb = np.cos(xincl)*np.sin(xdecl-xazim)
    ccc = np.sin(xincl)

    return aaa, bbb, ccc


def dat_extent(dat, axes):
    """
    Get the extent of the dat variable.

    Parameters
    ----------
    dat : PyGMI Data
        PyGMI raster dataset.
    axes : matplotlib.axes._subplots.AxesSubplot
        Matploltib axes.

    Returns
    -------
    left : float
        Left coordinate.
    right : float
        Right coordinate.
    bottom : float
        Bottom coordinate.
    top : float
        Top coordinate.

    """
    left, right, bottom, top = dat.extent

    if (right-left) > 10000 or (top-bottom) > 10000:
        axes.xaxis.set_label_text('Eastings (km)')
        axes.yaxis.set_label_text('Northings (km)')
        left /= 1000.
        right /= 1000.
        top /= 1000.
        bottom /= 1000.
    else:
        axes.xaxis.set_label_text('Eastings (m)')
        axes.yaxis.set_label_text('Northings (m)')

    return (left, right, bottom, top)


def test():
    """This routine is for testing purposes."""
#    from pygmi.pfmod.iodefs import ImportMod3D

# Import model file
#    filename = r'C:\Work\Programming\pygmi\data\Magmodel_Area3_Delph.npz'
#    imod = ImportMod3D(None)
#    imod.ifile = filename
#    imod.lmod.griddata.clear()
#    imod.lmod.lith_list.clear()
#    indict = np.load(filename)
#    imod.dict2lmod(indict)
#    calc_field(imod.lmod, magcalc=True)


# quick model
    lmod = quick_model(numx=300, numy=300, numz=30)
    lmod.lith_index[:, :, 0] = 1
#    lmod.lith_index[:, :, 10] = 1
    lmod.mht = 100
    calc_field(lmod, magcalc=True)

# Calculate the field

    magval = lmod.griddata['Calculated Magnetics'].data

#    plt.imshow(magval, cmap=cm.jet)
#    plt.show()


if __name__ == "__main__":
    test()
