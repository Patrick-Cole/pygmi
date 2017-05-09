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
""" Gravity and magnetic field calculations.
This uses the following algorithms:

References:
    Singh, B., Guptasarma, D., 2001. New method for fast computation of gravity
    and magnetic anomalies from arbitrary polyhedral. Geophysics 66, 521-526.

    Blakely, R.J., 1996. Potential Theory in Gravity and Magnetic Applications,
    1st edn. Cambridge University Press, Cambridge, UK, 441 pp. 200-201
    """

from __future__ import print_function

import copy
import tempfile
from math import sqrt
from multiprocessing import Pool
from PyQt5 import QtWidgets, QtCore

import numpy as np
import pylab as plt
from scipy.linalg import norm
from osgeo import gdal
from numba import jit
from matplotlib import cm
from pygmi.raster.dataprep import gdal_to_dat
from pygmi.raster.dataprep import data_to_gdal_mem
from pygmi.pfmod.datatypes import LithModel
from pygmi.misc import PTime


class GravMag(object):
    """This class holds the generic magnetic and gravity modelling routines

    Routine that will calculate the final versions of the field. Other,
    related code is here as well, such as the inversion routines.
    """
    def __init__(self, parent):

        self.parent = parent
        self.lmod1 = parent.lmod1
        self.lmod2 = parent.lmod2
        self.lmod = self.lmod1
        self.showtext = parent.showtext
        if hasattr(parent, 'pbars'):
            self.pbars = parent.pbars
        else:
            self.pbars = None
        self.oldlithindex = None
        self.mfname = self.parent.modelfilename
        self.tmpfiles = {}

        self.actionregionaltest = QtWidgets.QPushButton(self.parent)
        self.actioncalculate = QtWidgets.QPushButton(self.parent)
        self.actioncalculate2 = QtWidgets.QPushButton(self.parent)
        self.actioncalculate3 = QtWidgets.QPushButton(self.parent)
        self.actioncalculate4 = QtWidgets.QPushButton(self.parent)
        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.actionregionaltest.setText("Regional Test")
        self.actioncalculate.setText("Calculate Gravity (All)")
        self.actioncalculate2.setText("Calculate Magnetics (All)")
        self.actioncalculate3.setText("Calculate Gravity (Changes Only)")
        self.actioncalculate4.setText("Calculate Magnetics (Changes Only)")
        self.parent.toolbar.addWidget(self.actionregionaltest)
        self.parent.toolbar.addSeparator()
        self.parent.toolbar.addWidget(self.actioncalculate)
        self.parent.toolbar.addWidget(self.actioncalculate2)
        self.parent.toolbar.addWidget(self.actioncalculate3)
        self.parent.toolbar.addWidget(self.actioncalculate4)
        self.parent.toolbar.addSeparator()

        self.actionregionaltest.clicked.connect(self.test_pattern)
        self.actioncalculate.clicked.connect(self.calc_field_grav)
        self.actioncalculate2.clicked.connect(self.calc_field_mag)
        self.actioncalculate3.clicked.connect(self.calc_field_grav_changes)
        self.actioncalculate4.clicked.connect(self.calc_field_mag_changes)
        self.actioncalculate3.setEnabled(False)
        self.actioncalculate4.setEnabled(False)

    def calc_field_mag(self):
        """ Pre field-calculation routine """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.pview.viewmagnetics = True
        self.parent.profile.viewmagnetics = True

        self.lmod.lith_index_old[:] = -1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True, True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_plot()

        self.actioncalculate4.setEnabled(True)

    def calc_field_grav(self):
        """ Pre field-calculation routine """
        # Update this
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False
        self.parent.pview.viewmagnetics = False

        self.lmod.lith_index_old[:] = -1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_plot()

        self.actioncalculate3.setEnabled(True)

    def calc_field_mag_changes(self):
        """ calculates only mag changes """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.pview.viewmagnetics = True
        self.parent.profile.viewmagnetics = True

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True, True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_plot()

    def calc_field_grav_changes(self):
        """ calculates only grav changes """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False
        self.parent.pview.viewmagnetics = False

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

        if tlabel == 'Custom Profile Editor':
            self.parent.pview.update_plot()

    def calc_field2(self, showreports=False, magcalc=False):
        """ Calculate magnetic and gravity field """

        calc_field(self.lmod, pbars=self.pbars, showtext=self.showtext,
                   parent=self.parent, showreports=showreports,
                   magcalc=magcalc)

    def calc_regional(self):
        """
        Calculates a gravity and magnetic regional value based on a single
        solid lithology model. This gets used in tab_param. The principle is
        that the maximum value for a solid model with fixed extents and depth,
        using the most COMMON lithology, would be the MAXIMUM AVERAGE value for
        any model which we would do. Therefore the regional is simply:
        REGIONAL = OBS GRAVITY MEAN - CALC GRAVITY MAX
        This routine calculates the last term.
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

    def grd_to_lith(self, curgrid):
        """ Assign the DTM to the lithology model """
        d_x = curgrid.xdim
        d_y = curgrid.ydim
        utlx = curgrid.tlx
        utly = curgrid.tly
        gcols = curgrid.cols
        grows = curgrid.rows

        gxmin = utlx
        gymax = utly

        ndata = np.zeros([self.lmod.numy, self.lmod.numx])

        for i in range(self.lmod.numx):
            for j in range(self.lmod.numy):
                xcrd = self.lmod.xrange[0]+(i+.5)*self.lmod.dxy
                ycrd = self.lmod.yrange[1]-(j+.5)*self.lmod.dxy
                xcrd2 = int((xcrd-gxmin)/d_x)
                ycrd2 = int((gymax-ycrd)/d_y)
                if (ycrd2 >= 0 and xcrd2 >= 0 and ycrd2 < grows and
                        xcrd2 < gcols):
                    ndata[j, i] = curgrid.data.data[ycrd2, xcrd2]

        return ndata

    def test_pattern(self):
        """ Displays a test pattern of the data - an indication of the edge of
        model field decay. It gives an idea about how reliable the calculated
        field on the edge of the model is. """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1

        self.calc_regional()

        magtmp = self.lmod2.griddata['Calculated Magnetics'].data
        grvtmp = self.lmod2.griddata['Calculated Gravity'].data

        regplt = plt.figure()
        axes = plt.subplot(1, 2, 1)
        etmp = dat_extent(self.lmod2.griddata['Calculated Magnetics'], axes)
        plt.title('Magnetic Data')
        ims = plt.imshow(magtmp, extent=etmp)
        mmin = magtmp.mean()-2*magtmp.std()
        mmax = magtmp.mean()+2*magtmp.std()
        mint = (magtmp.std()*4)/10.
        if magtmp.ptp() > 0:
            csrange = np.arange(mmin, mmax, mint)
            cns = plt.contour(magtmp, levels=csrange, colors='b', extent=etmp)
            plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('nT')

        axes = plt.subplot(1, 2, 2)
        etmp = dat_extent(self.lmod2.griddata['Calculated Gravity'], axes)
        plt.title('Gravity Data')
        ims = plt.imshow(grvtmp, extent=etmp)
        mmin = grvtmp.mean()-2*grvtmp.std()
        mmax = grvtmp.mean()+2*grvtmp.std()
        mint = (grvtmp.std()*4)/10.

        if grvtmp.ptp() > 0:
            csrange = np.arange(mmin, mmax, mint)
            cns = plt.contour(grvtmp, levels=csrange, colors='y', extent=etmp)
            plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('mgal')

        regplt.show()

    def update_graph(self, grvval, magval, modind):
        """ Updates the graph """
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        self.lmod.lith_index = modind.copy()
        self.lmod.griddata['Calculated Gravity'].data = grvval.T.copy()
        self.lmod.griddata['Calculated Magnetics'].data = magval.T.copy()

        if tlabel == 'Layer Editor':
            self.parent.layer.combo()
        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot(slide=True)


class GeoData(object):
    """ Data layer class:
        This class defines each geological type and calculates the field
        for one cube from the standard definitions.

        The is a class which contains the geophysical information for a single
        lithology. This includes the final calculated field for that lithology
        only.
        """
    def __init__(self, parent, ncols=10, nrows=10, numz=10, dxy=10.,
                 d_z=10., mht=80., ght=0.):
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
        self.glayers = None

        self.x12 = None
        self.y12 = None
        self.z12 = None

        self.set_xyz(ncols, nrows, numz, dxy, mht, ght, d_z)

    def calc_origin(self):
        """ Calculate the field values for the lithologies"""

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -1*self.g_dxy/2,
                              -1*self.g_dxy, dtype=float)

            self.showtext('   Calculate gravity origin field')
            self.gboxmain(xdist, ydist, self.zobsg)

            self.modified = False
        return self.glayers, self.lith_index

    def calc_origin2(self):
        """ Calculate the field values for the lithologies"""

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
            self.gmmain(xdist, ydist)

            self.modified = False
        return self.mlayers, self.lith_index

    def netmagn(self):
        """ Calculate the net magnetization """
        theta = 0.
        fcx, fcy, fcz = dircos(self.finc, self.fdec, theta)
        unith = np.array([fcx, fcy, fcz])
        hintn = self.hintn * 10**-9          # in Tesla
        mu0 = 4*np.pi*10**-7
        hstrength = self.susc*hintn/mu0  # Induced magnetization, needs susc.
        ind_magn = hstrength*unith

#       B is Induced Field (Tesla)
#       M is Magnetization (A/m)
#       H is Magnetic Field (A/m)
#       k is Susceptibility and is M/H
#   `   B = mu0(H+M)
#       Q = Jr/Ji = mstrength/hstrength = Jr/kH

        mcx, mcy, mcz = dircos(self.minc, self.mdec, theta)
        unitm = np.array([mcx, mcy, mcz])
        rem_magn = self.mstrength*unitm   # Remnant magnetization
        net_magn = rem_magn+ind_magn      # Net magnetization
        netmagscalar = np.sqrt((net_magn**2).sum())
        return netmagscalar

    def rho(self):
        """ Returns the density contrast """
        return self.density - self.bdensity

    def set_xyz(self, ncols, nrows, numz, g_dxy, mht, ght, d_z, dxy=None,
                modified=True):
        """ Sets/updates xyz parameters again """
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
        """ Set x12, y12, z12. This is the limits of the cubes for the model"""

        numx = self.g_cols*self.g_dxy
        numy = self.g_rows*self.g_dxy
        numz = self.numz*self.d_z
        dxy = self.dxy
        d_z = self.d_z

        self.x12 = np.array([numx/2-dxy/2, numx/2+dxy/2])
        self.y12 = np.array([numy/2-dxy/2, numy/2+dxy/2])
        self.z12 = np.arange(-numz, numz+d_z, d_z)

    def gmmain(self, xobs, yobs):
        """ Algorithm for simultaneous computation of gravity and magnetic
            fields is based on the formulation published in GEOPHYSICS v. 66,
            521-526,2001. by Bijendra Singh and D. Guptasarma """
        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter

        x1 = float(self.x12[0])
        x2 = float(self.x12[1])
        y1 = float(self.y12[0])
        y2 = float(self.y12[1])
        z1 = 0.0
        z2 = self.d_z

        corner = np.array([[x1, y1, z1],
                           [x1, y2, z1],
                           [x2, y2, z1],
                           [x2, y1, z1],
                           [x1, y1, z2],
                           [x1, y2, z2],
                           [x2, y2, z2],
                           [x2, y1, z2]])

        nf = 6
        face = np.array([[0, 1, 2, 3],
                         [4, 7, 6, 5],
                         [1, 5, 6, 2],
                         [4, 0, 3, 7],
                         [3, 2, 6, 7],
                         [4, 5, 1, 0]])

        nedges = 4*nf
        edge = np.zeros([nedges, 8])
        # get edge lengths
        for f in range(nf):
            indx = face[f].tolist() + [face[f, 0]]
            for t in range(4):
                edgeno = f*4+t
                ends = indx[t:t+2]
                p1 = corner[ends[0]]
                p2 = corner[ends[1]]
                V = p2-p1
                L = norm(V)
                edge[edgeno, 0:3] = V
                edge[edgeno, 3] = L
                edge[edgeno, 6:8] = ends

        un = np.zeros([nf, 3])
        # get face normals
        for t in range(nf):
            ss = np.zeros([1, 3])
            for t1 in range(2):
                v1 = corner[face[t, t1+2]]-corner[face[t, 0]]
                v2 = corner[face[t, t1+1]]-corner[face[t, 0]]
                ss = ss+np.cross(v2, v1)
            un[t, :] = ss/norm(ss)

        # Define the survey grid
        X, Y = np.meshgrid(xobs, yobs)

        npro, nstn = X.shape
        # Initialise
        """
        Grav stuff
        Gc = 6.6732e-3            # Universal gravitational constant

        Mag stuff

        SI
        mu0 = 4*pi*10**-7  (Henry/m)
        B = mu0(H+M)   (Telsa, A/m)

        1 A/m = 4pi/1000 Oersted
        1 Gauss = 100000 gamma/nT
        1 Gauss = 1 Oersted
        1 A/m = 400pi  nT/gamma

        or (for conversion)

        1 A/m = Oersted*1000/4pi
        1 Gauss = gamma/nT*1/100000
        1 Gauss = Oersted*1
        1 A/m = nT/gamma*1/400pi
        1 Gauss = emu/cm3*4pi
        A/m  = emu/cm3 * 1000
        A/m = Gauss*1000/4pi
        Gauss = A/m*4pi/1000
        nT = A/m*100*4pi
        Mcgs = Msi / 1000

        CGS
        mu0 = 1
        B = H + 4*pi*M  (gauss, Oersted, emu/cm3)
        gauss == Oersted == 4*pi* emu/cm3
        B = H + 4*pi*M  (gauss, Oersted, emu/cm3)
        M = Mi + Mr
        M = k*H + Mr  (from Blakely)
        B = H + 4*pi*(k*H+Mr)
        B = H + 4*pi*k*H+4*pi*Mr

        if k is SI then this becomes:
        k(cgs) = k(SI)/(4*pi)

        B = H + k(SI)*H + 4*pi*Mr

        if Mr is in A/m, and H is in gauss, then Mr(cgs) = Mr(SI)/1000

        B = H + k*H + 4*pi*Mr(SI)/1000

        if H is in gamma (nT), then mult Mr term by 100000

        B = H + k(SI)*H + 400*pi*Mr(SI)

        Equations in code divide susc by 4pi because susc is SI. This is
        evident because of code comparison between two papers, one of which
        uses SI susc, and other uses CGS susc.

        However, the software uses M(CGS) only, i.e.

        M = Mi(CGS) + Mr(CGS)
          = H*k(CGS) + Mr(CGS)
          = H*k(SI)/4pi + Mr(SI)/1000  (H in gauss)
          = H*k(SI)/4pi + 100 * Mr(SI)  (H in nT/gamma)

        QED
        --->


        nT = 400*pi A/m
        mur = 1+k
        k = mur-1
        M = kH
        J = mu0M
        B = mu0(1+k)H
        B = mu0murH
        k(SI) = 4pi*k (cgs)

        M = B(mur-1)/mu0 * 10**-9  (if B is nT or gammas)
        M = kB/mu0 * 10**-9
        M = kB / 400pi

        B = mu0(H+M)
          = mu0(H+kH+Mr)

        B = mu0kH
          = 400pi*k*H  (H is A/m)


         1 Gauss is 100 000 nT

        """
        cx, cy, cz = dircos(self.finc, self.fdec, 90.0)

        uh = np.array([cx, cy, cz])
        H = self.hintn*uh               # The ambient magnetic field (nTesla)
        ind_magn = self.susc*H/(4*np.pi)   # Induced magnetization

        mcx, mcy, mcz = dircos(self.minc, self.mdec, 90.0)
        um = np.array([mcx, mcy, mcz])
#        rem_magn = (400*np.pi*self.mstrength)*um/(4*np.pi)
        rem_magn = (100*self.mstrength)*um     # Remanent magnetization

        net_magn = rem_magn+ind_magn  # Net magnetization
        pd = np.transpose(np.dot(un, net_magn.T))   # Pole densities

        # For each observation point do the following.
        # For each face find the solid angle.
        # For each side find p,q,r and add p,q,r of sides to get P,Q,R for the
        # face.
        # find hx,hy,hz.
        # find gx,gy,gz.
        # Add the components from all the faces to get Hx,Hy,Hz and Gx,Gy,Gz.

        mval = []
        newdepth = self.z12+abs(self.zobsm)
        indx = np.array([0, 1, 2, 3, 0, 1])

        for depth in piter(newdepth):
            if depth == 0.0:
                cor = (corner + [0., 0., depth+self.d_z/10000.])
            elif depth == (-1*self.d_z):
                cor = (corner + [0., 0., depth-self.d_z/10000.])
            else:
                cor = (corner + [0., 0., depth])

            if depth in newdepth:
                crs = np.zeros([4, 3])
                mgval = np.zeros([3, npro, nstn])

                mgval = gm3d(npro, nstn, X, Y, edge, cor, face, pd, un, indx,
                             crs, mgval)

#                Htot = np.sqrt((Hx+H[0])**2 + (Hy+H[1])**2 + (Hz+H[2])**2)
#                dta = Htot-self.hintn  # Correct, was originally dt
                dta = mgval[0]*cx + mgval[1]*cy + mgval[2]*cz
            else:
                dta = np.zeros(X.shape)

            mval.append(np.copy(dta.T))

        self.mlayers = np.array(mval)

    def gboxmain(self, xobs, yobs, zobs):
        """ Gbox routine by Blakely
            Note: xobs, yobs and zobs must be floats or there will be problems
            later.

        Subroutine GBOX computes the vertical attraction of a
        rectangular prism.  Sides of prism are parallel to x,y,z axes,
        and z axis is vertical down.

        Input parameters:
            Observation point is (x0,y0,z0).  The prism extends from x1
            to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
            directions, respectively.  Density of prism is rho.  All
            distance parameters in units of m;

        Output parameters:
            Vertical attraction of gravity, g, in mGal/rho.
            Must still be multiplied by rho outside routine.
            Done this way for speed. """

        glayers = []
        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter
#        piter = iter
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

        if z_0 == 0.:
            z1122 = np.arange(0., self.z12[-1]+self.d_z, self.d_z)
        for i in piter(z1122[:-1]):
            z12 = np.array([i, i+self.d_z])

            z_1 = float(z12[0])
            z_2 = float(z12[1])
            gval = np.zeros([self.g_cols, self.g_rows])

            gval = gboxmain2(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1,
                             x_2, y_2, z_2, np.ones(2), np.ones(2), np.ones(2),
                             np.array([-1, 1]))

            gval *= 6.6732e-3
            if z_0 != 0.:
                glayers.append(gval)
            else:
                glayers = [-gval]+glayers+[gval]
        self.glayers = np.array(glayers)


def save_layer(mlist):
    """ Routine saves the mlayer and glayer to a file """
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
    """ Matches the rows and columns of the second grid to the first
    grid """
    rgrv = lmod.griddata[rtxt]
    cgrv = lmod.griddata[ctxt]

    data = rgrv
    data2 = cgrv
    orig_wkt = data.wkt
    orig_wkt2 = data2.wkt

    doffset = 0.0
    if data.data.min() <= 0:
        doffset = data.data.min()-1.
        data.data -= doffset

    gtr0 = (data.tlx, data.xdim, 0.0, data.tly, 0.0, -data.ydim)
    gtr = (data2.tlx, data2.xdim, 0.0, data2.tly, 0.0, -data2.ydim)
    src = data_to_gdal_mem(data, gtr0, orig_wkt, data.cols, data.rows)
    dest = data_to_gdal_mem(data, gtr, orig_wkt2, data2.cols, data2.rows, True)

    gdal.ReprojectImage(src, dest, orig_wkt, orig_wkt2, gdal.GRA_Bilinear)

    dat = gdal_to_dat(dest, data.dataid)

    if doffset != 0.0:
        dat.data += doffset
        data.data += doffset

    return dat.data


def calc_field2(lmod, pbars=None, showtext=None, parent=None,
                showreports=False, magcalc=False):
    """ Calculate magnetic and gravity field

    This function calculates the magnetic and gravity field. It has two
    different modes of operation, by using the magcalc switch. If magcalc=True
    then magnetic fields are calculated, otherwize only gravity is calculated.

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
        if true, calculates magnetic data, otherwize only gravity.

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
        return

    # Init some variables for convenience
    lmod.update_lithlist()

    numx = int(lmod.numx)
    numy = int(lmod.numy)
    numz = int(lmod.numz)
    tmpfiles = {}

# model index
    modind = lmod.lith_index.copy()
    modindcheck = lmod.lith_index.copy()
    modind[modind == 0] = -1
    modindcheck[modind == 0] = -1

    if abs(np.sum(modind == -1)) == modind.size:
        showtext('No changes to model!')
        return

    for mlist in lmod.lith_list.items():
        mijk = mlist[1].lith_index
        if mijk not in modind:
            continue
        if mlist[0] != 'Background':
            mlist[1].modified = True
            showtext(mlist[0]+':')
            if parent is not None:
                mlist[1].parent = parent
                mlist[1].pbars = parent.pbars
                mlist[1].showtext = parent.showtext
            if magcalc:
                mlist[1].calc_origin2()
            else:
                mlist[1].calc_origin()
            tmpfiles[mlist[0]] = save_layer(mlist)

    if showreports is True:
        showtext('Summing data')

    QtCore.QCoreApplication.processEvents()
# get height corrections
    tmp = np.copy(lmod.lith_index)
    tmp[tmp > -1] = 0
    hcor = np.abs(tmp.sum(2))

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
        if mijk not in modind:
            continue
        tmpfiles[mlist[0]].seek(0)
        mfile = np.load(tmpfiles[mlist[0]])

        if magcalc:
            mglayers = mfile['mlayers']
        else:
            mglayers = mfile['glayers']*mlist[1].rho()

        showtext('Summing '+mlist[0]+' (PyGMI may become non-responsive' +
                 ' during this calculation)')

        if abs(np.sum(modind == -1)) < modind.size and mijk in modind:
            QtWidgets.QApplication.processEvents()
            i, j, k = np.nonzero(modind == mijk)
            iuni = np.array(np.unique(i), dtype=np.int32)
            juni = np.array(np.unique(j), dtype=np.int32)
            kuni = np.array(np.unique(k), dtype=np.int32)

            for k in kuni:
                baba = sum_fields(k, mgval, numx, numy, modind, aaa[0], aaa[1],
                                  mglayers, hcorflat, mijk, juni, iuni)
                mgvalin += baba

        showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtWidgets.QApplication.processEvents()

    mgvalin.resize([numx, numy])
    mgvalin = mgvalin.T
    mgvalin = mgvalin[::-1]
    mgvalin = np.ma.array(mgvalin)

    if magcalc:
        lmod.griddata['Calculated Magnetics'].data = mgvalin
    else:
        lmod.griddata['Calculated Gravity'].data = mgvalin

# This addoldcalc has has flaws w.r.t. regional if you change the regional
    if 'Gravity Regional' in lmod.griddata and not magcalc:
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

    return lmod.griddata


def calc_field(lmod, pbars=None, showtext=None, parent=None,
               showreports=False, magcalc=False):
    """ Calculate magnetic and gravity field

    This function calculates the magnetic and gravity field. It has two
    different modes of operation, by using the magcalc switch. If magcalc=True
    then magnetic fields are calculated, otherwize only gravity is calculated.

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
        if true, calculates magnetic data, otherwize only gravity.

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
        return

    ttt = PTime()
    # Init some variables for convenience
    lmod.update_lithlist()

    numx = int(lmod.numx)
    numy = int(lmod.numy)
    numz = int(lmod.numz)

    tmpfiles = {}

# model index
    modind = lmod.lith_index.copy()
    modindcheck = lmod.lith_index_old.copy()

    tmp = (modind == modindcheck)
# If modind and modindcheck have different shapes, then tmp == False. The next
# line checks for that.
    if not isinstance(tmp, bool):
        modind[tmp] = -1
        modindcheck[tmp] = -1

    if np.unique(modind).size == 1:
        showtext('No changes to model!')
        return

    if np.unique(modindcheck).size == 1 and np.unique(modindcheck)[0] == -1:
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
                    mlist[1].calc_origin2()
                else:
                    mlist[1].calc_origin()
                tmpfiles[mlist[0]] = save_layer(mlist)
        lmod.tmpfiles = tmpfiles

    if showreports is True:
        showtext('Summing data')

    QtCore.QCoreApplication.processEvents()
# get height corrections
    tmp = np.copy(lmod.lith_index)
    tmp[tmp > -1] = 0
    hcor = np.abs(tmp.sum(2))

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

        if np.unique(modind).size > 1 and mijk in modind:
            QtWidgets.QApplication.processEvents()
            i, j, k = np.nonzero(modind == mijk)
            iuni = np.array(np.unique(i), dtype=np.int32)
            juni = np.array(np.unique(j), dtype=np.int32)
            kuni = np.array(np.unique(k), dtype=np.int32)

            if i.size < 50000:
                for k in kuni:
                    baba = sum_fields(k, mgval, numx, numy, modind, aaa[0],
                                      aaa[1], mglayers, hcorflat, mijk, juni,
                                      iuni)
                    mgvalin += baba
            else:
                pool = Pool()
                baba = []

                for k in kuni:
                    baba.append(pool.apply_async(sum_fields,
                                                 args=(k, mgval, numx, numy,
                                                       modind, aaa[0], aaa[1],
                                                       mglayers, hcorflat,
                                                       mijk, juni, iuni,)))
                for p in baba:
                    mgvalin += p.get()
                pool.close()
                del baba

        if np.unique(modindcheck).size > 1 and mijk in modindcheck:
            QtWidgets.QApplication.processEvents()
            i, j, k = np.nonzero(modindcheck == mijk)
            iuni = np.array(np.unique(i), dtype=np.int32)
            juni = np.array(np.unique(j), dtype=np.int32)
            kuni = np.array(np.unique(k), dtype=np.int32)

            if i.size < 50000:
                for k in kuni:
                    baba = sum_fields(k, mgval, numx, numy, modindcheck,
                                      aaa[0],
                                      aaa[1], mglayers, hcorflat, mijk, juni,
                                      iuni)
                    mgvalin -= baba
            else:
                pool = Pool()
                baba = []

                for k in kuni:
                    baba.append(pool.apply_async(sum_fields,
                                                 args=(k, mgval, numx, numy,
                                                       modindcheck, aaa[0],
                                                       aaa[1],
                                                       mglayers, hcorflat,
                                                       mijk, juni, iuni,)))
                for p in baba:
                    mgvalin -= p.get()
                pool.close()
                del baba

        showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtWidgets.QApplication.processEvents()

    mgvalin.resize([numx, numy])
    mgvalin = mgvalin.T
    mgvalin = mgvalin[::-1]
    mgvalin = np.ma.array(mgvalin)

    if np.unique(modindcheck).size > 1:
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

    lmod.lith_index_old = np.copy(lmod.lith_index)

    showtext('Total Time: '+str(mins)+' minutes and '+str(secs)+' seconds')
    return lmod.griddata


@jit(nopython=True)
def sum_fields(k, mgval, numx, numy, modind, aaa0, aaa1, mlayers, hcorflat,
               mijk, jj, ii):
    """ Calculate magnetic and gravity field """

    b = numx*numy
    for j in range(b):
        mgval[j] = 0.

    for i in ii:
        xoff = numx-i
        for j in jj:
            yoff = numy-j
            if (modind[i, j, k] != mijk):
                continue
            for ijk in range(b):
                xoff2 = xoff + aaa0[ijk]
                yoff2 = aaa1[ijk]+yoff
                hcor2 = hcorflat[ijk]+k
                mgval[ijk] += mlayers[hcor2, xoff2, yoff2]

    return mgval


def quick_model(numx=50, numy=50, numz=50, dxy=1000, d_z=100,
                tlx=0, tly=0, tlz=0, mht=100, ght=0, finc=-67, fdec=-17,
                inputliths=None, susc=None, dens=None, minc=None, mdec=None,
                mstrength=None, hintn=30000):
    """ Create a quick model """
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


@jit(nopython=True)
def gboxmain2(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1, x_2, y_2, z_2,
              x, y, z, isign):
    """ Gbox routine by Blakely
        Note: xobs, yobs and zobs must be floats or there will be problems
        later.

    Subroutine GBOX computes the vertical attraction of a
    rectangular prism.  Sides of prism are parallel to x,y,z axes,
    and z axis is vertical down.

    Input parameters:
        Observation point is (x0,y0,z0).  The prism extends from x1
        to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
        directions, respectively.  Density of prism is rho.  All
        distance parameters in units of m;

    Output parameters:
        Vertical attraction of gravity, g, in mGal/rho.
        Must still be multiplied by rho outside routine.
        Done this way for speed. """

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


@jit(nopython=True)
def gm3d(npro, nstn, X, Y, edge, corner, face, pd, un, indx, crs, mgval):
    """ grvmag 3d. mgval MUST be zeros """

    for pr in range(npro):
        for st in range(nstn):
            x = X[pr, st]
            y = Y[pr, st]
            for f in range(6):  # 6 Faces
                for g in range(4):  # 4 points in a face
                    # correct corners so that we have distances from obs pnt
                    cindx = face[f, g]
                    crs[g, 0] = corner[cindx, 0] - x
                    crs[g, 1] = corner[cindx, 1] - y
                    crs[g, 2] = corner[cindx, 2]

                p = 0
                q = 0
                r = 0
                l = un[f, 0]
                m = un[f, 1]
                n = un[f, 2]

                p20 = crs[0, 0]
                p21 = crs[0, 1]
                p22 = crs[0, 2]
                r12b = np.sqrt(p20*p20+p21*p21+p22*p22)

                for t in range(4):  # 4 lines in a face
                    p20 = crs[indx[t+1], 0]
                    p21 = crs[indx[t+1], 1]
                    p22 = crs[indx[t+1], 2]

                    eno2 = 4*f+t   # Edge no
                    L = edge[eno2, 3]  # length of edge?
                    r12a = r12b
                    r12b = np.sqrt(p20*p20+p21*p21+p22*p22)

                    r12 = r12a+r12b
                    I = (1/L)*np.log((r12+L)/(r12-L))

                    p += I*edge[eno2, 0]
                    q += I*edge[eno2, 1]
                    r += I*edge[eno2, 2]

                # From omega, l, m, n PQR get components of field due to face f
                # dp1 is dot product between (l,m,n) and (x,y,z) or un and r.

                p10 = crs[0, 0]
                p11 = crs[0, 1]
                p12 = crs[0, 2]
                p20 = crs[1, 0]
                p21 = crs[1, 1]
                p22 = crs[1, 2]
                p30 = crs[2, 0]
                p31 = crs[2, 1]
                p32 = crs[2, 2]
                p40 = crs[3, 0]
                p41 = crs[3, 1]
                p42 = crs[3, 2]

                p1m = sqrt(p10**2 + p11**2 + p12**2)
                p2m = sqrt(p20**2 + p21**2 + p22**2)
                p3m = sqrt(p30**2 + p31**2 + p32**2)
                p4m = sqrt(p40**2 + p41**2 + p42**2)

                wn = (p30*(p11*p22 - p12*p21) + p31*(-p10*p22 + p12*p20) +
                      p32*(p10*p21 - p11*p20))
                wd = (p1m*p2m*p3m + p1m*(p20*p30 + p21*p31 + p22*p32) +
                      p2m*(p10*p30 + p11*p31 + p12*p32) +
                      p3m*(p10*p20 + p11*p21 + p12*p22))
                omega = -2*np.arctan2(wn, wd)

                wn = (p10*(p31*p42 - p32*p41) + p11*(-p30*p42 + p32*p40) +
                      p12*(p30*p41 - p31*p40))
                wd = (p1m*p3m*p4m + p1m*(p30*p40 + p31*p41 + p32*p42) +
                      p3m*(p10*p40 + p11*p41 + p12*p42) +
                      p4m*(p10*p30 + p11*p31 + p12*p32))

                omega += -2*np.arctan2(wn, wd)

                # l, m, n and components of unit normal to a face.
                gmtf1 = l*omega+n*q-m*r
                gmtf2 = m*omega+l*r-n*p
                gmtf3 = n*omega+m*p-l*q

                # gmtf are common to gravity and magnetic, so have no field
                # info. pd is the field contribution. f is face. pr is profile.
                # st is station.

                mgval[0, pr, st] += pd[f]*gmtf1  # Hx
                mgval[1, pr, st] += pd[f]*gmtf2  # Hy
                mgval[2, pr, st] += pd[f]*gmtf3  # Hz

    return mgval


def dircos(incl, decl, azim):
    """
    Subroutine DIRCOS computes direction cosines from inclination
    and declination.

    Input parameters:
        incl:  inclination in degrees positive below horizontal.
        decl:  declination in degrees positive east of true north.
        azim:  azimuth of x axis in degrees positive east of north.

        Output parameters:
        a,b,c:  the three direction cosines.
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
    """ Gets the extent of the dat variable """
    left = dat.tlx
    top = dat.tly
    right = left + dat.cols*dat.xdim
    bottom = top - dat.rows*dat.ydim

    if (right-left) > 10000 or (top-bottom) > 10000:
        axes.xaxis.set_label_text("Eastings (km)")
        axes.yaxis.set_label_text("Northings (km)")
        left /= 1000.
        right /= 1000.
        top /= 1000.
        bottom /= 1000.
    else:
        axes.xaxis.set_label_text("Eastings (m)")
        axes.yaxis.set_label_text("Northings (m)")

    return (left, right, bottom, top)


def test():
    """ This routine is for testing purposes """
    from pygmi.pfmod.iodefs import ImportMod3D

    ttt = PTime()
# Import model file
    filename = r'C:\Work\Programming\pygmi\data\Magmodel_South_Delph_copy.npz'
    imod = ImportMod3D(None)
    imod.ifile = filename
    imod.lmod.griddata.clear()
    imod.lmod.lith_list.clear()
    indict = np.load(filename)
    imod.dict2lmod(indict)

# Calculate the field
    calc_field(imod.lmod)
    ttt.since_last_call()


if __name__ == "__main__":
    test()
