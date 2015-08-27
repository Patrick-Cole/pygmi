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

from PyQt4 import QtGui, QtCore
import scipy.interpolate as si
import numpy as np
import pylab as plt
import copy
import tempfile
from scipy.linalg import norm
from pygmi.pfmod.datatypes import LithModel
from numba import jit
from matplotlib import cm


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

        self.actionregionaltest = QtGui.QPushButton(self.parent)
        self.actioncalculate = QtGui.QPushButton(self.parent)
        self.actioncalculate2 = QtGui.QPushButton(self.parent)
        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.actionregionaltest.setText("Regional Test")
        self.actioncalculate.setText("Calculate Gravity")
        self.actioncalculate2.setText("Calculate Magnetics")
        self.parent.toolbar.addWidget(self.actionregionaltest)
        self.parent.toolbar.addSeparator()
        self.parent.toolbar.addWidget(self.actioncalculate)
        self.parent.toolbar.addWidget(self.actioncalculate2)
        self.parent.toolbar.addSeparator()

        self.actionregionaltest.clicked.connect(self.test_pattern)
        self.actioncalculate.clicked.connect(self.calc_field)
        self.actioncalculate2.clicked.connect(self.calc_field_new)

    def calc_field_new(self):
        """ Pre field-calculation routine """
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

    def calc_field(self):
        """ Pre field-calculation routine """
        # Update this
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

        text, okay = QtGui.QInputDialog.getItem(
            self.parent, 'Regional Test',
            'Please choose the lithology to use:',
            ltmp)

        if not okay:
            return

        lmod1 = self.lmod1
        self.lmod2 = LithModel()
# This line ensures that lmod2 is different to lmod1
#        self.lmod2 = copy.copy(self.lmod2)
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
            ydist = np.arange(numy-self.g_dxy/2, -self.g_dxy/2, -self.g_dxy,
                              dtype=float)

#            self.showtext('   Calculate magnetic origin field')
#            self.mboxmain(xdist, ydist, self.zobsm)
            self.gboxmain(xdist, ydist, self.zobsg)

            self.modified = False
        else:
            pass
#            self.pbars.incrmain(2)

    def calc_origin2(self):
        """ Calculate the field values for the lithologies"""

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -self.g_dxy/2, -self.g_dxy,
                              dtype=float)

            self.gmmain(xdist, ydist)

            self.modified = False
        else:
            pass
#            self.pbars.incrmain(2)

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
#        z1 = float(self.z12[0])
#        z2 = float(self.z12[1])
        z1 = 0.0
        z2 = self.d_z

#        ncor = 8
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

#    nedges = sum(face(1:nf,1))
        nedges = 4*nf
        edge = np.zeros([nedges, 8])
        # get edge lengths
        for f in range(nf):
            indx = face[f].tolist() + [face[f, 0]]
            for t in range(4):
                # edgeno = sum(face(1:f-1,1))+t
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

        # Grav stuff
#        Gc = 6.6732e-3            # Universal gravitational constant
#        Gx = np.zeros(X.shape)
#        Gy = Gx.copy()
#        Gz = Gx.copy()

        # Mag stuff
        cx, cy, cz = dircos(self.finc, self.fdec, 90.0)
        uh = np.array([cx, cy, cz])
        H = self.hintn*uh               # The ambient magnetic field (nTesla)
        ind_magn = self.susc*H/(4*np.pi)   # Induced magnetization

        mcx, mcy, mcz = dircos(self.minc, self.mdec, 90.0)
        um = np.array([mcx, mcy, mcz])
        rem_magn = (400*np.pi*self.mstrength)*um     # Remanent magnetization

        net_magn = rem_magn+ind_magn  # Net magnetization
        Pd = np.transpose(np.dot(un, net_magn.T))   # Pole densities

        Hx = np.zeros(X.shape)
        Hy = Hx.copy()
        Hz = Hx.copy()

        # For each observation point do the following.
        # For each face find the solid angle.
        # For each side find p,q,r and add p,q,r of sides to get P,Q,R for the
        # face.
        # find hx,hy,hz.
        # find gx,gy,gz.
        # Add the components from all the faces to get Hx,Hy,Hz and Gx,Gy,Gz.


#        face = face.tolist()
#        un = un.tolist()
#        gval = []
        mval = []
        newdepth = self.z12+abs(self.zobsm)

        for depth in piter(newdepth):
            if depth == 0.0:
                cor = (corner + [0., 0., depth+self.d_z/10000.])
            elif depth == -self.d_z:
                cor = (corner + [0., 0., depth-self.d_z/10000.])
            else:
                cor = (corner + [0., 0., depth])

            if depth in newdepth:
                # Gx = np.zeros(X.shape)
                # Gy = Gx.copy()
                # Gz = Gx.copy()
                Hx = np.zeros(X.shape)
                Hy = Hx.copy()
                Hz = Hx.copy()

                indx = np.array([0, 1, 2, 3, 0, 1])
                crs = np.zeros([4, 3])
                p1 = np.zeros(3)
                p2 = np.zeros(3)
                p3 = np.zeros(3)
                mgval = np.zeros([3, npro, nstn])

                mgval = gm3d(npro, nstn, X, Y, edge, cor, face,  # Gx, Gy, Gz,
                             Hx, Hy, Hz, Pd, un, indx, crs, p1, p2, p3,
                             mgval)

                Hx = mgval[0]
                Hy = mgval[1]
                Hz = mgval[2]
                # Gx = mgval[3] * Gc
                # Gy = mgval[4] * Gc
                # Gz = mgval[5] * Gc

#                Htot = np.sqrt((Hx+H[0])**2 + (Hy+H[1])**2 + (Hz+H[2])**2)
#                dt = Htot-self.hintn
                dta = Hx*cx + Hy*cy + Hz*cz
            else:
                # Gz = np.zeros(X.shape)
                dta = np.zeros(X.shape)

#            gval.append(np.copy(Gz.T))
            mval.append(np.copy(dta.T))

#        self.glayers = np.array(gval)
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

        for i in piter(z1122[:-1]):
            z12 = np.array([i, i+self.d_z])

            z_1 = float(z12[0])
            z_2 = float(z12[1])
            gval = np.zeros([self.g_cols, self.g_rows])

            gval = gboxmain2(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1,
                             x_2, y_2, z_2, np.ones(2), np.ones(2), np.ones(2),
                             np.array([-1, 1]))

            gval *= 6.6732e-3
            glayers.append(gval)
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
    x = np.arange(rgrv.tlx, rgrv.tlx+rgrv.cols*rgrv.xdim,
                  rgrv.xdim)+0.5*rgrv.xdim
    y = np.arange(rgrv.tly-rgrv.rows*rgrv.ydim, rgrv.tly,
                  rgrv.xdim)+0.5*rgrv.ydim
    x_2, y_2 = np.meshgrid(x, y)
    z_2 = rgrv.data
    x_i = np.arange(cgrv.cols)*cgrv.xdim + cgrv.tlx + 0.5*cgrv.xdim
    y_i = np.arange(cgrv.rows)*cgrv.ydim + cgrv.tly - \
        cgrv.rows*cgrv.ydim + 0.5*cgrv.ydim
    xi2, yi2 = np.meshgrid(x_i, y_i)

    zfin = si.griddata((x_2.flatten(), y_2.flatten()), z_2.flatten(),
                       (xi2.flatten(), yi2.flatten()))
    zfin = np.ma.masked_invalid(zfin)
    zfin.shape = cgrv.data.shape

    return zfin


def calc_field(lmod, pbars=None, showtext=None, parent=None, showreports=False,
               magcalc=False):
    """ Calculate magnetic and gravity field

    This function calculates the magnetic and gravity field. It has two
    different modes of operation, by using the magcalc switch. If magcalc=True
    then magnetic fields are calculated, otherwize only
    gravity is calculated.

    Parameters
    ----------
    lmod : LithModel
        PyGMI lithological model
    pbars : module
        progress bar routine if available. (internal use)
    showtext : module
        showtext routine if available. (internal use)
    parent : parent
        parent function. (internal use)
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

# This forces a full recalc every time.
    lmod.clith_index[:] = 0
    lmod.cmagcalc = magcalc

#    if lmod.cmagcalc is not magcalc:
#        lmod.clith_index[:] = 0
#        lmod.cmagcalc = magcalc

    if lmod.clith_index.max() == 0 and magcalc:
        lmod.griddata['Calculated Magnetics'].data[:] = 0
    elif lmod.clith_index.max() == 0:
        lmod.griddata['Calculated Gravity'].data[:] = 0

# model index
    modind = lmod.lith_index.copy()
    modindcheck = lmod.lith_index.copy()
    cmodind = lmod.clith_index.copy()
    tmp = (modind == cmodind)
    modind[tmp] = -1
    cmodind[tmp] = -1
    modind[modind == 0] = -1
    cmodind[cmodind == 0] = -1
    modindcheck[modind == 0] = -1

    if (abs(np.sum(modind == -1)) == modind.size and
            abs(np.sum(cmodind == -1)) == cmodind.size):
        showtext('No changes to model!')
        return

    for mlist in lmod.lith_list.items():
        # if 'Background' != mlist[0] and mlist[1].modified is True:
        mijk = mlist[1].lith_index
        if mijk not in modind and mijk not in cmodind:
            continue
        if 'Background' != mlist[0]:  # and 'Penge' in mlist[0]:
            mlist[1].modified = True
            if showreports is True:
                showtext(mlist[0]+':')
            if parent is not None:
                mlist[1].parent = parent
                mlist[1].pbars = parent.pbars
                mlist[1].showtext = parent.showtext
            if magcalc:
                mlist[1].calc_origin2()
                if showreports is True:
                    showtext('   Calculate magnetic origin field')
            else:
                mlist[1].calc_origin()
                if showreports is True:
                    showtext('   Calculate gravity origin field')

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
    else:
        piter = iter

    magval = np.zeros([numx, numy])
    grvval = np.zeros([numx, numy])
    mtmp = magval.shape
    magval = magval.flatten()
    grvval = grvval.flatten()
    hcorflat = numz-hcor.flatten()
    aaa = np.reshape(np.mgrid[0:mtmp[0], 0:mtmp[1]], [2, numx*numy])

#    ttt = PTime()
    mgval = np.zeros([2, magval.size])

    for mlist in piter(lmod.lith_list.items()):
        if 'Background' == mlist[0]:
            continue
        mijk = mlist[1].lith_index
        if mijk not in modind and mijk not in cmodind:
            continue
        tmpfiles[mlist[0]].seek(0)
        mfile = np.load(tmpfiles[mlist[0]])

        if magcalc:
            mlayers = mfile['mlayers']

###############################################################################
# Not sure why I did this. It does not apply normally by it worries me.
        elif mfile['mlayers'].size > 1:
            showtext('warning, may be multiplying by netmag twice')
            mlayers = mfile['mlayers']*mlist[1].netmagn()
###############################################################################
        else:
            mlayers = np.zeros_like(mfile['glayers'])

        if not magcalc:
            glayers = mfile['glayers']*mlist[1].rho()
        elif mfile['glayers'].size > 1:
            glayers = mfile['glayers']*mlist[1].rho()
        else:
            glayers = np.zeros_like(mfile['mlayers'])


#        glayers = mfile['glayers']*mlist[1].rho()
        if showreports is True:
            showtext('Summing '+mlist[0]+' (PyGMI may become non-responsive' +
                     ' during this calculation)')

        if abs(np.sum(modind == -1)) < modind.size and mijk in modind:
            QtGui.QApplication.processEvents()
            i, j, k = np.nonzero(modind == mijk)
            iuni = np.array(np.unique(i), dtype=np.int32)
            juni = np.array(np.unique(j), dtype=np.int32)
            kuni = np.array(np.unique(k), dtype=np.int32)

            for k in kuni:
                baba = calc_fieldb(k, mgval, numx, numy, modind, aaa[0],
                                   aaa[1], mlayers, glayers, hcorflat, mijk,
                                   juni, iuni)
                magval += baba[0]
                grvval += baba[1]

        if abs(np.sum(cmodind == -1)) < cmodind.size and mijk in cmodind:
            if showreports is True:
                showtext('subtracting')
            QtGui.QApplication.processEvents()
            i, j, k = np.nonzero(cmodind == mijk)
            iuni = np.array(np.unique(i), dtype=np.int32)
            juni = np.array(np.unique(j), dtype=np.int32)
            kuni = np.array(np.unique(k), dtype=np.int32)

            for k in kuni:
                baba = calc_fieldb(k, mgval, numx, numy, modind, aaa[0],
                                   aaa[1], mlayers, glayers, hcorflat, mijk,
                                   juni, iuni)
                magval += baba[0]
                grvval += baba[1]

        if showreports is True:
            showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtGui.QApplication.processEvents()

    magval.resize(mtmp)
    grvval.resize(mtmp)
    magval = magval.T
    grvval = grvval.T
    magval = magval[::-1]
    grvval = grvval[::-1]

#    magval += lmod.griddata['Calculated Magnetics'].data
#    grvval += lmod.griddata['Calculated Gravity'].data

# Update variables
    lmod.griddata['Calculated Magnetics'].data += magval
    lmod.griddata['Calculated Gravity'].data += grvval

# This addoldcalc has has flaws w.r.t. regional if you change the regional
    if 'Gravity Regional' in lmod.griddata and lmod.clith_index.max() == 0:
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
        parent.outdata['Raster'] = list(lmod.griddata.values())

    if showreports is True:
        showtext('Calculation Finished')
    if pbars is not None:
        pbars.maxall()

    lmod.clith_index = lmod.lith_index.copy()

    return lmod.griddata


#@jit("f8[:,:](i4, f8[:,:], i4, i4, i4[:,:,:], i4[:], i4[:], " +
#     "f8[:,:,:], f8[:,:,:], i4[:], i4, i4[:], i4[:])", nopython=True)
@jit(nopython=True)
def calc_fieldb(k, mgval, numx, numy, modind, aaa0, aaa1, mlayers,
                glayers, hcorflat, mijk, jj, ii):
    """ Calculate magnetic and gravity field """

    b = numx*numy
    for i in range(2):
        for j in range(b):
            mgval[i, j] = 0.

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
                mgval[0, ijk] += mlayers[hcor2, xoff2, yoff2]
                mgval[1, ijk] += glayers[hcor2, xoff2, yoff2]

    return mgval


def quick_model(numx=50, numy=50, numz=50, dxy=1000, d_z=100,
                tlx=0, tly=0, tlz=0, mht=100, ght=0, finc=-67, fdec=-17,
                inputliths=['Generic'], susc=[0.01], dens=[3.0],
                minc=None, mdec=None, mstrength=None):
    """ Create a quick model """

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
        if mstrength is not None:
            lmod.lith_list[i].minc = minc[j-1]
            lmod.lith_list[i].mdec = mdec[j-1]
            lmod.lith_list[i].mstrength = mstrength[j-1]

    return lmod


#@jit(float64[:,:](float64[:,:], float64[:], float64[:], int32, int32, float64,
#     float64, float64, float64, float64, float64, float64,
#     float64[:], float64[:], float64[:], int32[:]), nopython=True)
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


#@jit("f8[:,:,:](i4, i4, f8[:,:], f8[:,:], f8[:,:], f8[:,:], i4[:,:], " +
#     "f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:], " +
#     "i4[:], f8[:,:], f8[:], f8[:], f8[:], f8[:,:,:])",
#     nopython=True)
@jit(nopython=True)
def gm3d(npro, nstn, X, Y, edge, corner, face, Hx, Hy, Hz, Pd, Un,
         indx, crs, p1, p2, p3, mgval):
    """ grvmag 3d """

    flimit = 64*np.spacing(1)
    omega = 0.0
    dp1 = 1.0
    I = 1.0

    for pr in range(npro):
        for st in range(nstn):
            x = X[pr, st]
            y = Y[pr, st]
            for f in range(6):  # 6 Faces
                for g in range(4):  # 4 points in a face
                    cindx = face[f, g]
                    crs[g, 0] = corner[cindx, 0] - x
                    crs[g, 1] = corner[cindx, 1] - y
                    crs[g, 2] = corner[cindx, 2]

                p = 0
                q = 0
                r = 0
                eno1 = 4*f
                W = -2*np.pi
                l = Un[f, 0]
                m = Un[f, 1]
                n = Un[f, 2]

                for t in range(4):
                    p1[0] = crs[indx[t], 0]
                    p1[1] = crs[indx[t], 1]
                    p1[2] = crs[indx[t], 2]
                    p2[0] = crs[indx[t+1], 0]
                    p2[1] = crs[indx[t+1], 1]
                    p2[2] = crs[indx[t+1], 2]
                    p3[0] = crs[indx[t+2], 0]
                    p3[1] = crs[indx[t+2], 1]
                    p3[2] = crs[indx[t+2], 2]
###############################################################################
                    ang = 0
                    anout = p1[0]*l+p1[1]*m+p1[2]*n

                    if anout > flimit:
                        n10 = p2[1]*p3[2] - p2[2]*p3[1]
                        n11 = p2[2]*p3[0] - p2[0]*p3[2]
                        n12 = p2[0]*p3[1] - p2[1]*p3[0]

                        n20 = p2[1]*p1[2] - p2[2]*p1[1]
                        n21 = p2[2]*p1[0] - p2[0]*p1[2]
                        n22 = p2[0]*p1[1] - p2[1]*p1[0]

                    else:
                        n10 = p2[1]*p1[2] - p2[2]*p1[1]
                        n11 = p2[2]*p1[0] - p2[0]*p1[2]
                        n12 = p2[0]*p1[1] - p2[1]*p1[0]

                        n20 = p2[1]*p3[2] - p2[2]*p3[1]
                        n21 = p2[2]*p3[0] - p2[0]*p3[2]
                        n22 = p2[0]*p3[1] - p2[1]*p3[0]

                    pn1 = np.sqrt(n10*n10+n11*n11+n12*n12)
                    pn2 = np.sqrt(n20*n20+n21*n21+n22*n22)

                    if (pn1 <= flimit) or (pn2 <= flimit):
                        ang = np.nan
                    else:
                        n10 = n10/pn1
                        n11 = n11/pn1
                        n12 = n12/pn1
                        n20 = n20/pn2
                        n21 = n21/pn2
                        n22 = n22/pn2

                        rrr = n10*n20+n11*n21+n12*n22
                        ang = np.arccos(rrr)

                        perp = n10*p3[0]+n11*p3[1]+n12*p3[2]
                        if anout > flimit:
                            perp = n10*p1[0]+n11*p1[1]+n12*p1[2]

                        if perp < -flimit:        # points p1,p2,p3 in cw order
                            ang = 2*np.pi-ang

                    if abs(anout) <= flimit:
                        ang = 0
###############################################################################
                    W += ang

                    eno2 = eno1+t   # Edge no
                    L = edge[eno2, 3]

                    r12 = (np.sqrt(p1[0]*p1[0]+p1[1]*p1[1]+p1[2]*p1[2]) +
                           np.sqrt(p2[0]*p2[0]+p2[1]*p2[1]+p2[2]*p2[2]))
                    I = (1/L)*np.log((r12+L)/(r12-L))

                    p += I*edge[eno2, 0]
                    q += I*edge[eno2, 1]
                    r += I*edge[eno2, 2]

        #        From omega, l, m, n PQR get components of field due to face f
                # dp1 is dot product between (l,m,n) and (x,y,z) or Un and r.
                dp1 = l*crs[0, 0]+m*crs[0, 1]+n*crs[0, 2]
                if dp1 < 0.:
                    omega = W
                else:
                    omega = -W

                # l, m, n and components of unit normal to a face.
                gmtf1 = l*omega+n*q-m*r
                gmtf2 = m*omega+l*r-n*p
                gmtf3 = n*omega+m*p-l*q

                Hx[pr, st] = Hx[pr, st]+Pd[f]*gmtf1
                Hy[pr, st] = Hy[pr, st]+Pd[f]*gmtf2
                Hz[pr, st] = Hz[pr, st]+Pd[f]*gmtf3

#                Gx[pr, st] = Gx[pr, st]-dp1*gmtf1
#                Gy[pr, st] = Gy[pr, st]-dp1*gmtf2
#                Gz[pr, st] = Gz[pr, st]-dp1*gmtf3

    for pr in range(npro):
        for st in range(nstn):
            mgval[0, pr, st] = Hx[pr, st]
            mgval[1, pr, st] = Hy[pr, st]
            mgval[2, pr, st] = Hz[pr, st]
#            mgval[3, pr, st] = Gx[pr, st]
#            mgval[4, pr, st] = Gy[pr, st]
#            mgval[5, pr, st] = Gz[pr, st]

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
