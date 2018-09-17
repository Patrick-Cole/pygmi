# -----------------------------------------------------------------------------
# Name:        tensor3d.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2017 Council for Geoscience
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

import pdb
import copy
import tempfile
from math import sqrt, atan2, log, atan
from multiprocessing import Pool
from PyQt5 import QtWidgets, QtCore
import winsound

import numpy as np
import scipy.interpolate as si
from osgeo import gdal
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
from pygmi.raster.dataprep import gdal_to_dat
from pygmi.raster.dataprep import data_to_gdal_mem
from pygmi.pfmod.datatypes import LithModel
from pygmi.misc import PTime


class TensorCube(object):
    """
    This class computes the forward modelled tensor responses for a cube.
    """
    def __init__(self):

        self.minc = -60.0
        self.mdec = -15.0
        self.mstrength = 0
        self.inc = -60.0
        self.dec = -15.0
        self.hintn = 28000
        self.azim = 90
        self.dxy = 10
        self.susc = 0.1
        self.dens = 2.85
        self.bdens = 2.67
        self.height = 0.0
        self.Gc = 6.6732e-3  # includes 100000 factor to convert to mGal


# the 0.5 values below are to avoid divide by zero errors later.
        self.u = [100, 300]
        self.v = [100, 300]
        self.w = [-20, -300]
        self.rc = 400

        self.cx = None
        self.cy = None
        self.cz = None
        self.pmag = None
        self.pbx = None
        self.pby = None
        self.pbz = None

        self.pgrv = None
        self.pgx = None
        self.pgy = None
        self.pgz = None

        self.bx = None
        self.by = None
        self.bz = None
        self.bxx = None
        self.byy = None
        self.bzz = None
        self.bxy = None
        self.byz = None
        self.bxz = None
        self.magval = None

        self.gx = None
        self.gy = None
        self.gz = None
        self.gxx = None
        self.gyy = None
        self.gzz = None
        self.gxy = None
        self.gyz = None
        self.gxz = None
        self.grvval = None
        self.xyall = None
        self.coords = None

    def calc_pygmi(self):
        """ Do pygmi calc """

        finc = self.inc
        fdec = self.dec
        dxy = self.dxy
        d_z = self.dxy
        numx = int(self.rc//dxy)
        numy = int(self.rc//dxy)
        numz = int(self.rc//d_z)
        numz = abs(int(self.w[1]//d_z))

        tlx = 0.
        tly = 0.
        tlz = 0.
        mht = 0
        ght = 0
        hintn = self.hintn
        u1 = int(self.u[0]//dxy)
        u2 = int(self.u[1]//dxy)
        v1 = int(self.v[0]//dxy)
        v2 = int(self.v[1]//dxy)
        w1 = abs(int(self.w[0]//dxy))
        w2 = abs(int(self.w[1]//dxy))

        inputliths = ['Generic']
        susc = [self.susc]
        dens = [self.dens]
        minc = [self.minc]
        mdec = [self.mdec]
        mstrength = [self.mstrength]

        lmod = quick_model(numx, numy, numz, dxy, d_z, tlx, tly, tlz, mht, ght,
                           finc, fdec, inputliths, susc, dens, minc, mdec,
                           mstrength, hintn)

        lmod.lith_index[u1:u2, v1:v2, w1:w2] = 1
        magobs = calc_field(lmod)
        self.pmag = magobs['Calculated Magnetics'].data
        self.pbx = magobs['Calculated Magnetics x']
        self.pby = magobs['Calculated Magnetics y']
        self.pbz = magobs['Calculated Magnetics z']

        self.pgrv = magobs['Calculated Gravity'].data
        self.pgx = magobs['Calculated Gravity x']
        self.pgy = magobs['Calculated Gravity y']
        self.pgz = magobs['Calculated Gravity z']

    def calc_mag(self, xobs=None, yobs=None):
        """ calc all """
        if xobs is None or yobs is None:
            return

        tmp = (xobs.size, yobs.size)

        self.bx = np.zeros(tmp)
        self.by = np.zeros(tmp)
        self.bz = np.zeros(tmp)
        self.bxx = np.zeros(tmp)
        self.byy = np.zeros(tmp)
        self.bzz = np.zeros(tmp)
        self.bxy = np.zeros(tmp)
        self.byz = np.zeros(tmp)
        self.bxz = np.zeros(tmp)

        ma, mb, mc = dircos(self.minc, self.mdec, self.azim)
        fa, fb, fc = dircos(self.inc, self.dec, self.azim)

        mr = self.mstrength*np.array([ma, mb, mc])*100
        mi = self.susc*self.hintn/(4*np.pi)*np.array([fa, fb, fc])
        m3 = mr+mi
        m = np.sqrt(m3 @ m3)
        m3 /= m
        self.cx, self.cy, self.cz = m3

        const = m

        for j, y in enumerate(yobs):
            for i, x in enumerate(xobs):
                self.bx[i, j] = self.fsum(self.Bx, x, y, self.height)
                self.by[i, j] = self.fsum(self.By, x, y, self.height)
                self.bz[i, j] = self.fsum(self.Bz, x, y, self.height)
                self.bxx[i, j] = self.fsum(self.Bxx, x, y, self.height)
                self.byy[i, j] = self.fsum(self.Byy, x, y, self.height)
                self.bzz[i, j] = self.fsum(self.Bzz, x, y, self.height)
                self.bxy[i, j] = self.fsum(self.Bxy, x, y, self.height)
                self.byz[i, j] = self.fsum(self.Byz, x, y, self.height)
                self.bxz[i, j] = self.fsum(self.Bxz, x, y, self.height)
#                self.coords[-i-1, j, 0] = x
#                self.coords[-i-1, j, 1] = y

        self.bx *= const
        self.by *= const
        self.bz *= const
        self.bxx *= const
        self.byy *= const
        self.bzz *= const
        self.byz *= const
        self.bxz *= const
        self.bxy *= const

        self.magval = self.cx*self.bx+self.cy*self.by+self.cz*self.bz
        self.magval2 = np.sqrt((self.bx+self.hintn*self.cx)**2 +
                               (self.by+self.hintn*self.cy)**2 +
                               (self.bz+self.hintn*self.cz)**2)-self.hintn

    def calc_grav(self, xobs=None, yobs=None):
        """ calc all """
        if xobs is None or yobs is None:
            return

        tmp = (xobs.size, yobs.size)

        self.gx = np.zeros(tmp)
        self.gy = np.zeros(tmp)
        self.gz = np.zeros(tmp)
        self.gxx = np.zeros(tmp)
        self.gyy = np.zeros(tmp)
        self.gzz = np.zeros(tmp)
        self.gxy = np.zeros(tmp)
        self.gyz = np.zeros(tmp)
        self.gxz = np.zeros(tmp)


        for i, x in enumerate(xobs):
            for j, y in enumerate(yobs):
                self.gx[-i-1, j] = self.fsum(self.Gx, x, y, 0.0)
                self.gy[-i-1, j] = self.fsum(self.Gy, x, y, 0.0)
                self.gz[-i-1, j] = self.fsum(self.Gz, x, y, 0.0)
                self.gxx[-i-1, j] = self.fsum(self.Gxx, x, y, 0.0)
                self.gyy[-i-1, j] = self.fsum(self.Gyy, x, y, 0.0)
                self.gzz[-i-1, j] = self.fsum(self.Gzz, x, y, 0.0)
                self.gxy[-i-1, j] = self.fsum(self.Gxy, x, y, 0.0)
                self.gyz[-i-1, j] = self.fsum(self.Gyz, x, y, 0.0)
                self.gxz[-i-1, j] = self.fsum(self.Gxz, x, y, 0.0)

        constg = (self.dens-self.bdens)*self.Gc
        self.gx *= constg
        self.gy *= constg
        self.gz *= constg
        self.gxx *= constg
        self.gyy *= constg
        self.gzz *= constg
        self.gyz *= constg
        self.gxz *= constg
        self.gxy *= constg

        self.grvval = self.gz

    def fsum(self, func, x, y, z):
        """ function """
        # y and z sign reversals convert this from ENU to ESD (dircos with 90)

        x1 = (x-self.u[0])
        x2 = (x-self.u[1])
        y1 = -(y-self.v[0])
        y2 = -(y-self.v[1])
        z1 = -(z-self.w[0])
        z2 = -(z-self.w[1])

        tmp = (func(x2, y2, z2) - func(x1, y2, z2) - func(x2, y1, z2) +
               func(x1, y1, z2) - func(x2, y2, z1) + func(x1, y2, z1) +
               func(x2, y1, z1) - func(x1, y1, z1))

        return tmp

    def getabg(self):
        """ gets alpha, beta and gamma """

        a, b, g = dircos(self.inc, self.dec, self.azim)

        return a, b, g

    def Gx(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -1*x*np.arctan2((y*z), (x*r))

        if y != 0:
            tmp += y*np.log(z+r)

        if z != 0:
            tmp += z*np.log(y+r)

        return tmp

    def Gy(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -1*y*np.arctan2(x*z, y*r)

        if x != 0:
            tmp += x*np.log(z+r)

        if z != 0:
            tmp += z*np.log(x+r)

        return tmp

    def Gz(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -1*z*np.arctan((x*y)/(z*r))+x*np.log(y+r)+y*np.log(x+r)

        return tmp

    def Gxx(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -np.arctan2((y*z), (x*r))

        return tmp

    def Gyy(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -np.arctan2((x*z), (y*r))

        return tmp

    def Gzz(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        tmp = -np.arctan2((x*y), (z*r))

        return tmp

    def Gxy(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)

        if z+r == 0:
            tmp = np.nan
        else:
            tmp = np.log(z+r)
        return tmp

    def Gyz(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)
        tmp = np.log(x+r)
        return tmp

    def Gxz(self, x, y, z):
        """ function """
        r = np.sqrt(x**2+y**2+z**2)
        tmp = np.log(y+r)
        return tmp

    def Bx(self, x, y, z):
        """ function """
        a, b, g = self.getabg()

        tmp = a*self.Gxx(x, y, z) + b*self.Gxy(x, y, z) + g*self.Gxz(x, y, z)

        return tmp

    def By(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)
        tmp = -(-a*np.log(z+r) + b*np.arctan2((x*z), (y*r)) - g*np.log(x+r))

        return tmp

    def Bz(self, x, y, z):
        """ function """
        a, b, g = self.getabg()

        tmp = a*self.Gxz(x, y, z) + b*self.Gyz(x, y, z) + g*self.Gzz(x, y, z)

        return tmp

    def Bxx(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = (a*y*z*(r**2 + x**2)/(r*(r**2*x**2 + y**2*z**2)) +
               b*x/(r**2 + r*z) + g*x/(r**2 + r*y))

        return tmp

    def Byy(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = (a*y/(r**2 + r*z) +
               b*x*z*(r**2 + y**2)/(r*(r**2*y**2 + x**2*z**2)) +
               g*y/(r**2 + r*x))

        return tmp

    def Bzz(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = (a*z/(r**2 + r*y) + b*z/(r**2 + r*x) +
               g*x*y*(r**2 + z**2)/(r*(r**2*z**2 + x**2*y**2)))
        return tmp

    def Bxy(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = -a*x*z/(r*(x**2 + y**2)) + b*y/(r**2 + r*z) + g/r

        return tmp

    def Byz(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = a/r - b*x*y/(r*(y**2 + z**2)) + g*z/(r**2 + r*x)

        return tmp

    def Bxz(self, x, y, z):
        """ function """
        a, b, g = self.getabg()
        r = np.sqrt(x**2+y**2+z**2)

        tmp = -a*x*y/(r*(x**2 + z**2)) + b/r + g*z/(r**2 + r*y)

        return tmp



class TensorCubeOld(object):
    """
    This class computes the forward modelled tensor responses for a cube.
    """
    def __init__(self):

        self.minc = -62.0
        self.mdec = -16.0
        self.mstrength = 0
        self.inc = -62.0
        self.dec = -16.0
        self.hintn = 28000
        self.azim = 90
        self.dxy = 10
        self.susc = 0.1
        self.dens = 3.0
        self.bdens = 2.67
        self.height = 0.0
        self.Gc = 6.6732e-3  # includes 100000 factor to convert to mGal

# the 0.5 values below are to avoid divide by zero errors later.
        self.u = [100.5, 300.5]
        self.v = [100.5, 300.5]
        self.w = [-20.5, -300.5]

        self.a, self.b, self.g = dircos(self.inc, self.dec, self.azim)

        self.pmag = None
        self.pbx = None
        self.pby = None
        self.pbz = None

        self.pgrv = None
        self.pgx = None
        self.pgy = None
        self.pgz = None

        self.bx = None
        self.by = None
        self.bz = None
        self.bxx = None
        self.byy = None
        self.bzz = None
        self.bxy = None
        self.byz = None
        self.bxz = None
        self.magval = None

        self.gx = None
        self.gy = None
        self.gz = None
        self.gxx = None
        self.gyy = None
        self.gzz = None
        self.gxy = None
        self.gyz = None
        self.gxz = None
        self.grvval = None
        self.xyall = None

    def calc_mag(self, xobs=None, yobs=None):
        """ calc mag """
        if xobs is None or yobs is None:
            return

        tmp = (xobs.size, yobs.size)

        self.bx = np.zeros(tmp)
        self.by = np.zeros(tmp)
        self.bz = np.zeros(tmp)
        self.bxx = np.zeros(tmp)
        self.byy = np.zeros(tmp)
        self.bzz = np.zeros(tmp)
        self.bxy = np.zeros(tmp)
        self.byz = np.zeros(tmp)
        self.bxz = np.zeros(tmp)

        ma, mb, mc = dircos(self.minc, self.mdec, self.azim)
        fa, fb, fc = dircos(self.inc, self.dec, self.azim)

        mr = self.mstrength*np.array([ma, mb, mc])
        mi = self.susc*self.hintn/(400*np.pi)*np.array([fa, fb, fc])
        m3 = mr+mi
        m = np.sqrt(m3 @ m3)
        m3 /= m
        self.a, self.b, self.g = m3

        hnew = m*(400*np.pi/self.susc)

        for i, x in enumerate(xobs):
            for j, y in enumerate(yobs):
                self.bx[i, j] = self.fsum(self.Bx, x, y, self.height)
                self.by[i, j] = self.fsum(self.By, x, y, self.height)
                self.bz[i, j] = self.fsum(self.Bz, x, y, self.height)
                self.bxx[i, j] = self.fsum(self.Bxx, x, y, self.height)
                self.byy[i, j] = self.fsum(self.Byy, x, y, self.height)
                self.bzz[i, j] = self.fsum(self.Bzz, x, y, self.height)
                self.bxy[i, j] = self.fsum(self.Bxy, x, y, self.height)
                self.byz[i, j] = self.fsum(self.Byz, x, y, self.height)
                self.bxz[i, j] = self.fsum(self.Bxz, x, y, self.height)

        const = self.susc*hnew/(4*np.pi)
        const = 100*m

#        self.bx = self.regrid(self.bx)*const
#        self.by = self.regrid(self.by)*const
#        self.bz = self.regrid(self.bz)*const
#        self.bxx = self.regrid(self.bxx)*const
#        self.byy = self.regrid(self.byy)*const
#        self.bzz = self.regrid(self.bzz)*const
#        self.bxy = self.regrid(self.bxy)*const
#        self.byz = self.regrid(self.byz)*const
#        self.bxz = self.regrid(self.bxz)*const

        self.bx = self.bx*const
        self.by = self.by*const
        self.bz = self.bz*const
        self.bxx = self.bxx*const
        self.byy = self.byy*const
        self.bzz = self.bzz*const
        self.bxy = self.bxy*const
        self.byz = self.byz*const
        self.bxz = self.bxz*const

        self.magval = self.a*self.bx+self.b*self.by+self.g*self.bz

    def calc_grav(self, xobs=None, yobs=None):
        """ calc grav """
        if xobs is None or yobs is None:
            return

        tmp = (xobs.size, yobs.size)

        self.gx = np.zeros(tmp)
        self.gy = np.zeros(tmp)
        self.gz = np.zeros(tmp)
        self.gxx = np.zeros(tmp)
        self.gyy = np.zeros(tmp)
        self.gzz = np.zeros(tmp)
        self.gxy = np.zeros(tmp)
        self.gyz = np.zeros(tmp)
        self.gxz = np.zeros(tmp)

        for i, x in enumerate(xobs):
            for j, y in enumerate(yobs):
                self.gx[i, j] = self.fsum(self.Gx, x, y, self.height)
                self.gy[i, j] = self.fsum(self.Gy, x, y, self.height)
                self.gz[i, j] = self.fsum(self.Gz, x, y, self.height)
                self.gxx[i, j] = self.fsum(self.Gxx, x, y, self.height)
                self.gyy[i, j] = self.fsum(self.Gyy, x, y, self.height)
                self.gzz[i, j] = self.fsum(self.Gzz, x, y, self.height)
                self.gxy[i, j] = self.fsum(self.Gxy, x, y, self.height)
                self.gyz[i, j] = self.fsum(self.Gyz, x, y, self.height)
                self.gxz[i, j] = self.fsum(self.Gxz, x, y, self.height)

        const = (self.dens-self.bdens)*self.Gc

        self.gx = self.gx*const
        self.gy = self.gy*const
        self.gz = self.gz*const
        self.gxx = self.gxx*const
        self.gyy = self.gyy*const
        self.gzz = self.gzz*const
        self.gxy = self.gxy*const
        self.gyz = self.gyz*const
        self.gxz = self.gxz*const
        self.grvval = self.gz

    def fsum(self, func, x, y, z):
        """ function """
        x1 = (x-self.u[0])
        x2 = (x-self.u[1])
        y1 = -(y-self.v[0])
        y2 = -(y-self.v[1])
        z1 = -(z-self.w[0])
        z2 = -(z-self.w[1])

        tmp = (func(x2, y2, z2) - func(x1, y2, z2) - func(x2, y1, z2) +
               func(x1, y1, z2) - func(x2, y2, z1) + func(x1, y2, z1) +
               func(x2, y1, z1) - func(x1, y1, z1))

        return tmp

    def getabg(self):
        """ gets alpha, beta and gamma """

        a, b, g = dircos(self.inc, self.dec, self.azim)
        return a, b, g

    def regrid(self, data):
        """ fills holes """
        mask = np.logical_not(np.isnan(data))

        xx, yy = np.meshgrid(np.arange(data.shape[1]),
                             np.arange(data.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
        data0 = np.ravel(data[:, :][mask])
        interp0 = si.NearestNDInterpolator(xym, data0)
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

        result0 = np.ma.masked_invalid(result0)

        return result0

    def Gx(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -x*atan2(y*z, x*r)

        if y != 0:
            tmp += y*log(z+r)

        if z != 0:
            tmp += z*log(y+r)

        return tmp

    def Gy(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -y*atan2(x*z, y*r)

        if x != 0:
            tmp += x*log(z+r)

        if z != 0:
            tmp += z*log(x+r)

        return tmp

    def Gz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -z*atan((x*y)/(z*r))

        if x != 0:
            tmp += x*log(y+r)
        if y != 0:
            tmp += y*log(x+r)

        return tmp

    def Gxx(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -atan2(y*z, x*r)
#        tmp = -atan((y*z)/(x*r))

        return tmp

    def Gyy(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -atan((x*z)/(y*r))
#        tmp = -np.arctan((x*z)/(y*r))

        return tmp

    def Gzz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -atan((x*y)/(z*r))

        return tmp

    def Gxy(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        if z+r == 0:
            tmp = np.nan
        else:
            tmp = log(z+r)
        return tmp

    def Gyz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)
        tmp = log(x+r)
        return tmp

    def Gxz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)
        tmp = log(y+r)
        return tmp

    def Bx(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = (self.a*self.Gxx(x, y, z) + self.b*self.Gxy(x, y, z) +
               self.g*self.Gxz(x, y, z))
#        tmp = -self.a*atan2(y*z, x*r) + self.b*log(z+r) + self.g*log(y+r)

        return tmp

    def By(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = (self.a*self.Gxy(x, y, z) + self.b*self.Gyy(x, y, z) +
               self.g*self.Gyz(x, y, z))
#        tmp = self.a*log(z+r) - self.b*atan2(x*z, y*r) + self.g*log(x+r)

        return tmp

    def Bz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = (self.a*self.Gxz(x, y, z) + self.b*self.Gyz(x, y, z) +
               self.g*self.Gzz(x, y, z))

#        tmp = (self.a*log(y+r) + self.b*log(x+r) - self.g*atan2(x*y, z*r))
        return tmp

    def Bxx(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        if x == 0 and y == 0:
            tmp = 0
        else:
            tmp = (self.a*y*z*(r**2 + x**2)/(r*(r**2*x**2 + y**2*z**2)) +
                   self.b*x/(r**2 + r*z) + self.g*x/(r**2 + r*y))

        return tmp

    def Byy(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        if x == 0 and y == 0:
            tmp = 0
        else:
            tmp = (self.a*y/(r**2 + r*z) +
                   self.b*x*z*(r**2 + y**2)/(r*(r**2*y**2 + x**2*z**2)) +
                   self.g*y/(r**2 + r*x))
        return tmp

    def Bzz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = (self.a*z/(r**2 + r*y) + self.b*z/(r**2 + r*x) +
               self.g*x*y*(r**2 + z**2)/(r*(r**2*z**2 + x**2*y**2)))
        return tmp

    def Bxy(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        if x == 0 and y == 0:
            tmp = self.g/r
        else:
            tmp = (-self.a*x*z/(r*(x**2 + y**2)) + self.b*y/(r**2 + r*z) +
                   self.g/r)

        return tmp

    def Byz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = self.a/r - self.b*x*y/(r*(y**2 + z**2)) + self.g*z/(r**2 + r*x)

        return tmp

    def Bxz(self, x, y, z):
        """ function """
        r = sqrt(x**2+y**2+z**2)

        tmp = -self.a*x*y/(r*(x**2 + z**2)) + self.b/r + self.g*z/(r**2 + r*y)

        return tmp


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
#        self.parent.pview.viewmagnetics = True
        self.parent.profile.viewmagnetics = True

        self.lmod.lith_index_old[:] = -1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True, True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_plot()

        self.actioncalculate4.setEnabled(True)

    def calc_field_grav(self):
        """ Pre field-calculation routine """
        # Update this
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False
#        self.parent.pview.viewmagnetics = False

        self.lmod.lith_index_old[:] = -1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_plot()

        self.actioncalculate3.setEnabled(True)

    def calc_field_mag_changes(self):
        """ calculates only mag changes """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
#        self.parent.pview.viewmagnetics = True
        self.parent.profile.viewmagnetics = True

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True, True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_plot()

    def calc_field_grav_changes(self):
        """ calculates only grav changes """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1
        self.parent.profile.viewmagnetics = False
#        self.parent.pview.viewmagnetics = False

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_model()

        # now do the calculations
        self.calc_field2(True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

#        if tlabel == 'Custom Profile Editor':
#            self.parent.pview.update_plot()

    def calc_field2(self, showreports=False, magcalc=False):
        """ Calculate magnetic and gravity field """

        if magcalc:
            calc_mag_field(self.lmod, pbars=self.pbars, showtext=self.showtext,
                           parent=self.parent, showreports=showreports)
        else:
            calc_grv_field(self.lmod, pbars=self.pbars, showtext=self.showtext,
                           parent=self.parent, showreports=showreports)

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
        axes = plt.subplot(121)
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

        axes = plt.subplot(122)
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
        self.mtmp = None
        self.glayers = None

        self.x12 = None
        self.y12 = None
        self.z12 = None

        self.set_xyz(ncols, nrows, numz, dxy, mht, ght, d_z)

    def calc_origin_grav(self, hcor=None):
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

            if hcor is None:
                hcor2 = 0
            else:
                hcor2 = int(self.numz-hcor.max())

            self.showtext('   Calculate gravity origin field')
            self.gboxmain(xdist, ydist, self.zobsg, hcor2)

            self.modified = False

    def calc_origin_mag(self, hcor=None):
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

            if hcor is None:
                hcor2 = 0
            else:
                hcor2 = int(self.numz-hcor.max())

            self.mboxmain(xdist, ydist, self.zobsm, hcor2)

            self.modified = False

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
        self.zobsm = mht
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

    def gboxmain(self, xobs, yobs, zobs, hcor):
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

        tcube = TensorCube()

        grvval = []
        gx = []
        gy = []
        gz = []
        gxx = []
        gxy = []
        gxz = []
        gyy = []
        gyz = []
        gzz = []

        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter

        tcube.bdens = self.bdensity
        tcube.dens = self.density
        tcube.dxy = self.dxy
        tcube.height = abs(zobs)
        tcube.u = self.x12
        tcube.v = self.y12

        z1122 = self.z12.copy()
        for z1 in piter(z1122[:-1]):
            if z1 < z1122[hcor]:
                grvval.append(np.zeros((self.g_cols, self.g_rows)))
                gx.append(np.zeros((self.g_cols, self.g_rows)))
                gy.append(np.zeros((self.g_cols, self.g_rows)))
                gz.append(np.zeros((self.g_cols, self.g_rows)))
                gxx.append(np.zeros((self.g_cols, self.g_rows)))
                gxy.append(np.zeros((self.g_cols, self.g_rows)))
                gxz.append(np.zeros((self.g_cols, self.g_rows)))
                gyy.append(np.zeros((self.g_cols, self.g_rows)))
                gyz.append(np.zeros((self.g_cols, self.g_rows)))
                gzz.append(np.zeros((self.g_cols, self.g_rows)))
                continue

            z2 = z1 + self.d_z

            print(-z1, -z2)
            tcube.w = [-z1, -z2]
            tcube.calc_grav(xobs, yobs)

            grvval.append(tcube.grvval)
            gx.append(tcube.gx)
            gy.append(tcube.gy.copy())
            gz.append(tcube.gz)
            gxx.append(tcube.gxx)
            gxy.append(tcube.gxy)
            gxz.append(tcube.gxz)
            gyy.append(tcube.gyy)
            gyz.append(tcube.gyz)
            gzz.append(tcube.gzz)

        self.mlayers = {}
        self.mlayers['Gravity'] = np.array(grvval)
        self.mlayers['gx'] = np.array(gx)
        self.mlayers['gy'] = np.array(gy)
        self.mlayers['gz'] = np.array(gz)
        self.mlayers['gxx'] = np.array(gxx)
        self.mlayers['gxy'] = np.array(gxy)
        self.mlayers['gxz'] = np.array(gxz)
        self.mlayers['gyy'] = np.array(gyy)
        self.mlayers['gyz'] = np.array(gyz)
        self.mlayers['gzz'] = np.array(gzz)


    def mboxmain(self, xobs, yobs, zobs, hcor):
        """ Mbox routine by Blakely
            Note: xobs, yobs and zobs must be floats or there will be problems
            later.

        Subroutine MBOX computes the total field anomaly of an infinitely
        extended rectangular prism.  Sides of prism are parallel to x,y,z
        axes, and z is vertical down.  Bottom of prism extends to infinity.
        Two calls to mbox can provide the anomaly of a prism with finite
        thickness; e.g.,

            call mbox(x0,y0,z0,x1,y1,z1,x2,y2,mi,md,fi,fd,m,theta,t1)
            call mbox(x0,y0,z0,x1,y1,z2,x2,y2,mi,md,fi,fd,m,theta,t2)
            t=t1-t2

        Requires subroutine DIRCOS.  Method from Bhattacharyya (1964).

        Input parameters:
            Observation point is (x0,y0,z0).  Prism extends from x1 to
            x2, y1 to y2, and z1 to infinity in x, y, and z directions,
            respectively.  Magnetization defined by inclination mi,
            declination md, intensity m.  Ambient field defined by
            inclination fi and declination fd.  X axis has declination
            theta. Distance units are irrelevant but must be consistent.
            Angles are in degrees, with inclinations positive below
            horizontal and declinations positive east of true north.
            Magnetization in A/m.

        Output paramters:
            Total field anomaly t, in nT."""

        tcube = TensorCube()

        magval = []
        bx = []
        by = []
        bz = []
        bxx = []
        bxy = []
        bxz = []
        byy = []
        byz = []
        bzz = []

        if self.pbars is not None:
            piter = self.pbars.iter
        else:
            piter = iter

        tcube.inc = self.finc
        tcube.dec = self.fdec
        tcube.susc = self.susc
        tcube.mstrength = self.mstrength
        tcube.mdec = self.mdec
        tcube.minc = self.minc
        tcube.hintn = self.hintn
        tcube.dxy = self.dxy
        tcube.height = abs(zobs)
        tcube.u = self.x12
        tcube.v = self.y12
        tcube.w = -self.z12
        z1122 = self.z12.copy()

        for z1 in piter(z1122[:-1]):
            if z1 < z1122[hcor]:
                magval.append(np.zeros((self.g_cols, self.g_rows)))
                bx.append(np.zeros((self.g_cols, self.g_rows)))
                by.append(np.zeros((self.g_cols, self.g_rows)))
                bz.append(np.zeros((self.g_cols, self.g_rows)))
                bxx.append(np.zeros((self.g_cols, self.g_rows)))
                bxy.append(np.zeros((self.g_cols, self.g_rows)))
                bxz.append(np.zeros((self.g_cols, self.g_rows)))
                byy.append(np.zeros((self.g_cols, self.g_rows)))
                byz.append(np.zeros((self.g_cols, self.g_rows)))
                bzz.append(np.zeros((self.g_cols, self.g_rows)))
                continue

#            z2 = z1 - self.d_z
            z2 = z1 + self.d_z
            print(-z1, -z2)
            tcube.w = [-z1, -z2]
            tcube.calc_mag(xobs, yobs)

            magval.append(tcube.magval)
            bx.append(tcube.bx)
            by.append(tcube.by)
            bz.append(tcube.bz)
            bxx.append(tcube.bxx)
            bxy.append(tcube.bxy)
            bxz.append(tcube.bxz)
            byy.append(tcube.byy)
            byz.append(tcube.byz)
            bzz.append(tcube.bzz)

        self.mlayers = {}
        self.mlayers['Magnetics'] = np.array(magval)
        self.mlayers['bx'] = np.array(bx)
        self.mlayers['by'] = np.array(by)
        self.mlayers['bz'] = np.array(bz)
        self.mlayers['bxx'] = np.array(bxx)
        self.mlayers['bxy'] = np.array(bxy)
        self.mlayers['bxz'] = np.array(bxz)
        self.mlayers['byy'] = np.array(byy)
        self.mlayers['byz'] = np.array(byz)
        self.mlayers['bzz'] = np.array(bzz)


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


def calc_mag_field(lmod, pbars=None, showtext=None, parent=None,
                   showreports=False):
    """ Calculate magnetic field

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

# get height corrections
    tmp = np.copy(lmod.lith_index)
    tmp[tmp > -1] = 0
    hcor = np.abs(tmp.sum(2))

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

                mlist[1].calc_origin_mag(hcor)

                outfile = tempfile.TemporaryFile()
                np.savez(outfile, **mlist[1].mlayers)
                outfile.seek(0)
                mlist[1].mlayers = None
                tmpfiles[mlist[0]] = outfile

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
    mgtext = ['Magnetics', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz',
              'bzz']

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

        showtext('Summing '+mlist[0]+' (PyGMI may become non-responsive' +
                 ' during this calculation)')

        msign = [1, -1]
        mgvalin = {}

        for mgtextind in mgtext:
            mglayers = mfile[mgtextind]
            mgvalin[mgtextind] = np.zeros(numx*numy)

            for mi, mtmp in enumerate([modind, modindcheck]):
                if np.unique(modind).size < 2 or mijk not in mtmp:
                    continue
                QtWidgets.QApplication.processEvents()
                i, j, k = np.nonzero(mtmp == mijk)
                iuni = np.array(np.unique(i), dtype=np.int32)
                juni = np.array(np.unique(j), dtype=np.int32)
                kuni = np.array(np.unique(k), dtype=np.int32)

                if i.size < 50000:
                    for k in kuni:
                        baba = sum_fields(k, mgval, numx, numy, modind, aaa[0],
                                          aaa[1], mglayers, hcorflat, mijk,
                                          juni, iuni)
                        mgvalin[mgtextind] += msign[mi]*baba
                else:
                    pool = Pool()
                    baba = []

                    for k in kuni:
                        baba.append(pool.apply_async(sum_fields,
                                                     args=(k, mgval,
                                                           numx, numy,
                                                           modind, aaa[0],
                                                           aaa[1],
                                                           mglayers, hcorflat,
                                                           mijk, juni, iuni,)))
                    for p in baba:
                        mgvalin[mgtextind] += msign[mi]*p.get()
                    pool.close()
                    del baba

            mgvalin[mgtextind].resize([numx, numy])
            mgvalin[mgtextind] = mgvalin[mgtextind].T
#            mgvalin[mgtextind] = mgvalin[mgtextind][::-1]
            mgvalin[mgtextind] = np.ma.array(mgvalin[mgtextind])

        showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtWidgets.QApplication.processEvents()

    if np.unique(modindcheck).size > 1:
        mgvalin['magval'] += lmod.griddata['Calculated Magnetics'].data

    for i in mgtext:
        if i != 'Magnetics':
            lmod.griddata['Calculated '+i] = \
                copy.deepcopy(lmod.griddata['Calculated Magnetics'])
        lmod.griddata['Calculated '+i].data = mgvalin[i]

        if lmod.lith_index.max() <= 0:
            lmod.griddata[i].data *= 0.

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


def calc_grv_field(lmod, pbars=None, showtext=None, parent=None,
                   showreports=False):
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

# get height corrections
    tmp = np.copy(lmod.lith_index)
    tmp[tmp > -1] = 0
    hcor = np.abs(tmp.sum(2))

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
                mlist[1].calc_origin_grav(hcor)

                outfile = tempfile.TemporaryFile()
                np.savez(outfile, **mlist[1].mlayers)
                outfile.seek(0)
                mlist[1].mlayers = None
                tmpfiles[mlist[0]] = outfile

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
    mgtext = ['Gravity', 'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz',
              'gzz']

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

        showtext('Summing '+mlist[0]+' (PyGMI may become non-responsive' +
                 ' during this calculation)')

        msign = [1, -1]
        mgvalin = {}

        for mgtextind in mgtext:
            mglayers = mfile[mgtextind]
            mgvalin[mgtextind] = np.zeros(numx*numy)

#            if mgtextind == 'gy':
#                plt.plot(mglayers[0,40])
#                plt.plot(mglayers[1,40])
#                plt.plot(mglayers[2,40])
#                tmp = mglayers[2]
#                plt.plot(tmp.T[40])
#                plt.plot(mglayers[4,40])
#                plt.plot(mglayers[5,40])
#                plt.show()
#                pdb.set_trace()

            for mi, mtmp in enumerate([modind, modindcheck]):
                if np.unique(modind).size < 2 or mijk not in mtmp:
                    continue
                QtWidgets.QApplication.processEvents()
                i, j, k = np.nonzero(mtmp == mijk)
                iuni = np.array(np.unique(i), dtype=np.int32)
                juni = np.array(np.unique(j), dtype=np.int32)
                kuni = np.array(np.unique(k), dtype=np.int32)

                if i.size < 50000:
                    for k in kuni:
                        baba = sum_fields(k, mgval, numx, numy, modind, aaa[0],
                                          aaa[1], mglayers, hcorflat, mijk,
                                          juni, iuni)
                        mgvalin[mgtextind] += msign[mi]*baba
                else:
                    pool = Pool()
                    baba = []

                    for k in kuni:
                        baba.append(pool.apply_async(sum_fields,
                                                     args=(k, mgval,
                                                           numx, numy,
                                                           modind, aaa[0],
                                                           aaa[1],
                                                           mglayers, hcorflat,
                                                           mijk, juni, iuni,)))
                    for p in baba:
                        mgvalin[mgtextind] += msign[mi]*p.get()
                    pool.close()
                    del baba

            mgvalin[mgtextind].resize([numx, numy])
            mgvalin[mgtextind] = mgvalin[mgtextind].T
            mgvalin[mgtextind] = mgvalin[mgtextind][::-1]
            mgvalin[mgtextind] = np.ma.array(mgvalin[mgtextind])

        showtext('Done')

        if pbars is not None:
            pbars.incrmain()
        QtWidgets.QApplication.processEvents()

    if np.unique(modindcheck).size > 1:
        mgvalin += lmod.griddata['Calculated Gravity'].data

    for i in mgtext:
        if i != 'Gravity':
            lmod.griddata['Calculated '+i] = \
                copy.deepcopy(lmod.griddata['Calculated Magnetics'])
        lmod.griddata['Calculated '+i].data = mgvalin[i]

        if lmod.lith_index.max() <= 0:
            lmod.griddata[i].data *= 0.

    if ('Gravity Regional' in lmod.griddata and
            np.unique(modindcheck).size == 1):
        zfin = gridmatch(lmod, 'Calculated Gravity', 'Gravity Regional')
        lmod.griddata['Calculated Gravity'].data += zfin

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


def quick_model(numx=50, numy=40, numz=5, dxy=100., d_z=100.,
                tlx=0., tly=0., tlz=0., mht=100., ght=0., finc=-67., fdec=-17.,
                inputliths=None, susc=None, dens=None, minc=None, mdec=None,
                mstrength=None, hintn=30000.):
    """ Create a quick model """
    if inputliths is None:
        inputliths = ['Generic']
    if susc is None:
        susc = [0.01]
    if dens is None:
        dens = [2.85]

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
    from pygmi.pfmod.iodefs import ImportTMod3D

# Import model file
    filename = r'C:\Work\Programming\pygmi\data\ptest1.npz'
    imod = ImportTMod3D(None)
    imod.ifile = filename
    imod.lmod.griddata.clear()
    imod.lmod.lith_list.clear()
    indict = np.load(filename)
    imod.dict2lmod(indict)
    calc_mag_field(imod.lmod)

    lmod = imod.lmod

# quick model
#    blx = 0  # bottom left x (min x)
#    bly = 0  # bottom left y (min y)
#    blz = 0  # surface z ( max z)
#
#    dxy = 10
#    d_z = 10
#
#    x1 = 150
#    x2 = 250
#    y1 = 150
#    y2 = 250
#
##    x1 = 200
##    x2 = 210
##    y1 = 200
##    y2 = 210
#
#    z1 = 0
#    z2 = -300
#
#    x1 = x1 - blx
#    x2 = x2 - blx
#    y1 = x1 - bly
#    y2 = x2 - bly
#    z1 = blz - z1
#    z2 = blz - z2
#
#    numx = 40
#    numy = 40
#    numz = int(z2/d_z)
#
#    lmod = quick_model(numx=numx, numy=numy, numz=numz, dxy=dxy, d_z=d_z,
#                       mht=0., ght=0., finc=45, fdec=30, susc=[0.1],
#                       hintn=28000, dens=[2.85])
#
#    xi1 = int(x1/dxy)
#    xi2 = int(x2/dxy)
#    yi1 = int(y1/dxy)
#    yi2 = int(y2/dxy)
#    zi1 = int(z1/d_z)
#    zi2 = int(z2/d_z)
#
#    lmod.lith_index[xi1:xi2, yi1:yi2, zi1:zi2] = 1
#
#    calc_mag_field(lmod)
##    calc_grv_field(lmod)

    prof = 20
    magval = lmod.griddata['Calculated Magnetics'].data
    bx = lmod.griddata['Calculated bx'].data
    by = lmod.griddata['Calculated by'].data
    bz = lmod.griddata['Calculated bz'].data
    bxx = lmod.griddata['Calculated bxx'].data
    byy = lmod.griddata['Calculated byy'].data
    bxy = lmod.griddata['Calculated bxy'].data
    byz = lmod.griddata['Calculated byz'].data
    bxz = lmod.griddata['Calculated bxz'].data
    bzz = lmod.griddata['Calculated bzz'].data


    x = np.arange(lmod.xrange[0], lmod.xrange[1], lmod.dxy)

    plt.figure(figsize=(8,11))
    plt.subplot(411)
    plt.title('$B_{tmi}$')
    plt.grid(True)
    plt.plot(x, magval[prof], label='Tensor')

    plt.subplot(412)
    plt.title('$B_{x}$')
    plt.grid(True)
    plt.plot(x, bx[prof], label='Tensor')

    plt.subplot(413)
    plt.title('$B_{y}$')
    plt.grid(True)
    plt.plot(x, by[prof], label='Tensor')

    plt.subplot(414)
    plt.title('$B_{z}$')
    plt.grid(True)
    plt.plot(x, bz[prof], label='Tensor')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,11))
    plt.subplot(511)
    plt.title('$B_{xx}$')
    plt.grid(True)
    plt.plot(x, bxx[prof], label='Tensor')

    plt.subplot(512)
    plt.title('$B_{xy}$')
    plt.grid(True)
    plt.plot(x, bxy[prof], label='Tensor')

    plt.subplot(513)
    plt.title('$B_{yy}$')
    plt.grid(True)
    plt.plot(x, byy[prof], label='Tensor')

    plt.subplot(514)
    plt.title('$B_{yz}$')
    plt.grid(True)
    plt.plot(x, byz[prof], label='Tensor')

    plt.subplot(515)
    plt.title('$B_{xz}$')
    plt.grid(True)
    plt.plot(x, bxz[prof], label='Tensor')

    plt.tight_layout()
    plt.show()

#    plt.title('Grvval')
#    plt.grid(True)
#    plt.plot(lmod.griddata['Calculated Gravity'].data[::-1][prof])
#    plt.show()
#
#    plt.title('gx')
#    plt.grid(True)
#    plt.plot(lmod.griddata['Calculated gx'].data[::-1][prof])
#    plt.show()
#
#    plt.title('gy')
#    plt.grid(True)
#    plt.plot(lmod.griddata['Calculated gy'].data[::-1][prof])
#    plt.show()
#
#    plt.title('gz')
#    plt.grid(True)
#    plt.plot(lmod.griddata['Calculated gz'].data[::-1][prof])
#    plt.show()
#
#    plt.title('gy')
#    plt.grid(True)
#    plt.imshow(lmod.griddata['Calculated gy'].data[::-1])
#    plt.show()

    plt.figure(figsize=(8,11))
    plt.subplot(4,3,10)
    plt.title('$B_{tmi}$')
    plt.imshow(magval, cmap=plt.cm.jet)

    plt.subplot(4,3,1)
    plt.title('$B_{x}$')
    plt.imshow(bx, cmap=plt.cm.jet)
    plt.subplot(4,3,2)
    plt.title('$B_{y}$')
    plt.imshow(by, cmap=plt.cm.jet)
    plt.subplot(4,3,3)
    plt.title('$B_{z}$')
    plt.imshow(bz, cmap=plt.cm.jet)
    plt.subplot(4,3,4)
    plt.title('$B_{xx}$')
    plt.imshow(bxx, cmap=plt.cm.jet)
    plt.subplot(4,3,5)
    plt.title('$B_{xy}$')
    plt.imshow(bxy, cmap=plt.cm.jet)
    plt.subplot(4,3,6)
    plt.title('$B_{xz}$')
    plt.imshow(bxz, cmap=plt.cm.jet)
    plt.subplot(4,3,8)
    plt.title('$B_{yy}$')
    plt.imshow(byy, cmap=plt.cm.jet)
    plt.subplot(4,3,9)
    plt.title('$B_{yz}$')
    plt.imshow(byz, cmap=plt.cm.jet)
    plt.subplot(4,3,12)
    plt.title('$B_{zz}$')
    plt.imshow(bzz, cmap=plt.cm.jet)

    plt.tight_layout()
    plt.show()


    print('Finished!')
    winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

    pdb.set_trace()


if __name__ == "__main__":
    test()
