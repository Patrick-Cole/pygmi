#------------------------------------------------------------------------------
# Name:        grvmagc.pyx (part of PyGMI)
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
#------------------------------------------------------------------------------
""" Gravity and magnetic field calculations. 
This uses the following algorithms:

Singh, B., Guptasarma, D., 2001. New method for fast computation of gravity
and magnetic anomalies from arbitrary polyhedral. Geophysics 66, 521 – 526.

Blakely, R.J., 1996. Potential Theory in Gravity and Magnetic Applications,
1st edn. Cambridge University Press, Cambridge, UK, 441 pp. 200-201

GravMag - Routine that will calculate the final versions of the field. Other,
related code is here as well, such as the inversion routines.

GeoData - The is a class which contains the geophysical information for a
single lithology. This includes the final calculated field for that lithology
only """

# To generate grvmagc.pyd:
# Open winpython command prompt (it has all python paths already)
# Type the following commands:
# cpp.bat:
#   CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64
#   cd "C:\Users\pcole.CGS\Work\Programming\PyGMI2\packages\pfmod"
# c1.bat:
#   cython -a grvmagc.pyx
# c2.bat:
#   cl  grvmagc.c /I"C:\WinPython-64bit-3.3.2.1\python-3.3.2.amd64\include"
#       /I"C:\WinPython-64bit-3.3.2.1\python-3.3.2.amd64\lib\site-packages\
#       numpy\core\include" /link /dll
#       /libpath:"C:\WinPython-64bit-3.3.2.1\python-3.3.2.amd64\libs"
#       /out:grvmagc.pyd

#define NPY_NO_DEPRECIATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
import cython
cimport cython
from libc.math cimport sqrt
from libc.math cimport log
from libc.math cimport atan2
from libc.math cimport acos

DTYPEI = np.int
ctypedef np.int_t DTYPEI_t
DTYPEF = np.float32
ctypedef np.float32_t DTYPEF_t
DTYPED = np.double
ctypedef np.double_t DTYPED_t

DEF pi = 3.141592653589793
##DEF Gc = 6.6732e-3 # 6.67384?


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)


def gm3d(int npro, int nstn,
         np.ndarray[DTYPED_t, ndim=2] X,
         np.ndarray[DTYPED_t, ndim=2] Y,
         np.ndarray[DTYPED_t, ndim=2] Edge,
         list Corner,
         list Face,
         np.ndarray[DTYPED_t, ndim=2] Gx,
         np.ndarray[DTYPED_t, ndim=2] Gy,
         np.ndarray[DTYPED_t, ndim=2] Gz,
         np.ndarray[DTYPED_t, ndim=2] Hx,
         np.ndarray[DTYPED_t, ndim=2] Hy,
         np.ndarray[DTYPED_t, ndim=2] Hz,
         np.ndarray[DTYPED_t, ndim=1] Pd,
         list Un):
    """ grvmag 3d """

    cdef int eno2
    cdef int eno1
    cdef double flimit
    cdef double gmtf1
    cdef double gmtf2
    cdef double gmtf3
    cdef double Omega
    cdef double l
    cdef double m
    cdef double n
    cdef double p
    cdef double q
    cdef double r
    cdef double r12
    cdef double L
    cdef double fsign
    cdef double Unf[3]
    cdef double p1[3]
    cdef double p2[3]
    cdef double p3[3]
    cdef double crs[4][3]
    cdef double V[24][3]

    cdef int g
    cdef int pr
    cdef int st
    cdef int f
    cdef int t
    cdef double x
    cdef double y
    cdef int cindx
    cdef double dp1
    cdef double W
    cdef int indx[6]

    flimit = 64*np.spacing(1)
    Omega = 0.0
    dp1 = 1.0
    I = 1.0

    indx[0] = 0
    indx[1] = 1
    indx[2] = 2
    indx[3] = 3
    indx[4] = 0
    indx[5] = 1

    for f in range(24):
        for g in range(3):
            V[f][g] = Edge[f, g]

    for pr in range(npro):
        for st in range(nstn):
            x = X[pr, st]
            y = Y[pr, st]
            for f in range(6):
                for g in range(4):
                    cindx = Face[f][g]
                    crs[g][0] = Corner[cindx][0] - x
                    crs[g][1] = Corner[cindx][1] - y
                    crs[g][2] = Corner[cindx][2]

                p = 0
                q = 0
                r = 0
                eno1 = 4*f
                W = -2*pi
                Unf[0] = Un[f][0]
                Unf[1] = Un[f][1]
                Unf[2] = Un[f][2]

                for t in range(4):
                    p1[0] = crs[indx[t]][0]
                    p1[1] = crs[indx[t]][1]
                    p1[2] = crs[indx[t]][2]
                    p2[0] = crs[indx[t+1]][0]
                    p2[1] = crs[indx[t+1]][1]
                    p2[2] = crs[indx[t+1]][2]
                    p3[0] = crs[indx[t+2]][0]
                    p3[1] = crs[indx[t+2]][1]
                    p3[2] = crs[indx[t+2]][2]
                    W += angle(p1, p2, p3, Unf, flimit)

                    eno2 = eno1+t   # Edge no
                    L = Edge[eno2, 3]

                    r12 = norm(p1)+norm(p2)

                    I = (1/L)*log((r12+L)/(r12-L))

                    p += I*V[eno2][0]
                    q += I*V[eno2][1]
                    r += I*V[eno2][2]

        #        From Omega, l, m, n PQR get components of field due to face f

                dp1 = dot(Unf, crs[0])
                if dp1 < 0.:
                    Omega = W
                else:
                    Omega = -W

                l = Unf[0]
                m = Unf[1]
                n = Unf[2]

                gmtf1 = l*Omega+n*q-m*r
                gmtf2 = m*Omega+l*r-n*p
                gmtf3 = n*Omega+m*p-l*q

                Hx[pr, st] = Hx[pr, st]+Pd[f]*gmtf1
                Hy[pr, st] = Hy[pr, st]+Pd[f]*gmtf2
                Hz[pr, st] = Hz[pr, st]+Pd[f]*gmtf3

                Gx[pr, st] = Gx[pr, st]-dp1*gmtf1
                Gy[pr, st] = Gy[pr, st]-dp1*gmtf2
                Gz[pr, st] = Gz[pr, st]-dp1*gmtf3

cdef inline double dot(double a[3], double b[3]):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

cdef inline double norm(double a[3]):
    return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

cdef inline double angle(double p1[3], double p2[3], double p3[3],
                         double Un[3], double flimit):
    """Angle.m finds the angle between planes O-p1-p2 and O-p2-p3 where p1
    p2 p3 are coordinates of three points taken in ccw order as seen from the
    origin O.
    This is used by gm3d for finding the solid angle subtended by a
    polygon at the origin. Un is the unit outward normal vector to the polygon.
    """

    cdef double anout
    cdef double pn1
    cdef double pn2
    cdef double ang
    cdef double perp
    cdef double r
    cdef double n10
    cdef double n11
    cdef double n12
    cdef double n20
    cdef double n21
    cdef double n22

    ang = 0
    anout = dot(p1, Un)

    if abs(anout) <= flimit:
        return 0

    if anout > flimit:    # face seen from inside, interchange p1 and p3
        p1, p3 = p3, p1

    n10 = p2[1]*p1[2] - p2[2]*p1[1]
    n11 = p2[2]*p1[0] - p2[0]*p1[2]
    n12 = p2[0]*p1[1] - p2[1]*p1[0]

    n20 = p2[1]*p3[2] - p2[2]*p3[1]
    n21 = p2[2]*p3[0] - p2[0]*p3[2]
    n22 = p2[0]*p3[1] - p2[1]*p3[0]

    pn1 = sqrt(n10*n10+n11*n11+n12*n12)
    pn2 = sqrt(n20*n20+n21*n21+n22*n22)

    if (pn1 <= flimit) or (pn2 <= flimit):
        ang = np.nan
    else:
        n10 = n10/pn1
        n11 = n11/pn1
        n12 = n12/pn1
        n20 = n20/pn2
        n21 = n21/pn2
        n22 = n22/pn2

        r = n10*n20+n11*n21+n12*n22
        ang = acos(r)

        perp = n10*p3[0]+n11*p3[1]+n12*p3[2]

        if perp < -flimit:        # points p1,p2,p3 in cw order
            ang = 2*pi-ang

##    if anout > flimit:    # face seen from inside, interchange p1 and p3
##        p3, p1 = p1, p3

    return ang


def calc_field2(int i, int numx, int numy, int numz,
                np.ndarray[DTYPEI_t, ndim=3] modind,
                np.ndarray[DTYPEI_t, ndim=2] hcor,
                np.ndarray[DTYPEI_t, ndim=1] aaa0,
                np.ndarray[DTYPEI_t, ndim=1] aaa1,
                np.ndarray[DTYPED_t, ndim=3] mlayers,
                np.ndarray[DTYPED_t, ndim=3] glayers,
                np.ndarray[DTYPED_t, ndim=1] magval,
                np.ndarray[DTYPED_t, ndim=1] grvval,
                np.ndarray[DTYPEI_t, ndim=1] hcorflat,
                int mijk):
    """ Calculate magnetic and gravity field """

    cdef int xoff
    cdef int yoff
    cdef int j
    cdef int k
    cdef int ijk
    cdef int igrd
    cdef int jgrd
    cdef np.ndarray[DTYPEI_t, ndim = 1] obs2
    cdef np.ndarray[DTYPED_t, ndim = 3] m2
    cdef np.ndarray[DTYPED_t, ndim = 3] g2
    cdef int b

    b = magval.size

    xoff = numx-i
    for j in range(numy):
        yoff = numy-j
        m2 = mlayers[:, xoff:xoff+numx, yoff:yoff+numy]
        g2 = glayers[:, xoff:xoff+numx, yoff:yoff+numy]
        for k in range(hcor[i, j], numz):
            if (modind[i, j, k] != mijk):
                continue
            obs2 = k+hcorflat
            for ijk in range(b):
                magval[ijk] += m2[obs2[ijk], aaa0[ijk], aaa1[ijk]]
                grvval[ijk] += g2[obs2[ijk], aaa0[ijk], aaa1[ijk]]


def gboxmain(np.ndarray[double, ndim=2] gval, np.ndarray[double, ndim=1] xobs,
             np.ndarray[double, ndim=1] yobs, int numx, int numy, double z_0,
             double x_1, double y_1, double z_1, double x_2, double y_2,
             double z_2, double pi):
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

    # include <math.h>
    cdef int isign[2]
    cdef double x[2]
    cdef double y[2]
    cdef double z[2]
    cdef double rijk
    cdef int ijk
    cdef double arg1
    cdef double arg2
    cdef double arg3
    cdef double sumi
    cdef int ii
    cdef int jj
    cdef int i
    cdef int j
    cdef int k

    isign[0] = -1
    isign[1] = 1
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
                        rijk = sqrt(x[i]*x[i]+y[j]*y[j]+z[k]*z[k])
                        ijk = isign[i]*isign[j]*isign[k]
                        arg1 = atan2(x[i]*y[j], z[k]*rijk)

                        if (arg1 < 0.):
                            arg1 = arg1 + 2 * pi
                        arg2 = rijk+y[j]
                        arg3 = rijk+x[i]
                        arg2 = log(arg2)
                        arg3 = log(arg3)
                        sumi += ijk*(z[k]*arg1-x[i]*arg2-y[j]*arg3)
            gval[ii, jj] = sumi


def mboxmain(int icnt, np.ndarray[double, ndim=1] z1mz0,
             np.ndarray[double, ndim=3] mval, np.ndarray[double, ndim=1] xobs,
             np.ndarray[double, ndim=1] yobs, int numx, int numy, int numz,
             double x_1, double y_1, double x_2, double y_2, double fm1,
             double fm2, double fm3, double fm4, double fm5, double fm6):

    """Subroutine MBOX computes the total field anomaly of an infinitely
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

    Note that in the case of no remanence: mi=fi and md=fd.

  Output paramters:
    Total field anomaly t, in nT. """

    # include <math.h>
    cdef double alpha[2]
    cdef double beta[2]
    cdef double h
    cdef double t
    cdef double hsq
    cdef double alphasq
    cdef double r0sq
    cdef double r0
    cdef double r0h
    cdef double alphabeta
    cdef double arg1
    cdef double arg2
    cdef double arg3
    cdef double arg4
    cdef double tatan
    cdef double sign
    cdef double tlog
    cdef int jj
    cdef int kk
    cdef int i
    cdef int j

    alpha[0] = x_1-xobs[icnt]
    alpha[1] = x_2-xobs[icnt]
    for jj in range(numy):
        beta[0] = y_1-yobs[jj]
        beta[1] = y_2-yobs[jj]
        for kk in range(numz):
            h = z1mz0[kk]
            hsq = h*h
            t = 0.
            for i in range(2):
                alphasq = alpha[i]*alpha[i]
                for j in range(2):
                    sign = 1.
                    if (i != j):
                        sign = -1.
                    r0sq = alphasq+beta[j]*beta[j]+hsq
                    r0 = sqrt(r0sq)
                    r0h = r0*h
                    alphabeta = alpha[i]*beta[j]
                    arg1 = (r0-alpha[i])/(r0+alpha[i])
                    arg2 = (r0-beta[j])/(r0+beta[j])
                    arg3 = alphasq+r0h+hsq
                    arg4 = r0sq+r0h-alphasq
                    tlog = +fm3*log(arg1)/2. + fm2*log(arg2)/2. - fm1*log(r0+h)
                    tatan = (-fm4*atan2(alphabeta, arg3) -
                             fm5*atan2(alphabeta, arg4) +
                             fm6*atan2(alphabeta, r0h))
                    t = t + sign*(tlog+tatan)
            mval[icnt, jj, kk] = t


##def gm3dbig(int npro, int nstn, int Ncor, int Nf, float dens, float depth,
##         np.ndarray[DTYPED_t, ndim=2] X,
##         np.ndarray[DTYPED_t, ndim=2] Y,
##         np.ndarray[DTYPED_t, ndim=2] Edge,
##         np.ndarray[DTYPED_t, ndim=2] Corner,
##         np.ndarray[DTYPEI_t, ndim=2] Face,
##         np.ndarray[DTYPED_t, ndim=2] Gx,
##         np.ndarray[DTYPED_t, ndim=2] Gy,
##         np.ndarray[DTYPED_t, ndim=2] Gz,
##         np.ndarray[DTYPED_t, ndim=2] Hx,
##         np.ndarray[DTYPED_t, ndim=2] Hy,
##         np.ndarray[DTYPED_t, ndim=2] Hz,
##         np.ndarray[DTYPED_t, ndim=1] Pd,
##         np.ndarray[DTYPED_t, ndim=2] Un):
##    """ grvmag 3d """
##
##    cdef np.ndarray[DTYPED_t, ndim=2] cor
##    cdef np.ndarray[DTYPED_t, ndim=2] crs
##    cdef np.ndarray[DTYPED_t, ndim=1] opt
##    cdef np.ndarray[DTYPED_t, ndim=1] p1
##    cdef np.ndarray[DTYPED_t, ndim=1] p2
##    cdef np.ndarray[DTYPED_t, ndim=1] p3
##    cdef np.ndarray[DTYPED_t, ndim=1] V
##    cdef np.ndarray[DTYPED_t, ndim=1] pqr
##    cdef int t
##    cdef int f
##    cdef int Eno
##    cdef float flimit
##    cdef float normp1
##    cdef float normp2
##    cdef float gmtf1
##    cdef float gmtf2
##    cdef float gmtf3
##    cdef float Omega
##    cdef float W
##    cdef float l
##    cdef float m
##    cdef float n
##    cdef float p
##    cdef float q
##    cdef float r
##    cdef float fsign
##    cdef int pr
##    cdef int st
##
##    flimit = 64*np.spacing(1)
##
##    for pr in range(npro):
##        for st in range(nstn):
##            opt = np.array([X[pr, st], Y[pr, st], -depth])
##            cor = Corner-opt # shift origin
##            Edge[:, 4:6] = 0  # clear record of integration
##            for f in range(Nf):
##                nsides = Face[f, 0]
##                cors = Face[f, 1:nsides+1]
##                indx = list(range(nsides))+[0, 1]
##                crs = cor[cors]
##                fsign = np.sign(np.dot(Un[f], crs[0]))
##                dp1 = np.dot(crs[indx[0]], Un[f])
##                if abs(dp1) <= flimit:
##                    Omega = 0.0
##                else:
##                    W = 0
##                    for t in range(nsides):
##                        p1 = crs[indx[t]]
##                        p2 = crs[indx[t+1]]
##                        p3 = crs[indx[t+2]]
##                        W = W + angle(p1, p2, p3, Un[f], flimit)
##                    W = W-(nsides-2)*pi
##                    Omega = -fsign*W
##        #       Integrate over each side if not done and save result
##                PQR = np.array([0, 0, 0])
##                for t in range(nsides):
##                    p1 = crs[indx[t]]
##                    p2 = crs[indx[t+1]]
##                    Eno = np.sum(Face[0:f, 0])+t   # Edge no
##                    V = Edge[Eno, 0:3]
##                    if Edge[Eno, 5] == 1:       # already done
##                        I = Edge[Eno, 4]
##                    if Edge[Eno, 5] != 1:       # not already done
##                        normp1 = sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
##                        normp2 = sqrt(p2[0]**2+p2[1]**2+p2[2]**2)
##                        if np.dot(p1, p2)/(normp1*normp2) == 1:
##        # origin, p1 and  p2 are on a straight line
##                            if normp1 > normp2:    # and p1 further than p2
##                                p1, p2 = p2, p1
##                        L = Edge[Eno, 3]
##                        normp1 = sqrt(p1[0]**2+p1[1]**2+p1[2]**2)
##                        normp2 = sqrt(p2[0]**2+p2[1]**2+p2[2]**2)
##                        r1 = normp1
##                        r2 = normp2
##                        I = (1/L)*log((r1+r2+L)/(r1+r2-L))  # mod. formula
##        #              Save edge integrals and mark as done
##                        s = np.nonzero(
##                            np.logical_and(Edge[:, 6] == Edge[Eno, 7],
##                                           Edge[:, 7] == Edge[Eno, 6]))
##                        Edge[Eno, 4] = I
##                        Edge[s, 4] = I
##                        Edge[Eno, 5] = 1
##                        Edge[s, 5] = 1
##                    pqr = I*V
##                    PQR = PQR+pqr
##
##        #       From Omega, l, m, n PQR get components of field due to face f
##                l, m, n = Un[f]
##                p, q, r = PQR
##
##                gmtf1 = (l*Omega+n*q-m*r)
##                gmtf2 = (m*Omega+l*r-n*p)
##                gmtf3 = (n*Omega+m*p-l*q)
##
##                Hx[pr, st] = Hx[pr, st]+Pd[f]*gmtf1
##                Hy[pr, st] = Hy[pr, st]+Pd[f]*gmtf2
##                Hz[pr, st] = Hz[pr, st]+Pd[f]*gmtf3
##
##                Gx[pr, st] = Gx[pr, st]-dens*Gc*dp1 * gmtf1
##                Gy[pr, st] = Gy[pr, st]-dens*Gc*dp1 * gmtf2
##                Gz[pr, st] = Gz[pr, st]-dens*Gc*dp1 * gmtf3
