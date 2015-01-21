#     ***************** Tlt_dpth_susc.for ***************************
#     Calculation of the depth to edges and mag suscept(SI units) to
#                        a vertical sided body.
#
#     Keyboard Input is the following info:
#     Incl, decl, line direction (clockwize from Geo North)
#     and Earth field strength in free format,ie space between values.
#
#     Data file is a *.csv dump from a GEOSOFT database.
#     Column sequence has! to be: Line_nr,x,y,tilt angle (deg),station
#     nr or (fid),dM/dh and dM/dz(i.e. vert deriv of RTP of TMF).
#
#     Tilt-angle works best on RTP data. So RTP before using it.
#
#     Works on the principle that distance between min and max
#     tilt angle is 4 times the depth to the magnetic interface
#     of a body with vertical! boundaries.
#
#     Susceptability is calc from Nabighian(1972),Geophys37,p507-517
#
#     Programmer: EHS; last changes on 12 Dec 2012
#
#     ************************************************************
#

"""
This is a collection of routines by Edgar Stettler

What I do is to calculate the RTP from a magnetic data set, then the AS,
then the VD of the AS, then the x,y and z derivatives of the RTP,
then the dMdh and then the tiltangle in degrees.

See the fortran text file for a description of some of the terms.
Then I display, as a ternary image, the VD of the AS in Cyan, the tilt angle in
Magenta and the dMdh  in yellow.
"""

from PyQt4 import QtGui, QtCore
import numpy as np
from . import cooper
# import scipy.signal as si
import copy


class DepthSusc(QtGui.QDialog):
    """
    Calculate depth and susceptibility

    Calculation of depth & suscep to the circumference of a vertical! sided
    body from the magnetic tilt angle.

    Algorithm needs the following keyboard input sequence of data:
        Incl(deg), decl,line direc (clkwize from G North),
        mag strength in free format (space between values).

    The data input file columns are:
        Line_nr,X,Y,tilt-angle(deg),Station_nr,dM/dh,dM/dz

    RTP total magnetic field (TMF) before calc derivs
        dM/dh=SQRT((dM/dx)*(dM/dx)+(dM/dy)*(dM/dy))
        dM/dz=vertical deriv of RTP TMF
        Tilt-angle=ATAN(deg)((dM/dz)/(dM/dh)

    Ref:
        Nabighian(1972),Geophys37,p507-517
        Salem et al., 2007, Leading Edge, Dec,p1502-5
    """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """

# For now take the first band of the data. We will need to choose on the
# main dialog in future.

        data = copy.deepcopy(self.indata['Raster'][0])
        t1, th, t2, ta, tdx = cooper.tilt1(data.data, 0, 0)
# A negative number implies we are straddling 0
        tmp = np.sign(t1[:, :-1]*t1[:, 1:])
# This tells the data is increasing (+ve) or decreasing (-ve)
        tmp2 = np.sign(t1[:, 1:]-t1[:, :-1])

# Store our results
        data.data *= 0
        data.data[:, 1:] = tmp
        data.data.mask[data.data>=0] = True

        self.outdata['Raster'] = [data]

        return True
# Input the incl, decl(deg) of earth field
        incl = -67.
        decl = -17.

# Input the line direction (90?), B-magfield

        line_direc = 90
        magfield = 30000

        incl = np.deg2rad(incl)  # convert to radians
        decl = np.deg2rad(decl)
        line_direc = np.deg2rad(line_direc)

# Get the following:
        line_nr, st_nr, x1, y1, z1, dMdh, dMdz = [1, 2, 3, 4, 5, 6, 7]

        flag = 0
        cur_ln_nr = line_nr

# First establish if ascending or descending sequence

# Get the following:
        line_nr, st_nr, x2, y2, z2, dMdh, dMdz = [1, 2, 3, 4, 5, 6, 7]

        xmin = x1
        ymin = y1
        zmin = z1
        xmax = x2
        ymax = y2
        zmax = z2
        z_previous = z2

#        if (Z2-Z1) < 0:
#            goto 60
#        if(Z2-Z1) == 0:
#            goto 70
#        if(Z2-Z1) > 0:
#            goto 80

#     Descending angles
        while (z2-z1) < 0 and cur_ln_nr == line_nr:
            # Get the following:
            line_nr, st_nr, x, y, z, dMdh, dMdz = [1, 2, 3, 4, 5, 6, 7]

            if z <= zmin:
                zmin = z
                if(z >= 0 and z_previous <= 0):
                    flag = 1
                    xloc = x
                    yloc = y
                    stloc = st_nr

                if(z <= 0 and z_previous >= 0):
                    flag = 1
                    xloc = x
                    yloc = y
                    stloc = st_nr

                z_previous = z
        xmin = x
        ymin = y

        if flag != 0:
            # write(12,*) xmin,xmax,ymin,ymax,zmin,zmax
            tilt_depth = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)/4.

#     Calculate the mag suseptibility
# Corrected edgar's dip calculation
            dip = z+2*incl-np.pi()/2

            mag_suscept = (dMdz * magfield * (1-np.cos(incl) * np.cos(incl) *
                           np.sin(line_direc+decl) * np.sin(line_direc+decl)) *
                           np.abs(np.sin(dip)))
            if(mag_suscept == 0.0):
                mag_suscept = .1
            mag_suscept = np.abs(dMdh/mag_suscept)
            if(mag_suscept == 0.):
                mag_suscept = .0001
            if(mag_suscept < 0.):
                mag_suscept = .0001

            flag = 0
# write(12,1000) line_nr, stloc, xloc, yloc, tilt_depth, mag_suscept*100.

            zmax = zmin
            xmax = xmin
            ymax = ymin


#     Ascending angles
        while (z2-z1) >= 0 and cur_ln_nr == line_nr:
            line_nr, st_nr, x, y, z, dMdh, dMdz = [1, 2, 3, 4, 5, 6, 7]

            if(z >= zmax):
                zmax = z
                if(z >= 0 and z_previous <= 0):
                    flag = 1
                    xloc = x
                    yloc = y
                    stloc = st_nr
                if(z <= 0 and z_previous >= 0):
                    flag = 1
                    xloc = x
                    yloc = y
                    stloc = st_nr
                z_previous = z
        xmax = x
        ymax = y

        if (flag != 0):
            tilt_depth = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)/4.
            dip = z+2*incl-np.pi()/2.
            mag_suscept = (dMdz * magfield * (1-np.cos(incl) * np.cos(incl) *
                           np.sin(line_direc+decl) * np.sin(line_direc+decl)) *
                           np.abs(np.sin(dip)))
            if(mag_suscept == 0.0):
                mag_suscept = .1
            mag_suscept = np.abs(dMdh/mag_suscept)
            if(mag_suscept == 0.0):
                mag_suscept = .00001
            if(mag_suscept < 0.):
                mag_suscept = .00001
#   output line_nr, stloc, xloc, yloc, tilt_depth,mag_suscept*100.
            flag = 0

            zmin = zmax
            xmin = xmax
            ymin = ymax
