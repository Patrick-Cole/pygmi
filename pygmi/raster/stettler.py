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

""" This is a collection of routines by Edgar Stettler """

# pylint: disable=E1101, C0103
from PyQt4 import QtGui, QtCore
import numpy as np
import scipy.signal as si
import copy


class DepthSusc(QtGui.QDialog):
    """ Calculate depth and susceptibility"""
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}

    def settings(self):
        """ Settings """

#      CHARACTER*35 IFILE,ofile,dummy*72
#      Character(132) String
#      INTEGER Line_nr, st_nr, cur_ln_nr,stloc,flag
#      REAL incl,line_direc,magfield,mag_suscept

        print('******************************************************')
        print('* Calculation of depth & suscep to the circumference *')
        print('*  of a vertical! sided body from the magnetic tilt  *')
        print('*                     angle.                         *')
        print('*                                                    *')
        print('* Algorithm needs the following keyboard input       *')
        print('*            sequence of data:                       *')
        print('* Incl(deg), decl,line direc (clkwize from G North), *')
        print('* mag strength in free format (space between values).*')
        print('*                                                    *')
        print('* The data input file columns are: Line_nr,X,Y,tilt- *')
        print('*    angle(deg),Station_nr,dM/dh,dM/dz as from a     *')
        print('*              GEOSOFT .csv output file.             *')
        print('*                                                    *')
        print('* RTP total magnetic field (TMF) before calc derivs  *')
        print('*                                                    *')
        print('*   dM/dh=SQRT((dM/dx)*(dM/dx)+(dM/dy)*(dM/dy))      *')
        print('*          dM/dz=vertical deriv of RTP TMF           *')
        print('*       Tilt-angle=ATAN(deg)((dM/dz)/(dM/dh)         *')
        print('*                                                    *')
        print('* Output file will be an ASCI text file, import      *')
        print('*    into a GEOSOFT data base to use further.        *')
        print('*                                                    *')
        print('* Ref: Nabighian(1972),Geophys37,p507-517            *')
        print('*      Salem et al., 2007, Leading Edge, Dec,p1502-5 *')
        print('*          EHS, last changes 12 Dec 2012             *')
        print('******************************************************')

        print('Input the incl, decl(deg) of earth field')
        incl = -67.
        decl = -17.

        print('Input the line direction (90?), B-magfield')
        line_direc = 90
        magfield = 30000

#      OPEN (UNIT=10,FILE=IFILE,STATUS='OLD',
#     $FORM='FORMATTED',ACCESS='SEQUENTIAL')
#      OPEN (UNIT=12,FILE=ofile,STATUS='unknown',
#     $FORM='FORMATTED',ACCESS='SEQUENTIAL')
        incl = incl * 180/np.pi()  # convert to radians
        decl = decl * 180/np.pi()
        line_direc = line_direc * 180/np.pi()

# 1000 format(1x,I12,I6,F13.2,F13.2,F13.3,F13.6)
# C 1001 format(1x,I12,I6,F13.2,F13.2,F6.1,F6.4,F6.4)

#      read(10,*) dummy  !reads column headings

#     Read a *.csv comma delimited input and change to space delimited

#    5 Read(10,'(132A)',end=200) string

#      call readcsvfrmat(string,line_nr,st_nr,x1,y1,z1,dMdh,dMdz)
#      write(6,1001) line_nr, st_nr, x1, y1, z1, dMdh, dMdz
        flag = 0
        cur_ln_nr = line_nr

#     First establish if ascending or descending sequence

#   50 read(10,'(132A)',end=200) string
#      call readcsvfrmat(string,line_nr,st_nr,x2,y2,z2,dMdh,dMdz)
#      write(6,1001) line_nr, st_nr, x2, y2, z2, dMdh, dMdz

        xmin = x1
        ymin = y1
        zmin = z1
        xmax = x2
        ymax = y2
        zmax = z2

#        if (Z2-Z1) < 0:
#            goto 60
#        if(Z2-Z1) == 0:
#            goto 70
#        if(Z2-Z1) > 0:
#            goto 80

#     Descending angles
        while (z2-z1) < 0 and cur_ln_nr == line_nr:
            pass
#            read(10,'(132A)',end=200) string
#            call readcsvfrmat(string,line_nr,st_nr,x,y,z,dMdh,dMdz)
#            write(12,1001) line_nr,st_nr,x,y,z,dMdh,dMdz

#            if (cur_ln_nr.eq.line_nr):
#                continue
#            else:
#                goto 5

#        write(12,*) xmin,xmax,ymin,ymax,zmin,zmax

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
        if flag == 1:
            # write(12,*) xmin,xmax,ymin,ymax,zmin,zmax
            tilt_depth = sqrt((xmax-xmin)**2+(ymax-ymin)**2)/4.

#     Calculate the mag suseptibility

#        c=1-cos(incl)*cos(incl)*sin(line_direc)*sin(line_direc)
#        z=z/57.2974
#        write(6,*) C,Z,dMdh,dMdz, tilt_depth
#        mag_suscept=(0.5*(dMdh-dMdz)*tilt_depth*(TAN(z)*TAN(z)+1))
#        write(6,*) mag_suscept
#        mag_suscept=mag_suscept/(1-TAN(z))
#        write(6,*) mag_suscept
#        mag_suscept=mag_suscept/magfield*C
#        write(6,*) mag_suscept
            dip = (z+2*incl-90.)*180/np.pi()
            mag_suscept = (dMdz * magfield * (1-cos(incl) * cos(incl) *
                           sin(line_direc+decl) * sin(line_direc+decl)) *
                           abs(sin(dip)))
            if(mag_suscept == 0.0):
                mag_suscept = .1
            mag_suscept = ABS(dMdh/mag_suscept)
            if(mag_suscept == 0.):
                mag_suscept = .0001
            if(mag_suscept < 0.):
                mag_suscept = .0001

            flag = 0
#            write(12,1000) line_nr, stloc, xloc, yloc, tilt_depth,
#                            mag_suscept*100.

            zmax = zmin
            xmax = xmin
            ymax = ymin


#     Ascending angles
        while (z2-z1) >= 0 and cur_ln_nr == line_nr:
            pass
#   80       read(10,'(132A)',end=200) string
#            call readcsvfrmat(string,line_nr,st_nr,x,y,z,dMdh,dMdz)
#        write(12,1000) line_nr, st_nr, x, y, z

#            if(cur_ln_nr.eq.line_nr):
#                continue
#            else
#                goto 5

#      write(12,*) xmin,xmax,ymin,ymax,zmin,zmax
            if(z >= zmax):
                Zmax = Z
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
        if (flag == 0):
            pass
#            goto 60
#      write(12,*) xmin,xmax,ymin,ymax,zmin,zmax
        Tilt_depth = sqrt((xmax-xmin)**2+(ymax-ymin)**2)/4.

#     Calculate the mag suseptibility

#      c=1-cos(incl)*cos(incl)*sin(line_direc)*sin(line_direc)
#      z=z/57.2974
#      write(6,*) C,Z,dMdh,dMdz, tilt_depth
#      mag_suscept=(0.5*(dMdh-dMdz)*tilt_depth*(TAN(z)*TAN(z)+1))
#      write(6,*) mag_suscept
#      mag_suscept=mag_suscept/(1-TAN(z))
#      write(6,*) mag_suscept
#      mag_suscept=mag_suscept/magfield*C
#      write(6,*) mag_suscept
        dip = (z+2*incl-90.)*57.2974
#        mag_suscept=dMdz*magfield*(1-COS(incl)*COS(incl)*SIN(line_direc+decl)*SIN(line_direc+decl))*ABS(SIN(dip))
        if(mag_suscept == 0.0):
            mag_suscept = .1
        mag_suscept = abs(dMdh/mag_suscept)
        if(mag_suscept == 0.0):
            mag_suscept = .00001
        if(mag_suscept < 0.):
            mag_suscept = .00001
#        write(12,1000) line_nr, stloc, xloc, yloc, tilt_depth,mag_suscept*100.
        flag = 0

        zmin = zmax
        xmin = xmax
        ymin = ymax

        # goto 60

#  200   close (10)
#        close (12)

#**********************************************************************
#      Subroutine readcsvfrmat(string,line_nr,st_nr,x,y,z,dMdh,dMdz)
#      CHARACTER*25 value1
#      Character(132) String
#      INTEGER Line_nr, st_nr
#      INTEGER string_len, variable, prev_i
#      REAL X1
#      Dimension val(7)
## 1000 format(132a)
## 1001 format(1x,I12,I6,F13.2,F13.2,F6.1,F6.4,F6.4)
#
#      String_len=LEN_TRIM(string)
#
##     Set initial conditions
#
#      prev_i=1  ! first char is L, so only from char 2
#      variable=1 !counter of how many variables, should be 7
#
##      Write(12,1000) string
#      Do 2 i=2,string_len
#      if(string(i:i).EQ.ACHAR(44)) then
##      write(12,*) i
#      value1=string(prev_i+1:i-1)
##      write(12,*) value1
#      READ(value1,*,err=10) x1
#      val(variable)=x1
#      variable=variable+1
#      prev_i=i
#      end if
#    2 continue
#      value1=string(prev_i+1:string_len)
##      write(12,*) prev_i+1, string_len
##      write(12,*) value1
#      READ(value1,*,err=10) x1
#      val(7)=x1
#
##     Convert
#
#   10 Line_nr=val(1)
#      St_nr=val(5)
#      X=val(2)
#      Y=val(3)
#      Z=val(4)
#      dMdh=val(6)
#      dMdz=val(7)
##      write(12,1001) line_nr,st_nr,x,y,z,dMdh,dMdz
#      return
#      end