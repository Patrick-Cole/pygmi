# -----------------------------------------------------------------------------
# Name:        cooper.py (part of PyGMI)
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
""" This is a collection of routines by Gordon Cooper """

# pylint: disable=E1101, C0103
from PyQt4 import QtGui, QtCore
import numpy as np
import scipy.signal as si
import copy

#        data = np.array([   [1, 2, 3, 4, 5, 6, 7, 8, 9],
#                            [1, 2, 3, 4, 5, 7, 6, 8, 9],
#                            [1, 2, 3, 4, 5, 7, 6, 8, 9],
#                            [4, 3, 2, 1, 0, 1, 2, 9, 0],
#                            [4, 3, 1, 1, 0, 1, 2, 3, 4],
#                            [4, 3, 1, 1, 0, 1, 1, 1, 2],
#                            [4, 3, 1, 1, 0, 1, 2, 3, 4],
#                            [4, 3, 1, 1, 0, 1, 1, 1, 2],
#                            [4, 3, 1, 1, 0, 1, 1, 2, 2]])


class Gradients(QtGui.QDialog):
    """ Gradients """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.azi = 45
        self.elev = 45
        self.order = 1

        self.gridlayout = QtGui.QGridLayout(self)
        self.sb_order = QtGui.QSpinBox(self)
        self.sb_elev = QtGui.QSpinBox(self)
        self.sb_azi = QtGui.QSpinBox(self)
        self.label = QtGui.QLabel(self)
        self.label_2 = QtGui.QLabel(self)
        self.label_3 = QtGui.QLabel(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)

        self.setupui()

        self.sb_azi.setValue(self.azi)
        self.sb_elev.setValue(self.elev)
        self.sb_order.setValue(self.order)

    def setupui(self):
        """ Setup UI """
#        self.resize(289, 166)
        self.sb_order.setMinimum(1)
        self.gridlayout.addWidget(self.sb_order, 3, 1, 1, 1)
        self.sb_elev.setMaximum(90)
        self.gridlayout.addWidget(self.sb_elev, 1, 1, 1, 1)
        self.sb_azi.setPrefix("")
        self.sb_azi.setMinimum(-360)
        self.sb_azi.setMaximum(360)
        self.gridlayout.addWidget(self.sb_azi, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.label, 1, 0, 1, 1)
        self.gridlayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout.addWidget(self.buttonbox, 4, 1, 1, 1)

        self.setWindowTitle("Gradient Calculation")
        self.label.setText("Elevation")
        self.label_2.setText("Azimuth")
        self.label_3.setText("Order")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return

        self.azi = self.sb_azi.value()
        self.elev = self.sb_elev.value()
        self.order = self.sb_order.value()

        data = copy.deepcopy(self.indata['Raster'])

        for i in range(len(data)):
            data[i].data = self.gradients(data[i].data, self.azi, self.elev,
                                          self.order)

        self.outdata['Raster'] = data

        return True

    def gradients(self, data, azi, elev, order):
        """
        # *****************************************************************
        # *** Compute different gradients of image data
        # *** GRJ Cooper July 2006
        # *** cooperg@geosciences.wits.ac.za , grcooper@iafrica.com
        # *** www.wits.ac.za/science/geophysics/gc.htm
        # *****************************************************************
        # *** azi       Filter direction (degrees)
        # *** elev      Elevation (for sunshading, degrees from horizontal)
        # *** order     Order of DR filter - see paper. Try 1 first.
        # *****************************************************************
        """
        # Directional derivative

        azi = azi*np.pi/180
        elev = elev*np.pi/180
        dx, dy = np.gradient(data)
        dt1 = -dy*np.sin(azi)-dx*np.cos(azi)

        # Sunshading

#        cAzi = np.cos(azi)
#        sAzi = np.sin(azi)
#        tElev = np.tan(elev)
#        top = (1.0-dx*cAzi*tElev-dy*sAzi*tElev)
#        bottom = np.sqrt(1.0+dx*dx+dy*dy)+np.sqrt(1.0+tElev*tElev)
#        sun = top/bottom

        # Derivative ratio

        dt2 = -dy*np.sin(azi+np.pi/2)-dx*np.cos(azi+np.pi/2)
        dr = np.arctan2(dt1, abs(dt2)**order)

        return dr


class Visibility2d(QtGui.QDialog):
    """ Compute visibility as a textural measure """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.wsize = 11
        self.dh = 10

        self.gridlayout = QtGui.QGridLayout(self)
        self.sb_dh = QtGui.QSpinBox(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.sb_wsize = QtGui.QSpinBox(self)
        self.label = QtGui.QLabel(self)
        self.label_2 = QtGui.QLabel(self)
        self.pbar = QtGui.QProgressBar(self)

        self.setupui()

        self.sb_wsize.setValue(self.wsize)
        self.sb_dh.setValue(self.dh)

    def setupui(self):
        """ Setup UI """
#        self.resize(289, 166)
        self.gridlayout.addWidget(self.label, 1, 0, 1, 1)
        self.sb_dh.setMinimum(1)
        self.sb_dh.setMaximum(10000)
        self.gridlayout.addWidget(self.sb_dh, 1, 1, 1, 1)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout.addWidget(self.buttonbox, 3, 0, 1, 2)
        self.sb_wsize.setPrefix("")
        self.sb_wsize.setMinimum(3)
        self.sb_wsize.setMaximum(100000)
        self.sb_wsize.setSingleStep(2)
        self.gridlayout.addWidget(self.sb_wsize, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.pbar.setProperty("value", 0)
        self.gridlayout.addWidget(self.pbar, 2, 0, 1, 2)

        self.setWindowTitle("Visibility")
        self.label.setText("Viewing Height (% std dev)")
        self.label_2.setText("Window Size (Odd)")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return

        self.wsize = self.sb_wsize.value()
        self.dh = self.sb_dh.value()

        data = copy.deepcopy(self.indata['Raster'])
        data2 = []

        self.pbar.setMinimum(0)
        self.pbar.setMaximum(len(data)+1)

        for i in range(len(data)):
            self.pbar.setValue(i+1)
            self.parent.showprocesslog(data[i].bandid+':')

            vtot, vstd, vsum = self.visibility2d(
                data[i].data, self.wsize, self.dh*data[i].data.std()/100.)
#            data[i].data = vtot
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2[-3].data = vtot
            data2[-2].data = vstd
            data2[-1].data = vsum
            data2[-3].bandid += ' Total Visibility'
            data2[-2].bandid += ' Visibility Variation'
            data2[-1].bandid += ' Visibility Vector Resultant'

        self.pbar.setValue(self.pbar.maximum())

        self.outdata['Raster'] = data2
        self.parent.showprocesslog('Finished!')

        return True

    def visibility2d(self, data, wsize, dh):
        """
        #***************************************************************
        # *** Compute visibility as a textural measure
        # *** G.R.J.Cooper August'02
        # *** cooperg@geosciences.wits.ac.za grcooper@iafrica.com
        # *** www.wits.ac.za/science/geophysics/gc.htm
        # **************************************************************
        # *** wsize must be odd
        # *** dh = height of observer above surface
        #***************************************************************
        # *** Compute vertical derivatives by calculating the visibility
        # *** at different heights above the surface (see paper)
        # **************************************************************
        """

        nr, nc = np.shape(data)
        wsize = abs(np.real(wsize))
        w2 = int(np.floor(wsize/2))
        vn = np.zeros([nr, nc])
        vs = np.zeros([nr, nc])
        ve = np.zeros([nr, nc])
        vw = np.zeros([nr, nc])
        vd1 = np.zeros([nr, nc])
        vd2 = np.zeros([nr, nc])
        vd3 = np.zeros([nr, nc])
        vd4 = np.zeros([nr, nc])
        vstd = np.zeros([nr, nc])
        mask = data.mask
        mean = data.mean()
        data = data.data
        data[mask] = mean

        self.parent.showprocesslog('NS')
        for J in range(nc):    # Columns
            for I in range(w2, nr-w2):
                d = data[I-w2:I+w2+1, J]
                vn[I, J] = self.visible1(d, wsize, w2+1, dh)
                vs[I, J] = self.visible2(d, wsize, w2+1, dh)

        self.parent.showprocesslog('EW')
        for J in range(w2, nc-w2):    # Rows
            for I in range(nr):
                d = data[I, J-w2:J+w2+1]
                ve[I, J] = self.visible1(d, wsize, w2+1, dh)
                vw[I, J] = self.visible2(d, wsize, w2+1, dh)

        self.parent.showprocesslog('Diag')
        for J in range(w2, nc-w2):
            for I in range(w2, nr-w2):
                d = np.zeros(wsize)
                for K in range(wsize):
                    d[K] = data[I-w2+K, J-w2+K]
                vd1[I, J] = self.visible1(d, wsize, w2+1, dh)
                vd2[I, J] = self.visible2(d, wsize, w2+1, dh)
                d = np.zeros(wsize)
                for K in range(wsize):
                    d[K] = data[I+w2-K, J-w2+K]
                vd3[I, J] = self.visible1(d, wsize, w2+1, dh)
                vd4[I, J] = self.visible2(d, wsize, w2+1, dh)

        self.parent.showprocesslog('Computing std of visibility')
        vtot = vn+vs+ve+vw+vd1+vd2+vd3+vd4
        vtot = vtot[w2:nr-w2, w2:nc-w2]

        for J in range(nc):
            for I in range(nr):
                vstd[I, J] = np.std([vn[I, J], vs[I, J], ve[I, J], vw[I, J],
                                    vd1[I, J], vd2[I, J], vd3[I, J],
                                    vd4[I, J]], ddof=1)

        vstd = vstd[w2:nr-w2, w2:nc-w2]

        dtr = np.pi/180
        c45 = np.cos(45*dtr)
        s45 = np.sin(45*dtr)
        vsumx = ve-vw+vd1*c45-vd2*c45+vd3*c45-vd4*c45
        vsumy = vn-vs+vd1*s45-vd2*s45-vd3*s45+vd4*s45
        vsum = np.sqrt(vsumx*vsumx+vsumy*vsumy)
        vsum = vsum[w2:nr-w2, w2:nc-w2]

        vtot = np.ma.array(vtot)
        vstd = np.ma.array(vstd)
        vsum = np.ma.array(vsum)
        vtot.mask = mask[w2:-w2, w2:-w2]
        vstd.mask = mask[w2:-w2, w2:-w2]
        vsum.mask = mask[w2:-w2, w2:-w2]

        return vtot, vstd, vsum

# -----------------------------------------------------------------------------
    def visible1(self, d, nr, cp, dh):
        """ Visible 1 """
        n = 1
#        d = d[d.nonzero()].tolist()

        if cp < nr-1 and len(d) > 0:
            n = 2
            cpn = cp-1
            thetamax = float(d[cpn+1]-d[cpn]-dh)
            for I in range(cpn+2, nr):
                theta = ((d[I]-d[cpn]-dh)/float(I-cpn))
                if theta >= thetamax:
                    n = n + 1
                    thetamax = theta

        return n

# *****************************************************************************
    def visible2(self, d, nr, cp, dh):
        """ Visible 2 """
        n = 0
#        d = d[d.nonzero()].tolist()

        if cp > 2 and len(d) > 0:
            n = 1
            cpn = cp-1
            thetamax = (d[cpn-1]-d[cpn]-dh)
            for I in range(cpn-2, -1, -1):
                theta = ((d[I]-d[cpn]-dh)/(cpn-I))
                if theta >= thetamax:
                    n = n + 1
                    thetamax = theta
        return n


class Tilt1(QtGui.QDialog):
    """ Tilt angle calculations """
    def __init__(self, parent):
        QtGui.QDialog.__init__(self, parent)

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.azi = 75
        self.smooth = 0

        self.gridlayout = QtGui.QGridLayout(self)
        self.label = QtGui.QLabel(self)
        self.buttonbox = QtGui.QDialogButtonBox(self)
        self.sb_azi = QtGui.QSpinBox(self)
        self.sb_s = QtGui.QSpinBox(self)
        self.label_2 = QtGui.QLabel(self)
        self.pbar = QtGui.QProgressBar(self)

        self.setupui()

        self.sb_s.setValue(self.smooth)
        self.sb_azi.setValue(self.azi)

    def setupui(self):
        """ Setup UI """

        self.gridlayout.addWidget(self.label, 1, 0, 1, 1)
        self.buttonbox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonbox.setStandardButtons(
            QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.Ok)
        self.gridlayout.addWidget(self.buttonbox, 3, 0, 1, 2)
        self.sb_azi.setMinimum(-360)
        self.sb_azi.setMaximum(360)
        self.sb_azi.setProperty("value", 0)
        self.gridlayout.addWidget(self.sb_azi, 1, 1, 1, 1)
        self.sb_s.setPrefix("")
        self.sb_s.setMinimum(0)
        self.sb_s.setMaximum(100000)
        self.sb_s.setSingleStep(1)
        self.gridlayout.addWidget(self.sb_s, 0, 1, 1, 1)
        self.gridlayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.pbar.setProperty("value", 0)
        self.gridlayout.addWidget(self.pbar, 2, 0, 1, 2)

        self.setWindowTitle("Tilt Angle")
        self.label.setText("Azimuth (degrees from east)")
        self.label_2.setText("Smoothing Matrix Size (Odd, 0 for None)")

        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("accepted()"),
                               self.accept)
        QtCore.QObject.connect(self.buttonbox, QtCore.SIGNAL("rejected()"),
                               self.reject)

    def settings(self):
        """ Settings """
        temp = self.exec_()
        if temp == 0:
            return

        self.smooth = self.sb_s.value()
        self.azi = self.sb_azi.value()

        data = copy.deepcopy(self.indata['Raster'])
        data2 = []

        self.pbar.setMinimum(0)
        self.pbar.setMaximum(len(data)+1)

        for i in range(len(data)):
            self.pbar.setValue(i+1)

            t1, th, t2, ta, tdx = self.tilt1(data[i].data, self.azi,
                                             self.smooth)
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2.append(copy.deepcopy(data[i]))
            data2[-5].data = t1
            data2[-4].data = th
            data2[-3].data = t2
            data2[-2].data = ta
            data2[-1].data = tdx
            data2[-5].bandid += ' Standard Tilt Angle'
            data2[-4].bandid += ' Hyperbolic Tilt Angle'
            data2[-3].bandid += ' 2nd Order Tilt Angle'
            data2[-2].bandid += ' Tilt Based Directional Derivative'
            data2[-1].bandid += ' Total Derivative'

        self.pbar.setValue(self.pbar.maximum())

        self.outdata['Raster'] = data2
        return True

    def tilt1(self, data, azi, s):
        """
        # ****************************************************************
        # *** Tilt angle calculations
        # *** GRJ Cooper 2005
        # *** School of Geosciences, University of the Witwatersrand
        # *** Johannesburg, South Africa
        # *** cooperg@geosciences.wits.ac.za
        # *** www.wits.ac.za/science/geophysics/gc.htm
        # ****************************************************************
        # *** Parameters
        # *** data      ; matrix of double to be filtered
        # *** azi       ; directional filter azimuth in degrees from East
        # *** thresh;   ; threshold for display of hyp tilt eg 70
        # *** map       ; colourmap to use eg 'jet'
        # *** trans     ; transparency for overlay of 2nd order tilt -
        # ***             0<trans<1
        # *** s         ; size of smoothing matrix to use - must be odd
        # ***             input 0 for no smoothing
        # ****************************************************************
        # *** NB Needs image processing toolbox for transparency overlay
        """

        dmin = data.min()
        dmax = data.max()
        dm = 0.5*(dmin+dmax)
        data[np.isnan(data)] = dm
        data[np.isinf(data)] = dm
        if s > 0:
            se = np.ones(s)/(s*s)
            data = si.convolve2d(data, se, 'valid')  # smooth

        nr, nc = data.shape
        dtr = np.pi/180.0
        azi = azi*dtr
#        thresh = thresh*dtr

        dy, dx = np.gradient(data)
        dxtot = np.sqrt(dx*dx+dy*dy)
        nmax = np.max([nr, nc])
        npts = int(2**self.nextpow2(nmax))
        dz = self.vertical(data, npts, nc, nr, 1)
        t1 = np.arctan(dz/dxtot)
        th = np.real(np.arctanh(np.nan_to_num(dz/dxtot)+(0+0j)))
        tdx = np.real(np.arctan(dxtot/abs(dz)))

        dx1 = dx*np.cos(azi)+dy*np.sin(azi)  # Standard directional derivative
        dx2 = dx*np.cos(azi+np.pi/2)+dy*np.sin(azi+np.pi/2)
        dxz = np.sqrt(dx2*dx2+dz*dz)
        ta = np.arctan(dx1/dxz)         # Tilt directional derivative

        # 2nd order Tilt angle

        ts = t1
        if s < 3:
            s = 3
        se = np.ones([s, s])/(s*s)
#        s2 = np.floor(s/2)
        ts = si.convolve2d(t1, se, 'same')
        [dxs, dys] = np.gradient(ts)
        dzs = self.vertical(ts, npts, nc, nr, 1)
        dxtots = np.sqrt(dxs*dxs+dys*dys)
        t2 = np.arctan(dzs/dxtots)

        # Standard tilt angle, hyperbolic tilt angle, 2nd order tilt angle,
        # Tilt Based Directional Derivative, Total Derivative
        t1 = np.ma.array(t1)
        th = np.ma.array(th)
        t2 = np.ma.array(t2)
        ta = np.ma.array(ta)
        tdx = np.ma.array(tdx)

        return t1, th, t2, ta, tdx

    def nextpow2(self, n):
        """ nextpow2 """
        m_i = np.ceil(np.log2(n))
        return m_i

# *****************************************************************************
    def vertical(self, data, npts, nc, nr, xint):
        """ Vertical """

        cdiff = int(np.floor((npts-nc)/2))
        rdiff = int(np.floor((npts-nr)/2))
        data1 = self.taper2d(data, npts, nc, nr, cdiff, rdiff)
        f = np.fft.fft2(data1)
        fz = f
        wn = 2.0*np.pi/(xint*(npts-1))
        f = np.fft.fftshift(f)
        cx = npts/2+1
        cy = cx
        for I in range(npts):
            freqx = (I+1-cx)*wn
            for J in range(npts):
                freqy = (J+1-cy)*wn
                freq = np.sqrt(freqx*freqx+freqy*freqy)
                fz[I, J] = f[I, J]*freq
        fz = np.fft.fftshift(fz)
        fzinv = np.fft.ifft2(fz)
        dz = np.real(fzinv[rdiff:nr+rdiff, cdiff:nc+cdiff])
        return dz

# *****************************************************************************
    def taper2d(self, g, npts, n, m, ndiff, mdiff):
        """ Taper 2D """

        npts2 = npts-1
        gm = g.mean()
        gf = np.zeros([npts, npts])
        gf[mdiff:mdiff+m, ndiff:ndiff+n] = g-gm
        for J in range(mdiff, mdiff+m):
            for I in range(ndiff):
                gf[I, J] = gf[I, J]*((1+np.sin(-np.pi/2+I*np.pi/ndiff))*0.5)
                gf[npts2-I, J] = gf[npts2-I, J]*((1+np.sin(
                                                 -np.pi/2+I*np.pi/ndiff))*0.5)

        for J in range(mdiff):
            tmp = ((1+np.sin(-np.pi/2+J*np.pi/(mdiff)))*0.5)
            for I in range(ndiff, ndiff+n):
                gf[I, J] = gf[I, J]*tmp
                gf[I, npts2-J-1] = gf[I, npts2-J-1]*tmp

        for I in range(ndiff):
            tmp = ((1+np.sin(-np.pi/2+I*np.pi/(ndiff)))*0.5)
            for J in range(mdiff):
                gf[I, J] = gf[I, J]*tmp
                gf[npts2-I, J] = gf[npts2-I, J]*tmp

        for I in range(ndiff):
            tmp = ((1+np.sin(-np.pi/2+I*np.pi/(ndiff)))*0.5)
            for J in range(mdiff+m-1, npts):
                gf[I, J] = gf[I, J]*tmp
                gf[npts2-I, J] = gf[npts2-I, J]*tmp

        for J in range(mdiff+m-1, npts):  # Corners
            for I in range(ndiff+n-1, npts):
                if ndiff == 0 or mdiff == 0:
                    gf[I, J] = np.nan
                else:
                    gf[I, J] = (gf[I, J] *
                                np.cos((I+1-ndiff-n)*np.pi/(2*ndiff)) *
                                np.cos((J+1-ndiff-m) * np.pi/(2*mdiff)))

        for J in range(mdiff):
            for I in range(ndiff):
                if ndiff == 0 or mdiff == 0:
                    gf[I, J] = np.nan
                else:
                    gf[I, J] = (gf[I, J] *
                                np.cos((I+1-ndiff)*np.pi/(2*ndiff)) *
                                np.cos((J+1-ndiff)*np.pi/(2*mdiff)))

        for J in range(mdiff):
            for I in range(ndiff+n-1, npts):
                if ndiff == 0 or mdiff == 0:
                    gf[I, J] = np.nan
                else:
                    gf[I, J] = (gf[I, J] *
                                np.cos((I+1-ndiff-n)*np.pi/(2*ndiff)) *
                                np.cos((J+1-ndiff)*np.pi/(2*mdiff)))

        for J in range(mdiff+m, npts):
            for I in range(ndiff):
                if ndiff == 0 or mdiff == 0:
                    gf[I, J] = np.nan
                else:
                    gf[I, J] = (gf[I, J] *
                                np.cos((I+1-ndiff)*np.pi/(2*ndiff)) *
                                np.cos((J+1-ndiff-m)*np.pi/(2*mdiff)))

        return gf
