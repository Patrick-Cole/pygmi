# -----------------------------------------------------------------------------
# Name:        beachball.py (part of PyGMI)
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
# Licence for original ModestImage code (modified version below)
# ModestImage
# Copyright (c) 2013 Chris Beaumont
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------------
""" Plot fault plane solutions.
The code below is translated from bb.m written by Andy Michael and Oliver Boyd
at http://www.ceri.memphis.edu/people/olboyd/Software/Software.html"""

import os
import numpy as np
import numexpr as ne
from PyQt4 import QtGui
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib import patches
import ogr
import osr
import scipy.spatial.distance as sdist


class MyMplCanvas(FigureCanvas):
    """Canvas for the actual plot"""
    def __init__(self, parent):
        fig = plt.figure()
        super(MyMplCanvas, self).__init__(fig)

        # figure stuff
        self.htype = 'Linear'
        self.cbar = cm.jet
        self.data = []
        self.gmode = None
        self.argb = [None, None, None]
        self.hhist = [None, None, None]
        self.hband = [None, None, None]
        self.htxt = [None, None, None]
        self.patches = []
        self.lines = []
        self.image = None
        self.cnt = None
        self.cntf = None
        self.background = None
        self.pwidth = 0.001

        self.axes = fig.add_subplot(111)
        self.axes.xaxis.set_visible(False)
        self.axes.yaxis.set_visible(False)

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def init_graph(self):
        """ Initialize the graph """

        self.axes.clear()
        self.axes.set_aspect("equal")

        maxdiam = self.pwidth*self.data[:, -1].max()
        xmin = self.data[:, 0].min()-maxdiam
        xmax = self.data[:, 0].max()+maxdiam
        ymin = self.data[:, 1].min()-maxdiam
        ymax = self.data[:, 1].max()+maxdiam

        self.axes.set_xlim((xmin, xmax))
        self.axes.set_ylim((ymin, ymax))

        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()

        for idat in self.data:
            pxy = idat[:2]
            np1 = idat[3:-1]
            pwidth = self.pwidth*idat[-1]
            xxx, yyy, xxx2, yyy2 = beachball(np1, pxy[0], pxy[1], pwidth)

            pvert1 = np.transpose([yyy, xxx])
            pvert0 = np.transpose([xxx2, yyy2])

            self.axes.add_patch(patches.Polygon(pvert1))
            self.axes.add_patch(patches.Polygon(pvert0, facecolor='none'))

        self.figure.canvas.draw()


class BeachBall(QtGui.QDialog):
    """ Create shapefiles with beachballs """
    def __init__(self, parent=None):
        super(BeachBall, self).__init__(parent)

        self.ifile = ""
        self.name = "Beachball: "
        self.ext = ""
        self.pbar = None
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.algorithm = 'FPFIT'
        self.nofps = False

        self.mmc = MyMplCanvas(self)
        self.btn_saveshp = QtGui.QPushButton()
        self.cbox_alg = QtGui.QComboBox()
        self.dsb_dist = QtGui.QDoubleSpinBox()

        self.setupui()

    def data_init(self):
        """ data init - entry point into routine """
        if 'Seis' not in self.indata:
            return

        data = self.indata['Seis']

        alist = []
        for i in data:
            alist += list(i['F'].keys())
        alist = sorted(set(alist))

        if len(alist) == 0:
            self.parent.showprocesslog('Error: no Fault Plane Solutions')
            self.nofps = True
            return False
        else:
            self.nofps = False

        try:
            self.cbox_alg.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.cbox_alg.clear()
        self.cbox_alg.addItems(alist)
        self.algorithm = alist[0]

        self.cbox_alg.currentIndexChanged.connect(self.change_alg)

        pwidth = 1.

        for j in alist:
            tmpxy = []
            tmpmag = []
            for i in data:
                if i['F'].get(j) is not None:
                    tmp = [i['1'].longitude,
                           i['1'].latitude,
                           i['1'].magnitude_1]
                    tmpxy.append([tmp[0], tmp[1]])
                    tmpmag.append(tmp[-1])

            if len(tmpmag) > 1:
                pwidth = min(np.median(sdist.pdist(tmpxy))/(2*max(tmpmag)),
                             pwidth)

        self.dsb_dist.setValue(pwidth)

        self.change_alg()

    def setupui(self):
        """ Setup UI """
        hbl_all = QtGui.QHBoxLayout(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        vbl_raster = QtGui.QVBoxLayout()
        label2 = QtGui.QLabel()
        label3 = QtGui.QLabel()

        self.dsb_dist.setDecimals(4)
        self.dsb_dist.setMinimum(0.0001)
        self.dsb_dist.setSingleStep(0.0001)
        self.dsb_dist.setProperty("value", 0.001)

        spacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum,
                                   QtGui.QSizePolicy.Expanding)

        self.setWindowTitle("Fault Plane Solution (FPS)")
        label2.setText('FPS Algorithm:')
        label3.setText('Width Scale Factor:')
        self.btn_saveshp.setText('Save Shapefile')

        vbl_raster.addWidget(label2)
        vbl_raster.addWidget(self.cbox_alg)
        vbl_raster.addWidget(label3)
        vbl_raster.addWidget(self.dsb_dist)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_saveshp)
        vbl_right = QtGui.QVBoxLayout()
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)
        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.btn_saveshp.clicked.connect(self.save_shp)
        self.dsb_dist.valueChanged.connect(self.change_alg)

    def save_shp(self):
        """Save Beachballs """
#        data = self.indata['Seis']

        ext = "Shape file (*.shp)"

        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(filename.rpartition('/')[0])

        self.ifile = str(filename)
        self.ext = filename[-3:]

        indata = self.mmc.data

        ifile_bnd = self.ifile[:-4]+'_bnd.shp'
        if os.path.isfile(self.ifile):
            tmp = self.ifile[:-4]
            os.remove(tmp+'.shp')
            os.remove(tmp+'.shx')
            os.remove(tmp+'.prj')
            os.remove(tmp+'.dbf')
        if os.path.isfile(ifile_bnd):
            tmp = ifile_bnd[:-4]
            os.remove(tmp+'.shp')
            os.remove(tmp+'.shx')
            os.remove(tmp+'.prj')
            os.remove(tmp+'.dbf')

        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(self.ifile)
        data_source2 = driver.CreateDataSource(ifile_bnd)

        # create the spatial reference, WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # create the layer
        layer = data_source.CreateLayer("Fault Plane Solution", srs,
                                        ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("Strike", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("Dip", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("Rake", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("Magnitude", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("Quadrant", ogr.OFTString))
        layer.CreateField(ogr.FieldDefn("Depth", ogr.OFTReal))

        layer2 = data_source2.CreateLayer("Fault Plane Solution Boundaries",
                                          srs, ogr.wkbPolygon)

        layer2.CreateField(ogr.FieldDefn("Strike", ogr.OFTReal))
        layer2.CreateField(ogr.FieldDefn("Dip", ogr.OFTReal))
        layer2.CreateField(ogr.FieldDefn("Rake", ogr.OFTReal))
        layer2.CreateField(ogr.FieldDefn("Magnitude", ogr.OFTReal))
        layer2.CreateField(ogr.FieldDefn("Quadrant", ogr.OFTString))
        layer2.CreateField(ogr.FieldDefn("Depth", ogr.OFTReal))

        # Calculate BeachBall
        for idat in indata:
            pxy = idat[:2]
            np1 = idat[3:-1]
            depth = idat[2]
            pwidth = self.mmc.pwidth*idat[-1]
            xxx, yyy, xxx2, yyy2 = beachball(np1, pxy[0], pxy[1], pwidth)

            pvert1 = np.transpose([yyy, xxx])
            pvert0 = np.transpose([xxx2, yyy2])

            # Create Geometry
            outring = ogr.Geometry(ogr.wkbLinearRing)
            for i in pvert1:
                outring.AddPoint(i[0], i[1])

            innerring = ogr.Geometry(ogr.wkbLinearRing)
            for i in pvert0:
                innerring.AddPoint(i[0], i[1])

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(outring)

            poly1 = ogr.Geometry(ogr.wkbPolygon)
            poly1.AddGeometry(innerring)

            feature = ogr.Feature(layer.GetLayerDefn())

            feature.SetField("Strike", np1[0])
            feature.SetField("Dip", np1[1])
            feature.SetField("Rake", np1[2])
            feature.SetField("Magnitude", idat[-1])
            feature.SetField("Quadrant", "Compressional")
            feature.SetField("Depth", depth)

            feature.SetGeometry(poly)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Destroy the feature to free resources

            feature2 = ogr.Feature(layer2.GetLayerDefn())
            feature2.SetField("Quadrant", "Tensional and Compressional")
            feature2.SetField("Depth", depth)
            feature2.SetGeometry(poly1)
            # Create the feature in the layer (shapefile)
            layer2.CreateFeature(feature2)
            layer2.CreateFeature(feature)
            # Destroy the feature to free resources
            feature.Destroy()
            feature2.Destroy()

        data_source.Destroy()
        data_source2.Destroy()

    def change_alg(self):
        """ Change algorithm """
        txt = str(self.cbox_alg.currentText())
        self.algorithm = txt
        data = self.indata['Seis']

        indata = []
        for i in data:
            if i['F'].get(self.algorithm) is not None:
                tmp = [i['1'].longitude, i['1'].latitude, i['1'].depth,
                       i['F'][self.algorithm].strike,
                       i['F'][self.algorithm].dip,
                       i['F'][self.algorithm].rake,
                       i['1'].magnitude_1]
                indata.append(tmp)

        self.mmc.data = np.array(indata)
        self.mmc.pwidth = self.dsb_dist.value()
        self.mmc.init_graph()

    def settings(self):
        """ run """
        if self.nofps:
            self.parent.showprocesslog('Error: no Fault Plane Solutions')
            return False

        self.show()
        QtGui.QApplication.processEvents()

        self.mmc.init_graph()


def beachball(fm, centerx, centery, diam):
    """
    Source code provided here are adopted from MatLab script
    `bb.m` written by Andy Michael and Oliver Boyd.

    function bb(fm, centerx, centery, diam, ta, color)
    Draws beachball diagram of earthquake double-couple focal mechanism(s).
    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    Parameters
    ----------
    fm : numpy array
        focal mechanism that is either number of mechnisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (mxx, myy, mzz, mxy, mxz, myz -
        the six independent components of the moment tensor). The strike is
        of the first plane, clockwise relative to north. The dip is of the
        first plane, defined clockwise and perpedicular to strike, relative
        to horizontal such that 0 is horizontal and 90 is vertical. The rake is
        of the first focal plane solution. 90 moves the hanging wall up-dip
        (thrust), 0 moves it in the strike direction (left-lateral), -90 moves
        it down-dip (normal), and 180 moves it opposite to strike
        (right-lateral).
    centerx : float
        place beachball(s) at position centerx
    centery : float
        place beachball(s) at position centery
    diam : float
        draw beachball with this diameter.

    Returns
    -------
    X : numpy array
        array of x coordinates for vertices
    Y : numpy array
        array of y coordinates for vertices
    xx : numpy array
        array of x coordinates for vertices
    yy : numpy array
        array of y coordinates for vertices
    """

    fm = np.array(fm)
    diam = np.array([diam])
    centerx = np.array([centerx])
    centery = np.array([centery])

    if fm.ndim == 1:
        ne1 = 1
        n = fm.size
        fm = np.array([fm])
    else:
        ne1, n = fm.shape
    if n == 6:
        s1 = np.zeros(ne)
        d1 = np.zeros(ne)
        r1 = np.zeros(ne)
        for j in range(ne):
            s1[j], d1[j], r1[j] = mij2sdr(fm[j, 0], fm[j, 1], fm[j, 2],
                                          fm[j, 3], fm[j, 4], fm[j, 5])
    else:
        s1 = fm[:, 0]
        d1 = fm[:, 1]
        r1 = fm[:, 2]

    # r2d = 180/np.pi
    d2r = np.pi/180
    ampy = np.cos(np.mean(centery)*d2r)

    if ne1 > 1:
        [_, i] = np.sort(diam, 1, 'descend')
        diam = diam[i]
        s1 = s1[i]
        d1 = d1[i]
        r1 = r1[i]
        centerx = centerx[i]
        centery = centery[i]

    mech = np.zeros([ne1, 1])
    j = np.nonzero(r1 > 180)
    r1[j] = r1[j] - 180
    mech[j] = 1
    j = np.nonzero(r1 < 0)
    r1[j] = r1[j] + 180
    mech[j] = 1

    # Get azimuth and dip of second plane
#    s2, d2, r2 = auxplane(s1, d1, r1)
    s2, d2, _ = auxplane(s1, d1, r1)

    for ev in range(ne1):
        S1 = s1[ev]
        D1 = d1[ev]
        S2 = s2[ev]
        D2 = d2[ev]
        P = r1[ev]
        CX = centerx[ev]
        CY = centery[ev]
        D = diam[ev]
        M = mech[ev]

    if M > 0:
        P = 2
    else:
        P = 1

    if D1 >= 90:
        D1 = 89.9999
    if D2 >= 90:
        D2 = 89.9999

    phi = np.arange(0, np.pi, .01)
    d = 90 - D1
    m = 90
    l1 = np.sqrt(d**2/(np.sin(phi)**2 + np.cos(phi)**2 * d**2/m**2))

    d = 90 - D2
    m = 90
    l2 = np.sqrt(d**2/(np.sin(phi)**2 + np.cos(phi)**2 * d**2/m**2))

    if D == 0:
        print('Enter a diameter for the beachballs!')
        return

    inc = 1
    X1, Y1 = pol2cart(phi+S1*d2r, l1)
    if P == 1:
        lo = S1 - 180
        hi = S2
        if lo > hi:
            inc = -inc
        th1 = np.arange(S1-180, S2, inc)
        Xs1, Ys1 = pol2cart(th1*d2r, 90*np.ones_like(th1))
        X2, Y2 = pol2cart(phi+S2*d2r, l2)
        th2 = np.arange(S2+180, S1, -inc)
    else:
        hi = S1 - 180
        lo = S2 - 180
        if lo > hi:
            inc = -inc
        th1 = np.arange(hi, lo, -inc)
        Xs1, Ys1 = pol2cart(th1*d2r, 90*np.ones_like(th1))
        X2, Y2 = pol2cart(phi+S2*d2r, l2)
        X2 = X2[::-1]
        Y2 = Y2[::-1]
        th2 = np.arange(S2, S1, inc)

    Xs2, Ys2 = pol2cart(th2*d2r, 90*np.ones_like(th2))

#    X = np.concatenate((X1, Xs1, X2, Xs2), 1)
#    Y = np.concatenate((Y1, Ys1, Y2, Ys2), 1)
    X = np.hstack((X1, Xs1, X2, Xs2))
    Y = np.hstack((Y1, Ys1, Y2, Ys2))

    ampy = 1.0
    if D > 0:
        X = ampy*X*D/90 + CY
        Y = Y*D/90 + CX
        phid = np.arange(0, 2*np.pi, 0.01)
        x, y = pol2cart(phid, 90)
        xx = x*D/90 + CX
        yy = ampy*y*D/90 + CY

    return X, Y, xx, yy


def pol2cart(phi, rho):
    """ Polar to cartesian coordinates """
    xxx = rho*np.cos(phi)
    yyy = rho*np.sin(phi)
    return xxx, yyy


def auxplane(s1, d1, r1):
    """
    function [strike, dip, rake] = auxplane(s1,d1,r1)
    Get Strike and dip of second plane, adapted from Andy Michael bothplanes.c
    """
    r2d = 180/np.pi

    z = (s1+90)/r2d
    z2 = d1/r2d
    z3 = r1/r2d
    # /* slick vector in plane 1 */
    sl1 = -np.cos(z3)*np.cos(z)-np.sin(z3)*np.sin(z)*np.cos(z2)
    sl2 = np.cos(z3)*np.sin(z)-np.sin(z3)*np.cos(z)*np.cos(z2)
    sl3 = np.sin(z3)*np.sin(z2)
    strike, dip = strikedip(sl2, sl1, sl3)

    n1 = np.sin(z)*np.sin(z2)  # /* normal vector to plane 1 */
    n2 = np.cos(z)*np.sin(z2)
#    n3 = np.cos(z2)
    h1 = -sl2  # /* strike vector of plane 2 */
    h2 = sl1
    # /* note h3=0 always so we leave it out */

    z = h1*n1 + h2*n2
    z = z/np.sqrt(h1*h1 + h2*h2)
    z = np.arccos(z)

    rake = np.zeros_like(strike)
    j = np.nonzero(sl3 > 0)
    rake[j] = z[j]*r2d
    j = np.nonzero(sl3 <= 0)
    rake[j] = -z[j]*r2d

    return strike, dip, rake


def strikedip(n, e, u):
    """
    function [strike, dip] = strikedip(n, e, u)
       Finds strike and dip of plane given normal vector having components n,
       e, and u
       Adapted from Andy Michaels stridip.c
    """
    r2d = 180/np.pi

    j = np.nonzero(u < 0)
    n[j] = -n[j]
    e[j] = -e[j]
    u[j] = -u[j]

    strike = np.arctan2(e, n)*r2d
    strike = strike - 90
    while strike >= 360:
        strike = strike - 360
    while strike < 0:
        strike = strike + 360

    x = np.sqrt(n**2 + e**2)
    dip = np.arctan2(x, u)*r2d
    return strike, dip


def mij2sdr(mxx, myy, mzz, mxy, mxz, myz):
    """
    function [str,dip,rake] = mij2sdr(mxx,myy,mzz,mxy,mxz,myz)
    Adapted from code, mij2d.f, created by Chen Ji

    Parameters
    ----------
    mxx - float
        independent component of the moment tensor
    myy - float
        independent component of the moment tensor
    mzz - float
        independent component of the moment tensor
    mxy - float
        independent component of the moment tensor
    mxz - float
        independent component of the moment tensor
    myz - float
        independent component of the moment tensor

    Returns
    -------
    strike : float
        strike of first focal plane (degrees)
    dip : float
        dip of first focal plane (degrees)
    rake : float
        rake of first focal plane (degrees)
    """

    a = np.array([[mxx, mxy, mxz],
                  [mxy, myy, myz],
                  [mxz, myz, mzz]])
    d, V = np.linalg.eig(a)

    D = np.array([d[1], d[0], d[2]])

    V = np.array([[-V[1, 1], V[1, 0], V[1, 2]],
                  [-V[2, 1], V[2, 0], V[2, 2]],
                  [V[0, 1], -V[0, 0], -V[0, 2]]])

    imax = np.nonzero(D == np.max(D))
    imin = np.nonzero(D == np.min(D))
    AE = (V[:, imax]+V[:, imin])/np.sqrt(2.0)
    AN = (V[:, imax]-V[:, imin])/np.sqrt(2.0)
    aer = np.sqrt(AE[0]**2+AE[1]**2+AE[2]**2)
    anr = np.sqrt(AN[0]**2+AN[1]**2+AN[2]**2)
    AE = AE/aer
    AN = AN/anr
    if AN[2] <= 0.:
        an1 = AN
        ae1 = AE
    else:
        an1 = -AN
        ae1 = -AE
    ft, fd, fl = TDL(an1, ae1)
    strike = 360 - ft
    dip = fd
    rake = 180 - fl
    return strike, dip, rake


def TDL(AN, BN):
    """
    TDL

    Parameters
    ----------
    AN : numpy array
        array comprising XN, YN, ZN
    BN : numpy array
        array comprising XE, YE, ZE

    Returns
    -------
    FT : float
        relates to strike (360 - ft)
    FD : float
        dip
    FL : float
        relates to rake (180 - fl)
    """
    XN, YN, ZN = AN.flatten()
    XE, YE, ZE = BN.flatten()
    aaa = 1.0E-06
    fdh = 57.2957795
    if abs(ZN) < aaa:
        FD = 90.
        axn = abs(XN)
        if axn > 1.0:
            axn = 1.0
        FT = np.arcsin(axn)*fdh
        ST = -XN
        CT = YN
        if ST >= 0. and CT < 0:
            FT = 180.-FT
        if ST < 0. and CT <= 0:
            FT = 180.+FT
        if ST < 0. and CT > 0:
            FT = 360.-FT
        FL = np.arcsin(abs(ZE))*fdh
        SL = -ZE
        if abs(XN) < aaa:
            CL = XE/YN
        else:
            CL = -YE/XN
        if SL >= 0. and CL < 0:
            FL = 180.-FL
        if SL < 0. and CL <= 0:
            FL = FL-180.
        if SL < 0. and CL > 0:
            FL = -FL
    else:
        if -ZN > 1.0:
            ZN = -1.0
        fdh = np.arccos(-ZN)
        FD = fdh*fdh
        SD = np.sin(fdh)
        if SD == 0:
            return
        ST = -XN/SD
        CT = YN/SD
        SX = abs(ST)
        if SX > 1.0:
            SX = 1.0
        FT = np.arcsin(SX)*fdh
        if ST >= 0. and CT < 0:
            FT = 180.-FT
        if ST < 0. and CT <= 0:
            FT = 180.+FT
        if ST < 0. and CT > 0:
            FT = 360.-FT
        SL = -ZE/SD
        SX = np.abs(SL)
        if SX > 1.0:
            SX = 1.0
        FL = np.arcsin(SX)*fdh
        if ST == 0:
            CL = XE/CT
        else:
            xxx = YN*ZN*ZE/SD/SD+YE
            CL = -SD*xxx/XN
            if CT == 0:
                CL = YE/ST
        if SL >= 0. and CL < 0:
            FL = 180.-FL
        if SL < 0. and CL <= 0:
            FL = FL-180.
        if SL < 0. and CL > 0:
            FL = -FL

    return FT, FD, FL
