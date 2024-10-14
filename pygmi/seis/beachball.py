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
"""
Plot fault plane solutions.

The code below is translated from bb.m written by Andy Michael and Oliver Boyd
at http://www.ceri.memphis.edu/people/olboyd/Software/Software.html
"""

import os
import numpy as np
import numexpr as ne
from PyQt5 import QtWidgets
import geopandas as gpd
from shapely import Polygon, make_valid
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib import patches
import scipy.spatial.distance as sdist

from pygmi.misc import BasicModule


class MyMplCanvas(FigureCanvasQTAgg):
    """Canvas for the actual plot."""

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)
        if parent is None:
            self.showlog = print
        else:
            self.showlog = parent.showlog

        # figure stuff
        self.htype = 'Linear'
        self.cbar = colormaps['jet']
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
        self.isgeog = True

        self.axes = fig.add_subplot(111)

        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def init_graph(self):
        """
        Initialize the graph.

        Returns
        -------
        None.

        """
        self.axes.clear()
        self.axes.set_aspect('equal')

        maxdiam = self.pwidth*self.data[:, -1].max()
        xmin = self.data[:, 0].min()-maxdiam
        xmax = self.data[:, 0].max()+maxdiam
        ymin = self.data[:, 1].min()-maxdiam
        ymax = self.data[:, 1].max()+maxdiam

        self.axes.set_xlim((xmin, xmax))
        self.axes.set_ylim((ymin, ymax))

        self.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        for idat in self.data:
            pxy = idat[:2]
            np1 = idat[3:-1]
            pwidth = self.pwidth*idat[-1]

            # patch = beach(np1, xy=pxy)

            # self.axes.add_collection(patch)

            xxx, yyy, xxx2, yyy2 = beachball(np1, pxy[0], pxy[1], pwidth,
                                             self.isgeog)

            pvert1 = np.transpose([yyy, xxx])
            pvert0 = np.transpose([xxx2, yyy2])

            self.axes.add_patch(patches.Polygon(pvert1,
                                                edgecolor=(0.0, 0.0, 0.0)))
            self.axes.add_patch(patches.Polygon(pvert0, facecolor='none',
                                                edgecolor=(0.0, 0.0, 0.0)))

        self.figure.canvas.draw()


class BeachBall(BasicModule):
    """Create shapefiles with beachballs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.algorithm = 'FPFIT'
        self.nofps = False
        self.stype = 'Seis'

        self.mmc = MyMplCanvas(self)
        self.btn_saveshp = QtWidgets.QPushButton('Save Shapefile')
        self.cmb_alg = QtWidgets.QComboBox()
        self.dsb_dist = QtWidgets.QDoubleSpinBox()
        self.rb_geog = QtWidgets.QRadioButton('Geographic Units')
        self.rb_proj = QtWidgets.QRadioButton('Projected Units')

        self.setupui()

    def data_init(self):
        """
        Initialise Data - entry point into routine.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Seis' in self.indata:
            self.stype = 'Seis'
        elif 'GenFPS' in self.indata:
            self.stype = 'GenFPS'
        else:
            return False

        data = self.indata[self.stype]

        alist = []
        for i in data:
            alist += list(i['F'].keys())
        alist = sorted(set(alist))

        if not alist:
            self.showlog('Error: no Fault Plane Solutions')
            self.nofps = True
            return False
        self.nofps = False

        try:
            self.cmb_alg.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.cmb_alg.clear()
        self.cmb_alg.addItems(alist)
        self.algorithm = alist[0]

        self.cmb_alg.currentIndexChanged.connect(self.change_alg)

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

        return True

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        hbl_all = QtWidgets.QHBoxLayout(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self)
        vbl_raster = QtWidgets.QVBoxLayout()
        lbl_2 = QtWidgets.QLabel('FPS Algorithm:')
        lbl_3 = QtWidgets.QLabel('Width Scale Factor:')

        self.dsb_dist.setDecimals(4)
        self.dsb_dist.setMinimum(0.0001)
        self.dsb_dist.setSingleStep(0.0001)
        self.dsb_dist.setProperty('value', 0.001)

        self.rb_geog.setChecked(True)

        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                       QtWidgets.QSizePolicy.Expanding)

        self.setWindowTitle('Fault Plane Solution (FPS)')

        vbl_raster.addWidget(lbl_2)
        vbl_raster.addWidget(self.cmb_alg)
        vbl_raster.addWidget(lbl_3)
        vbl_raster.addWidget(self.dsb_dist)
        vbl_raster.addWidget(self.rb_geog)
        vbl_raster.addWidget(self.rb_proj)
        vbl_raster.addItem(spacer)
        vbl_raster.addWidget(self.btn_saveshp)
        vbl_right = QtWidgets.QVBoxLayout()
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)
        hbl_all.addLayout(vbl_raster)
        hbl_all.addLayout(vbl_right)

        self.btn_saveshp.clicked.connect(self.save_shp)
        self.dsb_dist.valueChanged.connect(self.change_alg)
        self.rb_geog.toggled.connect(self.change_alg)

    def save_shp(self):
        """
        Save Beachballs.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        ext = 'Shape file (*.shp)'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save Shape File', '.', ext)
        if filename == '':
            return False
        os.chdir(os.path.dirname(filename))

        self.ifile = str(filename)

        indata = self.mmc.data

        if os.path.isfile(self.ifile):
            tmp = self.ifile[:-4]
            os.remove(tmp+'.shp')
            os.remove(tmp+'.shx')
            os.remove(tmp+'.prj')
            os.remove(tmp+'.dbf')

        layer = {'Strike': [],
                 'Dip': [],
                 'Rake': [],
                 'Magnitude': [],
                 'Quadrant': [],
                 'Depth': [],
                 'geometry': []}

        # Calculate BeachBall
        for i, idat in enumerate(indata):
            pxy = idat[:2]
            np1 = idat[3:-1]
            depth = idat[2]
            pwidth = self.mmc.pwidth*idat[-1]
            xxx, yyy, xxx2, yyy2 = beachball(np1, pxy[0], pxy[1], pwidth,
                                             self.mmc.isgeog,
                                             self.showlog)

            pvert1 = np.transpose([yyy, xxx])
            pvert0 = np.transpose([xxx2, yyy2])

            poly1 = Polygon(pvert1)
            poly0 = Polygon(pvert0)

            poly1 = make_valid(poly1)
            poly0 = make_valid(poly0)

            layer['geometry'].append(poly0)
            layer['Strike'].append(0)
            layer['Dip'].append(0)
            layer['Rake'].append(0)
            layer['Magnitude'].append(0)
            layer['Depth'].append(depth)
            layer['Quadrant'].append('Tensional (White)')

            layer['geometry'].append(poly1)
            layer['Strike'].append(np1[0])
            layer['Dip'].append(np1[1])
            layer['Rake'].append(np1[2])
            layer['Magnitude'].append(idat[-1])
            layer['Depth'].append(depth)
            layer['Quadrant'].append('Compressional (Colour)')

        gdf = gpd.GeoDataFrame(layer)
        gdf = gdf.set_crs(4326)
        gdf.to_file(self.ifile)

        return True

    def change_alg(self):
        """
        Change algorithm.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_alg.currentText())
        self.algorithm = txt
        data = self.indata[self.stype]

        indata = []
        for i in data:
            if i['F'].get(self.algorithm) is not None:
                tmp = [i['1'].longitude, i['1'].latitude, i['1'].depth,
                       i['F'][self.algorithm].strike,
                       i['F'][self.algorithm].dip,
                       i['F'][self.algorithm].rake,
                       i['1'].magnitude_1]
                indata.append(tmp)

        self.mmc.isgeog = self.rb_geog.isChecked()
        self.mmc.data = np.array(indata)
        self.mmc.pwidth = self.dsb_dist.value()
        self.mmc.init_graph()

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Parameters
        ----------
        nodialog : bool, optional
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if self.nofps:
            self.showlog('Error: no Fault Plane Solutions')
            return False

        QtWidgets.QApplication.processEvents()
        self.mmc.init_graph()

        if not nodialog:
            temp = self.exec()

            if temp == 0:
                return False

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
        self.saveobj(self.algorithm)
        self.saveobj(self.nofps)
        self.saveobj(self.stype)
        self.saveobj(self.cmb_alg)
        self.saveobj(self.dsb_dist)
        self.saveobj(self.rb_geog)
        self.saveobj(self.rb_proj)


def beachball(fm, centerx, centery, diam, isgeog, showlog=print):
    """
    Beachball.

    Source code provided here are adopted from MatLab script
    `bb.m` written by Andy Michael and Oliver Boyd.

    function bb(fm, centerx, centery, diam, ta, color)
    Draws beachball diagram of earthquake double-couple focal mechanism(s).
    S1, D1, and R1, the strike, dip and rake of one of the focal planes, can
    be vectors of multiple focal mechanisms.

    Parameters
    ----------
    fm : numpy array
        focal mechanism that is either number of mechanisms (NM) by 3
        (strike, dip, and rake) or NM x 6 (mxx, myy, mzz, mxy, mxz, myz -
        the six independent components of the moment tensor). The strike is
        of the first plane, clockwise relative to north. The dip is of the
        first plane, defined clockwise and perpendicular to strike, relative
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
    isgeog : bool
        True if in geographic coordinates, False otherwise.
    showlog : function, optional
        Routine to show text messages. The default is print.

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
        showlog('Enter a diameter for the beachballs!')
        return None

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

    X = np.hstack((X1, Xs1, X2, Xs2))
    Y = np.hstack((Y1, Ys1, Y2, Ys2))

    if isgeog:
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
    """
    Polar to cartesian coordinates.

    Parameters
    ----------
    phi : numpy array
        Polar angles in radians.
    rho : numpy array
        Polar r values.

    Returns
    -------
    xxx : numpy array
        X values.
    yyy : numpy array
        Y values.

    """
    xxx = rho*np.cos(phi)
    yyy = rho*np.sin(phi)
    return xxx, yyy


def auxplane(s1, d1, r1):
    """
    Get Strike and dip of second plane.

    Adapted from Andy Michael bothplanes.c

    Parameters
    ----------
    s1 : numpy array
        Strike 1.
    d1 : numpy array
        Dip 1.
    r1 : numpy array
        Rake 1.

    Returns
    -------
    strike : numpy array
        Strike of second plane.
    dip : numpy array
        Dip of second plane.
    rake : numpy array
        Rake of second plane.
    """
    r2d = 180/np.pi

    z = (s1+90)/r2d
    z2 = d1/r2d
    z3 = r1/r2d
    # slick vector in plane 1
    sl1 = -np.cos(z3)*np.cos(z)-np.sin(z3)*np.sin(z)*np.cos(z2)
    sl2 = np.cos(z3)*np.sin(z)-np.sin(z3)*np.cos(z)*np.cos(z2)
    sl3 = np.sin(z3)*np.sin(z2)
    strike, dip = strikedip(sl2, sl1, sl3)

    n1 = np.sin(z)*np.sin(z2)  # normal vector to plane 1
    n2 = np.cos(z)*np.sin(z2)
    # n3 = np.cos(z2)
    h1 = -sl2  # strike vector of plane 2
    h2 = sl1
    # note h3=0 always so we leave it out

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
    Find strike and dip of plane given normal vector.

    Adapted from Andy Michaels stridip.c

    Parameters
    ----------
    n : numpy array
        North coordinates for normal vector.
    e : numpy array
        East coordinate for normal vector.
    u : numpy array
        Up coordinate for normal vector.

    Returns
    -------
    strike : numpy array
        Strike of plane.
    dip : numpy array
        Dip of plane.

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
    Adapted from code, mij2d.f, created by Chen Ji.

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
    TDL.

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
            return None
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


def _testfn():
    """Test routine."""
    import sys
    from pygmi.seis.iodefs import ImportSeisan

    app = QtWidgets.QApplication(sys.argv)
    tmp = ImportSeisan()
    tmp.ifile = r"D:\workdata\PyGMI Test Data\Seismology\collect2.out"
    tmp.settings(True)

    outdata = tmp.outdata

    tmp = BeachBall()
    tmp.indata = outdata
    tmp.data_init()
    tmp.settings()


def _testfn2():
    """Test routine."""
    import matplotlib.pyplot as plt

    np1 = [20.77, 25, 0]
    # np1 = [150, 87, 1]
    xxx, yyy, xxx2, yyy2 = beachball(np1, 30, -30, 1, True)

    pvert1 = np.transpose([yyy, xxx])
    pvert0 = np.transpose([xxx2, yyy2])

    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')

    xmin = 29
    xmax = 31
    ymin = -31
    ymax = -29

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    patch = patches.Polygon(pvert1, edgecolor=(0.0, 0.0, 0.0))
    ax.add_patch(patch)

    plt.show()

    
if __name__ == "__main__":
    _testfn()
