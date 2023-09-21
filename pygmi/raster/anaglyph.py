# -----------------------------------------------------------------------------
# Name:        anaglyph.py (part of PyGMI)
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
"""Anaglyph routine."""

from PyQt5 import QtWidgets, QtCore
from scipy import ndimage
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import collections as mc
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

from pygmi.misc import frm, ContextModule


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    axes : matplotlib subplot
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        fig = Figure(layout='constrained')
        super().__init__(fig)

        self.axes = fig.add_subplot(111)
        self.scale = 7
        self.rotang = 10
        self.red1 = None
        self.blue1 = None
        self.red = None
        self.blue = None
        self.extent = None
        self.x = None
        self.z = None
        self.cnum = 10

        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def update_contours(self, data1, scale=7, rotang=10):
        """
        Update the contour plot.

        Parameters
        ----------
        data1 : PyGMI raster data.
            raster dataset to be used in contouring.
        scale : float, optional
            Scale. The default is 7.
        rotang : float, optional
            Rotation in degrees. The default is 10.

        Returns
        -------
        None.

        """
        self.scale = scale
        self.rotang = rotang
        self.extent = data1.extent

        self.z = data1.data*scale
        dxy = data1.xdim
        y, x = np.indices(self.z.shape)
        self.x = x*int(dxy) + data1.extent[0]
        y = data1.extent[-1] - y*int(dxy)

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        tmp = self.axes.contour(self.x, y, self.z, self.cnum)

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        # asegs = tmp.allsegs
        asegs = tmp.get_paths()
        lvls = tmp.levels
        xmin = self.x.min()
        xmax = self.x.max()
        a = np.deg2rad(rotang)
        m = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]

        lines = []
        for i, lvl in enumerate(asegs):
            cnt = np.split(lvl.vertices, np.where(lvl.codes == 1)[0][1:])
            for j in cnt:
                x1 = j[:, 0]
                y1 = j[:, 1]
                z1 = [lvls[i]]*x1.size
                x1 = x1-xmin
                t = np.transpose([x1, y1, z1])
                x1, y1, _ = np.transpose(t @ m)
                x1 = x1 + xmin
                lines.append(np.transpose([x1, y1]))

        lc = mc.LineCollection(lines, colors=[1, 0, 0, 0.5])
        self.axes.add_collection(lc)

        a = np.deg2rad(-rotang)
        m = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]

        lines = []
        for i, lvl in enumerate(asegs):
            cnt = np.split(lvl.vertices, np.where(lvl.codes == 1)[0][1:])
            for j in cnt:
                x1 = j[:, 0]
                y1 = j[:, 1]
                z1 = [lvls[i]]*x1.size
                x1 = x1-xmax
                t = np.transpose([x1, y1, z1])
                x1, y1, _ = np.transpose(t @ m)
                x1 = x1 + xmax
                lines.append(np.transpose([x1, y1]))

        lc = mc.LineCollection(lines, colors=[0, 1, 1, 0.5])
        self.axes.add_collection(lc)

        self.axes.autoscale()
        self.axes.set_aspect('equal')

        self.axes.set_xlabel('Eastings')
        self.axes.set_ylabel('Northings')

        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)
        self.axes.tick_params(axis='y', labelrotation=0)
        self.axes.tick_params(axis='x', labelrotation=90)

        self.figure.canvas.draw()

    def update_raster(self, data1, scale=7, rotang=10, atype='dubois',
                      cmap=colormaps['jet'], shade=False):
        """
        Update the raster plot.

        Parameters
        ----------
        data1 : PyGMI raster Data
            raster dataset to be used in contouring
        scale : float, optional
            Scale. The default is 7.
        rotang : float, optional
            Rotation in degrees. The default is 10.
        atype : str, optional
            Anaglyph type. The default is 'dubois'.
        cmap : matplotlib.colors.LinearSegmentedColormap, optional
            Matplotlib colormap. The default is jet.
        shade : bool, optional
            Option to choose sunshading. The default is False.

        Returns
        -------
        None.

        """
        self.scale = scale
        self.rotang = rotang

        self.extent = data1.extent

        self.z = data1.data*scale
        dxy = data1.xdim
        y, x = np.indices(self.z.shape)
        self.x = x*int(dxy)
        y = y*int(dxy)

        self.red1 = rot_and_clean(self.x, y, self.z, rotang, 'red')
        self.blue1 = rot_and_clean(self.x, y, self.z, rotang, 'blue')

        self.update_colors(shade, cmap, atype)

    def update_colors(self, doshade=False, cmap=colormaps['jet'],
                      atype='dubois'):
        """
        Update colors.

        Parameters
        ----------
        doshade : bool, optional
            Option to choose sunshading. The default is False.
        cmap : matplotlib.colors.LinearSegmentedColormap, optional
            Matplotlib colormap. The default is jet.
        atype : str, optional
            Anaglyph type. The default is 'dubois'.

        Returns
        -------
        None.

        """
        if not doshade:
            zmin = -2 * np.std(self.z)
            zmax = 2 * np.std(self.z)

            tmp = norm2(self.red1, zmin, zmax)
            tmp[tmp < 0] = 0.
            tmp[tmp > 1] = 1.
            self.red = cmap(tmp)
            self.red[:, :, 3] = np.logical_not(np.ma.getmaskarray(self.red1))

            tmp = norm2(self.blue1, zmin, zmax)
            tmp[tmp < 0] = 0.
            tmp[tmp > 1] = 1.
            self.blue = cmap(tmp)
            self.blue[:, :, 3] = np.logical_not(np.ma.getmaskarray(self.blue1))
        else:
            alpha = 0
            cell = 100
            azim = np.deg2rad(45)
            elev = np.deg2rad(45)
            self.red = sunshade(self.red1, azim=azim, elev=elev, alpha=alpha,
                                cell=cell, cmap=cmap)
            self.blue = sunshade(self.blue1, azim=azim, elev=elev, alpha=alpha,
                                 cell=cell, cmap=cmap)

        self.update_atype(atype)

    def update_atype(self, atype='dubois'):
        """
        Update anaglyph type.

        Parameters
        ----------
        atype : str, optional
            Anaglyph type. The default is 'dubois'.

        Returns
        -------
        None.

        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        adata = anaglyph(self.red, self.blue, atype)

        self.axes.imshow(adata, extent=self.extent, interpolation='nearest')

        self.axes.set_xlabel('Eastings')
        self.axes.set_ylabel('Northings')

        self.axes.tick_params(axis='y', labelrotation=0)
        self.axes.tick_params(axis='x', labelrotation=90)
        self.axes.xaxis.set_major_formatter(frm)
        self.axes.yaxis.set_major_formatter(frm)

        self.figure.canvas.draw()


class PlotAnaglyph(ContextModule):
    """Graph Window - The QDialog window which will contain our image."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Anaglyph (3D Image: Glasses Needed)')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Minimum)

        # Define Layouts
        hbl = QtWidgets.QHBoxLayout(self)  # self is where layout is assigned
        vbl_left = QtWidgets.QVBoxLayout()
        vbl_right = QtWidgets.QVBoxLayout()

        # Define Widgets
        self.mmc = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)
        self.cmb_1 = QtWidgets.QComboBox()
        self.cmb_2 = QtWidgets.QComboBox()
        self.cmb_cbar = QtWidgets.QComboBox()
        self.cb_shade = QtWidgets.QCheckBox('Sunshade:')
        self.rb_doimage = QtWidgets.QRadioButton('Full Image:')
        self.rb_docontour = QtWidgets.QRadioButton('Contour '
                                                '(Slider sets number):')
        self.slider_scale = QtWidgets.QSlider()
        self.slider_angle = QtWidgets.QSlider()
        self.slider_cnt = QtWidgets.QSlider()

        # Size policies
        self.cmb_1.setSizePolicy(sizepolicy)
        self.cmb_2.setSizePolicy(sizepolicy)
        self.cmb_cbar.setSizePolicy(sizepolicy)

        self.cb_shade.setSizePolicy(sizepolicy)
        self.rb_doimage.setSizePolicy(sizepolicy)
        self.rb_docontour.setSizePolicy(sizepolicy)
        self.slider_scale.setSizePolicy(sizepolicy)
        self.slider_angle.setSizePolicy(sizepolicy)
        self.slider_cnt.setSizePolicy(sizepolicy)

        # Configure Widgets
        self.cmb_2.addItem('Dubois (Red-Green)')
        self.cmb_2.addItem('Green-Magenta')
        self.cmb_2.addItem('Amber-Blue')
        self.cmb_2.addItem('True (Red-Green)')
        self.cmb_2.addItem('Gray (Red-Green)')
        self.cmb_2.addItem('Optimized (Red-Green)')

        maps = sorted(m for m in colormaps() if not
                      m.startswith(('spectral', 'Vega', 'jet')))

        self.cmb_cbar.addItem('jet')
        self.cmb_cbar.addItems(maps)
        self.rb_doimage.setChecked(True)
        self.slider_cnt.setMinimum(3)
        self.slider_cnt.setMaximum(30)
        self.slider_cnt.setOrientation(QtCore.Qt.Horizontal)
        self.slider_cnt.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_cnt.setValue(10)
        self.slider_scale.setMinimum(1)
        self.slider_scale.setMaximum(30)
        self.slider_scale.setOrientation(QtCore.Qt.Horizontal)
        self.slider_scale.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_scale.setValue(5)
        self.slider_angle.setMinimum(1)
        self.slider_angle.setMaximum(20)
        self.slider_angle.setOrientation(QtCore.Qt.Horizontal)
        self.slider_angle.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_angle.setValue(10)

        # Add widgets to layout
        vbl_left.addWidget(self.rb_doimage)
        vbl_left.addWidget(self.rb_docontour)
        vbl_left.addWidget(self.slider_cnt)
        vbl_left.addWidget(QtWidgets.QLabel('Bands:'))
        vbl_left.addWidget(self.cmb_1)
        vbl_left.addWidget(QtWidgets.QLabel('Type:'))
        vbl_left.addWidget(self.cmb_2)
        vbl_left.addWidget(QtWidgets.QLabel('Colour Bar:'))
        vbl_left.addWidget(self.cmb_cbar)
        vbl_left.addWidget(QtWidgets.QLabel('Scale (1-30):'))
        vbl_left.addWidget(self.slider_scale)
        vbl_left.addWidget(QtWidgets.QLabel('Image Angle (1-20):'))
        vbl_left.addWidget(self.slider_angle)
        vbl_left.addWidget(self.cb_shade)
        vbl_right.addWidget(self.mmc)
        vbl_right.addWidget(mpl_toolbar)
        hbl.addLayout(vbl_left)
        hbl.addLayout(vbl_right)

        self.setFocus()

        self.rb_doimage.released.connect(self.change_image)
        self.rb_docontour.released.connect(self.change_contours)
        self.cb_shade.stateChanged.connect(self.change_colors)
        self.cmb_1.currentIndexChanged.connect(self.change_all)
        self.cmb_2.currentIndexChanged.connect(self.change_atype)
        self.cmb_cbar.currentIndexChanged.connect(self.change_colors)
        self.slider_scale.sliderReleased.connect(self.change_all)
        self.slider_angle.sliderReleased.connect(self.change_all)
        self.slider_cnt.sliderReleased.connect(self.change_contours)

    def change_all(self):
        """
        Update from all combos.

        Returns
        -------
        None.

        """
        i = self.cmb_1.currentIndex()
        txt = str(self.cmb_cbar.currentText())
        cbar = colormaps[txt]
        shade = self.cb_shade.isChecked()
        scale = self.slider_scale.value()
        rotang = self.slider_angle.value()

        if self.rb_docontour.isChecked():
            data = self.indata['Raster']
            self.mmc.update_contours(data[i], scale=scale, rotang=rotang)
        else:
            data = self.indata['Raster']
            self.mmc.update_raster(data[i], atype=self.cmb_2.currentText(),
                                   cmap=cbar, shade=shade, scale=scale,
                                   rotang=rotang)

    def change_colors(self):
        """
        Update colour bar.

        Returns
        -------
        None.

        """
        txt = str(self.cmb_cbar.currentText())
        cbar = colormaps[txt]
        shade = self.cb_shade.isChecked()

        self.mmc.update_colors(atype=self.cmb_2.currentText(),
                               cmap=cbar, doshade=shade)

    def change_atype(self):
        """
        Update anaglyph type.

        Returns
        -------
        None.

        """
        self.mmc.update_atype(atype=self.cmb_2.currentText())

    def change_contours(self):
        """
        Update contours.

        Returns
        -------
        None.

        """
        self.rb_docontour.setChecked(True)

        i = self.cmb_1.currentIndex()
        scale = self.slider_scale.value()
        rotang = self.slider_angle.value()

        self.cb_shade.setDisabled(True)
        self.cmb_1.setDisabled(True)
        self.cmb_2.setDisabled(True)
        self.cmb_cbar.setDisabled(True)

        data = self.indata['Raster']

        self.mmc.cnum = self.slider_cnt.value()
        self.mmc.update_contours(data[i], scale=scale, rotang=rotang)

    def change_image(self):
        """
        Change Image, setting defaults.

        Returns
        -------
        None.

        """
        self.slider_scale.setValue(5)
        self.slider_angle.setValue(10)
        self.cb_shade.setEnabled(True)
        self.cmb_1.setEnabled(True)
        self.cmb_2.setEnabled(True)
        self.cmb_cbar.setEnabled(True)

        self.change_all()

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        if 'Raster' in self.indata:
            data = self.indata['Raster']
        else:
            return

        for i in data:
            self.cmb_1.addItem(i.dataid)
        self.change_all()


def sunshade(data, azim=-np.pi/4., elev=np.pi/4., alpha=1, cell=100,
             cmap=colormaps['terrain']):
    """
    Perform Sunshading on data.

    Parameters
    ----------
    data : numpy array
        input MxN data to be imaged.
    azim : float, optional
        Sun azimuth. The default is -np.pi/4..
    elev : float, optional
        Sun elevation. The default is np.pi/4..
    alpha : float, optional
        how much incident light is reflected (0 to 1). The default is 1.
    cell : float, optional
        between 1 and 100 - controls sunshade detail. The default is 100.
    cmap : matplotlib.colors.LinearSegmentedColormap, optional
        Matplotlib colormap.


    Returns
    -------
    colormap : numpy array
        Output colour mapped array (MxNx4).

    """
    mask = np.ma.getmaskarray(data)

    sunshader = currentshader(data, cell, elev, azim, alpha)
    snorm = norm2(sunshader)
    pnorm = np.uint8(norm2(histcomp(data))*255)

    colormap = cmap(pnorm)
    colormap[:, :, 0] = colormap[:, :, 0]*snorm
    colormap[:, :, 1] = colormap[:, :, 1]*snorm
    colormap[:, :, 2] = colormap[:, :, 2]*snorm
    colormap[:, :, 3] = np.logical_not(mask)

    return colormap


def norm2(dat, datmin=None, datmax=None):
    """
    Normalise vector.

    Parameters
    ----------
    dat : numpy array
        Vector to be normalised.
    datmin : float, optional
        Minimum dat value. The default is None.
    datmax : float, optional
        Maximum dat value. The default is None.

    Returns
    -------
    numpy array
        Normalised output data.

    """
    if datmin is None:
        datmin = np.min(dat)
    if datmax is None:
        datmax = np.max(dat)
    return (dat-datmin)/(datmax-datmin)


def currentshader(data, cell, theta, phi, alpha):
    """
    Blinn shader.

    Parameters
    ----------
    data : numpy array
        input MxN data to be imaged.
    cell : float
        between 1 and 100 - controls sunshade detail.
    theta : float
        sun elevation (also called g in code below).
    phi : float
        azimuth.
    alpha : float
        how much incident light is reflected.

    Returns
    -------
    R : numpy array
        Output data.

    """
    cdy = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    cdx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])

    dzdx = ndimage.convolve(data, cdx)  # Use convolve: matrix filtering
    dzdy = ndimage.convolve(data, cdy)  # 'valid' gets reduced array

    dzdx = dzdx/8.
    dzdy = dzdy/8.

    pinit = dzdx
    qinit = dzdy

    # Update cell
    p = pinit/cell
    q = qinit/cell
    sqrt_1p2q2 = np.sqrt(1+p**2+q**2)

    # Update angle
    cosg2 = np.cos(theta/2)
    p0 = -np.cos(phi)*np.tan(theta)
    q0 = -np.sin(phi)*np.tan(theta)
    sqrttmp = (1+np.sqrt(1+p0**2+q0**2))
    p1 = p0 / sqrttmp
    q1 = q0 / sqrttmp

    n = 2.0

    cosi = ((1+p0*p+q0*q)/(sqrt_1p2q2*np.sqrt(1+p0**2+q0**2)))
    coss = ((1+p1*p+q1*q)/(sqrt_1p2q2*np.sqrt(1+p1**2+q1**2)))
    Ps = coss**n
    R = ((1-alpha)+alpha*Ps)*cosi/cosg2
    return R


def histcomp(img, nbr_bins=256, perc=5.):
    """
    Histogram compaction.

    Parameters
    ----------
    img : numpy masked array
        Input data.
    nbr_bins : int, optional
        Number of bins. The default is 256.
    perc : float, optional
        Percentage to clip. The default is 5.

    Returns
    -------
    img2 : numpy array
        Output data.

    """
    tmp = img.compressed()

    imhist, bins = np.histogram(tmp, nbr_bins)

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / float(cdf[-1])  # normalize

    perc = perc/100.

    sindx = np.arange(nbr_bins)[cdf > perc][0]
    eindx = np.arange(nbr_bins)[cdf < (1-perc)][-1]+1
    svalue = bins[sindx]
    evalue = bins[eindx]

    scnt = perc*(nbr_bins-1)
    scnt = min(scnt, sindx)

    ecnt = perc*(nbr_bins-1)
    if ecnt > ((nbr_bins-1)-eindx):
        ecnt = (nbr_bins-1)-eindx

    img2 = np.empty_like(img, dtype=np.float32)
    np.copyto(img2, img)

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    return img2


def anaglyph(red, blue, atype='dubois'):
    """
    Colour Anaglyph.

    Parameters
    ----------
    red : numpy array
        Dataset for red channel.
    blue : numpy array
        Dataset for blue channel.
    atype : str, optional
        Anaglyph type. The default is 'dubois'.

    Returns
    -------
    rgb : numpy array
        Output dataset.

    """
    if 'Dubois' in atype:
        mat = np.array([[0.437, 0.449, 0.164, -0.011, -0.032, -0.007],
                        [-0.062, -0.062, -0.024, 0.377, 0.761, 0.009],
                        [-0.048, -0.050, -0.017, -0.026, -0.093, 1.234]])

        mat = np.array([[456, 500, 176, -43, -88, -2],
                        [-40, -38, -16, 378, 734, -18],
                        [-15, -21, -5, -72, -113, 1226]])/1000.
    elif 'Green-Magenta' in atype:
        mat = np.array([[-62, -158, -39, 529, 705, 24],
                        [284, 668, 143, -16, -15, -65],
                        [-15, -27, 21, 9, 75, 937]])/1000.
    elif 'Amber-Blue' in atype:
        mat = np.array([[1062, -205, 299, -16, -123, -17],
                        [-26, 908, 68, 6, 62, -17],
                        [-38, -173, 22, 94, 185, 911]])/1000.
    elif 'True' in atype:
        mat = np.array([[0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114]])
    elif 'Gray' in atype:
        mat = np.array([[0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114]])

    elif 'Optimized' in atype:
        mat = np.array([[0.0, 0.7, 0.3, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    newshape = (red.shape[0]*red.shape[1], 3)
    data1 = red[:, :, :3].copy()
    data2 = blue[:, :, :3].copy()
    data1.shape = newshape
    data2.shape = newshape
    mask = red[:, :, 3]
    data = np.transpose(np.hstack((data1, data2)))

    rgb = mat @ data
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1

    rgb = np.vstack((rgb, mask.flatten()))
    rgb = rgb.T
    rgb.shape = red.shape

#    red1 = RL*0.4154 + GL*0.4710 + BL*0.1669
#    red2 = (-RR*0.0109 - GR*0.0364 - BR*0.006)
#    green1 = -RL*0.0458 - GL*0.0484 - BL*0.0257
#    green2 = RR*0.3756 + GR*0.7333 + BR*0.0111
#    blue1 = -RL*0.0547 - GL*0.0615 + BL*0.0128
#    blue2 = (-RR*0.0651 - GR*0.1287 + BR*1.2971)

    return rgb


def rot_and_clean(x, y, z, rotang=5, rtype='red'):
    """
    Rotate and clean rotated data for 2d view.

    Parameters
    ----------
    x : numpy array
        X coordinates.
    y : numpy array
        Y coordinates.
    z : numpy array
        Z coordinates (or data values).
    rotang : float, optional
        Rotation angle. The default is 5.
    rtype : str, optional
        Rotation type. The default is 'red'.

    Returns
    -------
    zmap : numpy array
        Output data.

    """
    if rtype == 'red':
        rotang = -1. * abs(rotang)
    else:
        rotang = -1. * abs(rotang)
        z = z[:, ::-1]

    a = np.deg2rad(rotang)
    m = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]

    x = x-x.min()
    y = y-y.min()
    z = z-np.mean(z)

    z.set_fill_value(0)
    z = z.filled()

    t = np.transpose([x, y, z])
    x1, _, _ = np.transpose(t @ m)

    zmap = np.zeros(x.shape)

# Note that when you rotate about the y-axis, a peak will be rotated
# this means that you can have more that one solution for an x coordinate
# np.interp always takes the first solution, which co-incidentally happens
# to be the one we want in this case.

    zi = np.ma.filled(z)
    for j, xi in enumerate(x1):
        zmap[j] = np.interp(x[0], xi, zi[j])

    if rtype != 'red':
        zmap = zmap[:, ::-1]

    zmap = np.ma.masked_equal(zmap, 0)

    return zmap
