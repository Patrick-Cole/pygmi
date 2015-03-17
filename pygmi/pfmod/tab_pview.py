# -----------------------------------------------------------------------------
# Name:        tab_pview.py (part of PyGMI)
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
""" Profile Display Tab Routines """

from PyQt4 import QtGui, QtCore
import numpy as np
import scipy.ndimage as ndimage
from . import misc
import os

# from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as \
    NavigationToolbar


class ProfileDisplay(object):
    """ Widget class to call the main interface """
    def __init__(self, parent):
        self.parent = parent
        self.lmod1 = parent.lmod1
        self.lmod2 = parent.lmod2
        self.showtext = parent.showtext
        self.pbar = self.parent.pbar_sub
        self.xnodes = self.lmod1.custprofx
        self.ynodes = self.lmod1.custprofy
        self.curprof = 0
        self.pcntmax = len(self.xnodes)-1
#        self.xnodes = {0: self.lmod1.xrange}
#        self.ynodes = {0: [self.lmod1.yrange[0], self.lmod1.yrange[0]]}

        mainwindow = QtGui.QWidget()

        self.mmc = MyMplCanvas(self, self.lmod1)
        self.mpl_toolbar = NavigationToolbar(self.mmc, mainwindow)

        self.userint = mainwindow
        self.toolboxpage1 = QtGui.QWidget()
        self.groupbox = QtGui.QGroupBox(self.toolboxpage1)

        self.gridlayout = QtGui.QGridLayout(mainwindow)
        self.sb_profnum2 = QtGui.QSpinBox(mainwindow)
        self.hslider_profile2 = QtGui.QSlider(mainwindow)
        self.combo_profpic = QtGui.QComboBox(mainwindow)
        self.hs_ppic_opacity = QtGui.QSlider(mainwindow)
        self.toolbox = QtGui.QToolBox(mainwindow)

        self.verticallayout = QtGui.QVBoxLayout(self.groupbox)
        self.rb_axis_datamax = QtGui.QRadioButton(self.groupbox)
        self.rb_axis_profmax = QtGui.QRadioButton(self.groupbox)
        self.rb_axis_calcmax = QtGui.QRadioButton(self.groupbox)

        self.groupbox3 = QtGui.QGroupBox(self.toolboxpage1)
        self.sb_profile_linethick = QtGui.QSpinBox(self.toolboxpage1)
        self.gridlayout_20 = QtGui.QGridLayout(self.toolboxpage1)
        self.lw_prof_defs = QtGui.QListWidget(self.toolboxpage1)

        self.pb_add_prof = QtGui.QPushButton(self.toolboxpage1)
        self.pb_export_csv = QtGui.QPushButton(self.toolboxpage1)

        self.gridlayout4 = QtGui.QGridLayout(self.groupbox3)
        self.rb_magnetic = QtGui.QRadioButton(self.groupbox3)
        self.rb_gravity = QtGui.QRadioButton(self.groupbox3)

        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.sb_profnum2.setWrapping(True)
        self.sb_profnum2.setMaximum(999999999)
        self.gridlayout.addWidget(self.sb_profnum2, 0, 1, 1, 1)
        sizepolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Fixed)
        self.hslider_profile2.setSizePolicy(sizepolicy)
        self.hslider_profile2.setOrientation(QtCore.Qt.Horizontal)
        self.gridlayout.addWidget(self.hslider_profile2, 1, 1, 1, 1)
        self.gridlayout.addWidget(self.combo_profpic, 2, 1, 1, 1)
        self.hs_ppic_opacity.setSizePolicy(sizepolicy)
        self.hs_ppic_opacity.setMaximum(255)
        self.hs_ppic_opacity.setProperty("value", 255)
        self.hs_ppic_opacity.setOrientation(QtCore.Qt.Horizontal)
        self.hs_ppic_opacity.setTickPosition(QtGui.QSlider.TicksAbove)
        self.gridlayout.addWidget(self.hs_ppic_opacity, 3, 1, 1, 1)
        self.toolbox.setMaximumSize(QtCore.QSize(220, 16777215))
        self.gridlayout4.addWidget(self.rb_magnetic, 1, 0, 1, 1)
        self.gridlayout4.addWidget(self.rb_gravity, 4, 0, 1, 1)
        self.verticallayout.addWidget(self.rb_axis_datamax)
        self.verticallayout.addWidget(self.rb_axis_profmax)
        self.verticallayout.addWidget(self.rb_axis_calcmax)
        self.sb_profile_linethick.setMinimum(1)
        self.sb_profile_linethick.setMaximum(1000)

        self.gridlayout_20.addWidget(self.lw_prof_defs, 3, 0, 1, 1)
        self.gridlayout_20.addWidget(self.sb_profile_linethick, 4, 0, 1, 1)
        self.gridlayout_20.addWidget(self.groupbox3, 5, 0, 1, 1)
        self.gridlayout_20.addWidget(self.groupbox, 6, 0, 1, 1)
        self.gridlayout_20.addWidget(self.pb_add_prof, 7, 0, 1, 1)
        self.gridlayout_20.addWidget(self.pb_export_csv, 9, 0, 1, 1)
        self.toolbox.addItem(self.toolboxpage1, "")
        self.gridlayout.addWidget(self.toolbox, 5, 1, 1, 1)
        self.gridlayout.addWidget(self.mpl_toolbar, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.mmc, 1, 0, 5, 1)

        self.sb_profnum2.setPrefix("Custom Profile: ")
        self.groupbox3.setTitle("Profile Data")
        self.rb_magnetic.setText("Magnetic Data")
        self.rb_gravity.setText("Gravity Data")
        self.groupbox.setTitle("Profile Y-Axis Scale")
        self.rb_axis_datamax.setText("Scale to dataset maximum")
        self.rb_axis_profmax.setText("Scale to profile maximum")
        self.rb_axis_calcmax.setText("Scale to calculated maximum")
        self.sb_profile_linethick.setPrefix("Line Thickness: ")
        self.toolbox.setItemText(self.toolbox.indexOf(self.toolboxpage1),
                                 "General")
        self.pb_add_prof.setText("Add Custom Profile")
        self.pb_export_csv.setText("Export Profile")

    # Buttons etc
        self.rb_axis_datamax.setChecked(True)
        self.rb_gravity.setChecked(True)
        self.sb_profile_linethick.valueChanged.connect(self.width)
        self.lw_prof_defs.currentItemChanged.connect(self.change_defs)
        self.pb_add_prof.clicked.connect(self.addprof)
        self.rb_gravity.clicked.connect(self.rb_mag_grav)
        self.rb_magnetic.clicked.connect(self.rb_mag_grav)
        self.hslider_profile2.valueChanged.connect(self.hprofnum)
        self.sb_profnum2.valueChanged.connect(self.sprofnum)
        self.rb_axis_calcmax.clicked.connect(self.rb_plot_scale)
        self.rb_axis_profmax.clicked.connect(self.rb_plot_scale)
        self.rb_axis_datamax.clicked.connect(self.rb_plot_scale)
        self.pb_export_csv.clicked.connect(self.export_csv)
        self.hs_ppic_opacity.sliderMoved.connect(self.profpic_hs)
        self.combo_profpic.currentIndexChanged.connect(self.profpic_hs)

    def addprof(self):
        """ add another profile """
        self.update_model()
        (tx0, okay) = QtGui.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter first x coordinate', self.lmod1.xrange[0])
        if not okay:
            return
        (ty0, okay) = QtGui.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter first y coordinate', self.lmod1.yrange[0])
        if not okay:
            return
        (tx1, okay) = QtGui.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter last x coordinate', self.lmod1.xrange[-1])
        if not okay:
            return
        (ty1, okay) = QtGui.QInputDialog.getDouble(
            self.parent, 'Add Custom Profile',
            'Please enter last y coordinate', self.lmod1.yrange[-1])
        if not okay:
            return

        self.pcntmax += 1
        self.xnodes[self.pcntmax] = [float(tx0), float(tx1)]
        self.ynodes[self.pcntmax] = [float(ty0), float(ty1)]

        self.hslider_profile2.valueChanged.disconnect()
        self.combo_profpic.currentIndexChanged.disconnect()
        self.sb_profnum2.valueChanged.disconnect()

        self.hslider_profile2.setMaximum(self.pcntmax)
        self.sb_profnum2.setMaximum(self.pcntmax)

        self.sb_profnum2.valueChanged.connect(self.sprofnum)
        self.hslider_profile2.valueChanged.connect(self.hprofnum)
        self.combo_profpic.currentIndexChanged.connect(self.profpic_hs)

    def change_defs(self):
        """ List box in profile tab for definitions """

        i = self.lw_prof_defs.currentRow()
        if i == -1:
            misc.update_lith_lw(self.lmod1, self.lw_prof_defs)
            i = 0
        itxt = str(self.lw_prof_defs.item(i).text())

        if itxt not in self.lmod1.lith_list.keys():
            return

        lith = self.lmod1.lith_list[itxt]
        self.mmc.curmodel = lith.lith_index

    def change_model(self, slide=False):
        """ Change Model """

        bottom = self.lmod1.zrange[0]
        top = self.lmod1.zrange[1]

        data = self.lmod1.griddata['Calculated Gravity']
        xxx, yyy, right = self.cp_init(data)

        tmp = np.transpose([xxx, yyy]).astype(int)
        self.mmc.crd = tmp

        x = np.array(xxx).astype(int)
        y = np.array(yyy).astype(int)

        gtmp = []
        for i in range(self.lmod1.numz):
            gtmp.append(self.lmod1.lith_index[x, y, i])

        gtmp = np.array(gtmp[::-1])
    # First we plot the model stuff
        left = 0

        extent = (left, right, bottom, top)

        ctxt = str(self.combo_profpic.currentText())
        if len(self.lmod1.profpics) > 0 and ctxt != u'':
            gtmpl = self.lmod1.profpics[ctxt]
            opac = self.hs_ppic_opacity.value()/self.hs_ppic_opacity.maximum()
        else:
            gtmpl = None
            opac = 1.0

        if slide is True:
            self.mmc.slide_grid(gtmp, gtmpl, opac)
        else:
            self.mmc.init_grid(gtmp, extent, gtmpl, opac)

#        alt = self.lmod1.zrange[1]-self.lmod1.curlayer*self.lmod1.d_z

#        self.mmc.update_line(self.lmod1.xrange, [alt, alt])

    def cp_init(self, data):
        """ Initializes stuff for custom profile """
        x_0, x_1 = self.xnodes[self.curprof]
        y_0, y_1 = self.ynodes[self.curprof]

        bly = data.tly-data.ydim*data.rows
        x_0 = (x_0-data.tlx)/data.xdim
        x_1 = (x_1-data.tlx)/data.xdim
        y_0 = (y_0-bly)/data.ydim
        y_1 = (y_1-bly)/data.ydim
        rcell = int(np.sqrt((x_1-x_0)**2+(y_1-y_0)**2))
        rdist = np.sqrt((data.xdim*(x_1-x_0))**2+(data.ydim*(y_1-y_0))**2)

        xxx = np.linspace(x_0, x_1, rcell, False)
        yyy = np.linspace(y_0, y_1, rcell, False)

        return xxx, yyy, rdist

    def export_csv(self):
        """ Export Profile to csv """
        self.parent.pbars.resetall()
        filename = QtGui.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', 'Comma separated values (*.csv)')
        if filename == '':
            return
        os.chdir(filename.rpartition('/')[0])

        maggrid = self.lmod1.griddata['Calculated Magnetics'].data
        grvgrid = self.lmod1.griddata['Calculated Gravity'].data
        curprof = self.curprof

        cmag = maggrid[-curprof-1]
        cgrv = grvgrid[-curprof-1]
        xrng = (np.arange(cmag.shape[0])*self.lmod1.dxy +
                self.lmod1.xrange[0]+self.lmod1.dxy/2.)
        yrng = np.zeros_like(xrng)+self.lmod1.dxy/2.

        newdata = np.transpose([xrng, yrng, cgrv, cmag])

        fno = open(filename, 'wb')
        fno.write(b'"x","y","Gravity","Magnetics"\n')
        np.savetxt(fno, newdata, delimiter=',')
        fno.close()
        self.parent.pbars.incr()
        self.showtext('Profile save complete')

    def hprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.sb_profnum2.setValue(self.hslider_profile2.sliderPosition())
        self.profnum()

    def profpic_hs(self):
        """ Horizontal slider to change the profile """
        self.update_model()
        self.change_model(slide=True)
        self.update_plot(slide=True)

    def profnum(self):
        """ Routine to change a profile from spinbox"""
        self.update_model()

        self.curprof = self.sb_profnum2.value()
        self.change_model(slide=False)
        self.update_plot(slide=False)  # was True

    def sprofnum(self):
        """ Routine to change a profile from spinbox"""
        self.hslider_profile2.setValue(self.sb_profnum2.value())
        self.profnum()

    def rb_plot_scale(self):
        """ plot scale """
        self.change_model()
        self.update_plot()
        self.mpl_toolbar.update()

    def rb_mag_grav(self):
        """ Used to change the magnetic and gravity radiobutton """
        self.update_model()
        self.change_model()
        self.update_plot()

    def tab_activate(self):
        """ Runs when the tab is activated """
        self.lmod1 = self.parent.lmod1
        self.mmc.lmod = self.lmod1

        self.xnodes = self.lmod1.custprofx
        self.ynodes = self.lmod1.custprofy
        self.pcntmax = len(self.xnodes)-1

#        self.xnodes[0] = self.lmod1.xrange
#        self.ynodes[0] = [self.lmod1.yrange[0], self.lmod1.yrange[0]]

        misc.update_lith_lw(self.lmod1, self.lw_prof_defs)

        self.hslider_profile2.valueChanged.disconnect()
        self.combo_profpic.currentIndexChanged.disconnect()
        self.sb_profnum2.valueChanged.disconnect()

        self.hslider_profile2.setMinimum(0)
        self.hslider_profile2.setMaximum(self.pcntmax)
        self.sb_profnum2.setMaximum(self.pcntmax)

        if len(self.lmod1.profpics) > 0:
            self.combo_profpic.clear()
            self.combo_profpic.addItems(list(self.lmod1.profpics.keys()))
            self.combo_profpic.setCurrentIndex(0)

        self.change_model()  # needs to happen before profnum set value
        self.sb_profnum2.setValue(self.curprof)
        self.update_plot()
        self.sb_profnum2.valueChanged.connect(self.sprofnum)
        self.hslider_profile2.valueChanged.connect(self.hprofnum)
        self.combo_profpic.currentIndexChanged.connect(self.profpic_hs)

    def update_model(self):
        """ Update model itself """
#        tmp = self.lmod1.lith_index[:, self.curprof, ::-1]
#        if tmp.shape != self.mmc.mdata.T.shape:
#            return
#        self.lmod1.lith_index[:, self.curprof, ::-1] = (
#            self.mmc.mdata.T.copy())

        data = self.lmod1.griddata['Calculated Gravity']
        xxx, yyy = self.cp_init(data)[:2]

        gtmp = self.mmc.mdata[::-1].T.copy()
        rows, cols = gtmp.shape

        for j in range(cols):
            for i in range(rows):
                self.lmod1.lith_index[int(xxx[i]), int(yyy[i]), j] = gtmp[i, j]

    def update_plot(self, slide=False):
        """ Update the profile on the model view """

# Display the calculated profile
        data = None
        if self.rb_magnetic.isChecked():
            if 'Calculated Magnetics' in self.lmod1.griddata.keys():
                data = self.lmod1.griddata['Calculated Magnetics']
            self.mmc.ptitle = 'Magnetic Intensity: '
            self.mmc.punit = 'nT'
            regtmp = 0.0
        else:
            if 'Calculated Gravity' in self.lmod1.griddata.keys():
                data = self.lmod1.griddata['Calculated Gravity']
            self.mmc.ptitle = 'Gravity: '
            self.mmc.punit = 'mGal'
            regtmp = self.lmod1.gregional

        x_0, x_1 = self.xnodes[self.curprof]
        y_0, y_1 = self.ynodes[self.curprof]
        self.mmc.ptitle += str((x_0, y_0)) + ' to ' + str((x_1, y_1))

        if data is not None:
            xxx, yyy, rdist = self.cp_init(data)

            self.mmc.xlabel = "Eastings (m)"
            tmprng = np.linspace(0, rdist, len(xxx), False)
            tmpprof = ndimage.map_coordinates(data.data[::-1], [yyy, xxx],
                                              order=1, cval=np.nan)
            tmprng = tmprng[np.logical_not(np.isnan(tmpprof))]
            tmpprof = tmpprof[np.logical_not(np.isnan(tmpprof))]+regtmp
            extent = [0, rdist]

        extent = list(extent)+[tmpprof.min(), tmpprof.max()]

#        if self.rb_axis_datamax.isChecked():
#            extent = list(extent)+[data.data.min(), data.data.max()]
#        else:
#            extent = list(extent)+[tmpprof.min(), tmpprof.max()]

# Load in observed data - if there is any
        data2 = None
        tmprng2 = None
        tmpprof2 = None
        if ('Magnetic Dataset' in self.lmod1.griddata.keys() and
                self.rb_magnetic.isChecked()):
            data2 = self.lmod1.griddata['Magnetic Dataset']
        elif ('Gravity Dataset' in self.lmod1.griddata.keys() and
              self.rb_gravity.isChecked()):
            data2 = self.lmod1.griddata['Gravity Dataset']

        if data2 is not None:
            xxx, yyy, rdist = self.cp_init(data2)

            tmprng2 = np.linspace(0, rdist, len(xxx), False)
            tmpprof2 = ndimage.map_coordinates(data2.data[::-1], [yyy, xxx],
                                               order=1, cval=np.nan)

            tmprng2 = tmprng2[np.logical_not(np.isnan(tmpprof2))]
            tmpprof2 = tmpprof2[np.logical_not(np.isnan(tmpprof2))]

            if self.rb_axis_datamax.isChecked() or len(tmpprof2) == 0:
                extent[2:] = [data2.data.min(), data2.data.max()]
            elif self.rb_axis_profmax.isChecked():
                extent[2:] = [tmpprof2.min(), tmpprof2.max()]

        if slide is True:
            self.mmc.slide_plot(tmprng, tmpprof, extent, tmprng2, tmpprof2)
        else:
            self.mmc.init_plot(tmprng, tmpprof, extent, tmprng2, tmpprof2)

    def width(self, width):
        """ Sets the width of the edits on the profile view """

        self.mmc.width = width


class MyMplCanvas(FigureCanvas):
    """This is a QWidget"""
    def __init__(self, parent, lmod):
        # fig = Figure()
        fig = plt.figure()
        FigureCanvas.__init__(self, fig)

        self.parent = parent
        self.lmod = lmod
        self.cbar = plt.cm.jet
        self.curmodel = 0
        self.width = 1
        self.xold = None
        self.yold = None
        self.press = False
        self.newline = False
        self.mdata = np.zeros([10, 100])
        self.ptitle = ''
        self.punit = ''
        self.xlabel = 'Eastings (m)'
        self.plotisinit = False
        self.crd = None

# Events
        self.figure.canvas.mpl_connect('motion_notify_event', self.move)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self.button_release)

# Initial Images
        self.paxes = fig.add_subplot(211)
        self.paxes.yaxis.set_label_text("mGal")
        self.paxes.ticklabel_format(useOffset=False)

        self.cal = self.paxes.plot([], [])
        self.obs = plt.plot([], [], 'o')

        self.axes = fig.add_subplot(212)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.axes.yaxis.set_label_text("Altitude (m)")

        tmp = self.cbar(self.mdata)
        tmp[:, :, 3] = 0

        self.ims2 = plt.imshow(tmp.copy(), interpolation='nearest',
                               aspect='auto')
        self.ims = plt.imshow(tmp.copy(), interpolation='nearest',
                              aspect='auto')
        self.figure.canvas.draw()

        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.pbbox = self.figure.canvas.copy_from_bbox(self.paxes.bbox)
        self.prf = self.axes.plot([0, 0])
        self.figure.canvas.draw()
        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)

    def button_press(self, event):
        """ Button press """
        nmode = self.axes.get_navigate_mode()
        if event.button == 1 and nmode is None:
            self.press = True
            self.newline = True

    def button_release(self, event):
        """ Button press """
        nmode = self.axes.get_navigate_mode()
        if event.button == 1:
            self.press = False
            if nmode == 'ZOOM':
                extent = self.axes.get_xbound()
                self.paxes.set_xbound(extent[0], extent[1])
                self.figure.canvas.draw()
            else:
                self.parent.update_model()

    def move(self, event):
        """ Mouse is moving """
        if event.inaxes == self.axes and self.press is True:
            row = int((event.ydata - self.lmod.zrange[0])/self.lmod.d_z)
            col = int((event.xdata)/self.lmod.dxy)
#            col = int((event.xdata - self.lmod.xrange[0])/self.lmod.dxy)

            aaa = self.axes.get_xbound()[-1]
            bbb = self.mdata.shape[1]
            ccc = aaa/bbb
            col = int((event.xdata)/ccc)

            xdata = col
            ydata = row

            if self.newline is True:
                self.newline = False
                self.set_mdata(xdata, ydata)
            elif xdata != self.xold:
                mmm = float(ydata-self.yold)/(xdata-self.xold)
                ccc = ydata - mmm * xdata
                x_1 = min([self.xold, xdata])
                x_2 = max([self.xold, xdata])
                for i in range(x_1+1, x_2+1):
                    jold = int(mmm*(i-1)+ccc)
                    jnew = int(mmm*i+ccc)
                    if jold > jnew:
                        jold, jnew = jnew, jold
                    for j in range(jold, jnew+1):
                        self.set_mdata(i, j)

            elif ydata != self.yold:
                y_1 = min([self.yold, ydata])
                y_2 = max([self.yold, ydata])
                for j in range(y_1, y_2+1):
                    self.set_mdata(xdata, j)

            self.xold = xdata
            self.yold = ydata

            self.slide_grid(self.mdata)

    def set_mdata(self, xdata, ydata):
        """ Routine to 'draw' the line on mdata """

        gheight = self.mdata.shape[0]
        gwidth = self.mdata.shape[1]

        width = self.width-1  # 'pen' width
        xstart = xdata-width-1
        xend = xdata+width
        ystart = ydata-width-1
        yend = ydata+width
        if xstart < 0:
            xstart = 0
        if xend > gwidth:
            xend = gwidth
        if ystart < 0:
            ystart = 0
        if yend > gheight:
            yend = gheight

#        for i in range(xstart, xend):
        i = xstart
        while i < xend:
            tmp = (self.crd == self.crd[i])
            tmp = np.logical_and(tmp[:, 0], tmp[:, 1])
            tmp = tmp.nonzero()[0]
            if tmp[-1] >= xend:
                xend = tmp[-1]+1
            i += 1

        if xstart < xend and ystart < yend:
            mtmp = self.mdata[ystart:yend, xstart:xend]
            mtmp[mtmp != -1] = self.curmodel

    def luttodat(self, dat):
        """ lut to dat grid """
        mlut = self.lmod.mlut
        tmp = np.zeros([dat.shape[0], dat.shape[1], 4])

        for i in np.unique(dat):
            if i == -1:
                ctmp = [0, 0, 0, 0]
            else:
                ctmp = np.array(mlut[i]+[255])/255.

            tmp[dat[::-1] == i] = ctmp

        return tmp

    def init_grid(self, dat, extent, dat2, opac):
        """ Updates the single color map """
        # Note that because we clear the current axes, we must put the objects
        # back into it are the graph will go blank on a draw(). This is the
        # reason for the ims and prf commands below.

        self.paxes.set_xbound(extent[0], extent[1])
        self.ims.set_visible(False)
        self.axes.set_xlim(extent[0], extent[1])
        self.axes.set_ylim(extent[2], extent[3])

        if dat2 is not None:
            self.ims2.set_data(dat2.data)
            self.ims2.set_extent(self.dat_extent(dat2))
            self.ims2.set_clim(dat2.data.min(), dat2.data.max())

        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.bbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)

        self.ims.set_visible(True)
        self.ims.set_extent(extent)
        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)
        self.ims.set_alpha(opac)

        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()

        self.mdata = dat

    def slide_grid(self, dat, dat2=None, opac=1.0):
        """ Slider """
        self.mdata = dat
        tmp = self.luttodat(dat)
        self.ims.set_data(tmp)
        self.ims.set_alpha(opac)

        self.figure.canvas.restore_region(self.bbox)
        self.axes.draw_artist(self.ims)
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

        self.lbbox = self.figure.canvas.copy_from_bbox(self.axes.bbox)
        self.axes.draw_artist(self.prf[0])
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def update_line(self, xrng, yrng):
        """ Updates the line position """
        self.prf[0].set_data([xrng, yrng])
        self.figure.canvas.restore_region(self.lbbox)
        self.axes.draw_artist(self.prf[0])
#        self.figure.canvas.blit(self.axes.bbox)
        self.figure.canvas.update()

    def dat_extent(self, dat):
        """ Gets the extend of the dat variable """
        left = dat.tlx
        top = dat.tly
        right = left + dat.cols*dat.xdim
        bottom = top - dat.rows*dat.ydim
        return (left, right, bottom, top)

# This section is just for the profile line plot

    def extentchk(self, extent):
        """ Checks extent """
        dmin = extent[2]
        dmax = extent[3]
        if dmin == dmax:
            dmax = dmin+1
        return dmin, dmax

    def init_plot(self, xdat, dat, extent, xdat2, dat2):
        """ Updates the single color map """
        self.paxes.autoscale(False)
        dmin, dmax = self.extentchk(extent)
        self.paxes.cla()
        self.paxes.ticklabel_format(useOffset=False)
        self.paxes.set_title(self.ptitle)
        self.axes.xaxis.set_label_text(self.xlabel)
        self.paxes.yaxis.set_label_text(self.punit)
        self.paxes.set_ylim(dmin, dmax)
        self.paxes.set_xlim(extent[0], extent[1])
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.pbbox = self.figure.canvas.copy_from_bbox(self.paxes.bbox)

        self.paxes.set_autoscalex_on(False)
        self.cal = self.paxes.plot(xdat, dat)
        if xdat2 is not None:
            self.obs = self.paxes.plot(xdat2, dat2, 'o')
        else:
            self.obs = self.paxes.plot([], [], 'o')
        self.figure.canvas.draw()
        QtGui.QApplication.processEvents()
        self.plotisinit = True

    def slide_plot(self, xdat, dat, extent, xdat2, dat2):
        """ Slider """
        dmin, dmax = self.extentchk(extent)

        self.figure.canvas.restore_region(self.pbbox)
        self.cal[0].set_data([xdat, dat])
        if xdat2 is not None:
            self.obs[0].set_data([xdat2, dat2])
        else:
            self.obs[0].set_data([[], []])
        self.paxes.set_ylim(dmin, dmax)
        self.paxes.set_xlim(extent[0], extent[1])

        self.paxes.draw_artist(self.cal[0])
        if xdat2 is not None:
            self.paxes.draw_artist(self.obs[0])
#        self.figure.canvas.blit(self.paxes.bbox)
        self.figure.canvas.update()

        QtGui.QApplication.processEvents()
