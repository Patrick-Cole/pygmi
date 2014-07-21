# -----------------------------------------------------------------------------
# Name:        pfmod.py (part of PyGMI)
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
""" This is the main program for the modelling package """

# pylint: disable=E1101
import sys
from PySide import QtGui, QtCore

# Other dependancies
import pygmi.pfmod.misc as misc
import pygmi.pfmod.tab_ddisp as tab_ddisp
import pygmi.pfmod.tab_layer as tab_layer
import pygmi.pfmod.tab_prof as tab_prof
import pygmi.pfmod.tab_pview as tab_pview
import pygmi.pfmod.tab_param as tab_param
import pygmi.pfmod.tab_mext as tab_mext
import pygmi.pfmod.grvmag3d as grvmag3d
from .datatypes import LithModel


class MainWidget(QtGui.QMainWindow):
    """ Widget class to call the main interface """
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.indata = {}
        self.inraster = {}
        self.outdata = {}
        self.parent = parent

# General
        self.txtmsg = ''
        self.modelfilename = r'./tmp'

        self.centralwidget = QtGui.QWidget(self)
        self.toolbar = QtGui.QToolBar(self)
        self.statusbar = QtGui.QStatusBar(self)
        self.verticallayout = QtGui.QVBoxLayout(self.centralwidget)
        self.tabwidget = QtGui.QTabWidget(self.centralwidget)
        self.pbar_sub = QtGui.QProgressBar(self.centralwidget)
        self.pbar_main = QtGui.QProgressBar(self.centralwidget)
        self.textbrowser = QtGui.QTextBrowser(self.centralwidget)

        self.setupui()

        self.lmod1 = LithModel()  # actual model
        self.lmod2 = LithModel()  # regional model
        self.tabwidget.setCurrentIndex(0)
        self.oldtab = self.tabwidget.tabText(0)
        self.pbars = misc.ProgressBar(self.pbar_sub, self.pbar_main)
#        self.outdata['Model3D'] = self.lmod1
        self.outdata['Raster'] = list(self.lmod1.griddata.values())

# Model Extent Tab
        self.mext = tab_mext.MextDisplay(self)
        self.tabwidget.addTab(self.mext.userint, "Model Extent Parameters")

# Geophysical Parameters Tab
        self.param = tab_param.ParamDisplay(self)
        self.tabwidget.addTab(self.param.userint, "Geophysical Parameters")

# Data Display Tab
        self.ddisp = tab_ddisp.DataDisplay(self)
        self.tabwidget.addTab(self.ddisp.userint, "Data Display")

# Layer Editor Tab
        self.layer = tab_layer.LayerDisplay(self)
        self.tabwidget.addTab(self.layer.userint, "Layer Editor")

# Profile Editor Tab
        self.profile = tab_prof.ProfileDisplay(self)
        self.tabwidget.addTab(self.profile.userint, "Profile Editor")

# Profile Viewer Tab
        self.pview = tab_pview.ProfileDisplay(self)
        self.tabwidget.addTab(self.pview.userint, "Custom Profile Viewer")

# Gravity and magnetic modelling routines
        self.grvmag = grvmag3d.GravMag(self)

        self.tabwidget.currentChanged.connect(self.tab_change)

    def setupui(self):
        """ Setup for the GUI """
        self.resize(1024, 768)
        self.setCentralWidget(self.centralwidget)
#        self.setMenuBar(self.menubar)
        self.setStatusBar(self.statusbar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

        self.verticallayout.addWidget(self.tabwidget)
        self.textbrowser.setFrameShape(QtGui.QFrame.StyledPanel)
        self.verticallayout.addWidget(self.textbrowser)
        self.pbar_sub.setTextVisible(False)
        self.verticallayout.addWidget(self.pbar_sub)
        self.pbar_main.setTextVisible(False)
        self.verticallayout.addWidget(self.pbar_main)
        self.toolbar.setMovable(True)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.setWindowTitle("Potential Field Modelling")
#        self.setWindowIcon(QtGui.QIcon('pygmi.png'))

    def settings(self):
        """ Settings """
        if 'Raster' not in self.indata:
            self.indata['Raster'] = list(self.lmod1.griddata.values())

        for i in self.indata['Raster']:
            self.inraster[i.bandid] = i
        if 'Model3D' in self.indata.keys():
            self.lmod1 = self.indata['Model3D']
            self.lmod1.init_calc_grids()

        self.outdata['Model3D'] = self.lmod1
        self.mext.update_combos()
        self.mext.tab_activate()
        self.outdata['Raster'] = list(self.lmod1.griddata.values())

        if 'ProfPic' in self.indata:
            icnt = 0
            for i in self.indata['ProfPic']:
                icnt += 1
                self.lmod1.profpics['Profile: '+str(icnt)] = i

        self.show()

        return True

    def showtext(self, txt, replacelast=False):
        """ Show text on the text panel of the main user interface"""
        if replacelast is True:
            self.txtmsg = self.txtmsg[:self.txtmsg.rfind('\n')]
            self.txtmsg = self.txtmsg[:self.txtmsg.rfind('\n')]
            self.txtmsg += '\n'
        self.txtmsg += txt + '\n'
        self.textbrowser.setPlainText(self.txtmsg)
        tmp = self.textbrowser.verticalScrollBar()
        tmp.setValue(tmp.maximumHeight())
        self.repaint()

    def tab_change(self, index=None):
        """ This gets called any time we change a tab, and activates the
        routines behind the new tab """

        index = self.tabwidget.currentIndex()
        self.profile.change_defs()
        self.layer.change_defs()

        if self.oldtab == 'Layer Editor':
            self.layer.update_model()

        if self.oldtab == 'Profile Editor':
            self.profile.update_model()

        if self.oldtab == 'Custom Profile Viewer':
            self.pview.update_model()

        if self.tabwidget.tabText(index) == 'Geophysical Parameters':
            self.param.tab_activate()

        if self.tabwidget.tabText(index) == 'Model Extent Parameters':
            self.mext.tab_activate()

        if self.tabwidget.tabText(index) == 'Profile Editor':
            self.profile.tab_activate()

        if self.tabwidget.tabText(index) == 'Custom Profile Viewer':
            self.pview.tab_activate()

        if self.tabwidget.tabText(index) == 'Layer Editor':
            self.layer.tab_activate()

        if self.tabwidget.tabText(index) == 'Data Display':
            self.ddisp.tab_activate()

        self.oldtab = self.tabwidget.tabText(index)


def main():
    """ Main class of the PyGMI program """
    app = QtGui.QApplication(sys.argv)
    wid = MainWidget()
    wid.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
