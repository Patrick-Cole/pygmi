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

from PyQt4 import QtGui, QtCore

# Other dependancies
from . import misc
from . import tab_ddisp
from . import tab_layer
from . import tab_prof
from . import tab_pview
from . import tab_param
from . import tab_mext
from . import grvmag3d
from . import iodefs
from .datatypes import LithModel
import pygmi.menu_default as menu_default
import pygmi.misc as pmisc

class MainWidget(QtGui.QMainWindow):
    """ Widget class to call the main interface """
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.indata = {}
        self.inraster = {}
        self.outdata = {}
        self.parent = parent
        self.showprocesslog = self.parent.showprocesslog
        self.lmod1 = LithModel()  # actual model
        self.lmod2 = LithModel()  # regional model

# General
        self.txtmsg = ''
        self.modelfilename = r'./tmp'

        self.toolbar = QtGui.QToolBar()
        self.statusbar = QtGui.QStatusBar()
        self.tabwidget = QtGui.QTabWidget()
        self.pbar_sub = pmisc.ProgressBar()
        self.pbar_main = pmisc.ProgressBar()
        self.textbrowser = QtGui.QTextBrowser()
        self.actionsave = QtGui.QPushButton()

        self.tabwidget.setCurrentIndex(0)
        self.oldtab = self.tabwidget.tabText(0)
        self.pbars = misc.ProgressBar(self.pbar_sub, self.pbar_main)
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
        self.tabwidget.addTab(self.pview.userint, "Custom Profile Editor")

# Gravity and magnetic modelling routines
        self.grvmag = grvmag3d.GravMag(self)

        self.setupui()

    def setupui(self):
        """ Setup for the GUI """
        centralwidget = QtGui.QWidget(self)
        verticallayout = QtGui.QVBoxLayout(centralwidget)
        hlayout = QtGui.QHBoxLayout()

        helpdocs = menu_default.HelpButton()

        self.setCentralWidget(centralwidget)
        self.resize(1024, 768)
        self.toolbar.setStyleSheet('QToolBar{spacing:10px;}')
        self.setStatusBar(self.statusbar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
        self.textbrowser.setFrameShape(QtGui.QFrame.StyledPanel)
#        self.pbar_sub.setTextVisible(False)
#        self.pbar_main.setTextVisible(False)
        self.toolbar.setMovable(True)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolbar.addWidget(self.actionsave)

        self.setWindowTitle("Potential Field Modelling")
        self.actionsave.setText("Save Model")

        hlayout.addWidget(self.textbrowser)
        hlayout.addWidget(helpdocs)
        verticallayout.addWidget(self.tabwidget)
        verticallayout.addLayout(hlayout)
        verticallayout.addWidget(self.pbar_sub)
        verticallayout.addWidget(self.pbar_main)

        helpdocs.clicked.disconnect()
        helpdocs.clicked.connect(self.help_docs)
        self.actionsave.clicked.connect(self.savemodel)
        self.tabwidget.currentChanged.connect(self.tab_change)

    def savemodel(self):
        """ Model Save """
        tmp = iodefs.ExportMod3D(self)
        tmp.indata = self.outdata
        tmp.run()

        del tmp

    def help_docs(self):
        """
        Help Routine
        """
        index = self.tabwidget.currentIndex()
        htmlfile = ''

        if self.tabwidget.tabText(index) == 'Geophysical Parameters':
            htmlfile += 'pygmi.pfmod.param'

        if self.tabwidget.tabText(index) == 'Model Extent Parameters':
            htmlfile += 'pygmi.pfmod.mext'

        if self.tabwidget.tabText(index) == 'Profile Editor':
            htmlfile += 'pygmi.pfmod.prof'

        if self.tabwidget.tabText(index) == 'Custom Profile Editor':
            htmlfile += 'pygmi.pfmod.pview'

        if self.tabwidget.tabText(index) == 'Layer Editor':
            htmlfile += 'pygmi.pfmod.layer'

        if self.tabwidget.tabText(index) == 'Data Display':
            htmlfile += 'pygmi.pfmod.ddisp'

        menu_default.HelpDocs(self, htmlfile)

    def settings(self):
        """ Settings """
        if 'Raster' not in self.indata:
            self.indata['Raster'] = list(self.lmod1.griddata.values())

        for i in self.indata['Raster']:
            self.inraster[i.dataid] = i
        if 'Model3D' in self.indata.keys():
            self.lmod1 = self.indata['Model3D'][0]
#            self.lmod1.init_calc_grids()
            self.lmod2 = self.indata['Model3D'][-1]
#            self.lmod2.init_calc_grids()

        self.outdata['Model3D'] = [self.lmod1]
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

        if self.oldtab == 'Custom Profile Editor':
            self.pview.update_model()

        if self.tabwidget.tabText(index) == 'Geophysical Parameters':
            self.param.tab_activate()

        if self.tabwidget.tabText(index) == 'Model Extent Parameters':
            self.mext.tab_activate()

        if self.tabwidget.tabText(index) == 'Profile Editor':
            self.profile.tab_activate()

        if self.tabwidget.tabText(index) == 'Custom Profile Editor':
            self.pview.tab_activate()

        if self.tabwidget.tabText(index) == 'Layer Editor':
            self.layer.tab_activate()

        if self.tabwidget.tabText(index) == 'Data Display':
            self.ddisp.tab_activate()

        self.oldtab = self.tabwidget.tabText(index)
