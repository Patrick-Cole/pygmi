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
"""The main program for the modelling package."""

from PyQt5 import QtWidgets, QtCore

from pygmi.pfmod import misc
from pygmi.pfmod import tab_prof
from pygmi.pfmod import tab_param
from pygmi.pfmod import tab_mext
from pygmi.pfmod import grvmag3d
from pygmi.pfmod import iodefs
from pygmi.pfmod.datatypes import LithModel
from pygmi import menu_default
import pygmi.misc as pmisc


class MainWidget(QtWidgets.QMainWindow):
    """MainWidget - Widget class to call the main interface."""

    def __init__(self, parent=None):
        super().__init__(parent)

        if parent is None:
            self.showlog = print
        else:
            self.showlog = parent.showlog

        self.indata = {'tmp': True}
        self.inraster = {}
        self.outdata = {}
        self.parent = parent
        self.lmod1 = LithModel()  # actual model
        self.is_import = False

        # General
        self.txtmsg = ''

        self.toolbardock = QtWidgets.QToolBar()
        self.statusbar = QtWidgets.QStatusBar()

        self.pbar_sub = pmisc.ProgressBar()
        self.pbar_main = pmisc.ProgressBar()
        self.textbrowser = QtWidgets.QTextBrowser()
        self.actionsave = QtWidgets.QAction('Save Model')

        self.pbars = misc.ProgressBar(self.pbar_sub, self.pbar_main)
        tmp = list(set(self.lmod1.griddata.values()))
        self.outdata['Raster'] = tmp

        self.mext = tab_mext.MextDisplay(self)
        self.param = tab_param.ParamDisplay(self)
        self.lithnotes = tab_param.LithNotes(self)
        self.profile = tab_prof.ProfileDisplay(self)

        # Toolbars
        self.action_mext = QtWidgets.QAction('Model\nExtent\nParameters')
        self.toolbardock.addAction(self.action_mext)
        self.action_mext.triggered.connect(self.mext.tab_activate)

        self.action_param = QtWidgets.QAction('Geophysical\nParameters')
        self.toolbardock.addAction(self.action_param)
        self.action_param.triggered.connect(self.param.tab_activate)

        self.action_lnotes = QtWidgets.QAction('Lithology\nNotes')
        self.toolbardock.addAction(self.action_lnotes)
        self.action_lnotes.triggered.connect(self.lithnotes.tab_activate)

        # Dock Widgets
        dock = QtWidgets.QDockWidget('Editor')
        dock.setWidget(self.profile)
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
        self.toolbardock.addAction(dock.toggleViewAction())

        self.grvmag = grvmag3d.GravMag(self)

        self.setupui()

    def setupui(self):
        """
        GUI setup.

        Returns
        -------
        None.

        """
        centralwidget = QtWidgets.QWidget(self)
        verticallayout = QtWidgets.QVBoxLayout(centralwidget)
        hlayout = QtWidgets.QHBoxLayout()

        helpdocs = menu_default.HelpButton()

        self.setStatusBar(self.statusbar)
        self.setCentralWidget(centralwidget)

        self.addToolBar(self.toolbardock)
        self.textbrowser.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.toolbardock.addAction(self.actionsave)

        self.setWindowTitle('Potential Field Modelling')

        hlayout.addWidget(self.textbrowser)
        hlayout.addWidget(helpdocs)
        verticallayout.addLayout(hlayout)
        verticallayout.addWidget(self.pbar_sub)
        verticallayout.addWidget(self.pbar_main)

        helpdocs.clicked.disconnect()
        helpdocs.clicked.connect(self.help_docs)
        self.actionsave.triggered.connect(self.savemodel)

        if self.parent is not None:
            self.resize(self.parent.width(), self.parent.height())

    def savemodel(self):
        """
        Save model.

        Returns
        -------
        None.

        """
        self.showtext('Saving model, please do not close the interface...')
        tmp = iodefs.ExportMod3D(self.parent)
        tmp.indata = self.outdata
        tmp.run()

        del tmp
        self.showtext('Model save complete!')

    def help_docs(self):
        """
        Help documentation.

        Returns
        -------
        None.

        """
        menu_default.HelpDocs(self, 'pygmi.pfmod.prof')

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
        datatmp = list(set(self.lmod1.griddata.values()))

        if 'Raster' not in self.indata:
            self.indata['Raster'] = datatmp

        self.inraster = {}
        for i in self.indata['Raster']:
            self.inraster[i.dataid] = i
        if 'Model3D' in self.indata:
            self.lmod1 = self.indata['Model3D'][0]

        self.outdata['Model3D'] = [self.lmod1]
        self.mext.update_combos()
        self.mext.tab_activate()

        self.profile.change_defs()
        self.profile.tab_activate()

        self.outdata['Raster'] = datatmp

        self.show()

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """

    def data_reset(self):
        """
        Reset the data.

        Returns
        -------
        None.

        """
        if 'Model3D' in self.indata:
            self.lmod1 = self.indata['Model3D'][0]
        self.lmod1.griddata = {}
        self.lmod1.init_calc_grids()

    def showtext(self, txt, replacelast=False):
        """
        Show text on the text panel of the main user interface.

        Parameters
        ----------
        txt : str
            Text to display.
        replacelast : bool, optional
            Whether to replace the last text written. The default is False.

        Returns
        -------
        None.

        """
        if replacelast is True:
            self.txtmsg = self.txtmsg[:self.txtmsg.rfind('\n')]
            self.txtmsg = self.txtmsg[:self.txtmsg.rfind('\n')]
            self.txtmsg += '\n'
        self.txtmsg += txt + '\n'
        self.textbrowser.setPlainText(self.txtmsg)
        tmp = self.textbrowser.verticalScrollBar()
        tmp.setValue(tmp.maximumHeight())
        self.repaint()


def _testfn():
    """Test routine."""
    import sys
    from pygmi.pfmod.iodefs import ImportMod3D
    from pygmi.pfmod.misc import MergeMod3D
    app = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    ifile = r"D:\Workdata\modelling\mergetest\3dmodel_test.npz"
    ifile2 = r"D:\Workdata\modelling\mergetest\3dmodel_test2.npz"

    IO1 = ImportMod3D()
    IO1.ifile = ifile
    IO1.settings(True)

    IO2 = ImportMod3D()
    IO2.ifile = ifile2
    IO2.settings(True)

    data = {'Model3D': IO1.outdata['Model3D']+IO2.outdata['Model3D'],
            'Raster': IO1.outdata['Raster']+IO2.outdata['Raster']}

    MM = MergeMod3D()
    MM.indata = data
    MM.settings(True)

    for i in MM.outdata['Raster']:
        print(i.dataid)

    PF = MainWidget()
    PF.indata = MM.outdata
    PF.settings()

    app.exec_()


if __name__ == "__main__":
    _testfn()
