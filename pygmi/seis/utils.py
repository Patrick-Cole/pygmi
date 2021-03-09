# -----------------------------------------------------------------------------
# Name:        utils.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2020 Council for Geoscience
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
"""Module for miscellaneous utilities relating to earthquake seismology."""

import difflib
import os
from PyQt5 import QtWidgets, QtCore
# import pygmi.menu_default as menu_default


class CorrectDescriptions(QtWidgets.QDialog):
    """
    Correct SEISAN descriptions.

    This compares the descriptions found in SEISAN type 3 lines to a custom
    list.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
        else:
            self.showprocesslog = parent.showprocesslog

        self.indata = {}
        self.outdata = {}
        self.parent = parent

        idir = os.path.dirname(os.path.realpath(__file__))
        tfile = os.path.join(idir, r'descriptions.txt')

        self.textfile = QtWidgets.QLineEdit(tfile)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
#        helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        pb_textfile = QtWidgets.QPushButton('Load Description List')

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Correct Descriptions')

        gridlayout_main.addWidget(self.textfile, 0, 0, 1, 1)
        gridlayout_main.addWidget(pb_textfile, 0, 1, 1, 1)

#        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_textfile.pressed.connect(self.get_textfile)

    def get_textfile(self, filename=''):
        """
        Get description list filename.

        Parameters
        ----------
        filename : str, optional
            Filename submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        ext = ('Description list (*.txt)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        self.textfile.setText(filename)

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        if 'Seis' not in self.indata:
            return False

        tmp = self.exec_()

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def loadproj(self, projdata):
        """
        Load project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """
        return False

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        projdata : dictionary
            Project data to be saved to JSON project file.

        """
        projdata = {}

#        projdata['ftype'] = '2D Mean'

        return projdata

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        filename = self.textfile.text()
        with open(filename) as fno:
            tmp = fno.read()

        masterlist = tmp.split('\n')

        data = self.indata['Seis']

        nomatch = []
        correction = []

        for i in data:
            if '3' not in i:
                continue
            text = i['3'].region

            cmatch = difflib.get_close_matches(text, masterlist, 1, cutoff=0.7)
            if cmatch:
                cmatch = cmatch[0]
            else:
                # self.showprocesslog('No match found for '+text)
                nomatch.append(text)
                continue

            if cmatch != text:
                # self.showprocesslog('Correcting '+text+' to '+cmatch)
                correction.append(text+' to '+cmatch)
                i['3'].region = cmatch

        self.outdata['Seis'] = data


def test():
    """
    Test routine.

    Returns
    -------
    None.

    """
    import pygmi.seis.iodefs as iodefs

    idir = os.path.dirname(os.path.realpath(__file__))
    tfile = os.path.join(idir, r'descriptions.txt')

    with open(tfile) as fno:
        tmp = fno.read()

    masterlist = tmp.split('\n')

    ifile = r'C:\WorkData\seismology\2000to2018.out'

    IO = iodefs.ImportSeisan(None)
    IO.settings(ifile)
    data = IO.outdata['Seis']

    nomatch = []
    correction = []

    for i in data:
        if '3' not in i:
            continue
        text = i['3'].region

        cmatch = difflib.get_close_matches(text, masterlist, 1, cutoff=0.7)
        if cmatch:
            cmatch = cmatch[0]
        else:
            nomatch.append(text)
            continue

        if cmatch != text:
            correction.append(text+' to '+cmatch)

    print('len nomatch', len(nomatch))
    print('len correction', len(correction))

    nomatch = list(set(nomatch))
    correction = list(set(correction))

    print('len nomatch', len(nomatch))
    print('len correction', len(correction))


if __name__ == "__main__":
    test()
