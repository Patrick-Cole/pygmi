# -----------------------------------------------------------------------------
# Name:        birrp.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2019 Council for Geoscience
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
BIRRP -Bounded Influence Remote Reference Processing

BIRRP is developed by:
Dr Alan D. Chave
Woods Hole Oceanographic Institution
achave@whoi.edu

It requires an executable which must be obtained directly from Dr Chave.
Details can be found at:
    https://www.whoi.edu/science/AOPE/people/achave/Site/Next1.html

Conditions for the use of the BIRRP bounded influence remote reference
magnetotelluric processing program:

   1. The robust bounded influence magnetotelluric analysis program,
      hereinafter called BIRRP, is provided on a caveat emptor basis.
      The author of BIRRP is not responsible for or culpable in the event of
      errors in processing or interpretation resulting from use of this code.
   2. No payment will be accepted by any user for data processing with BIRRP.
   3. BIRRP will not be distributed to anyone. Interested persons should be
      referred to this website.
   4. The author will be notified of any possible coding errors that are
      encountered.
   5. The author will be informed of any improvements or additions that are
      made to BIRRP.
   6. The use of BIRRP will be acknowledged in any publications and
      presentations that ensue.

If these conditions are acceptable, send e-mail to achave@whoi.edu.
The body of the message should state "I accept the conditions under which
BIRRP is distributed" and copy the above six conditions.
A gzipped tar file containing the source code will be distributed by
return e-mail.

Note, it will still be necessary for the end-user to compile the code.

"""

import os
import functools
import subprocess
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pandas as pd
from pygmi.vector.datatypes import LData
import pygmi.menu_default as menu_default
import mtpy.processing.birrp as MTbp


class BIRRP(QtWidgets.QDialog):
    """
    Import Line Data.

    This class imports ASCII point data.

    Attributes
    ----------
    name : str
        item name
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        self.name = 'BIRRP'
        self.pbar = None  # self.parent.pbar
        self.parent = parent
        self.indata = {'tmp': True}
        self.outdata = {}
        self.ifile = ''

        self.df_gps = None

        self.ilev = QtWidgets.QComboBox()
        self.nout = QtWidgets.QComboBox()
        self.ninp = QtWidgets.QComboBox()
        self.tbw = QtWidgets.QLineEdit('2')
        self.deltat = QtWidgets.QLineEdit('1')
        self.nfft = QtWidgets.QLineEdit('65536')
        self.nsctmax = QtWidgets.QLineEdit('13')
        self.uin = QtWidgets.QLineEdit('0')
        self.ainuin = QtWidgets.QLineEdit('.9999')
        self.c2threshe = QtWidgets.QLineEdit('0')
        self.nz = QtWidgets.QComboBox()
        self.c2threshe1 = QtWidgets.QLineEdit('0')
        self.ofil = QtWidgets.QLineEdit('mt')
        self.nlev = QtWidgets.QComboBox()
        self.npcs = QtWidgets.QLineEdit('1')
        self.nar = QtWidgets.QLineEdit('15')
        self.imode = QtWidgets.QComboBox()
        self.jmode = QtWidgets.QComboBox()

        self.nread = QtWidgets.QLineEdit('1')

        self.nfil = {}
        self.fpar = {}
        self.cpar = {}
        self.arfilnam = {}
        self.filnam = {}
        self.nskip = {}
        self.dstim = {}
        self.wstim = {}
        self.wetim = {}
        self.npts = {}
        self.pb_arfilnam = {}
        self.pb_cpar = {}
        self.pb_filnam = {}

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            self.nfil[i] = QtWidgets.QLineEdit('0')
            self.fpar[i] = QtWidgets.QLineEdit('0')
            self.cpar[i] = QtWidgets.QLineEdit('filename')

            self.arfilnam[i] = QtWidgets.QLineEdit('filename')
            self.filnam[i] = QtWidgets.QLineEdit('filename')
            self.nskip[i] = QtWidgets.QLineEdit('0')
            self.dstim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')
            self.wstim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')
            self.wetim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')

            self.pb_arfilnam[i] = QtWidgets.QPushButton('ARFILNAM: '+i+' AR filter filename')
            self.pb_cpar[i] = QtWidgets.QPushButton('CPAR: '+i+' filter parameters filename')
            self.pb_filnam[i] = QtWidgets.QPushButton('FILNAM: '+i+' filename')

        self.thetae = QtWidgets.QLineEdit('0,90,0')
        self.thetab = QtWidgets.QLineEdit('0,90,0')
        self.thetar = QtWidgets.QLineEdit('0,90,0')

        self.setupui()

    def setupui(self):
        """Set up UI."""
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        pb_importbirrp = QtWidgets.QPushButton('Import BIRRP configuration file')
        pb_runbirrp = QtWidgets.QPushButton('Save BIRRP configuration file')  #' and run BIRRP')

        self.ilev.setDisabled(True)
        self.imode.setDisabled(True)
        self.ninp.setDisabled(True)
        self.nar.setValidator(QtGui.QIntValidator(self))
        self.nlev.setCurrentIndex(3)

        self.ilev.addItem('0 = basic')
        self.nout.addItems(['2 = EX, EY', '3 = EX, EY, BZ'])
        self.ninp.addItems(['2 = BX, BY'])
        self.imode.addItems(['0 = separate ASCII files',
                             '1 = separate binary files',
                             '2 = single ASCII file',
                             '3 = TS ASCII format'])
        self.jmode.addItems(['0 = by points', '1 = by date/time'])
        self.nz.addItems(['0 = separate from E',
                          '1 = E threshold',
                          '2 = E and B threshold'])
        self.nlev.addItems(['-3', '-2', '-1', '0', '1', '2', '3'])

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            self.nfil[i].setValidator(QtGui.QIntValidator(self))
            self.nfil[i].editingFinished.connect(self.nfil_changed)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
#        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'BIRRP Processing')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl1 = QtWidgets.QHBoxLayout()
        self.lay = QtWidgets.QFormLayout()
        self.lay2 = QtWidgets.QFormLayout()
        self.lay3 = QtWidgets.QFormLayout()

        self.lay.addRow("ILEV: input Level", self.ilev)
        self.lay.addRow("NOUT: number of output time series", self.nout)
        self.lay.addRow("NINP: number of input time series", self.ninp)
        self.lay.addRow("TBW: time bandwidth for prolate data window", self.tbw)
        self.lay.addRow("DELTAT: sample interval", self.deltat)
        self.lay.addRow("NFFT: initial section length", self.nfft)
        self.lay.addRow("NSCTMAX: maximum number of sections", self.nsctmax)
        self.lay.addRow("UIN: robustness parameter", self.uin)
        self.lay.addRow("AIUIN: leverage parameter", self.ainuin)
        self.lay.addRow("C2THRESHE: second stage coherence threshold", self.c2threshe)
        self.lay.addRow("OFIL: output filename root", self.ofil)
        self.lay.addRow("NLEV: output level", self.nlev)
        self.lay.addRow("NPCS: numer of data pieces", self.npcs)
        self.lay.addRow("NAR: length of ar filter (0 for none, <0 for filename)", self.nar)
        self.lay.addRow("IMODE: file mode", self.imode)
        self.lay.addRow("JMODE: input mode", self.jmode)
        self.lay.addRow("NREAD: number of data values to be read", self.nread)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for electrics", self.thetae)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for magnetics", self.thetab)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for calculation", self.thetar)

        for i in ['ex', 'ey']:
            self.lay2.addRow("NFIL: number filter parameters (<0 for filename) of "+i, self.nfil[i])
            self.lay2.addRow(self.pb_filnam[i], self.filnam[i])
            self.lay2.addRow("NSKIP: leading values to skip in "+i,
                             self.nskip[i])

        for i in ['hx', 'hy', 'rx', 'ry']:
            self.lay3.addRow("NFIL: number filter parameters (<0 for filename) of "+i, self.nfil[i])
            self.lay3.addRow(self.pb_filnam[i], self.filnam[i])
            self.lay3.addRow("NSKIP: leading values to skip in "+i,
                             self.nskip[i])

        hbl1.addLayout(self.lay)
        hbl1.addLayout(self.lay2)
        hbl1.addLayout(self.lay3)

        hbl2 = QtWidgets.QHBoxLayout()
#        hbl2.addWidget(helpdocs)
        hbl2.addWidget(buttonbox)

        vbl.addWidget(pb_importbirrp)
        vbl.addLayout(hbl1)
        vbl.addWidget(pb_runbirrp)
        vbl.addLayout(hbl2)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.nout.currentIndexChanged.connect(self.nout_changed)
        self.nz.currentIndexChanged.connect(self.nout_changed)
        self.jmode.currentIndexChanged.connect(self.jmode_changed)
        self.nar.editingFinished.connect(self.nar_changed)
        pb_importbirrp.pressed.connect(self.importbirrp)
        pb_runbirrp.pressed.connect(self.runbirrp)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            self.pb_filnam[i].pressed.connect(functools.partial(self.get_filename,
                                                                self.filnam[i]))
            self.pb_cpar[i].pressed.connect(functools.partial(self.get_filename,
                                                              self.cpar[i]))
            self.pb_arfilnam[i].pressed.connect(functools.partial(self.get_filename,
                                                                  self.arfilnam[i]))

    def importbirrp(self):
        """ imports a birrp config file """
        ext = ('*.birrp (*.birrp)')

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        with open(filename) as ifile:
            data = ifile.read()

        data = data.replace('\n', ' ')
        data = data.replace(',', ' ')

        data = data.split()

        ilev = data.pop(0)
        if int(ilev) != 0:
            print('not supported')
            return
        nout = int(data.pop(0))
        ninp = int(data.pop(0))
        tbw = data.pop(0)
        deltat = data.pop(0)
        nfft = data.pop(0)
        nsctmax = data.pop(0)
        yes = data.pop(0)
        uin = data.pop(0)
        ainuin = data.pop(0)
        c2threshe = data.pop(0)
        nz = 0
        if nout == 3:
            nz = int(data.pop(0))
        elif nout == 1:
            print('not supported')
            return
        c2threshe1 = ''
        if int(nout) == 3 and int(nz) == 0:
            c2threshe1 = data.pop(0)
        ofil = data.pop(0)
        nlev = int(data.pop(0))
        npcs = data.pop(0)
        nar = data.pop(0)
        imode = int(data.pop(0))
        jmode = int(data.pop(0))
        if imode != 0:
            print('not supported')
            return
        nread = ''
        if jmode == 0:
            nread = data.pop(0)
        else:
            print('not supported')
            return

        nfil = {}
        fpar = {}
        cpar = {}
        arfilnam = {}
        filnam = {}
        nskip = {}
        dstim = {}
        wstim = {}
        wetim = {}

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if nout == 2 and i == 'hz':
                continue
            nfil[i] = data.pop(0)
            fpar[i] = '0'
            if int(nfil[i]) > 0:
                fpar[i] = []
                for j in nfil[i]:
                    fpar[i].append(float(data.pop(0)))
            cpar[i] = 'filename'
            if int(nfil[i]) < 0:
                cpar[i] = data.pop(0)
            arfilnam[i] = 'filename'
            if int(nar) < 0:
                arfilnam[i] = data.pop(0)
            filnam[i] = data.pop(0)
            nskip[i] = 0
            dstim[i] = 'YYYY-MM-DD HH:MM:SS'
            wstim[i] = 'YYYY-MM-DD HH:MM:SS'
            wetim[i] = 'YYYY-MM-DD HH:MM:SS'
            if int(jmode) == 0:
                nskip[i] = data.pop(0)
            else:
                dstim[i] = data.pop(0)
                wstim[i] = data.pop(0)
                wetim[i] = data.pop(0)
        thetae = data.pop(0)+','+data.pop(0)+','+data.pop(0)
        thetab = data.pop(0)+','+data.pop(0)+','+data.pop(0)
        thetar = data.pop(0)+','+data.pop(0)+','+data.pop(0)

# Now we set controls
        self.nout.setCurrentIndex(nout-2)
        self.ninp.setCurrentIndex(ninp-2)
        self.tbw.setText(tbw)
        self.deltat.setText(deltat)
        self.nfft.setText(nfft)
        self.nsctmax.setText(nsctmax)
        self.uin.setText(uin)
        self.ainuin.setText(ainuin)
        self.c2threshe.setText(c2threshe)
        self.nz.setCurrentIndex(nz)
        self.c2threshe1.setText(c2threshe1)
        self.ofil.setText(ofil)
        self.nlev.setCurrentIndex(nlev+3)
        self.npcs.setText(npcs)
        self.nar.setText(nar)
        self.imode.setCurrentIndex(imode)
        self.jmode.setCurrentIndex(jmode)
        self.nread.setText(nread)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if nout == 2 and i == 'hz':
                continue
            self.nfil[i].setText(nfil[i])
            self.fpar[i].setText(fpar[i])
            self.cpar[i].setText(cpar[i])
            self.arfilnam[i].setText(arfilnam[i])
            self.filnam[i].setText(filnam[i])
            self.nskip[i].setText(nskip[i])
            self.dstim[i].setText(dstim[i])
            self.wstim[i].setText(wstim[i])
            self.wetim[i].setText(wetim[i])
        self.thetae.setText(thetae)
        self.thetab.setText(thetab)
        self.thetar.setText(thetar)

    def runbirrp(self):
        """ saves and runs a birrp config file """
        ext = ('*.birrp (*.birrp)')

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return

        birrp_path = os.path.dirname(__file__)[:-2]+r'\bin\birrp.exe'


        nout = self.nout.currentIndex()+2
        nz = self.nz.currentIndex()
        jmode = self.jmode.currentIndex()
        nar = int(self.nar.text())

        with open(filename, 'w+') as ofile:
            ofile.write('0\n')  # ilev == 0
            ofile.write(str(self.nout.currentIndex()+2)+'\n')
            ofile.write(str(self.ninp.currentIndex()+2)+'\n')

            ofile.write(self.tbw.text()+'\n')
            ofile.write(self.deltat.text()+'\n')
            ofile.write(self.nfft.text()+','+self.nsctmax.text()+'\n')
            ofile.write('y\n')
            ofile.write(self.uin.text()+','+self.ainuin.text()+'\n')
            ofile.write(self.c2threshe.text()+'\n')
            if nout == 3:
                ofile.write(str(self.nz.currentIndex())+'\n')
            if nout == 3 and nz == 0:
                ofile.write(self.c2threshe1.text()+'\n')
            ofile.write(self.ofil.text()+'\n')
            ofile.write(str(self.nlev.currentIndex()-3)+'\n')
            ofile.write(self.npcs.text()+'\n')
            ofile.write(self.nar.text()+'\n')
            ofile.write(str(self.imode.currentIndex())+'\n')
            ofile.write(str(self.jmode.currentIndex())+'\n')
            if jmode == 0:
                ofile.write(self.nread.text()+'\n')

            for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
                nfil = int(self.nfil[i].text()+'\n')
                if nout == 2 and i == 'hz':
                    continue
                ofile.write(self.nfil[i].text()+'\n')
                if nfil > 0:
                    ofile.write(self.fpar[i].text()+'\n')
                if nfil < 0:
                    ofile.write(self.cpar[i].text()+'\n')
                if nar < 0:
                    ofile.write(self.arfilnam[i].text()+'\n')
                ofile.write(self.filnam[i].text()+'\n')
                if jmode == 0:
                    ofile.write(self.nskip[i].text()+'\n')
                else:
                    ofile.write(self.dstim[i].text()+'\n')
                    ofile.write(self.wstim[i].text()+'\n')
                    ofile.write(self.wetim[i].text()+'\n')
            ofile.write(self.thetae.text()+'\n')
            ofile.write(self.thetab.text()+'\n')
            ofile.write(self.thetar.text()+'\n')

#        MTbp.run(birrp_path, filename)


    def get_filename(self, widget):
        """ get filename for a component """

        ext = ('*.* (*.*)')

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        widget.setText(filename)

    def nar_changed(self):
        """ nar changed """
        text = self.nar.text()
        val = int(text)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if i in ['ex', 'ey', 'hz']:
                lay = self.lay2
            else:
                lay = self.lay3
            row, _ = lay.getWidgetPosition(self.filnam[i])
            if row == -1:
                continue
            if val < 0:
                self.showrow(row, self.pb_arfilnam[i], self.arfilnam[i],
                             lay)
            else:
                self.removerow(self.arfilnam[i], lay)

    def nfil_changed(self):
        """ nfil changed """

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if i in ['ex', 'ey', 'hz']:
                lay = self.lay2
            else:
                lay = self.lay3

            text = self.nfil[i].text()
            val = int(text)
            filt = str([1.]*val)[1:-1]

            row, _ = lay.getWidgetPosition(self.nfil[i])

            if val > 0:
                self.fpar[i].setText(filt)
                self.showrow(row+1, "FPAR: vector of filter parameters",
                             self.fpar[i], lay)
                self.removerow(self.cpar[i], lay)
            elif val < 0:
                self.showrow(row+1, self.pb_cpar[i],
                             self.cpar[i], lay)
                self.removerow(self.fpar[i], lay)
            else:
                self.removerow(self.cpar[i], lay)
                self.removerow(self.fpar[i], lay)

    def imode_changed(self, indx):
        """ imode changed """
        row1, _ = self.lay.getWidgetPosition(self.nfil)
        row2, _ = self.lay.getWidgetPosition(self.cpar)
        row3, _ = self.lay.getWidgetPosition(self.fpar)

        row = max([row1, row2, row3])

        if indx > 0:
            self.showrow(row+1, "NBLOCK: size of data blocks", self.nblock,
                         self.lay)
        elif indx == 0:
            self.removerow(self.nblock, self.lay)

    def jmode_changed(self):
        """ jmode changed """
        row, _ = self.lay.getWidgetPosition(self.jmode)
        txt = self.jmode.currentText()

        if txt == '0 = by points':
            self.showrow(row+1, "NREAD: number of data values to be read",
                         self.nread, self.lay)
        else:
            self.removerow(self.nread, self.lay)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if i in ['ex', 'ey', 'hz']:
                lay = self.lay2
            else:
                lay = self.lay3

            row, _ = lay.getWidgetPosition(self.filnam[i])

            if txt == '0 = by points':
                self.removerow(self.dstim[i], lay)
                self.removerow(self.wstim[i], lay)
                self.removerow(self.wetim[i], lay)
            else:
                self.showrow(row+1, "DSTIM: data series start time",
                             self.dstim[i], lay)
                self.showrow(row+2, "WSTIM: processing window start time",
                             self.wstim[i], lay)
                self.showrow(row+3, "WETIM: processing window end time",
                             self.wetim[i], lay)

    def nout_changed(self):
        """ nout changed """
        row, _ = self.lay.getWidgetPosition(self.c2threshe)

        txt = self.nout.currentText()
        txt2 = self.nz.currentText()

# First do NZ
        if txt == '3 = EX, EY, BZ':
            self.showrow(row+1,
                         "NZ: threshold mode for vertical magnetic field",
                         self.nz, self.lay)
        else:
            self.removerow(self.nz, self.lay)

# Now do C2threshe1
        if txt == '3 = EX, EY, BZ' and txt2 == '0 = separate from E':
            self.showrow(row+2, "C2THRESHE1: coherence threshold for vertical magnetic field",
                         self.c2threshe1, self.lay)
        else:
            self.removerow(self.c2threshe1, self.lay)



# Now do file stuff
        if txt == '3 = EX, EY, BZ':
            row, _ = self.lay2.getWidgetPosition(self.nskip['ey'])

            self.showrow(row+1, "NFIL: number filter parameters (<0 for filename) of hz",
                         self.nfil['hz'], self.lay2)
            self.showrow(row+2, self.pb_filnam['hz'], self.filnam['hz'], self.lay2)
            self.showrow(row+3, "NSKIP: leading values to skip in hz",
                             self.nskip['hz'], self.lay2)
        else:
            self.removerow(self.nfil['hz'], self.lay2)
            self.removerow(self.filnam['hz'], self.lay2)
            self.removerow(self.nskip['hz'], self.lay2)


        self.nar_changed()
        self.nfil_changed()
        self.jmode_changed()

    def showrow(self, row, label, widget, lay):
        """ shows a row with a widget """
        if lay.getWidgetPosition(widget)[0] == -1:
            lay.insertRow(row, label, widget)
            widget.show()

    def removerow(self, widget, lay):
        """ removes a row """
        if lay.getWidgetPosition(widget)[0] > -1:
            widget.hide()
            lay.labelForField(widget).hide()
            lay.takeRow(widget)

    def settings(self, test=False):
        """Entry point into item. Data imported from here."""

        if not test:
            tmp = self.exec_()

            if tmp != 1:
                return tmp

        return True

    def get_gps(self, filename=''):
        """ Get GPS filename """
        ext = ('GPS comma delimited (*.csv)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))

        df2 = pd.read_csv(filename)

        df2['Station'] = pd.to_numeric(df2['Station'], errors='coerce')

        self.df_gps = df2

        self.gpsfile.setText(filename)

        ltmp = list(df2.columns)

        xind = 0
        yind = 1
        zind = 2
        for i, tmp in enumerate(ltmp):
            if 'lon' in tmp.lower():
                xind = i
            elif 'lat' in tmp.lower():
                yind = i
            elif 'elev' in tmp.lower() or 'alt' in tmp.lower():
                zind = i

        self.xchan.addItems(ltmp)
        self.ychan.addItems(ltmp)
        self.zchan.addItems(ltmp)

        self.xchan.setCurrentIndex(xind)
        self.ychan.setCurrentIndex(yind)
        self.zchan.setCurrentIndex(zind)

        self.xchan.setEnabled(True)
        self.ychan.setEnabled(True)
        self.zchan.setEnabled(True)


#def addrow(vbl, text, widget):
#    """ routine to simplify adding widgets with labels """
#    hbl = QtWidgets.QHBoxLayout()
#    label = QtWidgets.QLabel(text)
#
#    hbl.addWidget(label)
#    hbl.addWidget(widget)
#    vbl.addLayout(hbl)
#
#    return hbl
#
#
#class ComboBox(QtWidgets.QWidget):
#    """ Custom combo box with text label """
#    def __init__(self, text, items, parent=None):
#        QtWidgets.QWidget.__init__(self, parent=parent)
#        hbl = QtWidgets.QHBoxLayout(self)
#        label = QtWidgets.QLabel(text)
#        self.combo = QtWidgets.QComboBox()
#        for i in items:
#            self.combo.addItem(str(i))
#        hbl.addWidget(label)
#        hbl.addWidget(self.combo)
#
#
#class LineEdit(QtWidgets.QWidget):
#    """ Custom line edit with text label """
#    def __init__(self, text, item, parent=None):
#        QtWidgets.QWidget.__init__(self, parent=parent)
#        hbl = QtWidgets.QHBoxLayout(self)
#        label = QtWidgets.QLabel(text)
#        self.lineedit = QtWidgets.QLineEdit(str(item))
#        hbl.addWidget(label)
#        hbl.addWidget(self.lineedit)
