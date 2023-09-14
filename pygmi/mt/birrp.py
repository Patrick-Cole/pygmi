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
BIRRP -Bounded Influence Remote Reference Processing.

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
from PyQt5 import QtWidgets, QtCore, QtGui

from pygmi.misc import BasicModule


class BIRRP(BasicModule):
    """Class to export config file for BIRRP."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.indata = {'tmp': True}

        self.df_gps = None

        self.cmb_ilev = QtWidgets.QComboBox()
        self.cmb_nout = QtWidgets.QComboBox()
        self.cmb_ninp = QtWidgets.QComboBox()
        self.le_tbw = QtWidgets.QLineEdit('2')
        self.le_deltat = QtWidgets.QLineEdit('1')
        self.le_nfft = QtWidgets.QLineEdit('65536')
        self.le_nsctmax = QtWidgets.QLineEdit('13')
        self.le_uin = QtWidgets.QLineEdit('0')
        self.le_ainuin = QtWidgets.QLineEdit('.9999')
        self.le_c2threshe = QtWidgets.QLineEdit('0')
        self.cmb_nz = QtWidgets.QComboBox()
        self.le_c2threshe1 = QtWidgets.QLineEdit('0')
        self.le_ofil = QtWidgets.QLineEdit('mt')
        self.cmb_nlev = QtWidgets.QComboBox()
        self.le_npcs = QtWidgets.QLineEdit('1')
        self.le_nar = QtWidgets.QLineEdit('15')
        self.cmb_imode = QtWidgets.QComboBox()
        self.cmb_jmode = QtWidgets.QComboBox()

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
            self.le_nfil[i] = QtWidgets.QLineEdit('0')
            self.le_fpar[i] = QtWidgets.QLineEdit('0')
            self.le_cpar[i] = QtWidgets.QLineEdit('filename')

            self.le_arfilnam[i] = QtWidgets.QLineEdit('filename')
            self.le_filnam[i] = QtWidgets.QLineEdit('filename')
            self.le_nskip[i] = QtWidgets.QLineEdit('0')
            self.le_dstim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')
            self.le_wstim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')
            self.le_wetim[i] = QtWidgets.QLineEdit('YYYY-MM-DD HH:MM:SS')

            self.pb_arfilnam[i] = QtWidgets.QPushButton('ARFILNAM: ' + i +
                                                        ' AR filter filename')
            self.pb_cpar[i] = QtWidgets.QPushButton('CPAR: ' + i +
                                                    ' filter parameters '
                                                    'filename')
            self.pb_filnam[i] = QtWidgets.QPushButton('FILNAM: '+i+' filename')

        self.le_thetae = QtWidgets.QLineEdit('0,90,0')
        self.le_thetab = QtWidgets.QLineEdit('0,90,0')
        self.le_thetar = QtWidgets.QLineEdit('0,90,0')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        buttonbox = QtWidgets.QDialogButtonBox()
        # helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        pb_importbirrp = QtWidgets.QPushButton('Import BIRRP configuration '
                                               'file')
        pb_runbirrp = QtWidgets.QPushButton('Save BIRRP configuration file')

        self.cmb_ilev.setDisabled(True)
        self.cmb_imode.setDisabled(True)
        self.cmb_ninp.setDisabled(True)
        self.le_nar.setValidator(QtGui.QIntValidator(self))
        self.cmb_nlev.setCurrentIndex(3)

        self.cmb_ilev.addItem('0 = basic')
        self.cmb_nout.addItems(['2 = EX, EY', '3 = EX, EY, BZ'])
        self.cmb_ninp.addItems(['2 = BX, BY'])
        self.cmb_imode.addItems(['0 = separate ASCII files',
                             '1 = separate binary files',
                             '2 = single ASCII file',
                             '3 = TS ASCII format'])
        self.cmb_jmode.addItems(['0 = by points', '1 = by date/time'])
        self.cmb_nz.addItems(['0 = separate from E',
                          '1 = E threshold',
                          '2 = E and B threshold'])
        self.cmb_nlev.addItems(['-3', '-2', '-1', '0', '1', '2', '3'])

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            self.le_nfil[i].setValidator(QtGui.QIntValidator(self))
            self.le_nfil[i].editingFinished.connect(self.nfil_changed)

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
#        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'BIRRP Processing')

        vbl = QtWidgets.QVBoxLayout(self)
        hbl_1 = QtWidgets.QHBoxLayout()
        self.lay = QtWidgets.QFormLayout()
        self.lay2 = QtWidgets.QFormLayout()
        self.lay3 = QtWidgets.QFormLayout()

        self.lay.addRow("ILEV: input Level", self.cmb_ilev)
        self.lay.addRow("NOUT: number of output time series", self.cmb_nout)
        self.lay.addRow("NINP: number of input time series", self.cmb_ninp)
        self.lay.addRow("TBW: time bandwidth for prolate data window",
                        self.le_tbw)
        self.lay.addRow("DELTAT: sample interval", self.le_deltat)
        self.lay.addRow("NFFT: initial section length", self.le_nfft)
        self.lay.addRow("NSCTMAX: maximum number of sections", self.le_nsctmax)
        self.lay.addRow("UIN: robustness parameter", self.le_uin)
        self.lay.addRow("AIUIN: leverage parameter", self.le_ainuin)
        self.lay.addRow("C2THRESHE: second stage coherence threshold",
                        self.le_c2threshe)
        self.lay.addRow("OFIL: output filename root", self.le_ofil)
        self.lay.addRow("NLEV: output level", self.cmb_nlev)
        self.lay.addRow("NPCS: number of data pieces", self.le_npcs)
        self.lay.addRow("NAR: length of ar filter (0 for none, <0 for "
                        "filename)", self.le_nar)
        self.lay.addRow("IMODE: file mode", self.cmb_imode)
        self.lay.addRow("JMODE: input mode", self.cmb_jmode)
        self.lay.addRow("NREAD: number of data values to be read", self.nread)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for electrics",
                        self.le_thetae)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for magnetics",
                        self.le_thetab)
        self.lay.addRow("THETA1,THETA2,PHI: Rotation angles for calculation",
                        self.le_thetar)

        for i in ['ex', 'ey']:
            self.lay2.addRow("NFIL: number filter parameters "
                             "(<0 for filename) of "+i, self.le_nfil[i])
            self.lay2.addRow(self.pb_filnam[i], self.le_filnam[i])
            self.lay2.addRow("NSKIP: leading values to skip in "+i,
                             self.le_nskip[i])

        for i in ['hx', 'hy', 'rx', 'ry']:
            self.lay3.addRow("NFIL: number filter parameters "
                             "(<0 for filename) of "+i, self.le_nfil[i])
            self.lay3.addRow(self.pb_filnam[i], self.le_filnam[i])
            self.lay3.addRow("NSKIP: leading values to skip in "+i,
                             self.le_nskip[i])

        hbl_1.addLayout(self.lay)
        hbl_1.addLayout(self.lay2)
        hbl_1.addLayout(self.lay3)

        hbl_2 = QtWidgets.QHBoxLayout()
#        hbl_2.addWidget(helpdocs)
        hbl_2.addWidget(buttonbox)

        vbl.addWidget(pb_importbirrp)
        vbl.addLayout(hbl_1)
        vbl.addWidget(pb_runbirrp)
        vbl.addLayout(hbl_2)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.cmb_nout.currentIndexChanged.connect(self.cmb_nout_changed)
        self.cmb_nz.currentIndexChanged.connect(self.cmb_nout_changed)
        self.cmb_jmode.currentIndexChanged.connect(self.cmb_jmode_changed)
        self.le_nar.editingFinished.connect(self.le_nar_changed)
        pb_importbirrp.pressed.connect(self.importbirrp)
        pb_runbirrp.pressed.connect(self.runbirrp)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            self.pb_filnam[i].pressed.connect(functools.partial(self.get_filename,
                                                                self.le_filnam[i]))
            self.pb_cpar[i].pressed.connect(functools.partial(self.get_filename,
                                                              self.le_cpar[i]))
            self.pb_arfilnam[i].pressed.connect(functools.partial(self.get_filename,
                                                                  self.le_arfilnam[i]))

    def importbirrp(self):
        """
        Import a BIRRP config file.

        Returns
        -------
        None.

        """
        ext = '*.birrp (*.birrp)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        with open(filename, encoding='utf-8') as ifile:
            data = ifile.read()

        data = data.replace('\n', ' ')
        data = data.replace(',', ' ')

        data = data.split()

        ilev = data.pop(0)
        if int(ilev) != 0:
            self.showlog('not supported')
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
            self.showlog('not supported')
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
            self.showlog('not supported')
            return
        nread = ''
        if jmode == 0:
            nread = data.pop(0)
        else:
            self.showlog('not supported')
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
        self.cmb_nout.setCurrentIndex(nout-2)
        self.cmb_ninp.setCurrentIndex(ninp-2)
        self.le_tbw.setText(tbw)
        self.le_deltat.setText(deltat)
        self.le_nfft.setText(nfft)
        self.le_nsctmax.setText(nsctmax)
        self.le_uin.setText(uin)
        self.le_ainuin.setText(ainuin)
        self.le_c2threshe.setText(c2threshe)
        self.cmb_nz.setCurrentIndex(nz)
        self.le_c2threshe1.setText(c2threshe1)
        self.le_ofil.setText(ofil)
        self.cmb_nlev.setCurrentIndex(nlev+3)
        self.le_npcs.setText(npcs)
        self.le_nar.setText(nar)
        self.cmb_imode.setCurrentIndex(imode)
        self.cmb_jmode.setCurrentIndex(jmode)
        self.nread.setText(nread)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if nout == 2 and i == 'hz':
                continue
            self.le_nfil[i].setText(nfil[i])
            self.le_fpar[i].setText(fpar[i])
            self.le_cpar[i].setText(cpar[i])
            self.le_arfilnam[i].setText(arfilnam[i])
            self.le_filnam[i].setText(filnam[i])
            self.le_nskip[i].setText(nskip[i])
            self.le_dstim[i].setText(dstim[i])
            self.le_wstim[i].setText(wstim[i])
            self.le_wetim[i].setText(wetim[i])
        self.le_thetae.setText(thetae)
        self.le_thetab.setText(thetab)
        self.le_thetar.setText(thetar)

    def runbirrp(self):
        """
        Save and runs a birrp config file.

        Returns
        -------
        None.

        """
        ext = '*.birrp (*.birrp)'

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent, 'Save File', '.', ext)
        if filename == '':
            return

        birrp_path = os.path.dirname(__file__)[:-2]+r'\bin\birrp.exe'

        nout = self.cmb_nout.currentIndex()+2
        nz = self.cmb_nz.currentIndex()
        jmode = self.cmb_jmode.currentIndex()
        nar = int(self.le_nar.text())

        with open(filename, 'w+', encoding='utf-8') as ofile:
            ofile.write('0\n')  # ilev == 0
            ofile.write(str(self.cmb_nout.currentIndex()+2)+'\n')
            ofile.write(str(self.cmb_ninp.currentIndex()+2)+'\n')

            ofile.write(self.le_tbw.text()+'\n')
            ofile.write(self.le_deltat.text()+'\n')
            ofile.write(self.le_nfft.text()+','+self.le_nsctmax.text()+'\n')
            ofile.write('y\n')
            ofile.write(self.le_uin.text()+','+self.le_ainuin.text()+'\n')
            ofile.write(self.le_c2threshe.text()+'\n')
            if nout == 3:
                ofile.write(str(self.cmb_nz.currentIndex())+'\n')
            if nout == 3 and nz == 0:
                ofile.write(self.le_c2threshe1.text()+'\n')
            ofile.write(self.le_ofil.text()+'\n')
            ofile.write(str(self.cmb_nlev.currentIndex()-3)+'\n')
            ofile.write(self.le_npcs.text()+'\n')
            ofile.write(self.le_nar.text()+'\n')
            ofile.write(str(self.cmb_imode.currentIndex())+'\n')
            ofile.write(str(self.cmb_jmode.currentIndex())+'\n')
            if jmode == 0:
                ofile.write(self.nread.text()+'\n')

            for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
                nfil = int(self.le_nfil[i].text()+'\n')
                if nout == 2 and i == 'hz':
                    continue
                ofile.write(self.le_nfil[i].text()+'\n')
                if nfil > 0:
                    ofile.write(self.le_fpar[i].text()+'\n')
                if nfil < 0:
                    ofile.write(self.le_cpar[i].text()+'\n')
                if nar < 0:
                    ofile.write(self.le_arfilnam[i].text()+'\n')
                ofile.write(self.le_filnam[i].text()+'\n')
                if jmode == 0:
                    ofile.write(self.le_nskip[i].text()+'\n')
                else:
                    ofile.write(self.le_dstim[i].text()+'\n')
                    ofile.write(self.le_wstim[i].text()+'\n')
                    ofile.write(self.le_wetim[i].text()+'\n')
            ofile.write(self.le_thetae.text()+'\n')
            ofile.write(self.le_thetab.text()+'\n')
            ofile.write(self.le_thetar.text()+'\n')

#        MTbp.run(birrp_path, filename)

    def get_filename(self, widget):
        """
        Get filename for a component.

        Parameters
        ----------
        widget : widget
            widget whose text is set to filename..

        Returns
        -------
        None.

        """
        ext = '*.* (*.*)'

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent, 'Open File', '.', ext)
        if filename == '':
            return

        widget.setText(filename)

    def nar_changed(self):
        """
        Value of nar changed.

        Returns
        -------
        None.

        """
        text = self.le_nar.text()
        val = int(text)

        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if i in ['ex', 'ey', 'hz']:
                lay = self.lay2
            else:
                lay = self.lay3
            row, _ = lay.getWidgetPosition(self.le_filnam[i])
            if row == -1:
                continue
            if val < 0:
                self.showrow(row, self.pb_arfilnam[i], self.le_arfilnam[i],
                             lay)
            else:
                self.removerow(self.le_arfilnam[i], lay)

    def nfil_changed(self):
        """
        Value of nfil changed.

        Returns
        -------
        None.

        """
        for i in ['ex', 'ey', 'hz', 'hx', 'hy', 'rx', 'ry']:
            if i in ['ex', 'ey', 'hz']:
                lay = self.lay2
            else:
                lay = self.lay3

            text = self.le_nfil[i].text()
            val = int(text)
            filt = str([1.]*val)[1:-1]

            row, _ = lay.getWidgetPosition(self.le_nfil[i])

            if val > 0:
                self.le_fpar[i].setText(filt)
                self.showrow(row+1, "FPAR: vector of filter parameters",
                             self.le_fpar[i], lay)
                self.removerow(self.le_cpar[i], lay)
            elif val < 0:
                self.showrow(row+1, self.pb_cpar[i],
                             self.le_cpar[i], lay)
                self.removerow(self.le_fpar[i], lay)
            else:
                self.removerow(self.le_cpar[i], lay)
                self.removerow(self.le_fpar[i], lay)

    def imode_changed(self, indx):
        """
        Value of imode changed.

        Parameters
        ----------
        indx : int
            Index.

        Returns
        -------
        None.

        """
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
        """
        Value of jmode changed.

        Returns
        -------
        None.

        """
        row, _ = self.lay.getWidgetPosition(self.cmb_jmode)
        txt = self.cmb_jmode.currentText()

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

            row, _ = lay.getWidgetPosition(self.le_filnam[i])

            if txt == '0 = by points':
                self.removerow(self.le_dstim[i], lay)
                self.removerow(self.le_wstim[i], lay)
                self.removerow(self.le_wetim[i], lay)
            else:
                self.showrow(row+1, "DSTIM: data series start time",
                             self.le_dstim[i], lay)
                self.showrow(row+2, "WSTIM: processing window start time",
                             self.le_wstim[i], lay)
                self.showrow(row+3, "WETIM: processing window end time",
                             self.le_wetim[i], lay)

    def nout_changed(self):
        """
        Value of nout changed.

        Returns
        -------
        None.

        """
        row, _ = self.lay.getWidgetPosition(self.le_c2threshe)

        txt = self.cmb_nout.currentText()
        txt2 = self.cmb_nz.currentText()

# First do NZ
        if txt == '3 = EX, EY, BZ':
            self.showrow(row+1,
                         "NZ: threshold mode for vertical magnetic field",
                         self.cmb_nz, self.lay)
        else:
            self.removerow(self.cmb_nz, self.lay)

# Now do C2threshe1
        if txt == '3 = EX, EY, BZ' and txt2 == '0 = separate from E':
            self.showrow(row+2, "C2THRESHE1: coherence threshold for "
                         "vertical magnetic field",
                         self.le_c2threshe1, self.lay)
        else:
            self.removerow(self.le_c2threshe1, self.lay)

# Now do file stuff
        if txt == '3 = EX, EY, BZ':
            row, _ = self.lay2.getWidgetPosition(self.le_nskip['ey'])

            self.showrow(row+1, "NFIL: number filter parameters "
                         "(<0 for filename) of hz",
                         self.le_nfil['hz'], self.lay2)
            self.showrow(row+2, self.pb_filnam['hz'], self.le_filnam['hz'],
                         self.lay2)
            self.showrow(row+3, "NSKIP: leading values to skip in hz",
                         self.le_nskip['hz'], self.lay2)
        else:
            self.removerow(self.le_nfil['hz'], self.lay2)
            self.removerow(self.le_filnam['hz'], self.lay2)
            self.removerow(self.le_nskip['hz'], self.lay2)

        self.le_nar_changed()
        self.nfil_changed()
        self.cmb_jmode_changed()

    def showrow(self, row, label, widget, lay):
        """
        Show a row within a widget.

        Parameters
        ----------
        row : int
            Row number.
        label : str
            Row label.
        widget : Qt widget.
            Qt widget.
        lay : QtWidgets.QFormLayout
            Form Layout.

        Returns
        -------
        None.

        """
        if lay.getWidgetPosition(widget)[0] == -1:
            lay.insertRow(row, label, widget)
            widget.show()

    def removerow(self, widget, lay):
        """
        Remove a row.

        Parameters
        ----------
        widget : Qt widget.
            Qt widget.
        lay : QtWidgets.QFormLayout
            Form Layout.

        Returns
        -------
        None.

        """
        if lay.getWidgetPosition(widget)[0] > -1:
            widget.hide()
            lay.labelForField(widget).hide()
            lay.takeRow(widget)

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
        if not nodialog:
            tmp = self.exec()

            if tmp != 1:
                return False

        return True

    def saveproj(self):
        """
        Save project data from class.

        Returns
        -------
        None.

        """
