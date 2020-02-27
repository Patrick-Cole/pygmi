# -----------------------------------------------------------------------------
# Name:        tdem.py (part of PyGMI)
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
"""Time Domain EM. Currently just testing stuff"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..//..')))
import copy
import glob
from PyQt5 import QtWidgets, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import mtpy.modeling.occam1d as occam1d
import pygmi.menu_default as menu_default


class MyMplCanvas2(FigureCanvas):
    """
    MPL Canvas class.
    """

    def __init__(self, parent=None):
        fig = Figure()
        FigureCanvas.__init__(self, fig)

    def update_line(self, x, pdata, rdata, depths=None, res=None, title=None):
        """
        Update the plot from data.

        Parameters
        ----------
        x : numpy array
            X coordinates (period).
        pdata : numpy array
            Phase data.
        rdata : numpy array
            Apperent resistivity data.
        depths : numpy array, optional
            Model depths. The default is None.
        res : numpy array, optional
            Resistivities. The default is None.
        title : str or None, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 3)

        ax1 = self.figure.add_subplot(gs[:2, :2], label='Profile')
        self.figure.suptitle(title)
        ax1.grid(True, 'both')

        res1 = rdata[0]
        res2 = rdata[1]
        pha1 = pdata[0]
        pha2 = pdata[1]
        label1 = r'Measured'
        label2 = r'Modelled'

        ax1.plot(x, res1, 'b.', label=label1)
        ax1.plot(x, res2, 'r.', label=label2)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Period (s)')
        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')

        ax2 = self.figure.add_subplot(gs[2:, :2], sharex=ax1)
        ax2.grid(True, 'both')

        ax2.plot(x, pha1, 'b.')
        ax2.plot(x, pha2, 'r.')

        ax2.set_ylim(-180., 180.)

        ax2.set_xscale('log')
        ax2.set_yscale('linear')
        ax2.set_xlabel('Period (s)')
        ax2.set_ylabel(r'Phase (Degrees)')

        ax3 = self.figure.add_subplot(gs[:, 2])
        ax3.grid(True, 'both')
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.set_xlabel(r'Res. ($\Omega.m$)')
        ax3.set_ylabel(r'Depth (km)')

        if depths is not None:
            ax3.plot(res, np.array(depths)/1000)

        gs.tight_layout(self.figure)
        self.figure.canvas.draw()


class TDEM1D(QtWidgets.QDialog):
    """Occam 1D inversion."""

    def __init__(self, parent):
        super().__init__(parent)
        self.indata = {}
        self.outdata = {}
        self.data = None
        self.parent = parent
        self.cursoln = 0

        self.setWindowTitle('TDEM 1D Inversion')
        helpdocs = menu_default.HelpButton('pygmi.em.occam1d')

        sizepolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                           QtWidgets.QSizePolicy.Fixed)

        vbl = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout(self)
        hbl2 = QtWidgets.QHBoxLayout()
        gbl = QtWidgets.QGridLayout()
        gbl.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.mmc = MyMplCanvas2(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combobox1 = QtWidgets.QComboBox()
        self.combobox2 = QtWidgets.QComboBox()
        self.combomode = QtWidgets.QComboBox()
        self.combomode.addItems(['TE', 'TM', 'DET'])
        self.combomode.setCurrentIndex(0)
        self.errres = QtWidgets.QLineEdit('data')
        self.errres.setSizePolicy(sizepolicy)
        self.errphase = QtWidgets.QLineEdit('data')
        self.errphase.setSizePolicy(sizepolicy)
        self.errfloorres = QtWidgets.QLineEdit('4.')
        self.errfloorres.setSizePolicy(sizepolicy)
        self.errfloorphase = QtWidgets.QLineEdit('2.')
        self.errfloorphase.setSizePolicy(sizepolicy)

        self.targetdepth = QtWidgets.QLineEdit('40000.')
        self.targetdepth.setSizePolicy(sizepolicy)
        self.nlayers = QtWidgets.QLineEdit('100')
        self.nlayers.setSizePolicy(sizepolicy)
        self.bottomlayer = QtWidgets.QLineEdit('100000.')
        self.bottomlayer.setSizePolicy(sizepolicy)
        self.airlayer = QtWidgets.QLineEdit('10000.')
        self.airlayer.setSizePolicy(sizepolicy)
        self.z1layer = QtWidgets.QLineEdit('10.')
        self.z1layer.setSizePolicy(sizepolicy)
        self.maxiter = QtWidgets.QLineEdit('200')
        self.maxiter.setSizePolicy(sizepolicy)
        self.targetrms = QtWidgets.QLineEdit('1.')
        self.targetrms.setSizePolicy(sizepolicy)

        label1 = QtWidgets.QLabel('Station Name:')
        label1.setSizePolicy(sizepolicy)
        label3 = QtWidgets.QLabel('Mode:')
        label4 = QtWidgets.QLabel('Resistivity Errorbar (Data or %):')
        label5 = QtWidgets.QLabel('Phase Errorbar (Data or %):')
        label6 = QtWidgets.QLabel('Resistivity Error Floor (%):')
        label7 = QtWidgets.QLabel('Phase Error Floor (degrees):')
        label8 = QtWidgets.QLabel('Height of air layer:')
        label9 = QtWidgets.QLabel('Bottom of model:')
        label10 = QtWidgets.QLabel('Depth of target to investigate:')
        label11 = QtWidgets.QLabel('Depth of first layer:')
        label12 = QtWidgets.QLabel('Number of layers:')
        label13 = QtWidgets.QLabel('Maximum Iterations:')
        label14 = QtWidgets.QLabel('Target RMS:')

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        pb_apply = QtWidgets.QPushButton('Invert Station')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gbl.addWidget(label1, 0, 0)
        gbl.addWidget(self.combobox1, 0, 1)
        gbl.addWidget(label3, 2, 0)
        gbl.addWidget(self.combomode, 2, 1)
        gbl.addWidget(label4, 3, 0)
        gbl.addWidget(self.errres, 3, 1)
        gbl.addWidget(label5, 4, 0)
        gbl.addWidget(self.errphase, 4, 1)
        gbl.addWidget(label6, 5, 0)
        gbl.addWidget(self.errfloorres, 5, 1)
        gbl.addWidget(label7, 6, 0)
        gbl.addWidget(self.errfloorphase, 6, 1)
        gbl.addWidget(label8, 7, 0)
        gbl.addWidget(self.airlayer, 7, 1)
        gbl.addWidget(label9, 8, 0)
        gbl.addWidget(self.bottomlayer, 8, 1)
        gbl.addWidget(label10, 9, 0)
        gbl.addWidget(self.targetdepth, 9, 1)
        gbl.addWidget(label11, 10, 0)
        gbl.addWidget(self.z1layer, 10, 1)
        gbl.addWidget(label12, 11, 0)
        gbl.addWidget(self.nlayers, 11, 1)
        gbl.addWidget(label13, 12, 0)
        gbl.addWidget(self.maxiter, 12, 1)
        gbl.addWidget(label14, 13, 0)
        gbl.addWidget(self.targetrms, 13, 1)

        gbl.addWidget(pb_apply, 14, 0, 1, 2)
        gbl.addWidget(buttonbox, 15, 0, 1, 2)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(self.lbl_profnum)

        vbl.addWidget(self.mmc)
        vbl.addLayout(hbl2)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gbl)
        hbl.addLayout(vbl)

        pb_apply.clicked.connect(self.apply)
        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        self.combobox1.currentIndexChanged.connect(self.change_band)

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """
        self.outdata['MT - EDI'] = self.data

    def apply(self):
        """
        Apply.

        Returns
        -------
        None.

        """
        parm = {}

        parm['tdepth'] = tonumber(self.targetdepth.text())
        parm['nlayers'] = tonumber(self.nlayers.text())
        parm['blayer'] = tonumber(self.bottomlayer.text())
        parm['alayer'] = tonumber(self.airlayer.text())
        parm['z1layer'] = tonumber(self.z1layer.text())
        parm['miter'] = tonumber(self.maxiter.text())
        parm['trms'] = tonumber(self.targetrms.text())
        parm['rerr'] = tonumber(self.errres.text(), 'data')
        parm['perr'] = tonumber(self.errphase.text(), 'data')
        parm['perrflr'] = tonumber(self.errfloorphase.text())
        parm['rerrflr'] = tonumber(self.errfloorres.text())

        if -999 in parm.values():
            return

        mode = self.combomode.currentText()
        i = self.combobox1.currentText()
        edi_file = self.data[i].fn

        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            r = glob.glob(save_path+r'\*')
            for i in r:
                os.remove(i)
        else:
            os.makedirs(save_path)

        d1 = occam1d.Data()
        d1.write_data_file(edi_file=edi_file,
                           mode=mode,
                           save_path=save_path,
                           res_err=parm['rerr'],
                           phase_err=parm['perr'],
                           res_errorfloor=parm['rerrflr'],
                           phase_errorfloor=parm['perrflr'],
                           remove_outofquadrant=True
                           )

        m1 = occam1d.Model(target_depth=parm['tdepth'],
                           n_layers=parm['nlayers'],
                           bottom_layer=parm['blayer'],
                           z1_layer=parm['z1layer'],
                           air_layer_height=parm['alayer']
                           )
        m1.write_model_file(save_path=d1.save_path)

        s1 = occam1d.Startup(data_fn=d1.data_fn,
                             model_fn=m1.model_fn,
                             max_iter=parm['miter'],
                             target_rms=parm['trms'])

        s1.write_startup_file()

        self.mmc.figure.clear()
        self.mmc.figure.set_facecolor('r')
        self.mmc.figure.suptitle('Busy, please wait...', fontsize=14, y=0.5)
#        ax = self.mmc.figure.gca()
#        ax.text(0.5, 0.5, 'Busy, please wait...')
        self.mmc.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        occam_path = os.path.dirname(__file__)[:-2]+r'\bin\occam1d.exe'

        occam1d.Run(s1.startup_fn, occam_path, mode='TE')

        self.mmc.figure.set_facecolor('w')

        allfiles = glob.glob(save_path+r'\*.resp')
        self.hs_profnum.setMaximum(len(allfiles))
        self.hs_profnum.setMinimum(1)

        self.change_band()

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        self.data[i] = copy.deepcopy(self.indata['MT - EDI'][i])
        self.change_band()

    def change_band(self):
        """
        Combo to change band.

        Returns
        -------
        None.

        """
        i = self.combobox1.currentText()
        mode = self.combomode.currentText()
        n = self.hs_profnum.value()

        edi_file = self.data[i].fn
        save_path = edi_file[:-4]+'-'+mode

        if not os.path.exists(save_path):
            return
        if os.path.exists(save_path):
            r = glob.glob(save_path+r'\*.resp')
            if len(r) == 0:
                return

        iterfn = os.path.join(save_path, mode+'_'+f'{n:03}'+'.iter')
        respfn = os.path.join(save_path, mode+'_'+f'{n:03}'+'.resp')
        model_fn = os.path.join(save_path, 'Model1D')
        data_fn = os.path.join(save_path, 'Occam1d_DataFile_'+mode+'.dat')

        oc1m = occam1d.Model(model_fn=model_fn)
        oc1m.read_iter_file(iterfn)

        oc1d = occam1d.Data(data_fn=data_fn)
        oc1d.read_resp_file(respfn)

        rough = float(oc1m.itdict['Roughness Value'])
        rms = float(oc1m.itdict['Misfit Value'])
        rough = f'{rough:.1f}'
        rms = f'{rms:.1f}'

        title = 'RMS: '+rms+'    Roughness: '+rough

        depths = []
        res = []

        for i, val in enumerate(oc1m.model_res[:, 1]):
            if i == 0:
                continue
            if i > 1:
                depths.append(-oc1m.model_depth[i-1])
                res.append(val)

            depths.append(-oc1m.model_depth[i])
            res.append(val)

        x = 1/oc1d.freq
        rdata = [oc1d.data['resxy'][0], oc1d.data['resxy'][2]]
        pdata = [oc1d.data['phasexy'][0], oc1d.data['phasexy'][2]]

        self.mmc.update_line(x, pdata, rdata, depths, res, title)

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'MT - EDI' in self.indata:
            self.data = copy.deepcopy(self.indata['MT - EDI'])
        else:
            print('No EDI data')
            return False

        self.combobox1.currentIndexChanged.disconnect()
        self.combobox1.clear()

        for i in self.data:
            self.combobox1.addItem(i)

        self.combobox1.setCurrentIndex(0)
        self.combobox1.currentIndexChanged.connect(self.change_band)

        i = self.combobox1.currentText()
        mode = self.combomode.currentText()
        edi_file = self.data[i].fn
        save_path = edi_file[:-4]+'-'+mode

        if os.path.exists(save_path):
            allfiles = glob.glob(save_path+r'\*.resp')
            if len(allfiles) > 0:
                self.hs_profnum.setMaximum(len(allfiles))
                self.hs_profnum.setMinimum(1)

        self.change_band()

        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp


def tonumber(test, alttext=None):
    """
    Checks if something is a number or matches alttext

    Parameters
    ----------
    test : str
        Text to test.
    alttext : str, optional
        Alternate text to test. The default is None.

    Returns
    -------
    str or int or float
        Returns a lower case verion of alttext, or a number.

    """
    if alttext is not None and test.lower() == alttext.lower():
        return test.lower()

    if not test.replace('.', '', 1).isdigit():
        return -999

    if '.' in test:
        return float(test)

    return int(test)


def test():
    """ Test routine """
    import sys
    import pandas as pd
    from pygmi.vector.datatypes import PData

    from pygmi.vector import iodefs
    app = QtWidgets.QApplication(sys.argv)

    filename = r'C:\Work\Programming\EM\bookpurnong\SK655CS_Bookpurnong_ZX_HM_TxInc_newDTM.txt'

    xcol = 'E'
    ycol = 'N'
    nodata = -99999

    dat = []
    with open(filename) as fno:
        head = fno.readline()
        tmp = fno.read()

    head = head.split()
    tmp = tmp.lower()

    dtype = {}
    dtype['names'] = head
    dtype['formats'] = ['f4']*len(head)

    dat = {}
    tmp = tmp.split('\n')
    aaa = tmp[0].split('\t')
    breakpoint()
    tmp2 = np.genfromtxt(tmp, names=head)
    breakpoint()

    for i in range(0, len(tmp), 2):
        tmp2 = tmp[i+1]
        tmp2 = tmp2.split('\n')
        line = tmp[i]+tmp2.pop(0)
        tmp2 = np.genfromtxt(tmp2, names=head)
        dat[line] = tmp2


    dat2 = LData()
    dat2.xchannel = xcol
    dat2.ychannel = ycol
    dat2.data = dat
    dat2.nullvalue = nodata




    tmp = TDEM1D(None)
    tmp.indata['Point'] = dat
    tmp.settings()





if __name__ == "__main__":
    test()
