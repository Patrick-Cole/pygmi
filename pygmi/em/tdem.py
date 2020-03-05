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
import copy
import glob
from PyQt5 import QtWidgets, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from pymatsolver import Pardiso as Solver
from SimPEG import (Mesh, Maps, Utils, DataMisfit, Regularization,
                    Optimization, Inversion, InvProblem, Directives)
import SimPEG.EM as EM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..//..')))
import pygmi.menu_default as menu_default
from pygmi.vector.datatypes import LData


class MyMplCanvas2(FigureCanvas):
    """
    MPL Canvas class.
    """

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

    def update_line(self, sigma, z, times_off, zobs, zpred):
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
#        gs = self.figure.add_gridspec(3, 3)

        ax1 = self.figure.add_subplot(121, label='Profile')
#        self.figure.suptitle(title)
#        ax1.grid(True, 'both')

        ax1.semilogx(sigma, z, 'b', lw=2)
        ax1.set_ylim(-50, 0)
        ax1.grid(True)
        ax1.set_ylabel("Depth (m)")
        ax1.set_xlabel("Conductivity (S/m)")
#        ax1.legend(loc=3)
        ax1.set_title("Recovered Model")


#        ax1.plot(x, res1, 'b.', label=label1)
#        ax1.plot(x, res2, 'r.', label=label2)
#
#        ax1.set_xscale('log')
#        ax1.set_yscale('log')
#        ax1.legend(loc='upper left')
#        ax1.set_xlabel('Period (s)')
#        ax1.set_ylabel(r'App. Res. ($\Omega.m$)')

        ax2 = self.figure.add_subplot(122)
        ax2.grid(True, 'both')


        ax2.loglog(times_off, zobs, 'b-', label="Observed")
        ax2.loglog(times_off, zpred, 'bo', ms=4,
                   markeredgecolor='k', markeredgewidth=0.5, label="Predicted")

        ax2.set_xlim(times_off.min()*1.2, times_off.max()*1.1)

        ax2.set_xlabel(r"Time ($\mu s$)")
        ax2.set_ylabel("dBz / dt (V/A-m$^4$)")
        ax2.set_title("High-moment")
        ax2.grid(True)
        ax2.legend(loc=3)

#
#        ax2.set_ylim(-180., 180.)
#
#        ax2.set_xscale('log')
#        ax2.set_yscale('linear')
#        ax2.set_xlabel('Period (s)')
#        ax2.set_ylabel(r'Phase (Degrees)')

#        gs.tight_layout(self.figure)
        self.figure.tight_layout()
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

        self.comboline = QtWidgets.QComboBox()
        self.combofid = QtWidgets.QComboBox()
        self.combobalt = QtWidgets.QComboBox()
        self.txarea = QtWidgets.QLineEdit('313.98')
        self.txarea.setSizePolicy(sizepolicy)
        self.txofftime = QtWidgets.QLineEdit('0.0100286')
        self.txofftime.setSizePolicy(sizepolicy)
        self.txpeaktime = QtWidgets.QLineEdit('0.01')
        self.txpeaktime.setSizePolicy(sizepolicy)
        self.datachan = QtWidgets.QLineEdit('Z_Ch')
#        self.errfloorphase.setSizePolicy(sizepolicy)
#
#        self.targetdepth = QtWidgets.QLineEdit('40000.')
#        self.targetdepth.setSizePolicy(sizepolicy)
#        self.nlayers = QtWidgets.QLineEdit('100')
#        self.nlayers.setSizePolicy(sizepolicy)
#        self.bottomlayer = QtWidgets.QLineEdit('100000.')
#        self.bottomlayer.setSizePolicy(sizepolicy)
#        self.airlayer = QtWidgets.QLineEdit('10000.')
#        self.airlayer.setSizePolicy(sizepolicy)
#        self.z1layer = QtWidgets.QLineEdit('10.')
#        self.z1layer.setSizePolicy(sizepolicy)
#        self.maxiter = QtWidgets.QLineEdit('200')
#        self.maxiter.setSizePolicy(sizepolicy)
#        self.targetrms = QtWidgets.QLineEdit('1.')
#        self.targetrms.setSizePolicy(sizepolicy)

        label1 = QtWidgets.QLabel('Line Number:')
        label1.setSizePolicy(sizepolicy)
        label2 = QtWidgets.QLabel(r'Fid/Station Name:')
        label3 = QtWidgets.QLabel('Bird Height:')
        label4 = QtWidgets.QLabel('Tx Area:')
        label5 = QtWidgets.QLabel('Tx Off time:')
        label6 = QtWidgets.QLabel('Tx Peak Time:')
        label7 = QtWidgets.QLabel('Data channel prefix:')
#        label8 = QtWidgets.QLabel('Height of air layer:')
#        label9 = QtWidgets.QLabel('Bottom of model:')
#        label10 = QtWidgets.QLabel('Depth of target to investigate:')
#        label11 = QtWidgets.QLabel('Depth of first layer:')
#        label12 = QtWidgets.QLabel('Number of layers:')
#        label13 = QtWidgets.QLabel('Maximum Iterations:')
#        label14 = QtWidgets.QLabel('Target RMS:')

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        pb_apply = QtWidgets.QPushButton('Invert Station')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gbl.addWidget(label1, 0, 0)
        gbl.addWidget(self.comboline, 0, 1)
        gbl.addWidget(label2, 1, 0)
        gbl.addWidget(self.combofid, 1, 1)
        gbl.addWidget(label3, 2, 0)
        gbl.addWidget(self.combobalt, 2, 1)
        gbl.addWidget(label4, 3, 0)
        gbl.addWidget(self.txarea, 3, 1)
        gbl.addWidget(label5, 5, 0)
        gbl.addWidget(self.txofftime, 5, 1)
        gbl.addWidget(label6, 4, 0)
        gbl.addWidget(self.txpeaktime, 4, 1)
        gbl.addWidget(label7, 6, 0)
        gbl.addWidget(self.datachan, 6, 1)
#        gbl.addWidget(label8, 7, 0)
#        gbl.addWidget(self.airlayer, 7, 1)
#        gbl.addWidget(label9, 8, 0)
#        gbl.addWidget(self.bottomlayer, 8, 1)
#        gbl.addWidget(label10, 9, 0)
#        gbl.addWidget(self.targetdepth, 9, 1)
#        gbl.addWidget(label11, 10, 0)
#        gbl.addWidget(self.z1layer, 10, 1)
#        gbl.addWidget(label12, 11, 0)
#        gbl.addWidget(self.nlayers, 11, 1)
#        gbl.addWidget(label13, 12, 0)
#        gbl.addWidget(self.maxiter, 12, 1)
#        gbl.addWidget(label14, 13, 0)
#        gbl.addWidget(self.targetrms, 13, 1)

        gbl.addWidget(pb_apply, 14, 0, 1, 2)

        hbl2.addWidget(helpdocs)
        hbl2.addWidget(self.lbl_profnum)

        vbl.addWidget(self.mmc)
        vbl.addLayout(hbl2)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gbl)
        hbl.addLayout(vbl)

        pb_apply.clicked.connect(self.apply)
        self.comboline.currentIndexChanged.connect(self.change_line)

    def apply(self):
        """
        Invert the data.

        Returns
        -------
        None.

        """

        dprefix = (self.datachan.text()).lower()
        line = self.comboline.currentText()
        fid = float(self.combofid.currentText())
        balt = self.combobalt.currentText()
        txarea = float(self.txarea.text())
        offTime = float(self.txofftime.text())
        peakTime = float(self.txpeaktime.text())

        line = '2012.0'
        fid = 964.

        times = np.array([4.7000e-05, 5.9800e-05, 7.2600e-05, 8.8600e-05,
                          1.1180e-04, 1.4540e-04, 1.8520e-04, 2.3440e-04,
                          2.9520e-04, 3.7060e-04, 4.6440e-04, 5.8140e-04,
                          7.2780e-04, 9.1120e-04, 1.1170e-03, 1.4292e-03,
                          1.7912e-03, 2.2460e-03, 2.8174e-03, 3.5356e-03,
                          4.4388e-03, 5.5750e-03, 7.0000e-03, 8.8000e-03])
        times = times + peakTime
#        times = times + offTime
        a = 3.

        skytem = self.data.data[line][self.data.data[line]['fid'] == fid]
        channels = skytem.dtype.names
        datachans = []
        dchannum = []
        emdata = np.array([])
        for i in channels:
            if i.startswith(dprefix):
                datachans.append(i)
                dchannum.append(int(''.join(filter(str.isdigit, i))))
                emdata = np.append(emdata, skytem[i])

        if not datachans:
            text = 'Could not find data channels, your prefix may be wrong'
            QtWidgets.QMessageBox.warning(self.parent, 'Error', text,
                                          QtWidgets.QMessageBox.Ok)
            return

        # ------------------ Mesh ------------------ #
        # Step1: Set 2D cylindrical mesh
        cs, ncx, _, npad = 1., 10., 10., 20
        hx = [(cs, ncx), (cs, npad, 1.3)]
        npad = 12
        temp = np.logspace(np.log10(1.), np.log10(12.), 19)
        temp_pad = temp[-1] * 1.3 ** np.arange(npad)
        hz = np.r_[temp_pad[::-1], temp[::-1], temp, temp_pad]
        mesh = Mesh.CylMesh([hx, 1, hz], '00C')
        active = mesh.vectorCCz < 0.

        # Step2: Set a SurjectVertical1D mapping
        # Note: this sets our inversion model as 1D log conductivity
        # below subsurface

        active = mesh.vectorCCz < 0.
        actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8),
                                        nC=mesh.nCz)
        mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap
        sig_half = 1e-1
        sig_air = 1e-8
        sigma = np.ones(mesh.nCz)*sig_air
        sigma[active] = sig_half

        # Initial and reference model
        m0 = np.log(sigma[active])

        # ------------------ SkyTEM Forward Simulation ------------------ #
        # Step4: Invert SkyTEM data

        # Bird height from the surface
#        b_height_skytem = skytem["src_elevation"][()]
#        src_height = b_height_skytem[rxind_skytem]

        src_height = skytem[balt][0]
        srcLoc = np.array([0., 0., src_height])

        # Radius of the source loop
        area = txarea
        radius = np.sqrt(area/np.pi)
        rxLoc = np.array([[radius, 0., src_height]])

        # Parameters for current waveform
        # Note: we are Using theoretical VTEM waveform,
        # but effectively fits SkyTEM waveform

        dbdt_z = EM.TDEM.Rx.Point_dbdt(locs=rxLoc, times=times,
                                       orientation='z')  # vertical db_dt
        rxList = [dbdt_z]  # list of receivers
        wform = EM.TDEM.Src.VTEMWaveform(offTime=offTime, peakTime=peakTime,
                                         a=a)
        srcList = [EM.TDEM.Src.CircularLoop(rxList, loc=srcLoc, radius=radius,
                                            orientation='z', waveform=wform)]

        # solve the problem at these times
#        timeSteps = [(peakTime/5, 5),            # On time section
#                     ((offTime-peakTime)/5, 5),  # Off time section
#                     (1e-5, 5),
#                     (5e-5, 5),
#                     (1e-4, 10),
#                     (5e-4, 15)]

        dtimes = np.diff(times).tolist()
        timeSteps = ([peakTime/5]*5 + [(offTime-peakTime)/5]*5 +
                     dtimes + [dtimes[-1]])

        prob = EM.TDEM.Problem3D_e(mesh, timeSteps=timeSteps, sigmaMap=mapping,
                                   Solver=Solver)
        survey = EM.TDEM.Survey(srcList)
        prob.pair(survey)

        src = srcList[0]
        wave = []
        for time in prob.times:
            wave.append(src.waveform.eval(time))
        wave = np.hstack(wave)

        # Observed data
        dobs_sky = emdata * area

        # ------------------ SkyTEM Inversion ------------------ #
        # Uncertainty
        std = 0.12
        floor = 7.5e-12
        uncert = abs(dobs_sky) * std + floor

        # Data Misfit
        survey.dobs = -dobs_sky
        dmisfit = DataMisfit.l2_DataMisfit(survey)
        uncert = std*abs(dobs_sky) + floor
        dmisfit.W = Utils.sdiag(1./uncert)

        # Regularization
        regMesh = Mesh.TensorMesh([mesh.hz[mapping.maps[-1].indActive]])
        reg = Regularization.Simple(regMesh, mapping=Maps.IdentityMap(regMesh))

        # Optimization
        opt = Optimization.InexactGaussNewton(maxIter=5)

        # statement of the inverse problem
        invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

        # Directives and Inversion Parameters
        target = Directives.TargetMisfit()
        # betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
        invProb.beta = 20.
        inv = Inversion.BaseInversion(invProb, directiveList=[target])
        reg.alpha_s = 1e-1
        reg.alpha_x = 1.
        opt.LSshorten = 0.5
        opt.remember('xc')
    #    reg.mref = mopt_re  # Use RESOLVE model as a reference model

        # run the inversion
        mopt_sky = inv.run(m0)
        dpred_sky = invProb.dpred

        sigma = np.repeat(np.exp(mopt_sky), 2, axis=0)
        z = np.repeat(mesh.vectorCCz[active][1:], 2, axis=0)
        z = np.r_[mesh.vectorCCz[active][0], z, mesh.vectorCCz[active][-1]]

        times_off = ((times - offTime)*1e6)
        zobs = dobs_sky/area
        zpred = -dpred_sky/area

        self.mmc.update_line(sigma, z, times_off, zobs, zpred)

#        self.outdata['Line'] = self.data

    def reset_data(self):
        """
        Reset data.

        Returns
        -------
        None.

        """

    def change_line(self):
        """
        Combo to change line.

        Returns
        -------
        None.

        """

        self.combofid.clear()

        line = self.comboline.currentText()
        self.combofid.addItems(self.data.data[line]['fid'].astype(str))

    def update_plot(self):
        """
        Update the plot

        Returns
        -------
        None.

        """

#        self.mmc.update_line(x, pdata, rdata, depths, res, title)

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        if 'Line' in self.indata:
            self.data = copy.deepcopy(self.indata['Line'])
        else:
            print('No line data')
            return False

        self.comboline.currentIndexChanged.disconnect()
        self.comboline.clear()
        self.combofid.clear()
        self.combobalt.clear()

        cnames = list(self.data.data.values())[0].dtype.names
        self.combobalt.addItems(cnames)
        for i, tmp in enumerate(cnames):
            tmp = tmp.lower()
            if 'elev' in tmp or 'alt' in tmp or 'height' in tmp or 'radar' in tmp:
                self.combobalt.setCurrentIndex(i)
                break

        for i in self.data.data:
            self.comboline.addItem(i)

        self.comboline.setCurrentIndex(0)
        line = self.comboline.currentText()
        self.combofid.addItems(self.data.data[line]['fid'].astype(str))

        self.comboline.currentIndexChanged.connect(self.change_line)

        tmp = self.exec_()

        if tmp != 1:
            return False

#        self.acceptall()

        return True


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


def testrun():
    """ Test routine """
    app = QtWidgets.QApplication(sys.argv)

    # Load in line data
    filename = r'C:\Work\Programming\EM\bookpurnong\SK655CS_Bookpurnong_ZX_HM_TxInc_newDTM.txt'

    xcol = 'E'
    ycol = 'N'
    nodata = -99999

    dat = []
    with open(filename) as fno:
        head = fno.readline()
        tmp = fno.read()

    head = head.split()
    head = [i.lower() for i in head]
    tmp = tmp.lower()

    dtype = {}
    dtype['names'] = head
    dtype['formats'] = ['f4']*len(head)

    dat = {}
    tmp = tmp.split('\n')
    tmp2 = np.genfromtxt(tmp, names=head, delimiter='\t')
    lines = np.unique(tmp2['line'])

    for i in lines:
        dat[str(i)] = tmp2[tmp2['line'] == i]

    dat2 = LData()
    dat2.xchannel = xcol
    dat2.ychannel = ycol
    dat2.data = dat
    dat2.nullvalue = nodata

    # Run TEM1D routine

    tmp = TDEM1D(None)
    tmp.indata['Line'] = dat2
    tmp.settings()


if __name__ == "__main__":
    testrun()
