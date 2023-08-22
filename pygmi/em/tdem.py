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
"""Time Domain EM."""

import sys
import os
import copy
from contextlib import redirect_stdout
from PyQt5 import QtWidgets, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import discretize
from SimPEG import (maps, data_misfit, regularization,
                    optimization, inversion, inverse_problem, directives)
from SimPEG.electromagnetics import time_domain
import SimPEG.data as Sdata

from pygmi import menu_default
from pygmi.misc import QLabelVStack, BasicModule


class MyMplCanvas2(FigureCanvasQTAgg):
    """MPL Canvas class."""

    def __init__(self, parent=None):
        fig = Figure(layout='constrained')
        super().__init__(fig)

    def update_line(self, sigma, z, times_off, zobs, zpred):
        """
        Update the plot from data.

        Parameters
        ----------
        sigma : numpy array
            Conductivity values.
        z : numpy array
            Depth values.
        times_off : numpy array
            Time.
        zobs : numpy array
            Observed dBz/dt.
        zpred : numpy array
            Predicted dBz/dt.

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(121, label='Profile')
        ax1.semilogx(sigma, z, 'b', lw=2)
        ax1.grid(True)
        ax1.set_ylabel("Depth (m)")
        ax1.set_xlabel("Conductivity (S/m)")
        ax1.set_title("Recovered Model")

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

        # self.figure.tight_layout()
        self.figure.canvas.draw()

    def disp_wave(self, times, wave, title):
        """
        Display waveform.

        Parameters
        ----------
        times : numpy array
            Times.
        wave : numpy array
            Waveform amplitude.
        title : str
            Title.

        Returns
        -------
        None.

        """
        self.figure.clear()

        ax1 = self.figure.add_subplot(111)
        ax1.grid(True)
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('Time (s)')
        ax1.set_title(title)

        ax1.plot(times, wave)

        # self.figure.tight_layout()
        self.figure.canvas.draw()


class TDEM1D(BasicModule):
    """Occam 1D inversion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.cursoln = 0
        self.times = None

        self.setWindowTitle('TDEM 1D Inversion')
        helpdocs = menu_default.HelpButton('pygmi.em.tdem1d')

        vbl = QtWidgets.QVBoxLayout()
        hbl = QtWidgets.QHBoxLayout(self)
        gbl = QLabelVStack()

        self.mmc1 = MyMplCanvas2(self)
        self.mmc = MyMplCanvas2(self)
        mpl_toolbar = NavigationToolbar2QT(self.mmc, self.parent)

        self.combostype = QtWidgets.QComboBox()
        self.combostype.addItems(['CircularLoop',
                                  'MagDipole'])

        self.combowtype = QtWidgets.QComboBox()
        self.combowtype.addItems(['VTEMWaveform',
                                  'RampOffWaveform',
                                  'TrapezoidWaveform',
                                  'QuarterSineRampOnWaveform',
                                  'TriangularWaveform',
                                  'HalfSineWaveform'])
        self.combotxori = QtWidgets.QComboBox()
        self.combotxori.addItems(['z', 'x', 'y'])
        self.comborxori = QtWidgets.QComboBox()
        self.comborxori.addItems(['z', 'x', 'y'])
        self.comboline = QtWidgets.QComboBox()
        self.combofid = QtWidgets.QComboBox()
        self.combobalt = QtWidgets.QComboBox()
        self.loopturns = QtWidgets.QLineEdit('1.0')
        self.loopcurrent = QtWidgets.QLineEdit('1.0')
        self.mu = QtWidgets.QLineEdit('1.25663706212e-06')
        self.txarea = QtWidgets.QLineEdit('313.98')
        self.txofftime = QtWidgets.QLineEdit('0.0100286')
        self.txrampoff1 = QtWidgets.QLineEdit('0.01')
        self.txpeaktime = QtWidgets.QLineEdit('0.01')
        self.datachan = QtWidgets.QLineEdit('Z_Ch')
        self.sig_half = QtWidgets.QLineEdit('0.1')
        self.sig_air = QtWidgets.QLineEdit('1e-08')
        self.rel_err = QtWidgets.QLineEdit('12')
        self.noise_floor = QtWidgets.QLineEdit('7.5e-12')
        self.wfile = QtWidgets.QLineEdit('')
        self.maxiter = QtWidgets.QLineEdit('5')
        pb_wfile = QtWidgets.QPushButton('Load Window Times')
        pb_wdisp = QtWidgets.QPushButton('Refresh Waveform')

        self.mesh_cs = QtWidgets.QLineEdit('1')
        self.mesh_ncx = QtWidgets.QLineEdit('10')
        self.mesh_ncz = QtWidgets.QLineEdit('10')
        self.mesh_npad = QtWidgets.QLineEdit('20')
        self.mesh_padrate = QtWidgets.QLineEdit('1.3')

        self.lbl_profnum = QtWidgets.QLabel('Solution: 0')

        pb_apply = QtWidgets.QPushButton('Invert Station')

        buttonbox = QtWidgets.QDialogButtonBox()
        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        gbl.addWidget('Line Number:', self.comboline)
        gbl.addWidget(r'Fid/Station Name:', self.combofid)
        gbl.addWidget('Bird Height:', self.combobalt)
        gbl.addWidget('Data channel prefix:', self.datachan)
        gbl.addWidget('Waveform Type:', self.combowtype)
        gbl.addWidget('Tx Peak time:', self.txpeaktime)
        gbl.addWidget('Tx Ramp Off start time:', self.txrampoff1)
        gbl.addWidget('Tx Off time:', self.txofftime)
        gbl.addWidget('Source Type:', self.combostype)
        gbl.addWidget('Tx Orientation:', self.combotxori)
        gbl.addWidget('Tx Area:', self.txarea)
        gbl.addWidget('Number of turns in loop:', self.loopturns)
        gbl.addWidget('Current in loop:', self.loopcurrent)
        gbl.addWidget('Permeability of the background:', self.mu)
        gbl.addWidget('Rx Orientation:', self.comborxori)
        gbl.addWidget('Mesh cell size:', self.mesh_cs)
        gbl.addWidget('Mesh number cells in x direction:', self.mesh_ncx)
        gbl.addWidget('Mesh number cells in z direction:', self.mesh_ncz)
        gbl.addWidget('Mesh number of cells to pad:', self.mesh_npad)
        gbl.addWidget('Pad cell multiplier:', self.mesh_padrate)
        gbl.addWidget('Conductivity Air:', self.sig_air)
        gbl.addWidget('Conductivity Halfspace:', self.sig_half)
        gbl.addWidget('Data Relative Error (%):', self.rel_err)
        gbl.addWidget('Data Noise Floor:', self.noise_floor)
        gbl.addWidget('Optimization - maximum iterations:', self.maxiter)

        gbl.addWidget(pb_wfile, self.wfile)
        gbl.addWidget(helpdocs, pb_apply)

        vbl.addWidget(self.mmc1)
        vbl.addWidget(pb_wdisp)
        vbl.addWidget(self.mmc)
        vbl.addWidget(mpl_toolbar)

        hbl.addLayout(gbl.layout)
        hbl.addLayout(vbl)

        pb_apply.clicked.connect(self.apply)
        pb_wfile.pressed.connect(self.get_wfile)
        pb_wdisp.pressed.connect(self.disp_wave)

        self.comboline.currentIndexChanged.connect(self.change_line)
        self.combowtype.currentIndexChanged.connect(self.disp_wave)
        self.combostype.currentIndexChanged.connect(self.change_source)

        self.disp_wave()

    def apply(self):
        """
        Invert the data.

        Returns
        -------
        None.

        """
        if self.times is None:
            text = 'You need to load window times first.'
            QtWidgets.QMessageBox.warning(self.parent, 'Error', text,
                                          QtWidgets.QMessageBox.Ok)
            return

        self.disp_wave()

        dprefix = (self.datachan.text()).lower()
        line = self.comboline.currentText()
        fid = float(self.combofid.currentText())
        balt = self.combobalt.currentText()
        txarea = float(self.txarea.text())
        offtime = float(self.txofftime.text())
        peaktime = float(self.txpeaktime.text())
        loopturns = float(self.loopturns.text())
        loopcurrent = float(self.loopcurrent.text())
        mu = float(self.mu.text())
        npad = int(self.mesh_npad.text())
        padrate = float(self.mesh_padrate.text())
        cs = float(self.mesh_cs.text())
        ncx = float(self.mesh_ncx.text())
        ncz = float(self.mesh_ncz.text())
        npad = float(self.mesh_npad.text())
        sig_half = float(self.sig_half.text())
        sig_air = float(self.sig_air.text())
        rel_err = float(self.rel_err.text())/100.
        floor = float(self.noise_floor.text())
        maxiter = int(self.maxiter.text())

        txori = self.combotxori.currentText()
        rxori = self.comborxori.currentText()
        stype = self.combostype.currentText()

        times = self.times
        times = times + peaktime

        # Select line
        skytem = self.data[self.data.line.astype(str) == line]
        # Select fid
        skytem = skytem[skytem.fid == fid]

        emdata = skytem.loc[:, skytem.columns.str.startswith(dprefix)]
        datachans = list(emdata.columns.values)
        emdata = emdata.values.flatten()

        if not datachans:
            text = 'Could not find data channels, your prefix may be wrong'
            QtWidgets.QMessageBox.warning(self.parent, 'Error', text,
                                          QtWidgets.QMessageBox.Ok)
            return

        # ------------------ Mesh ------------------ #
        # Step1: Set 2D cylindrical mesh
        # hx and hz are not depths. They are lists of dx and dz.
        # cs is cell width.
        # ncx is number of mesh cells in x (or r in this case)
        # ncz is number ofmesh cells in z
        # npad is number of pad cells,
        # 1.3 is rate padded cells increase in size
        # 00C means mesh starts at 0 in x, y and centered in Z.
        # If N is above it means mesh negative.

        # cs, ncx, ncz, npad = 1., 10., 10., 20
        hx = [(cs, ncx), (cs, npad, padrate)]
        hz = [(cs, npad, -padrate), (cs, ncz*2), (cs, npad, padrate)]
        mesh = discretize.CylindricalMesh([hx, 1, hz], '00C')

        # Step2: Set a SurjectVertical1D mapping
        # Note: this sets our inversion model as 1D log conductivity
        # below subsurface

        active = mesh.cell_centers_z < 0.
        actmap = maps.InjectActiveCells(mesh, active, np.log(1e-8),
                                        nC=mesh.shape_cells[2])
        mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actmap
        sigma = np.ones(mesh.shape_cells[2]) * sig_air
        sigma[active] = sig_half

        # Initial and reference model
        m0 = np.log(sigma[active])

        # ------------------ SkyTEM Forward Simulation ------------------ #
        # Step4: Invert SkyTEM data

        # Bird height from the surface
        src_height = skytem[balt].iloc[0]
        srcloc = np.array([0., 0., src_height])

        # Radius of the source loop
        radius = np.sqrt(txarea/np.pi)
        rxloc = np.array([[radius, 0., src_height]])

        # Parameters for current waveform
        # Note: we are Using theoretical VTEM waveform,
        # but effectively fits SkyTEM waveform

        dbdt_z = time_domain.Rx.PointMagneticFluxTimeDerivative(
            locations=rxloc, times=times, orientation=rxori)  # vertical db_dt
        rxlist = [dbdt_z]  # list of receivers

        wform = self.update_wave()

        if stype == 'CircularLoop':
            srclist = [time_domain.Src.CircularLoop(rxlist,
                                                    N=loopturns,
                                                    mu=mu,
                                                    current=loopcurrent,
                                                    location=srcloc,
                                                    radius=radius,
                                                    orientation=txori,
                                                    waveform=wform)]
        elif stype == 'MagDipole':
            srclist = [time_domain.Src.MagDipole(rxlist,
                                                 location=srcloc,
                                                 mu=mu,
                                                 orientation=txori,
                                                 waveform=wform)]

        # solve the problem at these times
        dtimes = np.diff(times).tolist()

        timesteps = ([peaktime/5]*5 +               # On time section
                     [(offtime-peaktime)/5]*5 +     # Off time section
                     dtimes + [dtimes[-1]])         # Current zero from here

        survey = time_domain.Survey(srclist)
        sim = time_domain.Simulation3DElectricField(mesh,
                                                    time_steps=timesteps,
                                                    sigmaMap=mapping,
                                                    survey=survey)

        src = srclist[0]
        wave = []

        for time in sim.times:
            wave.append(src.waveform.eval(time))
        wave = np.hstack(wave)

        # Observed data
        dobs_sky = emdata * txarea

        # ------------------ SkyTEM Inversion ------------------ #
        # Data Misfit
        data = Sdata.Data(survey, dobs=-dobs_sky, relative_error=rel_err,
                          noise_floor=floor)
        dmisfit = data_misfit.L2DataMisfit(data=data, simulation=sim)

        # Regularization
        regmesh = discretize.TensorMesh([mesh.h[2][mapping.maps[-1].indActive]])
        # reg = regularization.Simple(regmesh, mapping=maps.IdentityMap(regmesh))
        # reg.alpha_s = 1e-2
        # reg.alpha_x = 1.

        # reg = regularization.Tikhonov(regmesh, alpha_s=1e-2)
        reg = regularization.WeightedLeastSquares(regmesh)

        # Optimization
        opt = optimization.InexactGaussNewton(maxIter=maxiter, LSshorten=0.5)

        # statement of the inverse problem
        invprob = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
        invprob.beta = 20.

        # Directives and Inversion Parameters
        target = directives.TargetMisfit()
        # beta = directives.BetaSchedule(coolingFactor=1, coolingRate=2)
        # betaest = directives.BetaEstimate_ByEig(beta0_ratio=1e0)
        # inv = inversion.BaseInversion(invprob, directiveList=[beta, betaest])
        inv = inversion.BaseInversion(invprob, directiveList=[target])

        # inv = inversion.BaseInversion(invprob, directiveList=[target])
        opt.remember('xc')

        # run the inversion
        try:
            with redirect_stdout(self.stdout_redirect):
                mopt_sky = inv.run(m0)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self.parent, 'Error', str(e),
                                          QtWidgets.QMessageBox.Ok)
            return

        dpred_sky = np.array(invprob.dpred)

        sigma = np.repeat(np.exp(mopt_sky), 2, axis=0)
        z = np.repeat(mesh.cell_centers_z[active][1:], 2, axis=0)
        z = np.r_[mesh.cell_centers_z[active][0], z,
                  mesh.cell_centers_z[active][-1]]

        times_off = (times - offtime)*1e6
        zobs = dobs_sky/txarea
        zpred = -dpred_sky/txarea

        self.mmc.update_line(sigma, z, times_off, zobs, zpred)

    def change_source(self):
        """
        Change Source.

        Returns
        -------
        None.

        """
        stype = self.combostype.currentText()

        if stype == 'CircularLoop':
            self.loopcurrent.setEnabled(True)
            self.loopturns.setEnabled(True)
            self.txarea.setEnabled(True)
        elif stype == 'MagDipole':
            self.loopcurrent.setDisabled(True)
            self.loopturns.setDisabled(True)
            self.txarea.setDisabled(True)

    def disp_wave(self):
        """
        Display waveform.

        Returns
        -------
        None.

        """
        offtime = float(self.txofftime.text())
        times = np.linspace(0, offtime, 1000)
        wtype = self.combowtype.currentText()

        wform = self.update_wave()

        if wtype == 'VTEMWaveform':
            title = 'VTEM Waveform'
        elif wtype == 'TrapezoidWaveform':
            title = 'Trapezoid Waveform'

        elif wtype == 'TriangularWaveform':
            title = 'Triangular Waveform'

        elif wtype == 'QuarterSineRampOnWaveform':
            title = 'Quarter Sine Ramp On Waveform'

        elif wtype == 'HalfSineWaveform':
            title = 'Half Sine Waveform'
        elif wtype == 'RampOffWaveform':
            title = 'Ramp Off Waveform'

        wave = [wform.eval(t) for t in times]

        self.mmc1.disp_wave(times, wave, title)

    def update_wave(self):
        """
        Update waveform.

        Returns
        -------
        wform : tdem waveform.
            Waveform.

        """
        starttime = 0.
        offtime = float(self.txofftime.text())
        peaktime = float(self.txpeaktime.text())
        rampoff1 = float(self.txrampoff1.text())

        rampon = np.array([starttime, peaktime])
        rampoff = np.array([rampoff1, offtime])

        wtype = self.combowtype.currentText()

        if wtype == 'VTEMWaveform':
            wform = time_domain.sources.VTEMWaveform(off_time=offtime,
                                                     peak_time=peaktime)
        elif wtype == 'TrapezoidWaveform':
            wform = time_domain.sources.TrapezoidWaveform(ramp_on=rampon,
                                                          ramp_off=rampoff)
        elif wtype == 'TriangularWaveform':
            wform = time_domain.sources.TriangularWaveform(start_time=starttime,
                                                           peak_time=peaktime,
                                                           off_time=offtime)
        elif wtype == 'QuarterSineRampOnWaveform':
            wform = time_domain.sources.QuarterSineRampOnWaveform(
                ramp_on=rampon, ramp_off=rampoff)
        elif wtype == 'HalfSineWaveform':
            wform = time_domain.sources.HalfSineWaveform(ramp_on=rampon,
                                                         ramp_off=rampoff)

        elif wtype == 'RampOffWaveform':
            wform = time_domain.sources.RampOffWaveform(offTime=offtime)

        return wform

    def get_wfile(self, filename=''):
        """
        Get window time filename.

        Parameters
        ----------
        filename : str, optional
            filename (txt). The default is ''.

        Returns
        -------
        None.

        """
        ext = 'Text file (*.txt)'

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        os.chdir(os.path.dirname(filename))
        self.times = np.loadtxt(filename)

        self.wfile.setText(filename)

    def change_line(self):
        """
        Combo to change line.

        Returns
        -------
        None.

        """
        self.combofid.clear()

        line = self.comboline.currentText()
        fid = self.data.fid[self.data.line.astype(str) == line].values.astype(str)
        self.combofid.addItems(fid)

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
        if 'Vector' in self.indata:
            self.data = copy.deepcopy(self.indata['Vector'][0])
        else:
            self.showlog('No line data')
            return False

        self.comboline.currentIndexChanged.disconnect()
        self.comboline.clear()
        self.combofid.clear()
        self.combobalt.clear()

        filt = ((self.data.columns != 'geometry') &
                (self.data.columns != 'line'))

        cnames = list(self.data.columns[filt])

        self.combobalt.addItems(cnames)
        for i, tmp in enumerate(cnames):
            tmp = tmp.lower()
            if ('elev' in tmp or 'alt' in tmp or 'height' in tmp or
                    'radar' in tmp):
                self.combobalt.setCurrentIndex(i)
                break

        lines = list(self.data.line.unique().astype(str))
        self.comboline.addItems(lines)

        self.comboline.setCurrentIndex(0)
        line = self.comboline.currentText()

        fid = self.data.fid[self.data.line.astype(str) == line].values.astype(str)
        self.combofid.addItems(fid)

        self.comboline.currentIndexChanged.connect(self.change_line)

        tmp = self.exec_()

        if tmp != 1:
            return False

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

        return projdata


def tonumber(test, alttext=None):
    """
    Check if something is a number or matches alttext.

    Parameters
    ----------
    test : str
        Text to test.
    alttext : str, optional
        Alternate text to test. The default is None.

    Returns
    -------
    str or int or float
        Returns a lower case version of alttext, or a number.

    """
    if alttext is not None and test.lower() == alttext.lower():
        return test.lower()

    if not test.replace('.', '', 1).isdigit():
        return -999

    if '.' in test:
        return float(test)

    return int(test)


def _testfn():
    """Test routine."""
    from pygmi.vector.iodefs import ImportXYZ

    app = QtWidgets.QApplication(sys.argv)

    # Load in line data
    filename = r'D:\Workdata\PyGMI Test Data\EM\SK655CS_Bookpurnong_ZX_HM_TxInc_newDTM.txt'
    wfile = r'D:\Workdata\PyGMI Test Data\EM\wtimes.txt'

    IO = ImportXYZ()
    IO.filt = 'Tab Delimited (*.txt)'
    IO.ifile = filename
    IO.xchan.setCurrentText('e')
    IO.ychan.setCurrentText('n')
    IO.nodata.setText('-99999')
    IO.settings(True)

    # Run TEM1D routine

    tmp = TDEM1D(None)
    tmp.get_wfile(wfile)

    tmp.indata = IO.outdata

    tmp.settings(True)


if __name__ == "__main__":
    _testfn()
