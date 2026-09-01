# -----------------------------------------------------------------------------
# Name:        ratios.py (part of PyGMI)
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
"""Calculate remote sensing ratios and condition indices."""

import os
import re
import sys
from collections.abc import Callable, Iterable

import numexpr as ne
import numpy as np
from PySide6 import QtWidgets

from pygmi.misc import BasicModule
from pygmi.raster.datatypes import Data
from pygmi.raster.iodefs import export_raster
from pygmi.raster.misc import lstack
from pygmi.rsense.iodefs import get_from_rastermeta, set_export_filename


class SatRatios(BasicModule):
    """
    GUI to calculate satellite ratios.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmb_sensor = QtWidgets.QComboBox()
        self.lw_ratios = QtWidgets.QListWidget()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        self.buttonbox.htmlfile = "rsense.dm.ratios"
        gl_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton("Invert Selection")
        lbl_sensor = QtWidgets.QLabel("Sensor:")
        lbl_ratios = QtWidgets.QLabel("Ratios:")

        self.lw_ratios.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )

        self.cmb_sensor.addItems(
            [
                "ASTER",
                "Landsat 8 and 9 (OLI)",
                "Landsat 7 (ETM+)",
                "Landsat 4 and 5 (TM)",
                "Sentinel-2",
                "WorldView",
                "EMIT",
                "Unknown",
            ]
        )

        self.setWindowTitle("Band Ratio Calculations")

        gl_main.addWidget(lbl_sensor, 0, 0, 1, 1)
        gl_main.addWidget(self.cmb_sensor, 0, 1, 1, 1)
        gl_main.addWidget(lbl_ratios, 1, 0, 1, 1)
        gl_main.addWidget(self.lw_ratios, 1, 1, 1, 1)
        gl_main.addWidget(btn_invert, 2, 0, 1, 2)

        gl_main.addWidget(self.buttonbox, 6, 0, 1, 2)

        # self.lw_ratios.clicked.connect(self.set_selected_ratios)
        self.cmb_sensor.currentIndexChanged.connect(self.setratios)
        btn_invert.clicked.connect(self.invert_selection)

    def settings(self, nodialog: bool = False) -> bool:
        """
        Entry point into item.

        Parameters
        ----------
        nodialog
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if "Raster" not in self.indata and "RasterFileList" not in self.indata:
            self.showlog("No Satellite Data")
            return False

        if "RasterFileList" in self.indata:
            dat = self.indata["RasterFileList"]
            instr = dat[0].sensor
        else:
            dat = self.indata["Raster"]
            instr = dat[0].metadata["Raster"]["Sensor"]

        if "ASTER" in instr:
            self.cmb_sensor.setCurrentText("ASTER")
        elif "LC08" in instr or "LC09" in instr:
            self.cmb_sensor.setCurrentText("Landsat 8 and 9 (OLI)")
        elif "LE07" in instr:
            self.cmb_sensor.setCurrentText("Landsat 7 (ETM+)")
        elif "LT04" in instr or "LT05" in instr:
            self.cmb_sensor.setCurrentText("Landsat 4 and 5 (TM)")
        elif "WorldView" in instr and "Multi" in instr:
            self.cmb_sensor.setCurrentText("WorldView")
        elif "Sentinel-2" in instr:
            self.cmb_sensor.setCurrentText("Sentinel-2")
        elif "EMIT" in instr:
            self.cmb_sensor.setCurrentText("EMIT")
        else:
            self.cmb_sensor.setCurrentText("Unknown")

        if self.lw_ratios.count() == 0:
            self.setratios()

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.cmb_sensor)
        self.saveobj(self.lw_ratios)

    def acceptall(self) -> bool:
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        sensor = self.cmb_sensor.currentText()

        if "RasterFileList" in self.indata:
            flist = self.indata["RasterFileList"]
            if sensor == "ASTER":
                flist = get_aster_list(flist)
            elif "Landsat" in sensor:
                flist = get_landsat_list(flist, sensor)
            elif "Sentinel-2" in sensor:
                flist = get_sentinel_list(flist)
            elif "EMIT" in sensor:
                flist = get_EMIT_list(flist)
            if not flist:
                self.showlog(
                    "Warning: This might not be "
                    + sensor
                    + " data. Will attempt to do calculation "
                    "anyway."
                )
                flist = self.indata["RasterFileList"]
        else:
            flist = [self.indata["Raster"]]

        rlist = []
        for i in self.lw_ratios.selectedItems():
            rlist.append(i.text())

        if not rlist:
            self.showlog("You need to select a ratio to calculate.")
            return False

        for ifile in flist:
            if "RasterFileList" in self.indata:
                dat = get_from_rastermeta(ifile, piter=self.piter, showlog=self.showlog)
            else:
                dat = ifile

            if dat is None:
                continue

            if sensor == "EMIT":
                dat = correct_EMIT_bands(rlist, dat)

            datfin = calc_ratios(
                dat, rlist, showlog=self.showlog, piter=self.piter, sensor=sensor
            )

            if datfin:
                odir = os.path.dirname(dat[0].filename)
                odir = os.path.join(odir, "ratios")

                os.makedirs(odir, exist_ok=True)

                ofile = set_export_filename(dat, odir, "ratio")

                self.showlog("Exporting to " + ofile)
                export_raster(
                    ofile,
                    datfin,
                    drv="GTiff",
                    piter=self.piter,
                    compression="DEFLATE",
                    showlog=self.showlog,
                )
                self.outdata["Raster"] = datfin

        return True

    def setratios(self):
        """
        Set the available ratios.

        The ratio definitions are for the ASTER satellite. Band 0 refers to
        an imaginary blue band.

        """
        sensor = self.cmb_sensor.currentText()

        rlist = []

        # carbonates/mafic minerals bands
        rlist += [
            r"(B7+B9)/B8 carbonate chlorite epidote",
            r"(B6+B9)/(B7+B8) epidote chlorite amphibole",
            r"(B6+B9)/B8 amphibole MgOH",
            r"B6/B8 amphibole",
            r"(B6+B8)/B7 dolomite",
            r"B13/B14 carbonate",
        ]

        # iron bands (All, but possibly only swir and vnir)
        rlist += [
            r"B2/B1 Ferric Iron Fe3+",
            r"B2/B0 Iron Oxide",
            r"B0/B2 Inverse Iron Oxide",
            r"B5/B3+B1/B2 Ferrous Iron Fe2+",
            r"B4/B5 Laterite or Alteration",
            r"B4/B2 Gossan",
            r"B5/B4 Ferrous Silicates (biotite, chloride, amphibole)",
            r"B4/B3 Ferric Oxides (can be ambiguous)",
            r"B4/B3A Mafic or Ultramafic Enhancement",
            r"B5/B3A Ferrous Iron Fe2+",
        ]  # lsat ferrous?

        # silicates bands
        rlist += [
            r"(B5+B7)/B6 sericite muscovite illite smectite",
            r"(B4+B6)/B5 alunite kaolinite pyrophyllite",
            r"B5/B6 phengitic or host rock",
            r"B7/B6 muscovite",
            r"B7/B5 kaolinite",
            r"(B5*B7)/(B6*B6) clay",
        ]

        # silica
        rlist += [
            r"B14/B12 quartz",
            r"B12/B13 basic degree index (gnt cpx epi chl) or SiO2",
            r"B13/B12 SiO2 same as B14/B12",
            r"(B11*B11)/(B10*B12) siliceous rocks",
            r"B11/B10 silica",
            r"B11/B12 silica",
            r"B13/B10 silica",
        ]

        # Other
        rlist += [
            r"B3/B2 Vegetation",
            r"(B3-B2)/(B3+B2) NDVI",
            r"(B3-B4)/(B3+B4) NDWI or NDMI water in leaves",
            r"(B1-B3)/(B1+B3) NDWI water bodies",
            r"2.5*(B3-B2)/(B3+6.0*B2-7.5*B0+1) EVI",
            r"B3/B1 GRVI",
            r"(B3-B2)/sqrt(B3+B2) RDVI",
            r"1.5*(B3-B2)/(B3+B2+0.5) SAVI",
            r"B3A/B1 GRVI Landslide",
            r"(B3A-B2)/sqrt(B3A+B2A) RDVI Landslide",
            r"1.5*(B3-B2)/(B3+B2A+0.5) SAVI Landslide",
            r"0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2",
            r"(B3A-B4+B5)/(B3A+B4-B5) NMDI",
            r"((B4+B2)-(B3+B0))/((B4+B2)+(B3+B0)) BSI",
        ]

        # EMIT
        rlist += [
            r"B1603/(B2185+B2225) Al-OH",
            r"B1603/(B2230+B2296) Fe-OH",
            r"B1603/(B2306+B2365) Mg-OH or CO3",
        ]

        # Colour composite

        rlist += [
            r"B5/B3 Used in colour composites",
            r"B4/B0 Used in colour composites",
            r"B5/B1 Used in colour composites",
            r"B4/B7 Used in colour composites",
            r"B12/B14 Used in colour composites",
            r"B3/B4 Used in colour composites",
        ]

        # Landslides

        # rlist += ['B0,B1,B2,B3,B4 Landslide Index']

        rlist2 = correct_bands(rlist, sensor)

        self.lw_ratios.clear()
        self.lw_ratios.addItems(rlist2)
        self.lw_ratios.selectAll()

    def invert_selection(self):
        """Invert the selected ratios."""
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(not item.isSelected())


class ConditionIndices(BasicModule):
    """
    GUI to calculate satellite condition indices.

    Parameters
    ----------
    parent
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.bfile = None

        self.cmb_index = QtWidgets.QComboBox()
        self.lw_ratios = QtWidgets.QListWidget()
        self.cmb_sensor = QtWidgets.QComboBox()

        self.setupui()

    def setupui(self):
        """Set up UI."""
        self.buttonbox.htmlfile = "rsense.dm.calccondind"
        gl_main = QtWidgets.QGridLayout(self)
        btn_invert = QtWidgets.QPushButton("Invert Selection")
        lbl_index = QtWidgets.QLabel("Index:")
        lbl_ratios = QtWidgets.QLabel("Condition Indices:")
        lbl_sensor = QtWidgets.QLabel("Sensor:")

        self.lw_ratios.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )

        self.cmb_index.addItems(["EVI", "NDVI", "MSAVI2"])

        self.cmb_sensor.addItems(
            [
                "ASTER",
                "Landsat 8 and 9 (OLI)",
                "Landsat 7 (ETM+)",
                "Landsat 4 and 5 (TM)",
                "Landsat (All)",
                "Sentinel-2",
                "WorldView",
                "Unknown",
            ]
        )

        self.setWindowTitle("Condition Indices Calculations")

        gl_main.addWidget(lbl_sensor, 0, 0, 1, 2)
        gl_main.addWidget(self.cmb_sensor, 0, 1, 1, 1)
        gl_main.addWidget(lbl_index, 1, 0, 1, 1)
        gl_main.addWidget(self.cmb_index, 1, 1, 1, 1)
        gl_main.addWidget(lbl_ratios, 2, 0, 1, 1)
        gl_main.addWidget(self.lw_ratios, 2, 1, 1, 1)
        gl_main.addWidget(btn_invert, 3, 0, 1, 2)

        gl_main.addWidget(self.buttonbox, 6, 0, 1, 2)

        self.lw_ratios.clicked.connect(self.set_selected_ratios)
        self.cmb_sensor.currentIndexChanged.connect(self.setratios)
        btn_invert.clicked.connect(self.invert_selection)

    def settings(self, nodialog: bool = False) -> bool:
        """
        Entry point into item.

        Parameters
        ----------
        nodialog
            Run settings without a dialog. The default is False.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        tmp = []
        if "RasterFileList" not in self.indata:
            self.showlog("You need a raster file list as input.")
            return False

        if len(self.indata["RasterFileList"]) < 2:
            self.showlog("You need more than one scene in your raster file list.")
            return False

        bfile = os.path.basename(self.indata["RasterFileList"][0].filename)
        self.bfile = bfile[:4]

        dat = self.indata["RasterFileList"][0]

        instr = dat.sensor

        if "ASTER" in instr:
            self.cmb_sensor.setCurrentText("ASTER")
        elif "LC08" in instr or "LC09" in instr:
            self.cmb_sensor.setCurrentText("Landsat 8 and 9 (OLI)")
        elif "LE07" in instr:
            self.cmb_sensor.setCurrentText("Landsat 7 (ETM+)")
        elif "LT04" in instr or "LT05" in instr:
            self.cmb_sensor.setCurrentText("Landsat 4 and 5 (TM)")
        elif "WorldView" in instr and "Multi" in instr:
            self.cmb_sensor.setCurrentText("WorldView")
        elif "Sentinel-2" in instr:
            self.cmb_sensor.setCurrentText("Sentinel-2")
        else:
            self.cmb_sensor.setCurrentText("Unknown")

        if self.lw_ratios.count() == 0:
            self.setratios()

        if not nodialog:
            tmp = self.exec()
        else:
            tmp = 1

        if tmp != 1:
            return False

        self.acceptall()

        return True

    def saveproj(self):
        """Save project data from class."""
        self.saveobj(self.cmb_sensor)
        self.saveobj(self.cmb_index)
        self.saveobj(self.lw_ratios)

    def acceptall(self) -> bool:
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        index = self.cmb_index.currentText()
        sensor = self.cmb_sensor.currentText()

        rlist1 = []
        for i in self.lw_ratios.selectedItems():
            rlist1.append(i.text())

        if not rlist1:
            self.showlog("You need to select a condition index to calculate.")
            return False

        rlist2 = []
        if "VCI" in rlist1 and "EVI" in index:
            rlist2 += [r"2.5*(B3-B2)/(B3+6.0*B2-7.5*B0+1) EVI"]
        elif "VCI" in rlist1 and "NDVI" in index:
            rlist2 += [r"(B3-B2)/(B3+B2) NDVI"]
        elif "VCI" in rlist1 and "MSAVI2" in index:
            rlist2 += [r"0.5*(2*B3+1-sqrt((2*B3+1)**2-8*(B3-B2))) MSAVI2"]

        evi = []
        tci = []
        vci = []
        vhi = []
        lst = []

        flist = self.indata["RasterFileList"]
        if sensor == "ASTER":
            flist = get_aster_list(flist)
        elif "Landsat" in sensor:
            flist = get_landsat_list(flist, sensor)
        elif "Sentinel-2" in sensor:
            flist = get_sentinel_list(flist)
        else:
            self.showlog(
                "Warning: This might not be "
                + sensor
                + " data. Will attempt to do calculation "
                "anyway."
            )
            flist = self.indata["RasterFileList"]

        for ifile in flist:
            dat = get_from_rastermeta(ifile, piter=self.piter, showlog=self.showlog)

            if dat is None:
                continue

            ofile = dat[0].filename

            # Prepare for layer stacking
            if sensor == "WorldView":
                wvlabels = {
                    "CoastalBlue": "B1",
                    "Blue": "B2",
                    "Green": "B3",
                    "Yellow": "B4",
                    "Red": "B5",
                    "RedEdge": "B6",
                    "NIR1": "B7",
                    "NIR2": "B8",
                }
                for i in dat:
                    if i.dataid.split()[0] in wvlabels:
                        i.dataid = wvlabels[i.dataid.split()[0]]

            bfile = os.path.basename(ifile.filename)
            rlist = correct_bands(rlist2, sensor, bfile)

            datsml = []
            for i in dat:
                txt = i.dataid.split()[0]

                if "Band" not in txt and "B" in txt:
                    txt = txt.replace("B", "Band")

                if "Band" not in txt and "LST" not in txt:
                    continue

                i.data = i.data.astype(float)
                i.data = i.data.filled(1e20)
                i.data = np.ma.masked_equal(i.data, 1e20)
                i.nodata = 1e20

                formula = ",".join(rlist)
                formula = re.sub(r"B(\d+)", r"Band\1", formula)

                if txt in formula or txt == "LST":
                    datsml.append(i)

            dat = lstack(datsml, piter=self.piter, showlog=self.showlog)

            del datsml

            # Correct band names
            datd = {}
            newmask = None
            for i in dat:
                tmp = i.dataid.split()
                txt = tmp[0]
                if txt == "Band":
                    txt = tmp[0] + tmp[1]

                if "Band" not in txt and "B" in txt and "," in txt:
                    txt = txt.replace("B", "Band")
                    txt = txt.replace(",", "")

                if "Band" not in txt and "B" in txt:
                    txt = txt.replace("B", "Band")

                if txt == "Band3N":
                    txt = "Band3"

                datd[txt] = i.data

                if "LST" in txt:
                    lst.append(i)

            # Calculate ratios
            for i in self.piter(rlist):
                self.showlog("Calculating " + i)
                formula = i.split(" ")[0]
                formula = re.sub(r"B(\d+)", r"Band\1", formula)
                blist = formula
                for j in ["/", "*", "+", "-", "(", ")"]:
                    blist = blist.replace(j, " ")
                blist = blist.split()
                blist = list(set(blist))
                blist = [i for i in blist if "Band" in i]

                abort = []
                for j in blist:
                    if "B" not in j:
                        continue
                    if j not in datd:
                        abort.append(j)
                if abort:
                    self.showlog("Error: " + " ".join(abort) + " missing.")
                    continue

                newmask = datd[blist[0]].mask
                for j in blist:
                    newmask = newmask | datd[j].mask

                if len(formula.split(r"/")) == 2:
                    f1, f2 = formula.split(r"/")
                    a1 = ne.evaluate(f1, datd)
                    a2 = ne.evaluate(f2, datd)

                    a2[np.isclose(a2, 0.0)] = 0.0
                    ratio = a1 / a2
                else:
                    ratio = ne.evaluate(formula, datd)

                newmask = newmask | (ratio < -1) | (ratio > 1)
                ratio = ratio.astype(np.float32)
                ratio[newmask] = 1e20
                ratio = np.ma.array(ratio, mask=newmask, fill_value=1e20)

                ratio = np.ma.fix_invalid(ratio)

                tmp = dat[0].copy(resetmeta=True)
                tmp.data = ratio
                tmp.nodata = 1e20
                evi.append(tmp)

        if lst:
            lst = lstack(lst, piter=self.piter, showlog=self.showlog, commonmask=True)
        if evi:
            evi = lstack(evi, piter=self.piter, showlog=self.showlog, commonmask=True)

        ofile = ""
        if ("TCI" in rlist1 or "VHI" in rlist1) and lst:
            tci = get_TCI(lst)
            ofile += "_TCI"
        if ("VCI" in rlist1 or "VHI" in rlist1) and evi:
            vci = get_VCI(evi, index)
            ofile += "_VCI_" + index
        if "VHI" in rlist1 and tci and vci:
            vhi = get_VHI(tci, vci)
            ofile += "_VHI"

        datfin = tci + vci + vhi

        for i in datfin:
            i.data = i.data.astype(np.float32)
            i.nodata = np.float32(i.nodata)

        if datfin:
            self.outdata["Raster"] = datfin

        return True

    def setratios(self):
        """Set the available indices."""
        sensor = self.cmb_sensor.currentText()
        rlist = []

        if "Unknown" not in sensor:
            rlist += ["VCI"]

        if "Landsat" in sensor:
            rlist += ["TCI", "VHI"]

        self.lw_ratios.clear()
        self.lw_ratios.addItems(rlist)

        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(True)

    def invert_selection(self):
        """Invert the selected ratios."""
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            item.setSelected(not item.isSelected())

        self.set_selected_ratios()

    def set_selected_ratios(self):
        """Set the selected ratios."""
        currentitem = self.lw_ratios.currentItem()

        idict = {}
        for i in range(self.lw_ratios.count()):
            item = self.lw_ratios.item(i)
            idict[item.text()] = i

        if currentitem.text() == "VHI" and currentitem.isSelected():
            for i in range(self.lw_ratios.count()):
                self.lw_ratios.item(i).setSelected(currentitem.isSelected())
        elif not currentitem.isSelected() and "VHI" in idict:
            self.lw_ratios.item(idict["VHI"]).setSelected(False)


def calc_ratios(
    dat: list[Data],
    rlist: list[str],
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
    sensor: str | None = None,
) -> list[Data]:
    """
    Calculate Band ratios.

    Note that this routine assumes that the ratio you supply is correct for
    your data.

    Parameters
    ----------
    dat
        List of PyGMI Data.
    rlist
        List of strings, containing ratios to calculate..
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.
    sensor
        The sensor being processed. The default is None.

    Returns
    -------
    list of Data
        List of PyGMI Data.

    """
    datsml = []

    for i in dat:
        tmp = i.dataid.split()
        txt = tmp[0]

        if "Band" not in txt and "B" in txt:
            txt = txt.replace("B", "Band")

        if "Band" not in txt and "LST" not in txt:
            continue

        formula = ",".join(rlist)
        formula = re.sub(r"B(\d+)", r"Band\1", formula)

        if txt == "Band3N":
            txt = "Band3"

        if txt in formula:
            datsml.append(i)

    dat = lstack(datsml, piter=piter, showlog=showlog)

    del datsml

    datd = {}
    newmask = None
    for i in dat:
        tmp = i.dataid.split()
        txt = tmp[0]
        if txt == "Band":
            txt = tmp[0] + tmp[1]

        if "Band" not in txt and "B" in txt and "," in txt:
            txt = txt.replace("B", "Band")
            txt = txt.replace(",", "")

        if "Band" not in txt and "B" in txt:
            txt = txt.replace("B", "Band")

        if txt == "Band3N":
            txt = "Band3"

        datd[txt] = i.data

    datfin = []
    for i in piter(rlist):
        showlog("Calculating " + i)
        if "Landslide Index" in i:
            rband = landslide_index(dat, sensor, showlog, piter)
            datfin += rband
            continue

        formula = i.split(" ")[0]
        formula = re.sub(r"B(\d+)", r"Band\1", formula)
        blist = formula
        for j in ["/", "*", "+", "-", "(", ")"]:
            blist = blist.replace(j, " ")
        blist = blist.split()
        blist = list(set(blist))
        blist = [i for i in blist if "Band" in i]

        abort = []
        for j in blist:
            if "B" not in j:
                continue
            if j not in datd:
                abort.append(j)
        if abort:
            showlog("Error: " + " ".join(abort) + " missing.")
            continue

        newmask = datd[blist[0]].mask
        for j in blist:
            newmask = newmask | datd[j].mask

        if len(formula.split(r"/")) == 2:
            f1, f2 = formula.split(r"/")
            a1 = ne.evaluate(f1, datd)
            a1 = a1.astype(np.float32)
            a2 = ne.evaluate(f2, datd)
            a2 = a2.astype(np.float32)

            a2[np.isclose(a2, 0.0)] = 0.0
            ratio = a1 / a2

            del a1
            del a2
        else:
            ratio = ne.evaluate(formula, datd)

        ratio = ratio.astype(np.float32)
        ratio[newmask] = np.float32(dat[0].nodata)
        ratio = np.ma.array(ratio, mask=newmask, fill_value=dat[0].nodata)

        ratio = np.ma.fix_invalid(ratio)

        rband = dat[0].copy(resetmeta=True)
        rband.data = ratio
        rband.dataid = i.replace(r"/", "div")
        datfin.append(rband)

    return datfin


def correct_bands(rlist: list[str], sensor: str, bfile: str | None = None) -> list[str]:
    """
    Correct the band designations.

    Ratio formula are defined in terms of ASTER bands. This converts that to
    the target sensor.

    Parameters
    ----------
    rlist
        List of input ratios.
    sensor
        Target sensor.
    bfile
        Data filename. The default is None.

    Returns
    -------
    list of str
        List of converted ratios.

    """
    sdict = {}

    sdict["ASTER"] = {
        "B1": "B1",
        "B2": "B2",
        "B3": "B3",
        "B4": "B4",
        "B3A": "B3",
        "B2A": "B2",
        "B5": "B5",
        "B6": "B6",
        "B7": "B7",
        "B8": "B8",
        "B9": "B9",
        "B10": "B10",
        "B11": "B11",
        "B12": "B12",
        "B13": "B13",
        "B14": "B14",
    }
    sdict["Landsat 8 and 9 (OLI)"] = {
        "B0": "B2",
        "B1": "B3",
        "B2": "B4",
        "B3": "B5",
        "B4": "B6",
        "B5": "B7",
        "B3A": "B5",
        "B2A": "B4",
    }
    sdict["Landsat 7 (ETM+)"] = {
        "B0": "B1",
        "B1": "B2",
        "B2": "B3",
        "B3": "B4",
        "B4": "B5",
        "B5": "B7",
        "B3A": "B4",
        "B2A": "B3",
    }
    sdict["Landsat 4 and 5 (TM)"] = sdict["Landsat 7 (ETM+)"]
    sdict["Sentinel-2"] = {
        "B0": "B2",
        "B1": "B3",
        "B2": "B4",
        "B3": "B8",
        "B4": "B11",
        "B5": "B12",
        "B3A": "B8A",
        "B2A": "B5",
    }
    sdict["WorldView"] = {
        "B0": "B2",
        "B1": "B3",
        "B2": "B5",
        "B3": "B7",
        "B3A": "B7",
        "B2A": "B5",
    }
    # sdict['EMIT'] = {'B1': 'B559', 'B2': 'B663', 'B3': 'B812',
    #                  'B4': 'B1647', 'B3A': 'B865', 'B2A': 'B700',
    #                   'B5': 'B2167', 'B6': 'B2204', 'B7': 'B2263',
    #                   'B8': 'B2330', 'B9': 'B2396', 'B0': 'B492',
    #                   'B1603': 'B1603', 'B2185':'B2189', 'B2225':'B2226',
    #                   'B2230':'B2234', 'B2296':'B2293', 'B2306':'B2308',
    #                   'B2365':'B2367'}

    sdict["EMIT"] = {
        "B0": "B492",
        "B1": "B559",
        "B2": "B664",
        "B2A": "B704",
        "B3": "B833",
        "B3A": "B865",
        "B4": "B1614",
        "B5": "B2167",
        "B6": "B2209",
        "B7": "B2262",
        "B8": "B2330",
        "B9": "B2400",
        "B1603": "B1603",
        "B2185": "B2185",
        "B2225": "B2225",
        "B2230": "B2230",
        "B2296": "B2296",
        "B2306": "B2306",
        "B2365": "B2365",
    }

    sdict["Unknown"] = {}

    if sensor == "Landsat (All)":
        if "LC09" in bfile or "LC08" in bfile:
            sensor = "Landsat 8 and 9 (OLI)"
        elif "LE07" in bfile:
            sensor = "Landsat 7 (ETM+)"
        else:
            sensor = "Landsat 4 and 5 (TM)"

    bandmap = sdict[sensor]
    # Sort the keys so we do long names like B3A first
    svalues = sorted(sorted(set(bandmap.keys())), key=lambda el: len(el))[::-1]
    rlist2 = []
    for i in rlist:
        formula = i.split(" ")[0]
        lbl = i[i.index(" ") :]
        bands = set(re.findall(r"B\d+\w?", formula))
        if bands.issubset(svalues):
            tmp = re.sub(r"B(\d+\w?)", r"tmpB\1", formula)
            for j in svalues:
                tmp = tmp.replace("tmp" + j, bandmap[j])

            rlist2.append(tmp + lbl)

    return rlist2


def correct_EMIT_bands(rlist: list[str], dat: list[Data]) -> list[Data]:
    """
    Correct EMIT band names.

    Parameters
    ----------
    rlist
        List of ratios.
    dat
        List of Data.

    Returns
    -------
    list of Data
        list of EMIT data bands.
    """
    blist1 = []
    for i in rlist:
        formula = i.split(" ")[0]
        formula = re.sub(r"B(\d+)", r"Band\1", formula)
        blist = formula
        for j in ["/", "*", "+", "-", "(", ")"]:
            blist = blist.replace(j, " ")
        blist = blist.split()
        blist = list(set(blist))
        blist1 += [i for i in blist if "Band" in i]
    blist = list(set(blist1))
    blist = [int(i[4:]) for i in blist]

    dlist = []
    for i in dat:
        dlist.append(i.metadata["Raster"]["wavelength"])

    dataids = {x: min(dlist, key=lambda y: abs(x - y)) for x in blist}

    dat1 = {}
    for i in dat:
        dat1[i.metadata["Raster"]["wavelength"]] = i

    dat2 = []
    for i in dataids:
        dat2.append(dat1[dataids[i]].copy())
        dat2[-1].dataid = f"B{i}"

    return dat2


def get_aster_list(flist: list[str]) -> list[str]:
    """
    Get ASTER files from a file list.

    Parameters
    ----------
    flist
        List of filenames.

    Returns
    -------
    list of str
        List of filenames.

    """
    flist2 = []
    for i in flist:
        if "ASTER" not in i.sensor:
            continue
        flist2.append(i)

    return flist2


def get_EMIT_list(flist: list[str]) -> list[str]:
    """
    Get EMIT files from a file list.

    Parameters
    ----------
    flist
        List of filenames.

    Returns
    -------
    list of str
        List of filenames.

    """
    flist2 = []
    for i in flist:
        if "EMIT" not in i.sensor:
            continue
        flist2.append(i)

    return flist2


def get_landsat_list(
    flist: list[str], sensor: str | None = None, allsats: bool = False
) -> list[str]:
    """
    Get Landsat files from a file list.

    Parameters
    ----------
    flist
        List of filenames.
    sensor
        Landsat satellite sensor, by default None.
    allsats
        use all Landsat sensors, by default False.


    Returns
    -------
    list of str
        List of filenames.

    """
    if isinstance(flist[0], list):
        bfile = os.path.basename(flist[0][0].filename)
        if bfile[:4] in ["LT04", "LT05", "LE07", "LC08", "LC09"]:
            return flist
        return []

    if allsats is True or sensor is None:
        fid = ["LT04", "LT05", "LE07", "LC08", "LC09"]
    elif sensor == "Landsat 8 and 9 (OLI)":
        fid = ["LC08", "LC09"]
    elif sensor == "Landsat 7 (ETM+)":
        fid = ["LE07"]
    elif sensor == "Landsat 4 and 5 (TM)":
        fid = ["LT04", "LT05"]
    else:
        return None

    flist2 = []
    for i in flist:
        for j in fid:
            if j not in i.sensor:
                continue
            if ".tif" in i.filename:
                continue
            flist2.append(i)

    return flist2


def get_sentinel_list(flist: list[str]) -> list[str]:
    """
    Get Sentinel-2 files from a file list.

    Parameters
    ----------
    flist
        List of filenames.

    Returns
    -------
    list of str
        List of filenames.

    """
    flist2 = []
    for i in flist:
        if "Sentinel-2" not in i.sensor:
            continue
        flist2.append(i)

    return flist2


def get_TCI(lst: list[Data]) -> list[Data]:
    """
    Calculate TCI.

    Parameters
    ----------
    lst
        list of PyGMI datasets - land surface temperatures.

    Returns
    -------
    list of Data
        output TCI datasets.

    """
    tci = []
    lst2 = []

    for j in lst:
        lst2.append(j.data)
    lst2 = np.ma.array(lst2)

    lstmax = lst2.max(0)
    lstmin = lst2.min(0)

    for dat in lst:
        tmp = dat.copy(resetmeta=True)

        tmp.data = (lstmax - dat.data) / (lstmax - lstmin)

        tmp.dataid = os.path.basename(dat.filename)[:-4] + "_TCI"
        tci.append(tmp)

    return tci


def get_VCI(evi: list[Data], index: str) -> list[Data]:
    """
    Calculate VCI.

    Parameters
    ----------
    evi
        list of EVI datasets.
    index
        index for dataid.

    Returns
    -------
    list of Data
        output VCI datasets.

    """
    evi2 = []
    for j in evi:
        evi2.append(j.data)

    evi2 = np.ma.array(evi2)

    evimax = evi2.max(0)
    evimin = evi2.min(0)

    vci = []
    for dat in evi:
        tmp = dat.copy(resetmeta=True)

        tmp.data = (dat.data - evimin) / (evimax - evimin)

        tmp.dataid = os.path.basename(dat.filename)[:-4] + "_VCI_" + index
        vci.append(tmp)

    return vci


def get_VHI(tci: list[Data], vci: list[Data], alpha: float = 0.5) -> list[Data]:
    """
    Calculate VHI.

    Parameters
    ----------
    tci
        TCI dataset list.
    vci
        VCI dataset list.
    alpha
        Weight for proportion of TCI and VCI. The default is 0.5.

    Returns
    -------
    list of Data
        Output VHI datasets.

    """
    vhi = []
    for tci1 in tci:
        for vci1 in vci:
            if tci1.filename == vci1.filename:
                tmp = tci1.copy(resetmeta=True)
                tmp.data = vci1.data * alpha + tci1.data * (1 - alpha)
                tmp.dataid = os.path.basename(tci1.filename)[:-4] + "_VHI"

                vhi.append(tmp)

    return vhi


def landslide_index(
    dat: list[Data],
    sensor: str | None = None,
    showlog: Callable[..., None] = print,
    piter: Iterable = iter,
) -> list[Data]:
    """
    Calculate Band ratios.

    Note that this routine assumes that the ratio you supply is correct for
    your data.

    Parameters
    ----------
    dat
        List of PyGMI Data.
    sensor
        The sensor being processed. The default is None.
    showlog
        Display information. The default is print.
    piter
        Progress bar iterator. The default is iter.

    Returns
    -------
    list of Data
        Red, green and blue PyGMI Data.

    """
    rlist = [
        r"(B3-B2)/(B3+B2) NDVI",
        r"(B1-B3)/(B1+B3) NDWI water bodies",
        r"B4 SWIR",
        r"((B4+B2)-(B3+B0))/((B4+B2)+(B3+B0)) BSI",
    ]

    if sensor is None:
        sensor = dat[0].metadata["Raster"]["Sensor"]
    rlist = correct_bands(rlist, sensor)

    datfin = calc_ratios(dat, rlist, showlog=showlog, piter=piter)

    NDVI = datfin[0].data
    NDWI = datfin[1].data
    SWIR = datfin[2].data
    BSI = datfin[3].data

    for i in datfin:
        if "NDVI" in i.dataid:
            NDVI = i.data
        elif "NDWI" in i.dataid:
            NDWI = i.data
        elif "SWIR" in i.dataid:
            SWIR = i.data
        elif "BSI" in i.dataid:
            BSI = i.data

    red = dat[0].copy(resetmeta=True)
    green = dat[0].copy(resetmeta=True)
    blue = dat[0].copy(resetmeta=True)

    red.data = red.data.astype(np.float32)
    green.data = green.data.astype(np.float32)
    blue.data = blue.data.astype(np.float32)

    red.data[:] = 3.5 * BSI
    green.data[~green.data.mask] = 0.3
    blue.data[~blue.data.mask] = 0.0

    filt = (SWIR > 0.8) | (NDVI < 0.15)
    red.data[filt] = 1.5
    green.data[filt] = 0.7
    blue.data[filt] = -1.0

    filt = NDVI > 0.25
    red.data[filt] = 0.0
    green.data[filt] = 0.2 * NDVI[filt]
    blue.data[filt] = 0.0

    filt = NDWI > 0.15
    red.data[filt] = 0.0
    green.data[filt] = 0.2
    blue.data[filt] = NDWI[filt]

    red.data = np.ma.masked_equal(red.data.filled(1e20), 1e20)
    red.nodata = 1e20

    green.data = np.ma.masked_equal(green.data.filled(1e20), 1e20)
    green.nodata = 1e20

    blue.data = np.ma.masked_equal(blue.data.filled(1e20), 1e20)
    blue.nodata = 1e20

    red.dataid = "Landslide Index Red"
    green.dataid = "Landslide Index Green"
    blue.dataid = "Landslide Index Blue"

    return [red, green, blue]


def _testfn():
    """Test routine."""
    from pygmi.rsense.iodefs import ImportBatch  # , ImportData

    # idir = r'D:\Workdata\PyGMI Test Data\Remote Sensing\Import\Landsat'
    # idir = r"D:\VMS\S2"
    idir = r"C:\Work\EMIT"

    os.chdir(idir)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    tmp1 = ImportBatch()
    tmp1.idir = idir
    # tmp1 = ImportData()
    tmp1.settings()

    SR = SatRatios()
    SR.indata = tmp1.outdata
    SR.settings()

    # dat2 = SR.outdata['Raster']
    # for i in dat2:
    #     plt.figure(dpi=150)
    #     plt.title(i.dataid)
    #     vmin = i.data.mean() - 2 * i.data.std()
    #     vmax = i.data.mean() + 2 * i.data.std()
    #     plt.imshow(i.data, vmin=vmin, vmax=vmax)
    #     plt.colorbar()
    #     plt.show()


def _testfn2():
    """Test routine."""
    import matplotlib.pyplot as plt

    from pygmi.rsense.iodefs import ImportBatch

    idir = r"D:\workdata\PyGMI Test Data\Remote Sensing\ConditionIndex"
    idir = r"D:\work\Programming\pygmi\pygmi\test\testdata"
    os.chdir(idir)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    tmp1 = ImportBatch()
    tmp1.idir = idir
    tmp1.settings()

    SR = ConditionIndices()
    SR.indata = tmp1.outdata
    SR.settings()

    dat2 = SR.outdata["Raster"]
    for i in dat2:
        plt.figure(dpi=150)
        plt.title(i.dataid)
        vmin = i.data.mean() - 2 * i.data.std()
        vmax = i.data.mean() + 2 * i.data.std()
        plt.imshow(i.data, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    _testfn()
