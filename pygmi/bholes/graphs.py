# -----------------------------------------------------------------------------
# Name:        boreholes/graphs.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2018 Council for Geoscience
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
"""Methods to plot borehole data via the context menu."""

import re
import textwrap
import xml

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.path import Path
from PySide6 import QtCore, QtWidgets

from pygmi.misc import ContextModule


class MyMplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for the actual plot."""

    def __init__(self):
        fig = Figure()
        super().__init__(fig)

    def update_legend(self, df, hatch, clith, col, stratcol):
        """
        Update the plot legend.

        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing the data.

        """
        fig = self.figure
        fig.clear()
        ax = fig.gca()

        pagewidth = 8
        pageheight = 4
        dpp = 14  # depth per page
        wpp = dpp * pagewidth / pageheight

        lith = np.array(df["Lithology"])
        strat = np.array(df["Stratigraphy"].replace(np.nan, "none"))
        rank = np.array(df["Rank"].replace(np.nan, "none"))

        rlookup = {
            "SUI": "Suite",
            "SBSUI": "Sub Suite",
            "FM": "Formation",
            "none": "",
            "NONE": "",
            "GRP": "Group",
            "MEMB": "Member",
            "SBGRP": "Sub Group",
            "SPGRP": "Super Group",
            "CPLX": "Complex",
        }

        ax.set_xlim((0, wpp))
        ax.set_ylim((0, dpp))
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        strat, idx = np.unique(strat, return_index=True)
        lith = np.unique(lith)
        rank = rank[idx]

        # Do hatch legend
        ax.text(0.5, 0.7, "Lithology", size=8)
        for j in np.arange(0, len(lith)):
            if lith[j] in hatch:
                pverts, pcodes = hatch[lith[j]]
            else:
                pverts, pcodes = hatch["NOR"]

            for k in pverts:
                pathfin = Path(pverts[k] + [0.5, j * 2 + 1], pcodes[k])
                pp1 = mpatches.PathPatch(pathfin, fc="w")
                if k == "#ffffff":
                    pp1.set_facecolor("w")
                elif k != "none":
                    pp1.set_facecolor("k")

                ax.add_patch(pp1)

            rect = mpatches.Rectangle([0.5, j * 2 + 2], 4.1, 3.1, fc="w", ec="none")
            ax.add_patch(rect)
            rect = mpatches.Rectangle([0.5, j * 2 + 1], 4, 1, fc="none", ec="k")
            ax.add_patch(rect)
            if lith[j] in clith:
                txt = clith[lith[j]]
            else:
                txt = lith[j]
            ax.text(4.7, j * 2 + 1.9, txt, size=6)

        # do color legend
        ax.text(15.0, 0.7, "Stratigraphy", size=8)
        for j in np.arange(0, len(strat)):
            scol = "#" + col[stratcol[strat[j]]]
            rect = mpatches.Rectangle([15, j * 2 + 1], 4, 1, fc=scol, ec="k")
            ax.add_patch(rect)
            if strat[j] == "none":
                ax.text(19.2, j * 2 + 1.9, strat[j].capitalize(), size=6)
            else:
                ax.text(
                    19.2,
                    j * 2 + 1.9,
                    strat[j].capitalize() + " " + rlookup[rank[j]],
                    size=6,
                )

        self.figure.canvas.draw()

    def update_log(self, df, hatch, col, stratcol):
        """
        Update the borehole log plot.

        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing the data.

        """
        fig = self.figure
        fig.clear()
        ax = fig.gca()
        fig.subplots_adjust(top=0.995)
        fig.subplots_adjust(bottom=0.005)
        fig.subplots_adjust(left=0.01)
        fig.subplots_adjust(right=0.3)

        pageheight = 8
        dpp = 25  # depth per page
        fontsize = 6
        dpi = (fontsize / 72) * (dpp / pageheight)

        depthfrom = -1 * np.array(df["Depth from"])
        depthto = -1 * np.array(df["Depth to"])
        lithd = np.array(df["Lithology description"].replace(np.nan, ""))
        lith = np.array(df["Lithology"])
        strat = np.array(df["Stratigraphy"].replace(np.nan, "none"))

        numpages = abs(depthto[-1] // dpp)

        ###########################################################################
        # Start of each borehole plot
        # Locations of the text lithology labels

        lithdpos = depthfrom
        yfin = lithdpos[0]
        if yfin == 0.0:
            yfin = -dpi

        for i, _ in enumerate(lithd):
            lithd[i] = commentprep(lithd[i])
            lithdpos[i] = min(lithdpos[i], yfin)
            if i < len(lithd) - 1 and (depthfrom[i] != depthfrom[i + 1]):
                yfin = lithdpos[i] - dpi * (1 + lithd[i].count("\n")) * 1.4

        # Start creating plots
        ax.set_ylim((-dpp * numpages, 0.0))
        ax.set_aspect("equal")
        ax.get_xaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.margins(x=0)

        for i in range(depthfrom.size):
            # This next line is to skip summary lines for a group.
            if i + 1 < depthfrom.size - 1 and (depthfrom[i] == depthfrom[i + 1]):
                continue

            patches = []
            if lith[i] in hatch:
                pverts, pcodes = hatch[lith[i]]
            else:
                pverts, pcodes = hatch["NOR"]

            scol = "#" + col[stratcol[strat[i]]]

            dfrom = depthfrom[i]
            dto = depthto[i]
            texty = lithdpos[i]
            ax.plot([4, 5], [dfrom, texty], "k", linewidth=1.0)
            ax.text(
                5.2,
                texty,
                f"{dfrom:.2f} {lithd[i]}",
                va="center",
                size=fontsize,
            )

            rect = mpatches.Rectangle([0, dto], 4, (dfrom - dto), fc=scol, ec="k")
            patches.append(rect)

            for j in np.arange(-dfrom, -dto, 4):
                for k in pverts:
                    pathfin = Path(pverts[k] - [0, j + 4], pcodes[k])
                    pp1 = mpatches.PathPatch(pathfin, fc=scol)

                    if k == "#ffffff":
                        pp1.set_facecolor(scol)
                    elif k != "none":
                        pp1.set_facecolor("k")
                    patches.append(pp1)

            rect = mpatches.Rectangle([0, dto - 4], 4.1, 4, fc="w")
            patches.append(rect)

            collection = PatchCollection(patches, match_original=True)
            ax.add_collection(collection)

            if lith[-1] == "NOR":
                ax.text(
                    5.2,
                    -dpp * numpages + dpi,
                    "(Last entry; log truncated due to length)",
                    va="center",
                    size=fontsize,
                )

            ax.hlines(dto, 0, 4, "k")  # Bottom of log

        self.figure.canvas.draw()


class PlotLog(ContextModule):
    """
    Class to plot the borehole log.

    Parameters
    ----------
    parent : pygmi.main.MainWidget, optional
        Reference to the parent routine. The default is None.

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.hatch = None
        self.clith = None
        self.col = None
        self.stratcol = None

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Borehole Log")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl_1 = QtWidgets.QHBoxLayout()
        hbl_2 = QtWidgets.QHBoxLayout()
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas()
        self.mmc2 = MyMplCanvas()

        self.lbl_topleft = QtWidgets.QLabel()
        self.lbl_topright = QtWidgets.QLabel()
        self.lbl_bottomleft = QtWidgets.QLabel()
        self.lbl_bottomright = QtWidgets.QLabel()
        self.lbl_topright.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.lbl_bottomright.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll.setWidget(self.mmc)

        self.buttonbox.htmlfile = "bholes.cm.showlog"
        self.buttonbox.buttonbox.hide()
        self.hbl.addWidget(self.buttonbox)

        self.cmb_1 = QtWidgets.QComboBox()
        self.lbl_1 = QtWidgets.QLabel("Borehole ID:")
        self.hbl.addWidget(self.lbl_1, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.hbl.addWidget(self.cmb_1)

        hbl_1.addWidget(self.lbl_topleft)
        hbl_1.addWidget(self.lbl_topright)
        hbl_2.addWidget(self.lbl_bottomleft)
        hbl_2.addWidget(self.lbl_bottomright)
        vbl.addLayout(hbl_1)
        vbl.addWidget(self.scroll, stretch=2)
        vbl.addLayout(hbl_2)
        vbl.addWidget(self.mmc2, stretch=1)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.cmb_1.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to choose the borehole to display.

        """
        i = self.cmb_1.currentText()

        data = self.indata["Borehole"]
        data = data.loc[data["Boreholeid"] == i]

        dfrom = data["Depth from"].iloc[0]
        dto = data["Depth to"].iloc[-1]
        depth = int((dto - dfrom) * 10.0)
        self.mmc.setFixedHeight(depth)

        hcompanyno = data["Companyno"].iloc[0]
        hfilt = (data["Companyno"] == hcompanyno).to_numpy().nonzero()[0][0]
        hrow = data.iloc[hfilt].astype(str)
        topleft = (
            hrow["Company"] + "\n" + hrow["Farmname"] + " (" + hrow["Farmno"] + ")"
        )
        topright = "Hole no: " + hrow["Companyno"] + "\n Sheet 1 of 1"
        if np.isnan(hrow["Drill date"]):
            bottomleft = "Drill date: None"
        else:
            bottomleft = "Drill date: " + hrow["Drill date"].split()[0]
        bottomleft += "\nDepth from: " + hrow["Depth from"]
        bottomleft += "\nDepth to: " + f"{dto}"
        bottomright = "Elevation: " + hrow["Elevation"]
        bottomright += "\nLatitude: " + hrow["Declat"]
        bottomright += "\nLongitude: " + hrow["Declon"]
        self.lbl_topleft.setText(topleft)
        self.lbl_topright.setText(topright)
        self.lbl_bottomleft.setText(bottomleft)
        self.lbl_bottomright.setText(bottomright)

        self.mmc2.update_legend(data, self.hatch, self.clith, self.col, self.stratcol)
        self.mmc.update_log(data, self.hatch, self.col, self.stratcol)

    def load_hatch(self):
        """
        Load all hatchings.

        """
        idir = __file__.rpartition("\\")[0]
        logfile = idir + "\\logplot.xlsx"

        xl = pd.ExcelFile(logfile)
        usgs = xl.parse("USGS")
        cgs = xl.parse("CGS")
        cgslookup = xl.parse("250K Lookup")
        colours = xl.parse("Colours")
        xl.close()

        usgs = usgs.set_index("code").to_dict()["description"]
        cgslookup["COLOR_CODE"] = cgslookup["COLOR_CODE"].astype(str)
        cgslookup["COLOR_CODE"] = cgslookup["COLOR_CODE"].apply("{0:0>3}".format)
        stratcol = cgslookup.set_index("LITHO_NAME").to_dict()["COLOR_CODE"]
        col = colours.set_index("code").to_dict()["colour"]
        clith = cgs.set_index("lithology").to_dict()["lithology description"]
        cgs = cgs.set_index("lithology").to_dict()["code"]
        col["none"] = "ffffff"
        col["nan"] = "ffffff"
        stratcol["none"] = "none"

        # Load in hatches
        self.hatch = {}
        for i in cgs:
            if np.isnan(cgs[i]):
                self.hatch[i] = [[], []]
                continue
            svgfile = idir + "\\svg\\" + str(int(cgs[i])) + ".svg"
            pverts, pcodes = gethatch(svgfile)
            self.hatch[i] = [pverts, pcodes]

        self.clith = clith
        self.col = col
        self.stratcol = stratcol

    def run(self):
        """
        Entry point into the routine, used to run context menu item.

        """
        data = []
        if "Borehole" in self.indata:
            data = self.indata["Borehole"]
        else:
            self.showlog("Error: You must have borehole data.")
            return

        self.cmb_update(self.cmb_1, data.Boreholeid.unique())

        self.showlog("Loading Hatching...")
        self.load_hatch()

        self.show()
        self.change_band()


def gethatch(svgfile):
    """
    Get hatching from an SVG file, to be used on the log.

    Parameters
    ----------
    svgfile : str
        SVG filename.

    Returns
    -------
    None.

    """
    tree = xml.etree.ElementTree.parse(svgfile)

    translate = []
    dpath = []
    style = []

    root = tree.getroot()
    defs = root.find("{http://www.w3.org/2000/svg}defs")

    for pat in defs.findall("{http://www.w3.org/2000/svg}pattern"):
        for child in pat:
            if child.tag == "{http://www.w3.org/2000/svg}g":
                tag = child.find("{http://www.w3.org/2000/svg}path")
                trans = child.get("transform")
                tmp = trans[10:-1].split(",")
                tmp = list(map(float, tmp))
                translate.append(tmp)
            elif child.tag == "{http://www.w3.org/2000/svg}path":
                tag = child
                translate.append([0.0, 0.0])
            else:
                continue

            dpath.append(tag.get("d"))
            stmp = {}
            for i in tag.get("style").split(";"):
                tmp = i.split(":")
                stmp[tmp[0]] = tmp[1]
            style.append(stmp)

    # translate = np.array(translate)
    pverts = {}
    pcodes = {}

    for idx, trans in enumerate(translate):
        tmp = re.split(r"(z|c|m|C|M|L|l)", dpath[idx])
        if tmp[0] == "":
            tmp.remove("")
        if tmp[-1] == "":
            tmp[-1] = "0., 0."

        # Start one graphics segment here
        rtmp = [0, 0]
        verts = []
        codes = []
        for i in range(0, len(tmp) - 1, 2):
            # Load in keys and values
            pkey = tmp[i]
            vtmp = re.split(r",| ", tmp[i + 1])
            while "" in vtmp:
                vtmp.remove("")
            vtmp = list(map(float, vtmp))
            pvals = np.reshape(vtmp, (len(vtmp) // 2, 2))

            # Correct relative coordinates

            if "m" in pkey or "l" in pkey:
                pvals = np.cumsum(pvals, axis=0) + rtmp

            if "c" in pkey:
                for k in range(0, pvals.shape[0], 3):
                    pvals[k : k + 3] += rtmp
                    rtmp = pvals[k + 2]

            # construct vertices and codes for paths
            if pkey.upper() == "M":
                verts += (pvals + trans).tolist()
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (len(pvals) - 1)

                if pvals.std() == 0.0 and pvals.size > 2:
                    verts[-2][0] += 0.5

            if pkey.upper() == "L":
                verts += (pvals + trans).tolist()
                codes += [Path.LINETO] * (len(pvals))

            if pkey.upper() == "C":
                verts += (pvals + trans).tolist()
                codes += [Path.CURVE4] * len(pvals)

            rtmp = pvals[-1]

        if style[idx]["fill"] not in pverts:
            pverts[style[idx]["fill"]] = verts
            pcodes[style[idx]["fill"]] = codes
        else:
            pverts[style[idx]["fill"]] += verts
            pcodes[style[idx]["fill"]] += codes

    for i, value in pverts.items():
        pverts[i] = np.array(value)
        pverts[i] /= np.max(pverts[i])
        pverts[i] *= 4

    return pverts, pcodes


def commentprep(mystring, slen=50):
    """
    Create the correct case for a string and inserts carriage returns.

    Parameters
    ----------
    mystring : str
        String to correct.
    slen : int, optional
        String length. The default is 50.

    Returns
    -------
    finstring : str
        Output string.

    """
    finstring = ""
    mystring = mystring.capitalize()
    for word in mystring.split():
        if re.search(r"\d", word):
            finstring += " " + word
        else:
            finstring += " " + word.capitalize()

    finstring = finstring.strip()
    finstring = textwrap.fill(finstring, slen)
    if "\n" in finstring:
        finstring = finstring[: finstring.index("\n")] + "..."

    return finstring


def chkname(iname):
    """
    Check a filename for illegal characters.

    Parameters
    ----------
    iname : str
        Input filename.

    Returns
    -------
    iname : str
        Corrected filename.

    """
    charlist = [
        ["#", "_hash_"],
        ["%", "_perc_"],
        ["&", "_amp_"],
        ["{", "_lb_"],
        ["}", "_rb_"],
        ["\\", "_bs_"],
        ["<", "_lrb_"],
        [">", "_rab_"],
        ["*", "_ast_"],
        ["?", "_q_"],
        ["/", "_fs_"],
        ["$", "_dol_"],
        ["!", "_exc_"],
        ['"', "_dq_"],
        ["'", "_sq_"],
        [":", "_col_"],
        ["@", "_at_"],
    ]
    for ichar, nchar in charlist:
        iname = iname.replace(ichar, nchar)

    return iname


def _testfn():
    """Test routine."""
    import sys

    from pygmi.bholes.iodefs import ImportData

    lfile = r"D:\workdata\PyGMI Test Data\boreholes\olma-coredata(lith).xlsx"
    hfile = r"D:\workdata\PyGMI Test Data\boreholes\olma-coredata(headers).xlsx"
    # lfile = r"D:\Sithilo Complex Data\Borehole Logs\Tugela Ultramafic Complexes - Boreholes - lith.xlsx"
    # hfile = r"D:\Sithilo Complex Data\Borehole Logs\Tugela Ultramafic Complexes - Boreholes.xlsx"
    # lfile = r"D:\workdata\PyGMI Test Data\boreholes\Marinda_actualCCUS_(Lithology).xlsx"
    # hfile = r"D:\workdata\PyGMI Test Data\boreholes\Marinda_actualCCUS_(headers).xlsx"

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    tmp1 = ImportData()
    tmp1.le_headfile.setText(hfile)
    tmp1.le_lithfile.setText(lfile)
    tmp1.fillcombos()
    tmp1.settings(True)

    tmp2 = PlotLog()
    tmp2.indata = tmp1.outdata
    tmp2.run()

    app.exec()


if __name__ == "__main__":
    _testfn()
