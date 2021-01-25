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
"""
Plot Borehole Data

This module provides a variety of methods to plot borehole data via the context
menu.
"""

import xml
import re
import textwrap
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MyMplCanvas(FigureCanvasQTAgg):
    """
    Canvas for the actual plot.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)

    def update_legend(self, data1):
        """
        Update legend.

        Parameters
        ----------
        data1 : dictionary
            Dictionary containing data.

        Returns
        -------
        None.

        """
        fig = self.figure
        fig.clear()
        ax = fig.gca()

        idir = __file__.rpartition('\\')[0]
        logfile = idir+'\\logplot.xlsx'

        pagewidth = 8
        pageheight = 4
        dpp = 14  # depth per page
        wpp = dpp*pagewidth/pageheight

    # #########################################################################
    # Load in logplot init
        xl = pd.ExcelFile(logfile)
        usgs = xl.parse('USGS')
        cgs = xl.parse('CGS')
        cgslookup = xl.parse('250K Lookup')
        colours = xl.parse('Colours')
        xl.close()

        usgs = usgs.set_index('code').to_dict()['description']
        cgslookup['COLOR_CODE'] = cgslookup['COLOR_CODE'].astype(str)
        cgslookup['COLOR_CODE'] = cgslookup['COLOR_CODE'].apply('{0:0>3}'.format)
        stratcol = cgslookup.set_index('LITHO_NAME').to_dict()['COLOR_CODE']
        col = colours.set_index('code').to_dict()['colour']
        clith = cgs.set_index('lithology').to_dict()['lithology description']
        cgs = cgs.set_index('lithology').to_dict()['code']
        col['none'] = 'ffffff'
        col['nan'] = 'ffffff'
        stratcol['none'] = 'none'

    # Load in hatches
        hatch = {}
        for i in cgs:
            if np.isnan(cgs[i]):
                hatch[i] = [[], []]
                continue
            svgfile = idir + '\\images\\' + str(int(cgs[i]))+'.svg'
            pverts, pcodes = gethatch(svgfile)
            hatch[i] = [pverts, pcodes]

        df = data1['log']
        hdf = data1['header']

        companyno = np.array(df['Companyno'])
        lith = np.array(df['Lithology'])
        strat = np.array(df['Stratigraphy'].replace(np.nan, 'none'))
        rank = np.array(df['Rank'].replace(np.nan, 'none'))

        hcompanyno = hdf['Companyno'].iloc[0]
        indx = np.nonzero(companyno == hcompanyno)[0]

        strat1 = strat
        lith1 = lith
        rank1 = rank

        rlookup = {'SUI': 'Suite',
                   'SBSUI': 'Sub Suite',
                   'FM': 'Formation',
                   'none': '',
                   'NONE': '',
                   'GRP': 'Group',
                   'MEMB': 'Member',
                   'SBGRP': 'Sub Group',
                   'SPGRP': 'Super Group',
                   'CPLX': 'Complex'}

        ax.set_xlim((0, wpp))
        ax.set_ylim((0, dpp))
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        strat, idx = np.unique(strat1[indx], return_index=True)
        lith = np.unique(lith1[indx])
        rank = rank1[indx][idx]

    # Do hatch legend
        ax.text(0.5, 0.7, 'Lithology', size=13)
        for j in np.arange(0, len(lith)):
            pverts, pcodes = hatch[lith[j]]

            for k in pverts:
                pathfin = Path(pverts[k]+[0.5, j*2+1], pcodes[k])
                pp1 = mpatches.PathPatch(pathfin, fc='w')
                if k == '#ffffff':
                    pp1.set_facecolor('w')
                elif k != 'none':
                    pp1.set_facecolor('k')

                ax.add_patch(pp1)

            rect = mpatches.Rectangle([0.5, j*2+2], 4.1, 3.1, fc='w',
                                      ec='none')
            ax.add_patch(rect)
            rect = mpatches.Rectangle([0.5, j*2+1], 4, 1, fc='none', ec='k')
            ax.add_patch(rect)
            ax.text(4.7, j*2+1.9, clith[lith[j]])

    # do color legend
        ax.text(15., 0.7, 'Stratigraphy', size=13)
        for j in np.arange(0, len(strat)):
            scol = '#'+col[stratcol[strat[j]]]
            rect = mpatches.Rectangle([15, j*2+1], 4, 1, fc=scol, ec='k')
            ax.add_patch(rect)
            if strat[j] == 'none':
                ax.text(19.2, j*2+1.9, strat[j].capitalize())
            else:
                ax.text(19.2, j*2+1.9, strat[j].capitalize()+' '+rlookup[rank[j]])

        self.figure.canvas.draw()

    def update_log(self, data1):
        """
        Update the raster plot.

        Parameters
        ----------
        data1 : dictionary.
            PyGMI log dataset to be used.

        Returns
        -------
        None.

        """

        fig = self.figure
        fig.clear()
        ax = fig.gca()
        fig.subplots_adjust(top=0.995)
        fig.subplots_adjust(bottom=0.005)
        # fig.subplots_adjust(left=0.005)
        # fig.subplots_adjust(right=0.25)
        fig.subplots_adjust(left=0.01)
        fig.subplots_adjust(right=0.3)

        idir = __file__.rpartition('\\')[0]
        logfile = idir+'\\logplot.xlsx'

        # pagewidth = 8
        pageheight = 8
        dpp = 25  # depth per page
        # wpp = dpp*pagewidth/pageheight
        fontsize = 10
        dpi = (fontsize/72)*(dpp/pageheight)

    ###########################################################################
    # Load in logplot init
        xl = pd.ExcelFile(logfile)
        usgs = xl.parse('USGS')
        cgs = xl.parse('CGS')
        cgslookup = xl.parse('250K Lookup')
        colours = xl.parse('Colours')
        xl.close()

        usgs = usgs.set_index('code').to_dict()['description']
        cgslookup['COLOR_CODE'] = cgslookup['COLOR_CODE'].astype(str)
        cgslookup['COLOR_CODE'] = cgslookup['COLOR_CODE'].apply('{0:0>3}'.format)
        stratcol = cgslookup.set_index('LITHO_NAME').to_dict()['COLOR_CODE']
        col = colours.set_index('code').to_dict()['colour']
        # clith = cgs.set_index('lithology').to_dict()['lithology description']
        cgs = cgs.set_index('lithology').to_dict()['code']
        col['none'] = 'ffffff'
        col['nan'] = 'ffffff'
        stratcol['none'] = 'none'

    # Load in hatches
        hatch = {}
        for i in cgs:
            if np.isnan(cgs[i]):
                hatch[i] = [[], []]
                continue
            svgfile = idir + '\\images\\' + str(int(cgs[i]))+'.svg'
            pverts, pcodes = gethatch(svgfile)
            hatch[i] = [pverts, pcodes]

        df = data1['log']
        hdf = data1['header']

        companyno = np.array(df['Companyno'])
        depthfrom = -1*np.array(df['Depth from'])
        depthto = -1*np.array(df['Depth to'])
        lithd = np.array(df['Lithology description'].replace(np.nan, ''))
        lith = np.array(df['Lithology'])
        strat = np.array(df['Stratigraphy'].replace(np.nan, 'none'))
        # rank = np.array(df['Rank'].replace(np.nan, 'none'))

        hcompanyno = hdf['Companyno'].iloc[0]
        indx = np.nonzero(companyno == hcompanyno)[0]
        numpages = abs(depthto[indx][-1]//dpp)

    ###########################################################################
    # Start of each borehole plot
    # Locations of the text lithology labels

        lithdpos = depthfrom[indx]
        yfin = lithdpos[0]
        if yfin == 0.:
            yfin = -dpi

        for i, _ in enumerate(lithd[indx]):
            lithd[indx[i]] = commentprep(lithd[indx[i]])
            if lithdpos[i] > yfin:
                lithdpos[i] = yfin
            if i < len(lithd[indx])-1:
                if depthfrom[indx][i] != depthfrom[indx][i+1]:
                    yfin = lithdpos[i] - dpi*(1+lithd[i].count('\n'))*1.4
    # Start creating plots
        #     pdf = PdfPages(odir+chkname(hrow['Companyno']+'.pdf'))
        # fig.set_figheight(pageheight*numpages)
        # fig.set_figwidth(pagewidth)

        # ax.set_xlim((0, wpp))
        ax.set_ylim((-dpp*numpages, 0.))
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.margins(x=0)

        for i in indx:
            # This next line is to skip summary lines for a group.
            if i < indx[-1]:
                if depthfrom[i] == depthfrom[i+1]:
                    continue

            pverts, pcodes = hatch[lith[i]]
            scol = '#'+col[stratcol[strat[i]]]

            dfrom = depthfrom[i]
            dto = depthto[i]
            texty = lithdpos[np.nonzero(indx == i)[0][0]]
            ax.plot([4, 5], [dfrom, texty], 'k', linewidth=1.0)
            ax.text(5.2, texty, '{0:.2f}'.format(dfrom)+' '+lithd[i],
                    va='center')

            rect = mpatches.Rectangle([0, dto], 4, (dfrom-dto), fc=scol,
                                      ec='k')
            ax.add_patch(rect)
            for j in np.arange(-dfrom, -dto, 4):
                for k in pverts:
                    pathfin = Path(pverts[k]-[0, j+4], pcodes[k])
                    pp1 = mpatches.PathPatch(pathfin, fc=scol)

                    if k == '#ffffff':
                        pp1.set_facecolor(scol)
                    elif k != 'none':
                        pp1.set_facecolor('k')
                    ax.add_patch(pp1)
            rect = mpatches.Rectangle([0, dto-4], 4.1, 4, fc='w')
            ax.add_patch(rect)

            if lith[indx[-1]] == 'NOR':
                ax.text(5.2, -dpp*numpages+dpi,
                        '(Last entry; log truncated due to length)',
                        va='center', size=fontsize)

            ax.hlines(dto, 0, 4)  # Bottom of log

        self.figure.canvas.draw()
        # self.figure.tight_layout()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Borehole Log')

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl1 = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        self.mmc2 = MyMplCanvas(self)
        # mpl_toolbar = NavigationToolbar(self.mmc, self.parent)
        # self.mmc.setFixedHeight(3000)
        self.label_topleft = QtWidgets.QLabel()
        self.label_topright = QtWidgets.QLabel()
        self.label_bottomleft = QtWidgets.QLabel()
        self.label_bottomright = QtWidgets.QLabel()
        self.label_topright.setAlignment(QtCore.Qt.AlignRight |
                                         QtCore.Qt.AlignVCenter)
        self.label_bottomright.setAlignment(QtCore.Qt.AlignRight |
                                            QtCore.Qt.AlignVCenter)

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.mmc)

        self.combobox1 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel('Borehole ID:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)

        hbl1.addWidget(self.label_topleft)
        hbl1.addWidget(self.label_topright)
        hbl2.addWidget(self.label_bottomleft)
        hbl2.addWidget(self.label_bottomright)
        vbl.addLayout(hbl1)
        vbl.addWidget(self.scroll)
        vbl.addLayout(hbl2)
        vbl.addWidget(self.mmc2)
        # vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        # self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """
        Combo box to change band.

        Returns
        -------
        None.

        """


class PlotLog(GraphWindow):
    """
    Plot Log Class.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.label2.hide()
        # self.combobox2.hide()
        self.indata = {}
        self.parent = parent

    def change_band(self):
        """ Combo box to choose band """
        i = self.combobox1.currentText()
        if 'Borehole' in self.indata:
            data = self.indata['Borehole'][i]
            hdf = data['header']
            dfrom = hdf['Depth from'].iloc[0]
            dto = hdf['Depth to'].iloc[0]
            depth = int((dto-dfrom)*10.)
            self.mmc.setFixedHeight(depth)

            hcompanyno = hdf['Companyno'].iloc[0]
            hfilt = (hdf['Companyno'] == hcompanyno).to_numpy().nonzero()[0][0]
            hrow = hdf.iloc[hfilt].astype(str)
            topleft = hrow['Company']+'\n'+hrow['Farmname']+' ('+hrow['Farmno']+')'
            topright = 'Hole no: '+hrow['Companyno']+'\n Sheet 1 of 1'
            bottomleft = 'Drill date: '+hrow['Drill date'].split()[0]
            bottomleft += '\nDepth from: '+hrow['Depth from']
            bottomleft += '\nDepth to: '+hrow['Depth to']
            bottomright = 'Elevation: '+hrow['Elevation']
            bottomright += '\nLatitude: '+hrow['Declat']
            bottomright += '\nLongitude: '+hrow['Declon']
            self.label_topleft.setText(topleft)
            self.label_topright.setText(topright)
            self.label_bottomleft.setText(bottomleft)
            self.label_bottomright.setText(bottomright)

            self.mmc2.update_legend(data)
            self.mmc.update_log(data)

    def run(self):
        """
        Run.

        Returns
        -------
        None.

        """
        self.show()
        if 'Borehole' in self.indata:
            data = self.indata['Borehole']

        for i in data:
            self.combobox1.addItem(i)
        self.change_band()


def gethatch(svgfile):
    """
    Get Hatching from SVG file.

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
    defs = root.find('{http://www.w3.org/2000/svg}defs')

    for pat in defs.findall('{http://www.w3.org/2000/svg}pattern'):
        for child in pat:
            if child.tag == '{http://www.w3.org/2000/svg}g':
                tag = child.find('{http://www.w3.org/2000/svg}path')
                trans = child.get('transform')
                tmp = trans[10:-1].split(',')
                tmp = list(map(float, tmp))
                translate.append(tmp)
            elif child.tag == '{http://www.w3.org/2000/svg}path':
                tag = child
                translate.append([0., 0.])
            else:
                continue

            dpath.append(tag.get('d'))
            # style.append(tag.get('style'))
            stmp = {}
            for i in tag.get('style').split(';'):
                tmp = i.split(':')
                stmp[tmp[0]] = tmp[1]
            style.append(stmp)

    # translate = np.array(translate)
    pverts = {}
    pcodes = {}

    for idx, trans in enumerate(translate):
        tmp = re.split(r'(z|c|m|C|M|L|l)', dpath[idx])
        if tmp[0] == '':
            tmp.remove('')
        if tmp[-1] == '':
            tmp[-1] = '0., 0.'

        # Start one graphics segment here
        rtmp = [0, 0]
        verts = []
        codes = []
        for i in range(0, len(tmp)-1, 2):
            # Load in keys and values
            pkey = tmp[i]
            vtmp = re.split(r',| ', tmp[i+1])
            while '' in vtmp:
                vtmp.remove('')
            vtmp = list(map(float, vtmp))
            pvals = np.reshape(vtmp, (len(vtmp)//2, 2))

            # Correct relative coordinates

            if 'm' in pkey or 'l' in pkey:
                pvals = np.cumsum(pvals, axis=0)+rtmp

            if 'c' in pkey:
                for k in range(0, pvals.shape[0], 3):
                    pvals[k:k+3] += rtmp
                    rtmp = pvals[k+2]

            # construct vertices and codes for paths
            if pkey.upper() == 'M':
                verts += (pvals+trans).tolist()
                codes += [Path.MOVETO]
                codes += [Path.LINETO]*(len(pvals)-1)

                if pvals.std() == 0.0 and pvals.size > 2:
                    verts[-2][0] += 0.5

            if pkey.upper() == 'L':
                verts += (pvals+trans).tolist()
                codes += [Path.LINETO]*(len(pvals))

            if pkey.upper() == 'C':
                verts += (pvals+trans).tolist()
                codes += [Path.CURVE4]*len(pvals)

            # if pkey.upper() == 'Z':
            #     verts += [trans]
            #     codes += [Path.LINETO]

            rtmp = pvals[-1]

        if style[idx]['fill'] not in pverts:
            pverts[style[idx]['fill']] = verts
            pcodes[style[idx]['fill']] = codes
        else:
            pverts[style[idx]['fill']] += verts
            pcodes[style[idx]['fill']] += codes

    for i in pverts:
        pverts[i] = np.array(pverts[i])
        pverts[i] /= np.max(pverts[i])
        pverts[i] *= 4

#  Draw here
#    fig, ax = plt.subplots(figsize=(6, 6))
#    for i in pverts:
#        pathfin = Path(pverts[i], pcodes[i])
#        pp1 = mpatches.PathPatch(pathfin, transform=ax.transData, fill=False)
#
#        if i == '#ffffff':
#            pp1.set_fill(True)
#            pp1.set_facecolor('w')
#        elif i != 'none':
#            pp1.set_fill(True)
#            pp1.set_facecolor('k')
#        ax.add_patch(pp1)
#    plt.axis('equal')
#    plt.tight_layout()
#    plt.show()

    return pverts, pcodes


def commentprep(mystring, slen=50):
    """
    Creates the correct case for a string and inserts carriage returns.

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
    finstring = ''
    mystring = mystring.capitalize()
    for word in mystring.split():
        if re.search(r'\d', word):
            finstring += ' ' + word
        else:
            finstring += ' ' + word.capitalize()

    finstring = finstring.strip()
    finstring = textwrap.fill(finstring, slen)
    if '\n' in finstring:
        finstring = finstring[:finstring.index('\n')]+'...'

    return finstring


def chkname(iname):
    """
    Checks filename for illegal characters.

    Parameters
    ----------
    iname : str
        Input filename.

    Returns
    -------
    iname : str
        Corrected filename.

    """
    charlist = [['#', '_hash_'],
                ['%', '_perc_'],
                ['&', '_amp_'],
                ['{', '_lb_'],
                ['}', '_rb_'],
                ['\\', '_bs_'],
                ['<', '_lrb_'],
                ['>', '_rab_'],
                ['*', '_ast_'],
                ['?', '_q_'],
                ['/', '_fs_'],
                ['$', '_dol_'],
                ['!', '_exc_'],
                ['"', '_dq_'],
                ["'", '_sq_'],
                [':', '_col_'],
                ['@', '_at_']]
    for ichar, nchar in charlist:
        iname = iname.replace(ichar, nchar)

    return iname
