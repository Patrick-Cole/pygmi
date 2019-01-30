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
menu. The following are supported:
"""


import os
import xml
import re
import textwrap
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as \
    NavigationToolbar


class MyMplCanvas(FigureCanvas):
    """
    Canvas for the actual plot

    Attributes
    ----------
    axes : matplotlib subplot
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        fig = Figure()
        self.parent = parent

        FigureCanvas.__init__(self, fig)

    def update_legend(self, data1):
        """
        Update legend
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
            svgfile = idir + '\\' + str(int(cgs[i]))+'.svg'
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
        Update the raster plot

        Parameters
        ----------
        data1 : PyGMI log data
            log dataset to be used
        """

        fig = self.figure
        fig.clear()
        ax = fig.gca()
        fig.subplots_adjust(top=0.995)
        fig.subplots_adjust(bottom=0.005)
        fig.subplots_adjust(left=0.005)
        fig.subplots_adjust(right=0.25)

        idir = __file__.rpartition('\\')[0]
        logfile = idir+'\\logplot.xlsx'

        pagewidth = 8
        pageheight = 8
        dpp = 25  # depth per page
        wpp = dpp*pagewidth/pageheight
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
#        clith = cgs.set_index('lithology').to_dict()['lithology description']
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
            svgfile = idir + '\\' + str(int(cgs[i]))+'.svg'
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
#        rank = np.array(df['Rank'].replace(np.nan, 'none'))

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
#            pdf = PdfPages(odir+chkname(hrow['Companyno']+'.pdf'))
#        fig.set_figheight(pageheight*numpages)
#        fig.set_figwidth(pagewidth)

#        ax.set_xlim((0, wpp))
        ax.set_ylim((-dpp*numpages, 0.))
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.set_frame_on(False)

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
#        self.figure.tight_layout()


class GraphWindow(QtWidgets.QDialog):
    """
    Graph Window - The QDialog window which will contain our image

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    """
    def __init__(self, parent=None):
        super(QtWidgets.QDialog, self).__init__(parent)
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Graph Window")

        vbl = QtWidgets.QVBoxLayout(self)  # self is where layout is assigned
        hbl1 = QtWidgets.QHBoxLayout()
        hbl2 = QtWidgets.QHBoxLayout()
        self.hbl = QtWidgets.QHBoxLayout()
        self.mmc = MyMplCanvas(self)
        self.mmc2 = MyMplCanvas(self)
        mpl_toolbar = NavigationToolbar(self.mmc, self.parent)
#        self.mmc.setFixedHeight(3000)
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
        self.combobox2 = QtWidgets.QComboBox()
        self.label1 = QtWidgets.QLabel()
        self.label2 = QtWidgets.QLabel()
        self.label1.setText('Bands:')
        self.label2.setText('Bands:')
        self.hbl.addWidget(self.label1)
        self.hbl.addWidget(self.combobox1)
        self.hbl.addWidget(self.label2)
        self.hbl.addWidget(self.combobox2)

        hbl1.addWidget(self.label_topleft)
        hbl1.addWidget(self.label_topright)
        hbl2.addWidget(self.label_bottomleft)
        hbl2.addWidget(self.label_bottomright)
        vbl.addLayout(hbl1)
        vbl.addWidget(self.scroll)
        vbl.addLayout(hbl2)
        vbl.addWidget(self.mmc2)
#        vbl.addWidget(mpl_toolbar)
        vbl.addLayout(self.hbl)

        self.setFocus()

        self.combobox1.currentIndexChanged.connect(self.change_band)
        self.combobox2.currentIndexChanged.connect(self.change_band)

    def change_band(self):
        """ Combo box to choose band """
        pass


class PlotLog(GraphWindow):
    """
    Plot Raster Class

    Attributes
    ----------
    label2 : QLabel
        reference to GraphWindow's label2
    combobox2 : QComboBox
        reference to GraphWindow's combobox2
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    """
    def __init__(self, parent):
        GraphWindow.__init__(self, parent)
        self.label2.hide()
        self.combobox2.hide()
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
            hfilt = (hdf['Companyno'] == hcompanyno).nonzero()[0][0]
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
        """ Run """
        self.show()
        if 'Borehole' in self.indata:
            data = self.indata['Borehole']

        for i in data:
            self.combobox1.addItem(i)
        self.change_band()


def main_pages():
    """ main """
    idir = r"C:\Work\Programming\Remote_Sensing\bh2\\"
    odir = idir+'pics\\'
    logfile = idir + r'logplot.xlsx'

#    allfiles = glob.glob(idir+'*(lith)*.xlsx')
    lithfile = idir+r'olma-coredata(lith).xlsx'
    headerfile = idir+r'olma-coredata(headers).xlsx'
    odir = idir+'olma-coredata\\'

    lithfile = idir+r'olma-coal(lith)x.xlsx'
    headerfile = idir+r'olma-coal(headers).xlsx'
    odir = idir+'olma-coal\\'

    pagewidth = 8
    pageheight = 11
    dpp = 25  # depth per page
    wpp = dpp*pagewidth/pageheight
    fontsize = 10
    dpi = (fontsize/72)*(dpp/pageheight)  # fontsize in inches * depth per inch

    # Create output directory, if it does not exist
    if not os.path.exists(odir):
        os.makedirs(odir)

#    scale = 750
#    scale = 75
#    pagey = -depthto[indx[-1]]*100/2.54/scale

###############################################################################
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
        svgfile = idir + str(int(cgs[i]))+'.svg'
        pverts, pcodes = gethatch(svgfile)
        hatch[i] = [pverts, pcodes]

# Load in lithology information
    xl = pd.ExcelFile(lithfile)
    df = xl.parse(xl.sheet_names[0])
    xl.close()

    xl = pd.ExcelFile(headerfile)
    hdf = xl.parse(xl.sheet_names[0])
    xl.close()

    companyno = np.array(df['Companyno'])
    depthfrom = -1*np.array(df['Depth from'])
    depthto = -1*np.array(df['Depth to'])
    lithd = np.array(df['Lithology description'].replace(np.nan, ''))
    lith = np.array(df['Lithology'])
    strat = np.array(df['Stratigraphy'].replace(np.nan, 'none'))
    rank = np.array(df['Rank'].replace(np.nan, 'none'))

    for hcompanyno in df['Companyno'].unique():
        print(hcompanyno)
#        if str(hcompanyno) != 'BN7':
#            continue

        indx = np.nonzero(companyno == hcompanyno)[0]
        numpages = int(abs(depthto[indx][-1]//dpp))
    # Load in header information
    # Farmname can be Farm name
    # Farm no can be Farm no
        hfilt = (hdf['Companyno'] == hcompanyno).nonzero()[0][0]
        hrow = hdf.iloc[hfilt].astype(str)

        topleft = hrow['Company']+'\n'+hrow['Farmname']+' ('+hrow['Farmno']+')'
        topright = 'Hole no: '+hrow['Companyno']+'\n Sheet 1 of 1'
        bottomleft = 'Drill date: '+hrow['Drill date'].split()[0]
        bottomleft += '\nDepth from: '+hrow['Depth from']
        bottomleft += '\nDepth to: '+hrow['Depth to']
        bottomright = 'Elevation: '+hrow['Elevation']
        bottomright += '\nLatitude: '+hrow['Declat']
        bottomright += '\nLongitude: '+hrow['Declon']

    ###########################################################################
    # Start of each borehole plot
    # Locations of the text lithology labels

        pltcnt = 1
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
        plt.figure(figsize=(pagewidth, pageheight))

        ax = plt.gca()
        ax.set_xlim((0, wpp))
        ax.set_ylim((-dpp*pltcnt, -dpp*(pltcnt-1)))
        ax.get_xaxis().set_visible(False)

    # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0., 1.0, topleft, bbox=props, va='bottom')
        ax.text(wpp, 1.0, topright, bbox=props, va='bottom', ha='right')
        ax.text(0., -dpp-1, bottomleft, bbox=props, va='top')
        ax.text(wpp, -dpp-1, bottomright, bbox=props, va='top', ha='right')

        for i in indx:
            # This next line is to skip summary lines for a group.
            if i < indx[-1]:
                if depthfrom[i] == depthfrom[i+1]:
                    continue

            depths = np.arange(0, -depthto[i], dpp)
            depths = depths[depths > -depthfrom[i]]
            depths = np.append(-depthfrom[i], depths)
            depths = depths[depths < -depthto[i]]
            depths = np.append(depths, -depthto[i])
            dfromlist = -depths[:-1]
            dtolist = -depths[1:]

            if i == indx[-1] and lith[i] == 'NOR':
                dfromlist = [dfromlist[0]]
                dtolist = [dtolist[0]]

            pverts, pcodes = hatch[lith[i]]
            scol = '#'+col[stratcol[strat[i]]]

            for dfrom, dto in np.transpose([dfromlist, dtolist]):
                if dfrom != dfromlist[0]:
                    plt.savefig(odir+chkname(hcompanyno+'_' +
                                             str(pltcnt)+'of'+str(numpages) +
                                             '.pdf'), bbox_inches='tight')
                    plt.close()
                    pltcnt += 1
                    plt.figure(figsize=(pagewidth, pageheight))
                    ax = plt.gca()
                    ax.set_xlim((0, wpp))
                    ax.set_ylim((-dpp*pltcnt, -dpp*(pltcnt-1)))
                    ax.get_xaxis().set_visible(False)

                    topright = ('Hole no: '+hcompanyno+'\n Sheet ' +
                                str(pltcnt) + ' of '+str(numpages))

                    ax.text(0., -dpp*(pltcnt-1)+1, topleft,
                            bbox=props, va='bottom')
                    ax.text(wpp, -dpp*(pltcnt-1)+1, topright, bbox=props,
                            va='bottom', ha='right')
                    ax.text(0., -dpp*(pltcnt)-1, bottomleft, bbox=props,
                            va='top')
                    ax.text(wpp, -dpp*(pltcnt)-1, bottomright, bbox=props,
                            va='top', ha='right')
                else:
                    va = 'center'
                    texty = lithdpos[np.nonzero(indx == i)[0][0]]
                    plt.plot([4, 5], [dfrom, texty], 'k', linewidth=1.0)
                    ax.text(5.2, texty, '{0:.2f}'.format(dfrom)+' '+lithd[i],
                            va='center')

                rect = mpatches.Rectangle([0, dto], 4,
                                          (dfrom-dto),
                                          fc=scol, ec='k')
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
            ax.text(5.2, -dpp*pltcnt+dpi,
                    '(Last entry; log truncated due to length)',
                    va=va, size=fontsize)

        ax.hlines(dtolist[0], 0, 4)  # Bottom of log
        plt.savefig(odir+chkname(hcompanyno+'_'+str(pltcnt)+'of' +
                                 str(numpages)+'.pdf'), bbox_inches='tight')
        plt.close()

        legend(pagewidth, pageheight, dpp, hdf, strat, lith, indx, props,
               stratcol, hatch, clith, col, rank, odir, dpi, hcompanyno)
        return

#    scale = -depthto[indx[-1]]*100/2.54
#    print('scale: 1:', scale)


def main_single():
    """ main """
    idir = r"C:\Work\Programming\Remote_Sensing\bh2\\"
    odir = idir+'pics\\'
    logfile = idir + r'logplot.xlsx'

    lithfile = idir+r'olma-coredata(lith).xlsx'
    headerfile = idir+r'olma-coredata(headers).xlsx'
#    lithfile = idir+r'olma-coal(lith)x.xlsx'
#    headerfile = idir+r'olma-coal(headers).xlsx'

    pagewidth = 8
    pageheight = 11
    dpp = 25  # depth per page
    wpp = dpp*pagewidth/pageheight
    fontsize = 10
    dpi = (fontsize/72)*(dpp/pageheight)  # fontsize in inches * depth per inch

###############################################################################
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
        svgfile = idir + str(int(cgs[i]))+'.svg'
        pverts, pcodes = gethatch(svgfile)
        hatch[i] = [pverts, pcodes]

# Load in lithology information
    xl = pd.ExcelFile(lithfile)
    df = xl.parse(xl.sheet_names[0])
    xl.close()

    xl = pd.ExcelFile(headerfile)
    hdf = xl.parse(xl.sheet_names[0])
    xl.close()

    companyno = np.array(df['Companyno'])
    depthfrom = -1*np.array(df['Depth from'])
    depthto = -1*np.array(df['Depth to'])
    lithd = np.array(df['Lithology description'].replace(np.nan, ''))
    lith = np.array(df['Lithology'])
    strat = np.array(df['Stratigraphy'].replace(np.nan, 'none'))
    rank = np.array(df['Rank'].replace(np.nan, 'none'))

    for hcompanyno in df['Companyno'].unique():
        print(hcompanyno)
#        if str(hcompanyno) != '411':
#            continue
        indx = np.nonzero(companyno == hcompanyno)[0]
        numpages = abs(depthto[indx][-1]//dpp)

    # Load in header information
        hfilt = (hdf['Companyno'] == hcompanyno).nonzero()[0][0]
        hrow = hdf.iloc[hfilt].astype(str)

        topleft = hrow['Company']+'\n'+hrow['Farmname']+' ('+hrow['Farmno']+')'
        topright = 'Hole no: '+hrow['Companyno']+'\n Sheet 1 of 1'
        bottomleft = 'Drill date: '+hrow['Drill date'].split()[0]
        bottomleft += '\nDepth from: '+hrow['Depth from']
        bottomleft += '\nDepth to: '+hrow['Depth to']
        bottomright = 'Elevation: '+hrow['Elevation']
        bottomright += '\nLatitude: '+hrow['Declat']
        bottomright += '\nLongitude: '+hrow['Declon']

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
        pdf = PdfPages(odir+chkname(hrow['Companyno']+'.pdf'))

        plt.figure(figsize=(pagewidth, pageheight*numpages))

        ax = plt.gca()
        ax.set_xlim((0, wpp))
        ax.set_ylim((-dpp*numpages, 0.))
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)

    # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0., 1.0, topleft, bbox=props, va='bottom')
        ax.text(wpp, 1.0, topright, bbox=props, va='bottom', ha='right')
        ax.text(0., -dpp*numpages-1, bottomleft, bbox=props, va='top')
        ax.text(wpp, -dpp*numpages-1, bottomright, bbox=props, va='top',
                ha='right')

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
            plt.plot([4, 5], [dfrom, texty], 'k', linewidth=1.0)
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

#        if lith[indx[-1]] == 'NOR':
#            ax.text(5.2, -dpp*numpages+dpi,
#                    '(Last entry; log truncated due to length)',
#                    va='center', size=fontsize)

        ax.hlines(dto, 0, 4)  # Bottom of log
#        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
    # Start Legend
        legend(pagewidth, pageheight, dpp, hdf, strat, lith, indx, props,
               stratcol, hatch, clith, col, rank, odir, dpi, hcompanyno, pdf)
        pdf.close()


def legend(pagewidth, pageheight, dpp, hdf, strat1, lith1, indx, props,
           stratcol, hatch, clith, col, rank1, odir, dpi, hcompanyno,
           pdf=None):
    """ Plot legend """

    wpp = dpp*pagewidth/pageheight
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

    hfilt = (hdf['Companyno'] == hcompanyno).nonzero()[0][0]
    hrow = hdf.iloc[hfilt].astype(str)

#    indx = np.nonzero(companyno == hcompanyno)[0]

    topleft = hrow['Company']+'\n'+hrow['Farmname']+' ('+hrow['Farmno']+')'
    ltopright = 'Legend: '+hrow['Companyno']+'\n'
    bottomleft = 'Drill date: '+hrow['Drill date'].split()[0]
    bottomleft += '\nDepth from: '+hrow['Depth from']
    bottomleft += '\nDepth to: '+hrow['Depth to']
    bottomright = 'Elevation: '+hrow['Elevation']
    bottomright += '\nLatitude: '+hrow['Declat']
    bottomright += '\nLongitude: '+hrow['Declon']

    plt.figure(figsize=(pagewidth, pageheight))
    ax = plt.gca()
    ax.set_xlim((0, wpp))
    ax.set_ylim((0, dpp))
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.text(0., -1.0, topleft, bbox=props, va='bottom')
    ax.text(wpp, -1.0, ltopright, bbox=props, va='bottom', ha='right')
    ax.text(0., dpp+1, bottomleft, bbox=props, va='top')
    ax.text(wpp, dpp+1, bottomright, bbox=props, va='top', ha='right')

    strat, idx = np.unique(strat1[indx], return_index=True)
    lith = np.unique(lith1[indx])
    rank = rank1[indx][idx]

# Do hatch legend
    ax.text(0.5, 0.7, 'Lithology', size=14)
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
        ax.text(4.7, j*2+1.6, clith[lith[j]])

# do color legend
    ax.text(10., 0.7, 'Stratigraphy', size=14)
    for j in np.arange(0, len(strat)):
        scol = '#'+col[stratcol[strat[j]]]
        rect = mpatches.Rectangle([10, j*2+1], 4, 1, fc=scol, ec='k')
        ax.add_patch(rect)
        if strat[j] == 'none':
            ax.text(14.2, j*2+2.0, strat[j].capitalize()+'\n')
        else:
            ax.text(14.2, j*2+2.0, strat[j].capitalize()+'\n' +
                    rlookup[rank[j]])

    ax.text(0.5, dpp-dpi, 'Log descriptions may be truncated. ' +
            'If so, refer to tabular log for full description.')

    if pdf is None:
        plt.savefig(odir+chkname(hcompanyno+'_legend.pdf'),
                    bbox_inches='tight')
    else:
        pdf.savefig(bbox_inches='tight')
    plt.close()
    return


def gethatch(svgfile):
    """ Test stuff """
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
#            style.append(tag.get('style'))
            stmp = {}
            for i in tag.get('style').split(';'):
                tmp = i.split(':')
                stmp[tmp[0]] = tmp[1]
            style.append(stmp)

#    translate = np.array(translate)
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

#            if pkey.upper() == 'Z':
#                verts += [trans]
#                codes += [Path.LINETO]

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
    """ creates the correct case for a string and inserts carriage returns"""
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
    """ checks filename for illegal characters """
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
