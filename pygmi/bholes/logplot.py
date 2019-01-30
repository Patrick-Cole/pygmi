"""
Logplot
-------

Software to plot CGS log files.
"""

import glob
import os
import xml
import re
import textwrap
import numpy as np
import pandas as pd
import ogr
import gdal
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pygmi.raster.iodefs import Data


def get_tiff(ifile, bandid=None):
    """
    Gets sentinel Data from envi

    Parameters
    ----------
    ifile : str
        filename to import

    Returns
    -------
    dat : PyGMI raster Data
        dataset imported
    """

    datall = {}
    ifile = ifile[:]

    dataset = gdal.Open(ifile, gdal.GA_ReadOnly)
    gtr = dataset.GetGeoTransform()

    for i in range(dataset.RasterCount):

        rtmp = dataset.GetRasterBand(i+1)
        nval = rtmp.GetNoDataValue()
#        bandmeta = rtmp.GetMetadata()
        bandid = rtmp.GetDescription()

        dat = Data()
        dat.data = rtmp.ReadAsArray()
        dat.data = np.ma.masked_equal(dat.data, nval)

        if dat.data.dtype.kind == 'i':
            if nval is None:
                nval = 999999
            nval = int(nval)
        elif dat.data.dtype.kind == 'u':
            if nval is None:
                nval = 0
            nval = int(nval)
        else:
            if nval is None:
                nval = 1e+20
            nval = float(nval)

        dat.tlx = gtr[0]
        dat.tly = gtr[3]
        dat.dataid = bandid

        dat.nullvalue = nval
        dat.rows = dataset.RasterYSize
        dat.cols = dataset.RasterXSize
        dat.xdim = abs(gtr[1])
        dat.ydim = abs(gtr[5])
        dat.wkt = dataset.GetProjection()
        datall[i+1] = dat

    if datall == {}:
        datall = None

    dataset = None
    return datall


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


def temp():
    """ Test stuff """
#    svgfile = r'C:\Work\Programming\Remote_Sensing\bh2\USGS-patterns.svg'
#    tree = xml.etree.ElementTree.parse(svgfile)
#
#    ofile = open(r'C:\Work\Programming\Remote_Sensing\bh2\test.txt', 'w')
#
#    root = tree.getroot()
#    meta = root.find('{http://www.w3.org/2000/svg}g')
#
#    id1 = []
#    for pat in meta.findall('{http://www.w3.org/2000/svg}text'):
#        for child in pat:
#            if child.tag == '{http://www.w3.org/2000/svg}tspan':
#                if child.text.isdigit():
#                    id1.append(child.text+',')
#                else:
#                    id1[-1] += ' ' + child.text
#
#    id1 = [i+'\n' for i in id1]
#
#    ofile = open(r'C:\Work\Programming\Remote_Sensing\bh2\test.csv', 'w')
#    ofile.writelines(id1)
#    ofile.close()

    ofile = open(r'C:\Work\Programming\Remote_Sensing\bh2\colors.txt')

    lines = ofile.readlines()

    ymck = []
    rgb = []
    hexi = []

    for i, line in enumerate(lines):
        lines[i] = line[:-1]
        if len(lines[i]) == 3:
            lines[i] = lines[i]+'0'
        y = lines[i][0]
        m = lines[i][1]
        c = lines[i][2]
        k = lines[i][3]

        if y == 'X':
            y = '10'
        if m == 'X':
            m = '10'
        if c == 'X':
            c = '10'
        if k == 'X':
            k = '10'

        y = int(y)*10
        m = int(m)*10
        c = int(c)*10
        k = int(k)*10
        ymck.append([y, m, c, k])

        r = int(255*(1-c/100)*(1-k/100))
        g = int(255*(1-m/100)*(1-k/100))
        b = int(255*(1-y/100)*(1-k/100))
        rgb.append([r, g, b])
        hexi.append(hex(r)[2:].zfill(2) +
                    hex(g)[2:].zfill(2) +
                    hex(b)[2:].zfill(2))

    aaa = open(r'C:\Work\Programming\Remote_Sensing\bh2\colors.csv', 'w')

    for i in hexi:
        aaa.write(i+'\n')
    aaa.close()


def mainmerge():
    """ merge log header and core files """
    idir = r"C:\Work\Programming\Remote_Sensing\bh2\\"
    odir = idir+'pics\\'
#    logfile = idir + r'logplot.xlsx'

#    allfiles = glob.glob(idir+'*(lith)*.xlsx')
    lithfile = idir+r'olma-coredata(lith).xlsx'
    headerfile = idir+r'olma-coredata(headers).xlsx'
    odir = idir+'olma-coredata\\'
    lithfile = idir+r'olma-coal(lith)x.xlsx'
    headerfile = idir+r'olma-coal(headers).xlsx'
    odir = idir+'olma-coal\\'

    # Create output directory, if it does not exist
    if not os.path.exists(odir):
        os.makedirs(odir)
# Load in lithology information
    xl = pd.ExcelFile(lithfile)
    df = xl.parse(xl.sheet_names[0])
    xl.close()

    xl = pd.ExcelFile(headerfile)
    hdf = xl.parse(xl.sheet_names[0])
    xl.close()

    df['Depth to'] = df['Depth from']-df['Depth to']
#    df = df.assign(thick=(df['Depth to']-df['Depth from']))
#    sLength = len(df1['Companyno'])

    dfhdf = pd.merge(df, hdf, on='Companyno')

    dfhdf.to_csv(idir+'out.csv')


def profile():
    """ Creates a profile from borehole logs """
#Line 1 –Van Ryn
    boreholes = [42688, 4020770, 4020785]
#Line 2 Alexander Dam
#    boreholes = [4004109, 15754, 4064621, 16795, 15611, 4004619]
#Line 3 Blesbokspruit
#    boreholes = [4004033, 4004029, 21120, 21113]
#Line 4 Geduld Dam
#    boreholes = [4199734, 15659]
#Line 5
#    boreholes = [14425, 15554, 15754, 4064633]

    idir = r"C:\Work\Programming\Remote_Sensing\bh2\\"
    dfile = idir+'dem\\s27_e028_TM29.tif'
    sfile = idir+'line1.shp'

    lithfile1 = idir+r'olma-coredata(lith).xlsx'
    headerfile1 = idir+r'olma-coredata(headers).xlsx'
    lithfile2 = idir+r'olma-coal(lith)x.xlsx'
    headerfile2 = idir+r'olma-coal(headers).xlsx'

    logfile = idir + r'logplot.xlsx'

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

    xl = pd.ExcelFile(logfile)
    cgs = xl.parse('CGS')
    xl.close()
    clith = cgs.set_index('lithology').to_dict()['lithology description']

    dtm = get_tiff(dfile)[1]

    xl = pd.ExcelFile(lithfile1)
    df1 = xl.parse(xl.sheet_names[0])
    xl.close()
    xl = pd.ExcelFile(lithfile2)
    df2 = xl.parse(xl.sheet_names[0])
    xl.close()

    df = df1.append(df2)

    xl = pd.ExcelFile(headerfile1)
    hdf1 = xl.parse(xl.sheet_names[0])
    xl.close()
    xl = pd.ExcelFile(headerfile2)
    hdf2 = xl.parse(xl.sheet_names[0])
    xl.close()

    hdf = hdf1.append(hdf2)

   # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(sfile):
        shpDriver.DeleteDataSource(sfile)
    outDataSource = shpDriver.CreateDataSource(sfile)
    outLayer = outDataSource.CreateLayer(sfile, geom_type=ogr.wkbLineString)

   # create a field
    idField = ogr.FieldDefn('Lithology', ogr.OFTString)
    outLayer.CreateField(idField)
    idField = ogr.FieldDefn('Strat', ogr.OFTString)
    outLayer.CreateField(idField)
    idField = ogr.FieldDefn('Boreholeid', ogr.OFTString)
    outLayer.CreateField(idField)

    # Create the feature and set values
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)

    x = None
    y = None
    elev = None
    rdist = None
    hdf['Lox'] = -1* hdf['Lox']  # convert to TM sign
    hdf['Loy'] = -1* hdf['Loy']  # convert to TM sign

    hdf = hdf.sort_values(['Lox', 'Loy'])

    hfilt = []
    for bid in boreholes:
        try:
            hfilt.append((hdf['Boreholeid'] == bid).nonzero()[0][0])
        except:
            print('Error with boreholeid ',bid)
            return

    for bid in boreholes:
        hfilt = (hdf['Boreholeid'] == bid).nonzero()[0][0]
        hrow = hdf.iloc[hfilt]
        dfilt = (df['Boreholeid'] == bid).nonzero()[0]
        drows = df.iloc[dfilt]
#        drows['Stratigraphy'] = drows['Stratigraphy'].replace(np.nan, 'none')

        if x is None:
            rdist = [0]
            elev = [hrow['Elevation']]
            x = [hrow['Loy']]
            y = [hrow['Lox']]
        else:
            rdist.append(np.sqrt((x[-1]-hrow['Loy'])**2+(y[-1]-hrow['Lox'])**2))
            rdist[-1] = rdist[-1] + rdist[-2]
            x.append(hrow['Loy'])
            y.append(hrow['Lox'])
            elev.append(hrow['Elevation'])

        if drows['Depth from'].iloc[0] != 0.:
            line = None
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(rdist[-1], elev[-1]-drows['Depth from'].iloc[0])
            line.AddPoint(rdist[-1], elev[-1])

            outFeature.SetGeometry(line)
            outFeature.SetField('Lithology', clith['NOR'])
            outFeature.SetField('Strat', 'No Information')
            outFeature.SetField('Boreholeid', str(bid))
            outLayer.CreateFeature(outFeature)

        for _, row in drows.iterrows():
            line = None
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(rdist[-1], elev[-1]-row['Depth from'])
            line.AddPoint(rdist[-1], elev[-1]-row['Depth to'])

            strat = str(row['Stratigraphy']).capitalize()
            rank = str(row['Rank'])
            if strat == 'Nan':
                strat = 'No Information'
            if rank == 'nan':
                rank = 'none'
            rank = rlookup[rank]
            strat = strat+' '+rank
            strat = strat.title()

            outFeature.SetGeometry(line)
            outFeature.SetField('Lithology', clith[row['Lithology']])
            outFeature.SetField('Strat', strat)
            outFeature.SetField('Boreholeid', str(bid))
            outLayer.CreateFeature(outFeature)

    for i in range(1, len(x)):
        c1 = (x[i-1]-dtm.tlx)/dtm.xdim
        c2 = (x[i]-dtm.tlx)/dtm.xdim
        r1 = (dtm.tly-y[i-1])/dtm.ydim
        r2 = (dtm.tly-y[i])/dtm.ydim

        numpnts = int(max(abs(c1-c2), abs(r1-r2)))

        xvals = np.linspace(c1, c2, numpnts)
        yvals = np.linspace(r1, r2, numpnts)

        rows = np.round(yvals).astype(int)
        cols = np.round(xvals).astype(int)

        zvals = dtm.data[rows, cols]
        zvals = zvals.data.astype(float)
        rvals = np.linspace(rdist[i-1], rdist[i], numpnts)

        line = None
        line = ogr.Geometry(ogr.wkbLineString)

        for j in range(numpnts):
            line.AddPoint(rvals[j], zvals[j])

        outFeature.SetGeometry(line)
        outFeature.SetField('Lithology', 'DTM')
        outFeature.SetField('Strat', 'DTM')
        outFeature.SetField('Boreholeid', 'DTM')
        outLayer.CreateFeature(outFeature)

    outFeature = None
    print('Finished!')


if __name__ == "__main__":
    profile()
#    mainmerge()
#    main_pages()
#    temp()
