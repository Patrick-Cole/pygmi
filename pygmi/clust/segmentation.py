# -----------------------------------------------------------------------------
# Name:        segmentation.py (part of PyGMI)
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
"""Image segmentation routines."""
import copy
import numpy as np
from sklearn.cluster import DBSCAN
import skimage
import sklearn.preprocessing as skp
from numba import jit
from PyQt5 import QtWidgets, QtCore, QtGui
import pygmi.menu_default as menu_default


class ImageSeg(QtWidgets.QDialog):
    """
    Image Segmentation.

    Attributes
    ----------
    pbar : progressbar
        reference to a progress bar.
    parent : parent
        reference to the parent routine
    outdata : dictionary
        dictionary of output datasets
    ifile : str
        input file name. Used in main.py
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if parent is None:
            self.showprocesslog = print
            self.pbar = None
        else:
            self.showprocesslog = parent.showprocesslog
            self.pbar = parent.pbar

        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.ifile = ''

        self.scale = QtWidgets.QLineEdit('1000')
        self.wcompact = QtWidgets.QLineEdit('0.5')
        self.wcolor = QtWidgets.QLineEdit('0.9')
        self.eps = QtWidgets.QLineEdit('0.1')
        self.dbscan = QtWidgets.QCheckBox('Use DBSCAN to group segments')

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
        helpdocs = menu_default.HelpButton('pygmi.clust.segmentation')

        lbl_wcompact = QtWidgets.QLabel('Compactness weight')
        lbl_wcolor = QtWidgets.QLabel('Color weight')
        lbl_scale = QtWidgets.QLabel('Maximum allowable cost function')
        lbl_eps = QtWidgets.QLabel('DBSCAN eps')

        val = QtGui.QDoubleValidator(0.0, 1.0, 2)
        val.setNotation(QtGui.QDoubleValidator.StandardNotation)
        val.setLocale(QtCore.QLocale(QtCore.QLocale.C))

        self.wcompact.setValidator(val)
        self.wcolor.setValidator(val)
        self.dbscan.setChecked(False)

        val = QtGui.QDoubleValidator()
        val.setBottom = 0
        self.scale.setValidator(QtGui.QIntValidator(self))

        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Image Segmentation')

        gridlayout_main.addWidget(lbl_wcompact, 0, 0, 1, 1)
        gridlayout_main.addWidget(self.wcompact, 0, 1, 1, 1)

        gridlayout_main.addWidget(lbl_wcolor, 1, 0, 1, 1)
        gridlayout_main.addWidget(self.wcolor, 1, 1, 1, 1)

        gridlayout_main.addWidget(lbl_scale, 2, 0, 1, 1)
        gridlayout_main.addWidget(self.scale, 2, 1, 1, 1)

        gridlayout_main.addWidget(self.dbscan, 3, 0, 1, 2)

        gridlayout_main.addWidget(lbl_eps, 4, 0, 1, 1)
        gridlayout_main.addWidget(self.eps, 4, 1, 1, 1)

        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)

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
        if 'Raster' not in self.indata:
            self.showprocesslog('No Raster Data.')
            return False

        data1 = []
        for i in self.indata['Raster']:
            data1.append(i.data.data)

        data1 = np.array(data1)
        data1 = np.moveaxis(data1, 0, -1)

        if not nodialog:
            tmp = self.exec_()

            if tmp != 1:
                return False

        scale = float(self.scale.text())
        wcolor = float(self.wcolor.text())
        wcompact = float(self.wcompact.text())
        eps = float(self.eps.text())

        doshape = True

        omap = self.segment1(data1, scale=scale, wcolor=wcolor,
                             wcompact=wcompact, doshape=doshape)

        odat = copy.deepcopy(self.indata['Raster'][0])
        odat.data = np.ma.array(omap, mask=self.indata['Raster'][0].data.mask)
        odat.dataid = 'Segments'

        self.outdata['Raster'] = [odat]

        if not self.dbscan.isChecked():
            return True

        means = []
        for i in range(odat.data.max()+1):
            tmp = data1[odat.data == i]
            if tmp.size == 0:
                continue
            means.append(tmp.mean(0))

        means = np.array(means)
        means = skp.StandardScaler().fit_transform(means)
        dbout = DBSCAN(eps=eps).fit_predict(means)

        data2 = odat.data.copy()

        newmax = dbout.max()+1
        for i, val in enumerate(dbout):
            filt = (odat.data == i)
            if val == -1:
                data2[filt] = newmax
                newmax += 1
                continue
            data2[filt] = val

        odat.data = data2
        self.outdata['Raster'] = [odat]

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
        self.scale.setText(projdata['scale'])
        self.scale.setText(projdata['wcolor'])
        self.scale.setText(projdata['wcompact'])

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

        projdata['scale'] = self.scale.text()
        projdata['wcolor'] = self.wcolor.text()
        projdata['wcompact'] = self.wcompact.text()

        return projdata

    def segment1(self, data, scale=500, wcolor=0.5, wcompact=0.5,
                 doshape=True):
        """
        Segment Part 1.

        Parameters
        ----------
        data : numpy array
            Input data.
        scale : TYPE, optional
            Scale. The default is 500.
        wcolor : float, optional
            Color weight. The default is 0.5.
        wcompact : float, optional
            Compactness weight. The default is 0.5.
        doshape : bool, optional
            Perform shape segmentation. The default is True.

        Returns
        -------
        omap : numpy array
            Output data.

        """
        rows, cols, bands = data.shape

        self.showprocesslog('Initialising...')

        olist = {}
        slist = {}
        mlist = {}
        nlist = {}
        omap = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                tmp = []
                for ii in range(max(i-1, 0), min(i+2, rows)):
                    for jj in range(max(j-1, 0), min(j+2, cols)):
                        if ii == i and jj == j:
                            continue
                        tmp.append(ii*cols+jj)
                olist[i*cols+j] = set(tmp)
                for k in range(bands):
                    slist[(k, i*cols+j)] = 0
                    mlist[(k, i*cols+j)] = data[i, j, k]
                    nlist[(k, i*cols+j)] = 1
                omap[i, j] = i*cols+j

        self.showprocesslog('merging...')

        omap = self.segment2(omap, olist, slist, mlist, nlist, bands,
                             doshape, wcompact, wcolor, scale)

        self.showprocesslog('renumbering...')
        tmp = np.unique(omap)

        for i, val in enumerate(tmp):
            omap[omap == val] = i

        return omap.astype(int)

    def segment2(self, omap, olist, slist, mlist, nlist, bands, doshape,
                 wcompact, wcolor, scale):
        """
        Segment Part 2.

        Parameters
        ----------
        omap : numpy array
            DESCRIPTION.
        olist : dictionary
            DESCRIPTION.
        slist : dictionary
            DESCRIPTION.
        mlist : dictionary
            DESCRIPTION.
        nlist : dictionary
            DESCRIPTION.
        bands : int
            Number of bands in data.
        doshape : bool, optional
            Perform shape segmentation. The default is True.
        wcompact : float, optional
            Compactness weight. The default is 0.5.
        wcolor : float, optional
            Color weight. The default is 0.5.
        scale : TYPE, optional
            Scale. The default is 500.


        Returns
        -------
        omap : TYPE
            DESCRIPTION.

        """
        wband = np.ones(bands)/bands

        cnt = 0
        oldlen = len(olist.keys())+1

        _, cols = omap.shape
        rminmax = {}
        cminmax = {}
        for i in olist:
            rminmax[i] = [i//cols]*2
            cminmax[i] = [i-rminmax[i][0]*cols]*2

        while len(olist.keys()) != oldlen:
            oldlen = len(olist.keys())
            cnt += 1
            elist = set(olist.keys())

            clen = len(elist)
            if self.pbar is not None:
                self.pbar.setMaximum(clen)
                self.pbar.setMinimum(0)
                self.pbar.setValue(0)
            self.showprocesslog('Iteration number: '+str(cnt))
            oldperc = 0

            olist3 = olist.copy()

            while elist:
                i = elist.pop()

                if not olist3[i]:
                    continue

                hcolor = 0.
                sm2 = []
                nm2 = []
                mean2 = []
                ollist = list(olist3[i])
                for k in range(bands):
                    s2 = np.array([slist[(k, j)] for j in ollist])
                    x2 = np.array([mlist[(k, j)] for j in ollist])
                    n2 = np.array([nlist[(k, j)] for j in ollist])

                    n1 = nlist[(k, i)]
                    x1 = mlist[(k, i)]
                    s1 = slist[(k, i)]
                    nm = (n1+n2)

                    mean = (n1*x1+n2*x2)/nm
                    sm = np.sqrt((n1*(s1**2+(x1-mean)**2) +
                                  n2*(s2**2+(x2-mean)**2))/nm)
                    hcolor += np.abs(wband[k]*(nm*sm-(n1*s1+n2*s2)))

                    sm2.append(sm)
                    mean2.append(mean)
                    nm2.append(nm)

                if cnt > 1 and doshape is True:
                    rmin, rmax = rminmax[i]
                    cmin, cmax = cminmax[i]

                    somap = omap[max(0, rmin-1):rmax+2, max(0, cmin-1):cmax+2]

                    l1 = get_l(somap == i)
                    b1 = (rmax-rmin+cmax-cmin+2)*2

                    l1, b1 = 2, 4

                    l2 = []
                    b2 = []
                    lm = []
                    bm = []

                    for ol in ollist:
                        rmin1, rmax1 = rminmax[ol]
                        cmin1, cmax1 = cminmax[ol]

                        somap = omap[max(0, rmin1-1):rmax1+2,
                                     max(0, cmin1-1):cmax1+2]

                        ltmp = get_l(somap == ol)
                        btmp = (rmax1-rmin1+cmax1-cmin1+2)*2

                        ltmp, btmp = 4, 2

                        l2.append(ltmp)
                        b2.append(btmp)

                        rmin2 = min(rmin1, rmin)
                        rmax2 = max(rmax1, rmax)
                        cmin2 = min(cmin1, cmin)
                        cmax2 = max(cmax1, cmax)

                        somap = omap[max(0, rmin2-1):rmax2+2,
                                     max(0, cmin2-1):cmax2+2]

                        filt = (somap == ol) + (somap == i)
                        ltmp2 = get_l(filt)
                        btmp2 = (rmax2-rmin2+cmax2-cmin2+2)*2

                        ltmp2, btmp2 = ltmp+l1, btmp+b1

                        lm.append(ltmp2)
                        bm.append(btmp2)

                    l2 = np.array(l2)
                    b2 = np.array(b2)
                    lm = np.array(lm)
                    bm = np.array(bm)

                    hsmooth = nm*lm/bm-(n1*l1/b1+n2*l2/b2)
                    hcompact = np.sqrt(nm)*lm-(np.sqrt(n1)*l1+np.sqrt(n2)*l2)

                    hshape = wcompact*hcompact + (1-wcompact)*hsmooth
                    hdiff = wcolor*hcolor+(1-wcolor)*hshape

                else:
                    hdiff = hcolor

                if hdiff.min() > scale:
                    continue

                mindiff = hdiff.argmin()
                hind = ollist[mindiff]

                olist[i] = olist[i] | olist[hind]
                olist[i].remove(i)
                olist[i].remove(hind)
                olist3[i] = olist[i].copy()

                rmm1 = min(rminmax[i][0], rminmax[hind][0])
                rmm2 = max(rminmax[i][1], rminmax[hind][1])
                rminmax[i] = [rmm1, rmm2]

                cmm1 = min(cminmax[i][0], cminmax[hind][0])
                cmm2 = max(cminmax[i][1], cminmax[hind][1])
                cminmax[i] = [cmm1, cmm2]

                for k in range(bands):
                    slist[(k, i)] = sm2[k][mindiff]
                    mlist[(k, i)] = mean2[k][mindiff]
                    nlist[(k, i)] = nm2[k][mindiff]
                    del slist[(k, hind)]
                    del mlist[(k, hind)]
                    del nlist[(k, hind)]

                for j in olist[hind]:
                    if j == i:
                        continue
                    olist[j].discard(hind)
                    olist[j].add(i)

                    olist3[j] = olist[j].copy()
                    olist3[j].discard(i)

                del olist[hind]

                elist.discard(hind)

                cnow = clen-len(elist)
                if cnow*1000//clen-oldperc > 0:
                    if self.pbar is not None:
                        self.pbar.setValue(cnow)
                    oldperc = cnow*1000//clen

                rmin, rmax = rminmax[i]
                cmin, cmax = cminmax[i]

                omap[rmin:rmax+1, cmin:cmax+1][omap[rmin:rmax+1, cmin:cmax+1] == hind] = i

        return omap


@jit(nopython=True, fastmath=True)
def get_l(data):
    """
    Get bounding box length.

    Parameters
    ----------
    data : numpy array
        Input data.

    Returns
    -------
    ltmp : int
        Bounding box length.

    """
    rows, cols = data.shape
    ltmp = 0
    for i in range(rows):
        for j in range(cols-1):
            ltmp += abs(data[i, j+1]-data[i, j])

    for i in range(rows-1):
        for j in range(cols):
            ltmp += abs(data[i+1, j]-data[i, j])

    return ltmp


def _testfn():
    """Test routine."""
    import sys
    import matplotlib.pyplot as plt
    from pygmi.raster.datatypes import Data
    from pygmi.misc import PTime
    from pygmi.raster import iodefs
    from matplotlib import rcParams

    rcParams['figure.dpi'] = 300

    APP = QtWidgets.QApplication(sys.argv)  # Necessary to test Qt Classes

    data1 = skimage.data.coffee()  # 400x600 48.5 secs

    # data1 = data1[:, :300]  # 26

    plt.imshow(data1)
    plt.axis('off')
    plt.show()

    wcolor = 0.5
    wcompact = 0.5
    doshape = True
    scale = 1000

    b1 = Data()
    b1.data = np.ma.array(data1[:, :, 0])
    b2 = Data()
    b2.data = np.ma.array(data1[:, :, 1])
    b3 = Data()
    b3.data = np.ma.array(data1[:, :, 2])

    data = [b1, b2, b3]

    # ifile = r'E:\Workdata\testdata.hdr'

    # data = iodefs.get_raster(ifile)

    ttt = PTime()
    IS = ImageSeg()
    IS.indata = {'Raster': data}
    # IS.settings(True)
    IS.settings()
    ttt.since_last_call()

    odata = IS.outdata['Raster'][0]

    plt.imshow(odata.data)
    plt.axis('off')
    plt.show()

    # means = []
    # for i in range(odata.data.max()+1):
    #     tmp = data1[odata.data==i]
    #     means.append(tmp.mean(0))

    # means = np.array(means)
    # means = skp.StandardScaler().fit_transform(means)

    # dbout = DBSCAN(eps=0.15).fit_predict(means)

    # data2 = odata.data.copy()

    # newmax = dbout.max()+1
    # for i, val in enumerate(dbout):
    #     filt = (odata.data==i)
    #     if val == -1:
    #         data2[filt] = newmax
    #         newmax += 1
    #         continue
    #     data2[filt] = val

    # plt.imshow(data2, cmap='inferno')
    # plt.colorbar()
    # plt.show()

    # omap = segment1(data1, scale=scale, wcolor=wcolor,
    #                 wcompact=wcompact, doshape=doshape)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(omap)
    # plt.show()

#    xall = []
#    yall = []
#    for shp, _ in rasterio.features.shapes(omap):
#        for crds in shp['coordinates']:
#            x = np.array(crds)[:, 0]
#            y = np.array(crds)[:, 1]
#            xall.append(x)
#            yall.append(y)
#
#    plt.figure(figsize=(8, 8))
#    for i, _ in enumerate(xall):
#        plt.plot(xall[i], yall[i], 'k')
#    plt.imshow(data1)
#    plt.show()


if __name__ == "__main__":
    _testfn()

    print('Finished!')
