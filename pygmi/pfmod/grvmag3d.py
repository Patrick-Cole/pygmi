# -----------------------------------------------------------------------------
# Name:        grvmag3d.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2013 Council for Geoscience
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
""" Gravity and magnetic field calculations.
This uses the following algorithms:

Singh, B., Guptasarma, D., 2001. New method for fast computation of gravity
and magnetic anomalies from arbitrary polyhedral. Geophysics 66, 521 – 526.

Blakely, R.J., 1996. Potential Theory in Gravity and Magnetic Applications,
1st edn. Cambridge University Press, Cambridge, UK, 441 pp. 200-201



GravMag - Routine that will calculate the final versions of the field. Other,
related code is here as well, such as the inversion routines.

GeoData - The is a class which contains the geophysical information for a
single lithology. This includes the final calculated field for that lithology
only """

# pylint: disable=E1101, W0612, W0613
from PyQt4 import QtGui, QtCore
import scipy.interpolate as si
import numpy as np
import pylab as plt
import copy
import tempfile
import sys
from scipy.linalg import norm

if sys.platform.startswith('win'):
    if sys.maxsize > 2**32:
        if sys.version_info.major == 2:
            import pygmi.pfmod.grvmagc_27_x64 as grvmagc
        else:
            import pygmi.pfmod.grvmagc_33_x64 as grvmagc
    else:
        if sys.version_info.major == 2:
            import pygmi.pfmod.grvmagc_27_x86 as grvmagc
        else:
            import pygmi.pfmod.grvmagc_33_x86 as grvmagc

else:
    import pyximport
    pyximport.install()
    import pygmi.pfmod.grvmagc as grvmagc


class GravMag(object):
    """This class holds the generic magnetic and gravity modelling routines """
    def __init__(self, parent):

        self.parent = parent
        self.lmod1 = parent.lmod1
        self.lmod2 = parent.lmod2
        self.lmod = self.lmod1
        self.showtext = parent.showtext
        if hasattr(parent, 'pbars'):
            self.pbars = parent.pbars
        else:
            self.pbars = None
        self.oldlithindex = None
        self.mfname = self.parent.modelfilename
        self.tmpfiles = {}

        self.actionregionaltest = QtGui.QAction(self.parent)
        self.actioncalculate = QtGui.QAction(self.parent)
        self.actioncalculate2 = QtGui.QAction(self.parent)
        self.actioninvgrav = QtGui.QAction(self.parent)
        self.actioninvmag = QtGui.QAction(self.parent)
        self.setupui()

    def setupui(self):
        """ Setup UI """
        self.actionregionaltest.setText("Regional Test")
        self.actioncalculate.setText("Calculate Gravity at Gravity Height")
        self.actioncalculate2.setText(
            "Calculate Magnetics and Gravity at Magnetic Height")
        self.actioninvgrav.setText("Auto Model Gravity")
        self.actioninvmag.setText("Auto Model Magnetics")
        self.parent.toolbar.addAction(self.actionregionaltest)
        self.parent.toolbar.addAction(self.actioncalculate)
        self.parent.toolbar.addAction(self.actioncalculate2)
#        self.parent.toolbar.addAction(self.actioninvgrav)
#        self.parent.toolbar.addAction(self.actioninvmag)

        self.actionregionaltest.triggered.connect(self.test_pattern)
        self.actioncalculate.triggered.connect(self.calc_field)
        self.actioncalculate2.triggered.connect(self.calc_field_new)
#        self.actioninvgrav.triggered.connect(self.invgrav)
#        self.actioninvmag.triggered.connect(self.invmag)

    def calc_field_new(self):
        """ Pre field-calculation routine """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        # now do the calculations
        self.calc_field2(True, True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

    def calc_field(self):
        """ Pre field-calculation routine """
        # Update this
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1

        # Update the model from the view
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        # now do the calculations
        self.calc_field2(True)

        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot()

    def calc_field2(self, showreports=False, altcalc=False):
        """ Calculate magnetic and gravity field """
        if self.pbars is not None:
            self.pbars.resetall(mmax=2*(len(self.lmod.lith_list)-1)+1)
        if np.max(self.lmod.lith_index) == -1:
            self.showtext('Error: Create a model first')
            return
# Init some variables for convenience
        numx = int(self.lmod.numx)
        numy = int(self.lmod.numy)
        numz = int(self.lmod.numz)
        tmpfiles = self.tmpfiles

        for mlist in self.lmod.lith_list.items():
#            if 'Background' != mlist[0] and mlist[1].modified is True:
            if 'Background' != mlist[0]:
                mlist[1].modified = True
                self.showtext(mlist[0]+':')
                mlist[1].parent = self.parent
                mlist[1].pbars = self.parent.pbars
                mlist[1].showtext = self.parent.showtext
                if altcalc:
                    mlist[1].calc_origin2()
                else:
                    mlist[1].calc_origin()
                tmpfiles[mlist[0]] = self.save_layer(mlist)

        if showreports is True:
            self.showtext('Summing data')

        QtCore.QCoreApplication.processEvents()
# get height corrections
        tmp = np.copy(self.lmod.lith_index)
        tmp[tmp > -1] = 0
        hcor = np.abs(tmp.sum(2))

# model index
        modind = self.lmod.lith_index.copy()
        modind[modind == 0] = -1

# Get mlayers and glayers with correct rho and netmagn
        magval = np.zeros([numx, numy])
        grvval = np.zeros([numx, numy])

        if self.pbars is not None:
            self.pbars.resetsub(maximum=numx*(len(self.lmod.lith_list)-1))

        mtmp = magval.shape
        magval = magval.flatten()
        grvval = grvval.flatten()
        hcorflat = numz-hcor.flatten()

        for mlist in self.lmod.lith_list.items():
            if 'Background' == mlist[0]:
                continue
#            mfile = np.load(self.mfname+'_'+mlist[0]+'_tmp.npz')
            tmpfiles[mlist[0]].seek(0)
            mfile = np.load(tmpfiles[mlist[0]])
            if altcalc:
                mlayers = mfile['mlayers']
            elif mfile['mlayers'].size > 1:
                mlayers = mfile['mlayers']*mlist[1].netmagn()
            else:
                mlayers = np.zeros_like(mfile['glayers'])
            glayers = mfile['glayers']*mlist[1].rho()
            mijk = mlist[1].lith_index

            aaa = np.reshape(np.mgrid[0:mtmp[0], 0:mtmp[1]], [2, magval.size])

            for i in range(numx):
                if self.pbars is not None:
                    self.pbars.incr()
                grvmagc.calc_field2(i, numx, numy, numz, modind, hcor, aaa[0],
                                    aaa[1], mlayers, glayers, magval, grvval,
                                    hcorflat, mijk)

        magval.resize(mtmp)
        grvval.resize(mtmp)
        magval = magval.T
        grvval = grvval.T
        magval = magval[::-1]
        grvval = grvval[::-1]

# Update variables
        self.lmod.griddata['Calculated Magnetics'].data = magval
        self.lmod.griddata['Calculated Gravity'].data = grvval
        self.grav_regional()

        if self.lmod.lith_index.max() <= 0:
            self.lmod.griddata['Calculated Magnetics'].data *= 0.
            self.lmod.griddata['Calculated Gravity'].data *= 0.

        if 'Magnetic Dataset' in self.lmod.griddata:
            ztmp = self.gridmatch('Magnetic Dataset', 'Calculated Magnetics')
            self.lmod.griddata['Magnetic Residual'] = copy.deepcopy(
                self.lmod.griddata['Magnetic Dataset'])
            self.lmod.griddata['Magnetic Residual'].data = (
                self.lmod.griddata['Magnetic Dataset'].data - ztmp)
            self.lmod.griddata['Magnetic Residual'].bandid = \
                'Magnetic Residual'

        if 'Gravity Dataset' in self.lmod.griddata:
            ztmp = self.gridmatch('Gravity Dataset', 'Calculated Gravity')
            self.lmod.griddata['Gravity Residual'] = copy.deepcopy(
                self.lmod.griddata['Gravity Dataset'])
            self.lmod.griddata['Gravity Residual'].data = (
                self.lmod.griddata['Gravity Dataset'].data - ztmp)
            self.lmod.griddata['Gravity Residual'].bandid = 'Gravity Residual'

        self.parent.outdata['Raster'] = list(self.lmod1.griddata.values())
#        self.parent.outdata['Raster'] = \
#            [self.lmod1.griddata['Calculated Magnetics']]
        self.showtext('Calculation Finished')
        if self.pbars is not None:
            self.pbars.maxall()

    def save_layer(self, mlist):
        """ Routine saves the mlayer and glayer to a file """
#        self.mfname = self.parent.modelfilename

#        filename = self.mfname+'_'+mlist[0]+'_tmp.npz'

#        self.showtext('Saving '+filename.split('/')[-1] +
#                      ' (Can be deleted later if you wish)')

        outfile = tempfile.TemporaryFile()

        outdict = {}

        outdict['mlayers'] = mlist[1].mlayers
        outdict['glayers'] = mlist[1].glayers

        np.savez(outfile, **outdict)
        outfile.seek(0)

        mlist[1].mlayers = None
        mlist[1].glayers = None

        return outfile

    def calc_regional(self):
        """ Calculates a gravity regional value based on a single
        solid lithology model. This gets used in tab_param. The principle is
        that the maximum value for a solid model with fixed extents and depth,
        using the most COMMON lithology, would be the MAXIMUM AVERAGE value for
        any model which we would do. Therefore the regional is simply:
                    REGIONAL = OBS GRAVITY MEAN - CALC GRAVITY MAX
        This routine calculates the last term """

        ltmp = list(self.lmod1.lith_list.keys())
        ltmp.pop(ltmp.index('Background'))

        text, okay = QtGui.QInputDialog.getItem(
            self.parent, 'Regional Test',
            'Please choose the lithology to use:',
            ltmp)

        if not okay:
            return

        lmod1 = self.lmod1
        self.lmod2.lith_list.clear()

        numlayers = lmod1.numz
        layerthickness = lmod1.d_z

        self.lmod2.update(lmod1.numx, lmod1.numy, numlayers, lmod1.xrange[0],
                          lmod1.yrange[1], lmod1.zrange[1], lmod1.dxy,
                          layerthickness)

        self.lmod2.lith_index = self.lmod1.lith_index.copy()
        self.lmod2.lith_index[self.lmod2.lith_index != -1] = 1

        self.lmod2.lith_list['Background'] = GeoData(
            self.parent, lmod1.numx, lmod1.numy, self.lmod2.numz, lmod1.dxy,
            self.lmod2.d_z, lmod1.mht, lmod1.ght)

        self.lmod2.lith_list['Regional'] = GeoData(
            self.parent, lmod1.numx, lmod1.numy, self.lmod2.numz, lmod1.dxy,
            self.lmod2.d_z, lmod1.mht, lmod1.ght)

        lithn = self.lmod2.lith_list['Regional']
        litho = self.lmod1.lith_list[text]
        lithn.hintn = litho.hintn
        lithn.finc = litho.finc
        lithn.fdec = litho.fdec
        lithn.zobsm = litho.zobsm
        lithn.susc = litho.susc
        lithn.mstrength = litho.mstrength
        lithn.qratio = litho.qratio
        lithn.minc = litho.minc
        lithn.mdec = litho.mdec
        lithn.density = litho.density
        lithn.bdensity = litho.bdensity
        lithn.zobsg = litho.zobsg
        lithn.lith_index = 1

        self.lmod = self.lmod2
        self.calc_field2(False, True)
        self.lmod = self.lmod1

    def grd_to_lith(self, curgrid):
        """ Assign the DTM to the lithology model """
        d_x = curgrid.xdim
        d_y = curgrid.ydim
        utlx = curgrid.tlx
        utly = curgrid.tly
        gcols = curgrid.cols
        grows = curgrid.rows

        gxmin = utlx
        gymax = utly

        ndata = np.zeros([self.lmod.numy, self.lmod.numx])

        for i in range(self.lmod.numx):
            for j in range(self.lmod.numy):
                xcrd = self.lmod.xrange[0]+(i+.5)*self.lmod.dxy
                ycrd = self.lmod.yrange[1]-(j+.5)*self.lmod.dxy
                xcrd2 = int((xcrd-gxmin)/d_x)
                ycrd2 = int((gymax-ycrd)/d_y)
                if (ycrd2 >= 0 and xcrd2 >= 0 and ycrd2 < grows and
                        xcrd2 < gcols):
                    ndata[j, i] = curgrid.data.data[ycrd2, xcrd2]

        return ndata

    def grav_regional(self):
        """ If there is a gravity regional, then add it """
        if 'Gravity Regional' not in self.lmod.griddata:
            return
        zfin = self.gridmatch('Calculated Gravity', 'Gravity Regional')
        self.lmod.griddata['Calculated Gravity'].data += zfin

    def gridmatch(self, ctxt, rtxt):
        """ Matches the rows and columns of the second grid to the first
        grid """
        rgrv = self.lmod.griddata[rtxt]
        cgrv = self.lmod.griddata[ctxt]
        x = np.arange(rgrv.tlx, rgrv.tlx+rgrv.cols*rgrv.xdim,
                      rgrv.xdim)+0.5*rgrv.xdim
        y = np.arange(rgrv.tly-rgrv.rows*rgrv.ydim, rgrv.tly,
                      rgrv.xdim)+0.5*rgrv.ydim
        x_2, y_2 = np.meshgrid(x, y)
        z_2 = rgrv.data
        x_i = np.arange(cgrv.cols)*cgrv.xdim + cgrv.tlx + 0.5*cgrv.xdim
        y_i = np.arange(cgrv.rows)*cgrv.ydim + cgrv.tly - \
            cgrv.rows*cgrv.ydim + 0.5*cgrv.ydim
        xi2, yi2 = np.meshgrid(x_i, y_i)

        zfin = si.griddata((x_2.flatten(), y_2.flatten()), z_2.flatten(),
                           (xi2.flatten(), yi2.flatten()))
        zfin = np.ma.masked_invalid(zfin)
        zfin.shape = cgrv.data.shape

        return zfin

    def hist_eq(self, img, nbr_bins=256):
        """ Routine to calculate histogram equalization """
        imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
        im2 = np.interp(img.flatten(), bins[:-1], cdf)

        return im2.reshape(img.shape)

    def test_pattern(self):
        """ Displays a test pattern of the data - an indication of the edge of
        model field decay. It gives an idea aabout how reliable the calculated
        field on the edge of the model is. """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1

        self.calc_regional()

        magtmp = self.lmod2.griddata['Calculated Magnetics'].data
        grvtmp = self.lmod2.griddata['Calculated Gravity'].data

        regplt = plt.figure()
        axes = plt.subplot(1, 2, 1)
        etmp = self.dat_extent(self.lmod2.griddata['Calculated Magnetics'],
                               axes)
        plt.title('Magnetic Data')
        ims = plt.imshow(magtmp, extent=etmp)
        mmin = magtmp.mean()-2*magtmp.std()
        mmax = magtmp.mean()+2*magtmp.std()
        mint = (magtmp.std()*4)/10.
        csrange = np.arange(mmin, mmax, mint)
        cns = plt.contour(magtmp, levels=csrange, colors='b', extent=etmp)
        plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('nT')

        axes = plt.subplot(1, 2, 2)
        etmp = self.dat_extent(self.lmod2.griddata['Calculated Gravity'],
                               axes)
        plt.title('Gravity Data')
        ims = plt.imshow(grvtmp, extent=etmp)
        mmin = grvtmp.mean()-2*grvtmp.std()
        mmax = grvtmp.mean()+2*grvtmp.std()
        mint = (grvtmp.std()*4)/10.
        csrange = np.arange(mmin, mmax, mint)
        cns = plt.contour(grvtmp, levels=csrange, colors='y', extent=etmp)

        plt.clabel(cns, inline=1, fontsize=10)
        cbar = plt.colorbar(ims, orientation='horizontal')
        cbar.set_label('mgal')

        regplt.show()

    def dat_extent(self, dat, axes):
        """ Gets the extent of the dat variable """
        left = dat.tlx
        top = dat.tly
        right = left + dat.cols*dat.xdim
        bottom = top - dat.rows*dat.ydim

        if (right-left) > 10000 or (top-bottom) > 10000:
            axes.xaxis.set_label_text("Eastings (km)")
            axes.yaxis.set_label_text("Northings (km)")
            left /= 1000.
            right /= 1000.
            top /= 1000.
            bottom /= 1000.
        else:
            axes.xaxis.set_label_text("Eastings (m)")
            axes.yaxis.set_label_text("Northings (m)")

        return (left, right, bottom, top)

    def calc_field_only(self, modindo, mijk):
        """ Calculates only the field """
        numx = int(self.lmod.numx)
        numy = int(self.lmod.numy)
        numz = int(self.lmod.numz)

# get height corrections
        tmp = np.copy(self.lmod.lith_index)
        tmp[tmp > -1] = 0
        hcor = np.abs(tmp.sum(2))

# model index
        modind = modindo.copy()
        modind[modind == 0] = -1

        magval = np.zeros([numx, numy])
        grvval = np.zeros([numx, numy])

#        if self.pbars != None:
#           self.pbars.resetall(maximum=numx*(len(self.lmod.lith_list)-1),
#                      mmax=1)

        for mlist in self.lmod.lith_list.items():
            if 'Background' == mlist[0]:
                continue
            mfile = np.load(self.mfname+'_'+mlist[0]+'_tmp.npz')
            mlayers = mfile['mlayers']*mlist[1].netmagn()
            glayers = mfile['glayers']*mlist[1].rho()
            mijk = mlist[1].lith_index

            for i in range(numx):
                grvmagc.calc_field2(i, numx, numy, numz, modind, hcor,
                                    mlayers, glayers, magval, grvval, mijk)
#                calc_field2(i, numx, numy, numz, modind, hcor,
#                        mlayers, glayers, magval, grvval, mijk)

        return magval, grvval

    def invgrav(self):
        """ Calculate magnetic and gravity field """
        self.lmod1 = self.parent.lmod1
        self.lmod2 = self.parent.lmod2
        self.lmod = self.lmod1

        ltmp = list(self.lmod1.lith_list.keys())
        ltmp.pop(ltmp.index('Background'))

        text, okay = QtGui.QInputDialog.getItem(
            self.parent, 'Auto Model Gravity',
            'Please choose the lithology to use:', ltmp)

        if not okay:
            return

        # Update the model from the view
        # get height corrections
        tmp = np.copy(self.lmod.lith_index)
        tmp[tmp > -1] = 0
#        hcor = np.abs(tmp.sum(2))

        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        if tlabel == 'Layer Editor':
            self.parent.layer.update_model()

        if tlabel == 'Profile Editor':
            self.parent.profile.update_model()

        self.calc_field2(True)

# Init some variables
#        numx = int(self.lmod.numx)
#        numy = int(self.lmod.numy)
        numz = int(self.lmod.numz)
        modind = self.lmod.lith_index.copy()
        modindo = self.lmod.lith_index.copy()

        gdata = self.grd_to_lith(self.lmod1.griddata['Gravity Dataset'])
        gdata = gdata.T - self.lmod1.gregional

        self.showtext('Beginning Inversion')
        mlist = [text, self.lmod.lith_list[text]]
#        for mlist in self.lmod.lith_list.items():
        modindo = modind.copy()
        self.showtext(mlist[0])
#        mfile = np.load(self.mfname+'_'+mlist[0]+'_tmp.npz')
#        mlayers = mfile['mlayers']*mlist[1].netmagn()
#        glayers = mfile['glayers']*mlist[1].rho()
        mijk = mlist[1].lith_index
        if self.pbars is not None:
            self.pbars.resetall(maximum=numz, mmax=2)

        for k in range(numz):
            if self.pbars is not None:
                self.pbars.incr()
            magvalo, grvvalo = self.calc_field_only(modind, mijk)
            modind[:, :, k] = mijk
            magval, grvval = self.calc_field_only(modind, mijk)
            tmp = modind[:, :, k]
            tmp2 = modindo[:, :, k]
            tmp[abs(grvval-gdata) > abs(grvvalo-gdata)] = tmp2[
                abs(grvval-gdata) > abs(grvvalo-gdata)]
            tmp[grvval > gdata] = tmp2[grvval > gdata]

            magval, grvval = self.calc_field_only(modind, mijk)
            self.update_graph(grvval, magval, modind)

        for k in range(numz):
            if self.pbars is not None:
                self.pbars.incr()
            magval, grvval = self.calc_field_only(modind, mijk)
            tmp = modind[:, :, k]
            tmp2 = modindo[:, :, k]
            tmp[grvval > gdata] = tmp2[grvval > gdata]

            magval, grvval = self.calc_field_only(modind, mijk)
            self.update_graph(grvval, magval, modind)

        self.showtext('Calculation Finished')

    def update_graph(self, grvval, magval, modind):
        """ Updates the graph """
        indx = self.parent.tabwidget.currentIndex()
        tlabel = self.parent.tabwidget.tabText(indx)

        self.lmod.lith_index = modind.copy()
        self.lmod.griddata['Calculated Gravity'].data = grvval.T.copy()
        self.lmod.griddata['Calculated Magnetics'].data = magval.T.copy()

        if tlabel == 'Layer Editor':
            self.parent.layer.combo()
        if tlabel == 'Profile Editor':
            self.parent.profile.update_plot(slide=True)


class GeoData(object):
    """ Data layer class:
        This class defines each geological type and calculates the field
        for one cube from the standard definitions. """
    def __init__(self, parent, ncols=10, nrows=10, numz=10, dxy=10.,
                 d_z=10., mht=80., ght=0.):
        self.hintn = 30000.
        self.susc = 0.01
        self.mstrength = 0.
        self.finc = -63.
        self.fdec = -17.
        self.minc = -63.
        self.mdec = -17.
        self.theta = 90.
        self.bdensity = 2.67
        self.density = 2.85
        self.qratio = 0.0
        self.lith_index = 0
        self.parent = parent
        if hasattr(parent, 'pbars'):
            self.pbars = parent.pbars
        else:
            self.pbars = None
        self.showtext = self.parent.showtext

    # ncols and nrows are the smaller dimension of the original grid.
    # numx, numy, numz are the dimensions of the larger grid to be used as a
    # template.

        self.modified = True
        self.g_cols = None
        self.g_rows = None
        self.g_dxy = None
        self.numz = None
        self.dxy = None
        self.d_z = None
        self.zobsm = None
        self.zobsg = None

        self.mlayers = None
        self.glayers = None

        self.x12 = None
        self.y12 = None
        self.z12 = None

        self.set_xyz(ncols, nrows, numz, dxy, mht, ght, d_z)

    def calc_origin(self):
        """ Calculate the field values for the lithologies"""

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -self.g_dxy/2, -self.g_dxy,
                              dtype=float)

#            self.showtext('   Calculate magnetic origin field')
#            self.mboxmain(xdist, ydist, self.zobsm)
            self.showtext('   Calculate gravity origin field')
            self.gboxmain(xdist, ydist, self.zobsg)

            self.modified = False
        else:
            pass
#            self.pbars.incrmain(2)

    def calc_origin2(self):
        """ Calculate the field values for the lithologies"""

        if self.modified is True:
            numx = self.g_cols*self.g_dxy
            numy = self.g_rows*self.g_dxy

# The 2 lines below ensure that the profile goes over the center of the grid
# cell
            xdist = np.arange(self.g_dxy/2, numx+self.g_dxy/2, self.g_dxy,
                              dtype=float)
            ydist = np.arange(numy-self.g_dxy/2, -self.g_dxy/2, -self.g_dxy,
                              dtype=float)

            self.showtext('   Calculate magnetic and gravity origin field ' +
                          ' at magnetic height')
            self.gmmain(xdist, ydist, self.zobsm)

            self.modified = False
        else:
            pass
#            self.pbars.incrmain(2)

    def dircos(self, incl, decl, azim):

        """
    Subroutine DIRCOS computes direction cosines from inclination
        and declination.

    Input parameters:
        incl:  inclination in degrees positive below horizontal.
        decl:  declination in degrees positive east of true north.
        azim:  azimuth of x axis in degrees positive east of north.

        Output parameters:
        a,b,c:  the three direction cosines.
        """

        d2rad = np.pi/180.
        xincl = incl*d2rad
        xdecl = decl*d2rad
        xazim = azim*d2rad
        aaa = np.cos(xincl)*np.cos(xdecl-xazim)
        bbb = np.cos(xincl)*np.sin(xdecl-xazim)
        ccc = np.sin(xincl)
        return aaa, bbb, ccc

    def netmagn(self):
        """ Calculate the net magnetization """
        theta = 0.
        fcx, fcy, fcz = self.dircos(self.finc, self.fdec, theta)
        unith = np.array([fcx, fcy, fcz])
        hintn = self.hintn * 10**-9          # in Tesla
        mu0 = 4*np.pi*10**-7
        hstrength = self.susc*hintn/mu0  # Induced magnetization, needs susc.
        ind_magn = hstrength*unith

#       B is Induced Field (Tesla)
#       M is Magnetization (A/m)
#       H is Magnetic Field (A/m)
#       k is Susceptibility and is M/H
#   `   B = mu0(H+M)
#       Q = Jr/Ji = mstrength/hstrength = Jr/kH

        mcx, mcy, mcz = self.dircos(self.minc, self.mdec, theta)
        unitm = np.array([mcx, mcy, mcz])
        rem_magn = self.mstrength*unitm   # Remnant magnetization
        net_magn = rem_magn+ind_magn      # Net magnetization
        netmagscalar = np.sqrt((net_magn**2).sum())
        return netmagscalar

    def rho(self):
        """ Returns the density contrast """
        return self.density - self.bdensity

    def set_xyz(self, ncols, nrows, numz, g_dxy, mht, ght, d_z, dxy=None,
                modified=True):
        """ Sets/updates xyz parameters again """
        self.modified = modified
        self.g_cols = ncols*2+1
        self.g_rows = nrows*2+1
        self.numz = numz
        self.g_dxy = g_dxy
        self.d_z = d_z
        self.zobsm = -mht
        self.zobsg = -ght

        if dxy is None:
            self.dxy = g_dxy  # This must be a multiple of g_dxy or equal to it
        else:
            self.dxy = dxy  # This must be a multiple of g_dxy or equal to it.

        self.set_xyz12()

    def set_xyz12(self):
        """ Set x12, y12, z12. This is the limits of the cubes for the model"""

        numx = self.g_cols*self.g_dxy
        numy = self.g_rows*self.g_dxy
        numz = self.numz*self.d_z
        dxy = self.dxy
        d_z = self.d_z

        self.x12 = np.array([numx/2-dxy/2, numx/2+dxy/2])
        self.y12 = np.array([numy/2-dxy/2, numy/2+dxy/2])
        self.z12 = np.arange(-numz, numz+d_z, d_z)

    def gmmain(self, xobs, yobs, zobs):
        """ Algorithm for simultaneous computation of gravity and magnetic
            fields is based on the formulation published in GEOPHYSICS v. 66,
            521-526,2001. by Bijendra Singh and D. Guptasarma """

        x1 = float(self.x12[0])
        x2 = float(self.x12[1])
        y1 = float(self.y12[0])
        y2 = float(self.y12[1])
#        z1 = float(self.z12[0])
#        z2 = float(self.z12[1])
        z1 = 0.0
        z2 = self.d_z

        ncor = 8
        corner = np.array([[x1, y1, z1],
                           [x1, y2, z1],
                           [x2, y2, z1],
                           [x2, y1, z1],
                           [x1, y1, z2],
                           [x1, y2, z2],
                           [x2, y2, z2],
                           [x2, y1, z2]])

        nf = 6
        face = np.array([[0, 1, 2, 3],
                         [4, 7, 6, 5],
                         [1, 5, 6, 2],
                         [4, 0, 3, 7],
                         [3, 2, 6, 7],
                         [4, 5, 1, 0]])

#    nedges = sum(face(1:nf,1))
        nedges = 4*nf
        edge = np.zeros([nedges, 8])
        # get edge lengths
        for f in range(nf):
            indx = face[f].tolist() + [face[f, 0]]
            for t in range(4):
    #            edgeno = sum(face(1:f-1,1))+t
                edgeno = f*4+t
                ends = indx[t:t+2]
                p1 = corner[ends[0]]
                p2 = corner[ends[1]]
                V = p2-p1
                L = norm(V)
                edge[edgeno, 0:3] = V
                edge[edgeno, 3] = L
                edge[edgeno, 6:8] = ends

        un = np.zeros([nf, 3])
        # get face normals
        for t in range(nf):
            ss = np.zeros([1, 3])
            for t1 in range(2):
                v1 = corner[face[t, t1+2]]-corner[face[t, 0]]
                v2 = corner[face[t, t1+1]]-corner[face[t, 0]]
                ss = ss+np.cross(v2, v1)
            un[t, :] = ss/norm(ss)

        # Define the survey grid
        X, Y = np.meshgrid(xobs, yobs)

        npro, nstn = X.shape
        # Initialise

        # Grav stuff
        Gc = 6.6732e-3            # Universal gravitational constant
        Gx = np.zeros(X.shape)
        Gy = Gx.copy()
        Gz = Gx.copy()

        # Mag stuff
        cx, cy, cz = self.dircos(self.finc, self.fdec, 90.0)
        uh = np.array([cx, cy, cz])
        H = self.hintn*uh               # The ambient magnetic field (nTesla)
        ind_magn = self.susc*H/(4*np.pi)   # Induced magnetization

        mcx, mcy, mcz = self.dircos(self.minc, self.mdec, 90.0)
        um = np.array([mcx, mcy, mcz])
        rem_magn = (400*np.pi*self.mstrength)*um     # Remanent magnetization

        net_magn = rem_magn+ind_magn  # Net magnetization
        Pd = np.transpose(np.dot(un, net_magn.T))   # Pole densities

        Hx = np.zeros(X.shape)
        Hy = Hx.copy()
        Hz = Hx.copy()

        # For each observation point do the following.
        # For each face find the solid angle.
        # For each side find p,q,r and add p,q,r of sides to get P,Q,R for the
        # face.
        # find hx,hy,hz.
        # find gx,gy,gz.
        # Add the components from all the faces to get Hx,Hy,Hz and Gx,Gy,Gz.

        if self.pbars is not None:
            self.pbars.resetsub(len(self.z12)-1)

        face = face.tolist()
        un = un.tolist()
        gval = []
        mval = []
        newdepth = self.z12+abs(self.zobsm)

        for depth in newdepth:
            if self.pbars is not None:
                self.pbars.incr()
            if depth == 0.0:
                cor = (corner + [0., 0., depth+self.d_z/10000.]).tolist()
            elif depth == -self.d_z:
                cor = (corner + [0., 0., depth-self.d_z/10000.]).tolist()
            else:
                cor = (corner + [0., 0., depth]).tolist()

            if depth in newdepth:
                Gx = np.zeros(X.shape)
                Gy = Gx.copy()
                Gz = Gx.copy()
                Hx = np.zeros(X.shape)
                Hy = Hx.copy()
                Hz = Hx.copy()

                grvmagc.gm3d(npro, nstn, X, Y, edge, cor, face, Gx, Gy, Gz,
                             Hx, Hy, Hz, Pd, un)
                Gx *= Gc
                Gy *= Gc
                Gz *= Gc
                Htot = np.sqrt((Hx+H[0])**2 + (Hy+H[1])**2 + (Hz+H[2])**2)
                dt = Htot-self.hintn
                dta = Hx*cx + Hy*cy + Hz*cz
            else:
                Gz = np.zeros(X.shape)
                dta = np.zeros(X.shape)

            gval.append(np.copy(Gz.T))
            mval.append(np.copy(dta.T))

        self.glayers = np.array(gval)
        self.mlayers = np.array(mval)

    def gboxmain(self, xobs, yobs, zobs):
        """ Gbox routine by Blakely
            Note: xobs, yobs and zobs must be floats or there will be problems
            later.

        Subroutine GBOX computes the vertical attraction of a
        rectangular prism.  Sides of prism are parallel to x,y,z axes,
        and z axis is vertical down.

        Input parameters:
            Observation point is (x0,y0,z0).  The prism extends from x1
            to x2, from y1 to y2, and from z1 to z2 in the x, y, and z
            directions, respectively.  Density of prism is rho.  All
            distance parameters in units of m;

        Output parameters:
            Vertical attraction of gravity, g, in mGal/rho.
            Must still be multiplied by rho outside routine.
            Done this way for speed. """

        glayers = []
        if self.pbars is not None:
            self.pbars.resetsub(len(self.z12)-1)
        z1122 = self.z12.copy()
        x_1 = float(self.x12[0])
        y_1 = float(self.y12[0])
        x_2 = float(self.x12[1])
        y_2 = float(self.y12[1])
        z_0 = float(zobs)
        pi = np.pi
        numx = int(self.g_cols)
        numy = int(self.g_rows)

        if zobs == 0:
            zobs = -0.01

        for i in z1122[:-1]:
            if self.pbars is not None:
                self.pbars.incr()

            z12 = np.array([i, i+self.d_z])

            z_1 = float(z12[0])
            z_2 = float(z12[1])
            gval = np.zeros([self.g_cols, self.g_rows])
            grvmagc.gboxmain(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1,
                             x_2, y_2, z_2, pi)
#            gboxmain(gval, xobs, yobs, numx, numy, z_0, x_1, y_1, z_1,
#                    x_2, y_2, z_2, pi)

            gval *= 6.6732e-3
            glayers.append(gval)
        self.glayers = np.array(glayers)
#        self.glayers = np.array(glayers, dtype=np.float32)

    def mboxmain(self, xobs, yobs, zobs):

        """Subroutine MBOX computes the total field anomaly of an infinitely
      extended rectangular prism.  Sides of prism are parallel to x,y,z
      axes, and z is vertical down.  Bottom of prism extends to infinity.
      Two calls to mbox can provide the anomaly of a prism with finite
      thickness; e.g.,

         call mbox(x0,y0,z0,x1,y1,z1,x2,y2,mi,md,fi,fd,m,theta,t1)
         call mbox(x0,y0,z0,x1,y1,z2,x2,y2,mi,md,fi,fd,m,theta,t2)
         t=t1-t2

      Requires subroutine DIRCOS.  Method from Bhattacharyya (1964).

      Input parameters:
        Observation point is (x0,y0,z0).  Prism extends from x1 to
        x2, y1 to y2, and z1 to infinity in x, y, and z directions,
        respectively.  Magnetization defined by inclination mi,
        declination md, intensity m.  Ambient field defined by
        inclination fi and declination fd.  X axis has declination
        theta. Distance units are irrelevant but must be consistent.
        Angles are in degrees, with inclinations positive below
        horizontal and declinations positive east of true north.
        Magnetization in A/m.

        Note that in the case of no remanence: mi=fi and md=fd.

      Output paramters:
        Total field anomaly t, in nT. """

        mma, mmb, mmc = self.dircos(self.minc, self.mdec, self.theta)
        ffa, ffb, ffc = self.dircos(self.finc, self.fdec, self.theta)
        fm1 = mma*ffb+mmb*ffa
        fm2 = (mma*ffc+mmc*ffa)/2
        fm3 = (mmb*ffc+mmc*ffb)/2
        fm4 = mma*ffa
        fm5 = mmb*ffb
        fm6 = mmc*ffc

        x_1 = float(self.x12[0])
        y_1 = float(self.y12[0])
        x_2 = float(self.x12[1])
        y_2 = float(self.y12[1])
        z1mz0 = self.z12 - zobs
        fm1 = float(fm1)
        fm2 = float(fm2)
        fm3 = float(fm3)
        fm4 = float(fm4)
        fm5 = float(fm5)
        fm6 = float(fm6)

#        theta = self.theta
        numx = int(self.g_cols)
        numy = int(self.g_rows)
        numz = len(z1mz0)
        mval = np.zeros([numx, numy, numz])

        if self.pbars is not None:
            self.pbars.resetsub(numx)
        for icnt in range(numx):
            if self.pbars is not None:
                self.pbars.incr()
            grvmagc.mboxmain(icnt, z1mz0, mval, xobs, yobs, numx, numy, numz,
                             x_1, y_1, x_2, y_2, fm1, fm2, fm3, fm4, fm5, fm6)

        mval = mval*10**-7*10**9
        mval = mval[:, :, :-1]-mval[:, :, 1:]
        self.mlayers = np.transpose(mval, (2, 0, 1))
