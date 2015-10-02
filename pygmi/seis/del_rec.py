# -----------------------------------------------------------------------------
# Name:        del_rec.py (part of PyGMI)
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
""" This program deletes seisan records """

import numpy as np
from PyQt4 import QtGui
import os
import matplotlib.pyplot as plt


class DeleteRecord(object):
    """ Main form which does the GUI and the program """
    def __init__(self, parent=None):
        # Initialize Variables
        self.parent = parent
        self.indata = {'tmp': True}
        self.outdata = {}
        self.showtext = self.parent.showprocesslog

        self.settings()

    def settings(self):
        """ Settings """
        self.showtext('Delete Rows starting')

        ifile = QtGui.QFileDialog.getOpenFileName()
        if ifile == '':
            return
        os.chdir(ifile.rpartition('/')[0])

        self.delrec(ifile)

        return True

    def delrec(self, ifile):
        """ Deletes record """

        ofile = ifile[:-4]+'_new.out'

        self.showtext('Input Filename: '+ifile)
        self.showtext('Output Filename: '+ofile)

        outputf = open(ofile, 'w')
        inputf = open(ifile)

        skey = QtGui.QInputDialog.getText(
            self.parent, 'Delete Criteria',
            'Please input the terms used to decide on lines to delete',
            QtGui.QLineEdit.Normal, 'AML, IAML')[0]

        skey = str(skey).upper()

        self.showtext('Delete Criteria: '+skey)
        self.showtext('Working...')

        skey = skey.replace(' ', '')
        skey = skey.split(',')

        idata = inputf.readlines()
        odata = idata
        for j in skey:
            odata = [i for i in odata if i.find(j) < 0]

        outputf.writelines(odata)  # Insert a blank line

# Close files
        inputf.close()
        outputf.close()

        self.showtext('Completed!')


class Quarry(object):
    """ Main form which does the GUI and the program """
    def __init__(self, parent=None):
        # Initialize Variables
        self.parent = parent
        self.indata = {'tmp': True}
        self.outdata = {}
        self.showtext = self.parent.showprocesslog

        self.settings()

    def settings(self):
        """ Settings """
        self.showtext('Delete quarry events starting')

        ifile = QtGui.QFileDialog.getOpenFileName()[0]
        if ifile == '':
            return

        os.chdir(ifile.rpartition('/')[0])

        self.calcrq2(ifile)

        return True

    def calcrq2(self, ifile):
        """ Calculates the Rq value """
        ofile = ifile[:-4]+'_new.out'
        ofile2 = ifile[:-4]+'_del.out'

        inputf = open(ifile)

        self.showtext('Working...')

        idata = inputf.readlines()
        date = []
        hour = []
        lat = []
        lon = []
        for i in idata:
            date.append(i[0:21])
            hour.append(int(i[17:19]))
            lat.append(float(i[29:36]))
            lon.append(float(i[38:44]))

        day = [6, 19]
        hour = np.array(hour)
        hour[hour < day[0]] = -99
        hour[hour > day[1]] = -99
        hour[hour != -99] = True
        hour[hour == -99] = False
        idata = np.array(idata)

        lon = np.array(lon)
        lat = np.array(lat)
        ld = day[1]-day[0]
        ln = 24-ld
        rdist = 0.2
        nmin = 50
        nmax = 400
        nstep = 50
        nrange = list(range(nmin, nmax+nstep, nstep))
        rlyrs = len(nrange)
        stayinloop = True

        ilat = []
        ilon = []
        ihour = []
        iidata = []

        rperc = self.randrq(nmax, nstep, nrange, day)
        self.showtext('Calculating Rq values')

        while stayinloop:
            cnt = hour.shape[0]
            nd = np.zeros([cnt, rlyrs])
            nn = np.zeros([cnt, rlyrs])
            mask = np.ones(cnt).astype(bool)

            for i in range(cnt):
                londiff = lon-lon[i]
                latdiff = lat-lat[i]
                r = np.sqrt(londiff**2+latdiff**2)
                if r.min() > rdist:
                    nn[i] = 1.
                    continue
                rs = np.argsort(r)
                rs = rs[:nmax]
                r = r[rs]
                rs = rs[r < rdist]
                hrs = hour[rs]
                for N in nrange:
                    ndx = N/nstep-1
                    nd[i, ndx] = hrs[:N].sum()
                    nn[i, ndx] = N-nd[i, ndx]
                    if nn[i, ndx] == 0:
                        mask[i] = False
                        nn[i, ndx] = N
                        nd[i, ndx] = 0

            rq = (nd*ln)/(nn*ld)

            for ndx in range(len(nrange)):
                rq[:, ndx][rq[:, ndx] > rperc[ndx][1]] = rperc[ndx][1]
                rq[:, ndx] -= rperc[ndx][0]
                rq[:, ndx] /= (rperc[ndx][1]-rperc[ndx][0])
                rq[:, ndx] *= 100.

            tmpcnt = []
            for i in range(rlyrs):
                tmpcnt.append(np.where(rq[:, i] > 99.)[0].shape[0])

            self.showtext(str(tmpcnt)+' possible eliminations in '
                          + ' event groups: ' + str(nrange), True)

            tmax = np.transpose(np.where(rq == rq.max()))[0]
            i, ndx = tmax

            if rq[i, ndx] > 99.:
                londiff = lon-lon[i]
                latdiff = lat-lat[i]
                r = np.sqrt(londiff**2+latdiff**2)
                rs = np.argsort(r)
                rs = rs[:(ndx+1)*nstep]
                r = r[rs]
                rs = rs[r < rdist]

                mask[rs] = False
                ilat += lat[np.logical_not(mask)].tolist()
                ilon += lon[np.logical_not(mask)].tolist()
                ihour += hour[np.logical_not(mask)].tolist()
                iidata += idata[np.logical_not(mask)].tolist()
                lat = lat[mask]
                lon = lon[mask]
                hour = hour[mask]
                idata = idata[mask]
            else:
                stayinloop = False

        plt.plot(lon, lat, 'r.')
        plt.plot(ilon, ilat, 'b.')
        plt.show()

        outputf2 = open(ofile2, 'w')
        outputf = open(ofile, 'w')

        outputf2.writelines(iidata)
        outputf.writelines(idata)

        inputf.close()
        outputf.close()
        outputf2.close()

        self.showtext('Completed!')

#    def calcrq(self, ifile):
#        """ Calculates the Rq value """
#        ofile = ifile[:-4]+'_new.out'
#
#        outputf = open(ofile, 'w')
#        inputf = open(ifile)
#
#        self.showtext('Working...')
#
#        idata = inputf.readlines()
# ##        odata = idata
#        date = []
#        hour = []
#        lat = []
#        lon = []
#        for i in idata:
#            date.append(i[0:21])
#            hour.append(int(i[17:19]))
#            lat.append(float(i[29:36]))
#            lon.append(float(i[38:44]))
#
#        day = [6, 19]
#        hour = np.array(hour)
#        hour[hour < day[0]] = -99
#        hour[hour > day[1]] = -99
#        hour[hour != -99] = True
#        hour[hour == -99] = False
#        idata = np.array(idata)
#
#        lon = np.array(lon)
#        lat = np.array(lat)
#        ld = day[1]-day[0]
#        ln = 24-ld
#        dxy = 0.1
#        dxyd2 = np.sqrt(2)*dxy
#        lonrange = lon.max()-lon.min()+dxy
#        latrange = lat.max()-lat.min()+dxy
#        tlx = lon.min()
#        tly = lat.max()
#        rows = int(latrange/dxy)
#        cols = int(lonrange/dxy)
#        nmin = 50
#        nmax = 400
#        nstep = 50
#        nrange = list(range(nmin, nmax+nstep, nstep))
#        rlyrs = len(nrange)
#        stayinloop = True
#
#        ttt.since_last_call()
#        rperc = self.randrq(nmax, nstep, nrange, day)
#        self.showtext('Calculating Rq values')
#
#        ttt.since_last_call('Begin of while loop')
#
#        plt.figure(1)
#        plt.plot(lon, lat, 'b.')
#        plt.show()
#        QtGui.QApplication.processEvents()
#
#        while stayinloop:
#            nd = np.zeros([rows, cols, rlyrs])
#            nn = np.zeros([rows, cols, rlyrs])
#            cnt = hour.shape[0]
#            mask = np.ones(cnt).astype(bool)
#
#            for i in range(cols):
#                londiff = (tlx+i*dxy)-lon
#                for j in range(rows):
#                    latdiff = (tly-j*dxy)-lat
#                    r = np.sqrt(londiff**2+latdiff**2)
#                    if r.min() > dxyd2:
#                        nn[j, i] = 1.
#                        continue
#                    rs = np.argsort(r)
#                    rs = rs[:nmax]
#                    r = r[rs]
#                    rs = rs[r < dxyd2]
#                    hrs = hour[rs]
#                    for N in nrange:
#                        ndx = N/nstep-1
#                        nd[j, i, ndx] = hrs[:N].sum()
#                        nn[j, i, ndx] = N-nd[j, i, ndx]
#                        if nn[j, i, ndx] == 0:
#                            mask[rs[:N]] = False
#                            nn[j, i, ndx] = N
#                            nd[j, i, ndx] = 0
#
#            rq = (nd*ln)/(nn*ld)
# ##            rq[rq == np.inf] = max(rq[rq < np.inf].max(), 100.)
#
#            for ndx in range(len(nrange)):
#                rq[:, :, ndx][rq[:, :, ndx] > rperc[ndx][1]] = rperc[ndx][1]
#                rq[:, :, ndx] -= rperc[ndx][0]
#                rq[:, :, ndx] /= (rperc[ndx][1]-rperc[ndx][0])
#                rq[:, :, ndx] *= 100.
#
#            tmax = np.transpose(np.where(rq == rq.max()))[0]
#            j, i, ndx = tmax
#            self.showtext('Rq percentile:'+str(rq[j, i, ndx]), True)
#
#            ttt.since_last_call('section 2')
#
#            if rq[j, i, ndx] > 99.:
#                londiff = (tlx+i*dxy)-lon
#                latdiff = (tly-j*dxy)-lat
#                r = np.sqrt(londiff**2+latdiff**2)
#                rs = np.argsort(r)
#                rs = rs[:(ndx+1)*nstep]
#                r = r[rs]
#                rs = rs[r < dxyd2]
#
# ##                cnt = hour.shape[0]
# ##                mask = np.ones(cnt).astype(bool)
#                mask[rs] = False
#                lat = lat[mask]
#                lon = lon[mask]
#                hour = hour[mask]
#                idata = idata[mask]
#            else:
#                stayinloop = False
#            ttt.since_last_call('section 3')
#
#        ttt.since_first_call('end')
#
#        plt.figure(1)
#        plt.plot(lon, lat, 'r.')
#        plt.show()
#
#        outputf.writelines(idata)
#
# # Close files
#
#        inputf.close()
#        outputf.close()
#
#        self.showtext('Completed!')

    def randrq(self, nmax, nstep, nrange, day):
        """ Calculates random Rq values """

        self.showtext('Calculating random Rq values for calibration')
        rperc = []
        nd = 0
        ld = day[1]-day[0]
        ln = 24-ld

        for N in nrange:
            self.showtext(str(N)+' of '+str(nmax), True)
            tmp = np.random.rand(1000000, nstep)
            tmp *= 24
            tmp[tmp < day[0]] = -99
            tmp[tmp > day[1]] = -99
            tmp[tmp != -99] = True
            tmp[tmp == -99] = False

            nd += tmp.sum(1)
            nn = N-nd
            rq = (nd*ln)/(nn*ld)
            rperc.append(np.percentile(rq, [99, 100]))

        return rperc
