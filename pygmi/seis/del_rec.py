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

import os
import numpy as np
from PyQt5 import QtWidgets
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

        ifile, _ = QtWidgets.QFileDialog.getOpenFileName()
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

        skey = QtWidgets.QInputDialog.getText(
            self.parent, 'Delete Criteria',
            'Please input the terms used to decide on lines to delete',
            QtWidgets.QLineEdit.Normal, 'AML, IAML')[0]

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
        self.indata = {}
        self.outdata = {}
        self.showtext = self.parent.showprocesslog
        self.name = "Quarry: "
        self.pbar = None

        self.events = []

    def settings(self):
        """ Settings """
        self.showtext('Delete quarry events starting')

        if 'Seis' not in self.indata:
            return

        data = self.indata['Seis']

        alist = []
        for i in data:
            if '1' in i:
                alist.append(i['1'])

        if not alist:
            self.showtext('Error: no Type 1 records')
            return False

        self.events = alist

        data = self.calcrq2()
        self.outdata['Seis'] = data

        return True

    def calcrq2(self):
        """ Calculates the Rq value """

        self.showtext('Working...')

        hour = []
        lat = []
        lon = []
        for i in self.events:
            if i.latitude is None or i.longitude is None:
                continue
            hour.append(i.hour)
            lat.append(i.latitude)
            lon.append(i.longitude)

        day = [6, 19]  # daytime start at 6am and ends at 7pm
        hour = np.array(hour)
        hour[hour < day[0]] = -99
        hour[hour > day[1]] = -99
        hour[hour != -99] = True
        hour[hour == -99] = False

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
                rq[:, ndx] = rq[:, ndx] - rperc[ndx][0]
                rq[:, ndx] = rq[:, ndx] / (rperc[ndx][1]-rperc[ndx][0])
                rq[:, ndx] = rq[:, ndx] * 100.

            tmpcnt = []
            for i in range(rlyrs):
                tmpcnt.append(np.where(rq[:, i] > 99.)[0].shape[0])

            self.showtext(str(tmpcnt)+' possible eliminations in '
                          ' event groups: ' + str(nrange), True)

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
                ilat = ilat + lat[np.logical_not(mask)].tolist()
                ilon = ilon + lon[np.logical_not(mask)].tolist()
                ihour = ihour + hour[np.logical_not(mask)].tolist()
                lat = lat[mask]
                lon = lon[mask]
                hour = hour[mask]
            else:
                stayinloop = False

        plt.plot(lon, lat, 'r.')
        plt.plot(ilon, ilat, 'b.')
        plt.show()

        self.showtext('Completed!')

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
            tmp = tmp * 24
            tmp[tmp < day[0]] = -99
            tmp[tmp > day[1]] = -99
            tmp[tmp != -99] = True
            tmp[tmp == -99] = False

            nd = nd + tmp.sum(1)
            nn = N-nd
            rq = (nd*ln)/(nn*ld)
            rperc.append(np.percentile(rq, [99, 100]))

        return rperc
