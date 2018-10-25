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
from sklearn.cluster import DBSCAN
import scipy.spatial.distance as ssd
from pygmi.misc import PTime


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
        self.name = "Quarry: "
        self.pbar = None

        if parent is None:
            self.showtext = print
        else:
            self.showtext = self.parent.showprocesslog

        self.events = []
        self.day = [10, 16]  # daytime start at 6am and ends at 7pm

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

        data = self.calcrq2b()
        self.outdata['Seis'] = data

        return True

    def calcrq2(self):
        """ Calculates the Rq value """

        ttt = PTime()
        self.showtext('Working...')


        hour = []
        lat = []
        lon = []
        newevents = []

        for i in self.events:
            if np.isnan(i.latitude) or np.isnan(i.longitude):
                continue
            hour.append(i.hour)
            lat.append(i.latitude)
            lon.append(i.longitude)
            newevents.append(i)

        day = self.day
        ehour = np.array(hour)
        ehourall = ehour.copy()
        hour = np.logical_and(ehour >= day[0], ehour <= day[1])

        lon = np.array(lon)
        lat = np.array(lat)
        ld = day[1]-day[0]  # number of daylight hours
        ln = 24-ld  # number of nightime hours
        rdist = 0.2  # max distance for event to qualify. In degrees
        stayinloop = True
        N = 50

        nmin = 50
        nmax = 400
        nstep = 50
        nrange = list(range(nmin, nmax+nstep, nstep))
        rperc = self.randrq(nmax, nstep, nrange, day)

#        self.showtext('Calculating clusters')
#
#        # use DBscan to identify cluster centers and numbers of clusters.
#        # eps is max distance between samples
#
#        X = np.transpose([lon, lat])
#        db = DBSCAN(eps=0.01, min_samples=10).fit(X)
#        labels = db.labels_  # noisy samples are -1
#
#        print('now calculate means')
#        clusters = []
#        for i in np.unique(labels):
#            if i == -1:
#                continue
#            lontmp = lon[labels == i].mean()
#            lattmp = lat[labels == i].mean()
#            clusters.append([lontmp, lattmp])
#
#        clusters = np.array(clusters)
        self.showtext('Calculating Rq values')

        while stayinloop:
#            ttt.since_last_call('iteration')
            lls = np.transpose([lat, lon])
#            lls2 = lls[hour]
            cnt = lls.shape[0]
            nd = []
            rstot = []
            print('daylight events left:', hour.sum(), 'of', hour.size)

            # instead of a grid, we are using an actual event location
            # instead of centering on every event, we should use only daytime
            # events
            # also, it must not use an event if it is further than a certain
            # distance
            # also, perhaps if total events less than 50, is that even allowed?

            for i in range(cnt):  # i is node number, centered on an event
                r = ((lls-lls[i])**2).sum(1)

                rs = np.argpartition(r, N)[:N]
                hrs = hour[rs]  # daylight hours for this node
                rstot.append(rs[hrs])
                nd.append(hrs)

            nd = np.sum(nd, 1)
            nn = N-nd
            nd[nn == 0] = 0
            nn[nn == 0] = N
#            ttt.since_last_call('for loop')

            rq = (nd*ln)/(nn*ld)

            rstot = np.array(rstot)

            maxel = np.argmax(rq-rperc[0])

            if rq[maxel]-rperc[0] > 0:
#                maxel = np.nonzero(hour)[0][maxel]
                lat = np.delete(lat, rstot[maxel])
                lon = np.delete(lon, rstot[maxel])
                hour = np.delete(hour, rstot[maxel])
                ehour = np.delete(ehour, rstot[maxel])
                newevents = np.delete(newevents, rstot[maxel])
            else:
                stayinloop = False

        ttt.since_last_call('Total')
        self.showtext('Completed!')

        plt.hist(ehourall, 24)
        plt.show()

        plt.hist(ehour, 24)
        plt.show()


        breakpoint()

        return newevents.tolist()

    def calcrq2b(self):
        """ Calculates the Rq value """

        ttt = PTime()
        self.showtext('Working...')


        hour = []
        lat = []
        lon = []
        newevents = []

        for i in self.events:
            if np.isnan(i.latitude) or np.isnan(i.longitude):
                continue
            hour.append(i.hour)
            lat.append(i.latitude)
            lon.append(i.longitude)
            newevents.append(i)

        day = self.day

        ehour = np.array(hour)
        ehourall = ehour.copy()
        hour = np.logical_and(ehour >= day[0], ehour <= day[1])

        lon = np.array(lon)
        lat = np.array(lat)
        ld = day[1]-day[0]  # number of daylight hours
        ln = 24-ld  # number of nightime hours
        rdist = 0.2  # max distance for event to qualify. In degrees
        N = 50

        rperc = self.randrqb(N, day, ehourall.shape[0])
        rperc = 3.0

        self.showtext('Calculating Rq values')

        lls = np.transpose([lat, lon])
        cnt = lls.shape[0]
        nd = []
        rstot = []
        print('daylight events:', hour.sum(), 'of', hour.size)

        for i in range(cnt):  # i is node number, centered on an event
            r = np.sqrt(((lls-lls[i])**2).sum(1))

            rs = np.argpartition(r, N)[:N]

            if r[rs].max() > rdist:
                continue

            hrs = hour[rs]  # daylight hours for this node
            rstot.append(rs[hrs])
            nd.append(hrs)

        nd = np.sum(nd, 1)
        nn = (N-nd).astype(float)
        nn[nn == 0] = 0.00001
        rq = (nd*ln)/(nn*ld)

        rstot = np.array(rstot)

        filt = (rq-rperc) > 0

        rstot2 = []
        for i in rstot[filt]:
            rstot2 += i.tolist()

        maxel = np.unique(rstot2)

        lat = np.delete(lat, maxel)
        lon = np.delete(lon, maxel)
        hour = np.delete(hour, maxel)
        ehour = np.delete(ehour, maxel)
        newevents = np.delete(newevents, maxel)

        ttt.since_last_call('Total')
        self.showtext('Completed!')

        print('New total number of events:', ehour.size)

        plt.hist(ehourall, 24)
        plt.show()

        plt.hist(ehour, 24)
        plt.show()


        breakpoint()

        return newevents.tolist()


    def randrq(self, nmax, nstep, nrange, day):
        """ Calculates random Rq values """
        rperc = [1.97435897, 1.64253394, 1.46153846, 1.41025641, 1.35737179,
                 1.3234714, 1.28444936, 1.26923077]
#        rperc[0] = 2.5

        self.showtext('Calculating random Rq values for calibration')
        rperc = []
        nd = 0
        ld = day[1]-day[0]
        ln = 24-ld

        nrange = [10]
        for N in nrange:
            self.showtext(str(N)+' of '+str(nmax), True)
            tmp = np.random.rand(1000000, nstep)
            tmp *= 24

            tmp = np.logical_and(tmp >= day[0], tmp <= day[1])

            nd += tmp.sum(1)
            nn = N-nd
            rq = (nd*ln)/(nn*ld)
            rperc.append(np.percentile(rq, 99))

        return rperc

    def randrqb(self, N, day, num):
        """ Calculates random Rq values """

        self.showtext('Calculating random Rq values for calibration')
#        nd = 0
        elist = [50, 100, 150, 200]
#        elist = [150]

        for N in elist:
            rqmean = []
            for ld in range(1, 24):
                day = (0, ld)
    #            ld = day[1]-day[0]
                ln = 24-ld

                ln_over_ld = ln/ld

    #        self.showtext(str(N))


 #               print('random '+str(N)+' dat hours '+str(ld))

                tmp = np.random.rand(1000000, N)
                tmp *= 24

#                tmp2 = tmp.flatten()[:num]
#                plt.title('random '+str(N)+' dat hours '+str(ld))
#                plt.hist(tmp2, 24)
#                plt.show()

                tmp = np.logical_and(tmp >= day[0], tmp <= day[1])

                nd = tmp.sum(1)
                nn = (N-nd).astype(float)


#                nd = nd[nn!=0]
#                nn = nn[nn!=0]

#                nn[nn == 0] = 0.00001

    #            rq = (nd*ln)/(nn*ld)
    #            rq = ln_over_ld*nd/nn
                rq = nd/nn
                rperc = np.percentile(rq, 99)
                rperc = rperc*ln_over_ld

#                print(rperc)
#                print(np.mean(rq))
#                rqmean.append(np.mean(rq))
#                rqmean.append(np.median(rq))
                rqmean.append(rperc)

            plt.plot(rqmean)
            plt.xlim(0,24)
            plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
            plt.grid(True)
            plt.show()


        breakpoint()


        return rperc


def main():
    import iodefs

    ifile = r'C:\Work\Programming\pygmi\data/pygmi.out'

    quarry = Quarry()

    dat = iodefs.ImportSeisan()
    dat.settings(ifile)

    quarry.indata = dat.outdata
    quarry.settings()


#    breakpoint()


# search for events closer than a certain distance (0.2 deg)
# order by highest number of events to lowest
# eliminate nodes with only daytime, then one nighttime etc until
#    a horizontal distibution is achieved.


if __name__ == "__main__":
    main()

