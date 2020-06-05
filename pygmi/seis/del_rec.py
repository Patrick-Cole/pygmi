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
"""This program deletes seisan records."""

import os
import numpy as np
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
#from pygmi.misc import PTime
import pygmi.seis.iodefs as iodefs
#from sklearn.cluster import DBSCAN
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt


class DeleteRecord():
    """Main form which does the GUI and the program."""

    def __init__(self, parent=None):
        # Initialize Variables
        self.parent = parent
        self.indata = {'tmp': True}
        self.outdata = {}

        self.settings()

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        print('Delete Rows starting')

        ifile, _ = QtWidgets.QFileDialog.getOpenFileName()
        if ifile == '':
            return False
        os.chdir(ifile.rpartition('/')[0])

        self.delrec(ifile)

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata

    def delrec(self, ifile):
        """
        Delete record.

        Parameters
        ----------
        ifile : str
            Input filename.

        Returns
        -------
        None.

        """
        ofile = ifile[:-4]+'_new.out'

        print('Input Filename: '+ifile)
        print('Output Filename: '+ofile)

        outputf = open(ofile, 'w')
        inputf = open(ifile)

        skey = QtWidgets.QInputDialog.getText(
            self.parent, 'Delete Criteria',
            'Please input the terms used to decide on lines to delete',
            QtWidgets.QLineEdit.Normal, 'AML, IAML')[0]

        skey = str(skey).upper()

        print('Delete Criteria: '+skey)
        print('Working...')

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

        print('Completed!')


class Quarry():
    """Main form which does the GUI and the program."""

    def __init__(self, parent=None):
        # Initialize Variables
        self.parent = parent
        self.indata = {}
        self.outdata = {}
        self.name = 'Quarry: '
        self.pbar = None

        self.events = []
        self.day = [10, 16]  # daytime start at 6am and ends at 7pm
        self.day = [9, 19]  # daytime start at 6am and ends at 7pm

    def settings(self, nodialog=False):
        """
        Entry point into item.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        print('Delete quarry events starting')

        if 'Seis' not in self.indata:
            return False

        data = self.indata['Seis']

        alist = []
        for i in data:
            if '1' in i:
#                alist.append(i['1'])
                alist.append(i)

        if not alist:
            print('Error: no Type 1 records')
            return False

        self.events = alist

#        data = self.calcrq2()
        data = self.calcrq2b()
        self.outdata['Seis'] = data

        return True

    def loadproj(self, projdata):
        """
        Loads project data into class.

        Parameters
        ----------
        projdata : dictionary
            Project data loaded from JSON project file.

        Returns
        -------
        chk : bool
            A check to see if settings was successfully run.

        """

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

#        projdata['ftype'] = '2D Mean'

        return projdata

    def calcrq2(self):
        """
        Calculate the Rq value.

        Returns
        -------
        newevents : list
            New events

        """
        print('Working...')

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
#        ehourall = ehour.copy()
        hour = np.logical_and(ehour >= day[0], ehour <= day[1])

        lon = np.array(lon)
        lat = np.array(lat)
        ld = day[1]-day[0]  # number of daylight hours
        ln = 24-ld  # number of nightime hours
#        rdist = 0.2  # max distance for event to qualify. In degrees
        stayinloop = True
        N = 50

        nmin = 50
        nmax = 400
        nstep = 50
        nrange = list(range(nmin, nmax+nstep, nstep))
        rperc = self.randrq(nmax, nstep, nrange, day)

#        print('Calculating clusters')
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
        print('Calculating Rq values')

        while stayinloop:
            lls = np.transpose([lat, lon])
#            lls2 = lls[hour]
            cnt = lls.shape[0]
            nd = []
            rstot = []
            print('daylight events left: '+str(hour.sum())+' of '+str(hour.size))

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

            rq = (nd*ln)/(nn*ld)

            rstot = np.array(rstot)

            maxel = np.argmax(rq-rperc[0])

            if rq[maxel]-rperc[0] > 0:
                lat = np.delete(lat, rstot[maxel])
                lon = np.delete(lon, rstot[maxel])
                hour = np.delete(hour, rstot[maxel])
                ehour = np.delete(ehour, rstot[maxel])
                newevents = np.delete(newevents, rstot[maxel])
#                maxel = np.nonzero(hour)[0][maxel]
            else:
                stayinloop = False
            stayinloop = False

        print('Completed!')

        # plt.hist(ehourall, 24)
        # plt.show()

        # plt.hist(ehour, 24)
        # plt.show()

#        return newevents.tolist()
        return newevents

    def calcrq2b(self):
        """
        Calculate the Rq value.

        Returns
        -------
        newevents : list
            New events

        """
#        ttt = PTime()
        print('Working...')

        hour = []
        lat = []
        lon = []
        newevents = []

        for i2 in self.events:
            i = i2['1']
            if np.isnan(i.latitude) or np.isnan(i.longitude):
                continue
            hour.append(i.hour)
            lat.append(i.latitude)
            lon.append(i.longitude)
            newevents.append(i2)

        day = self.day

        ehour = np.array(hour)
#        ehourall = ehour.copy()
        hour = np.logical_and(ehour >= day[0], ehour <= day[1])

        lon = np.array(lon)
        lat = np.array(lat)
        ld = day[1]-day[0]  # number of daylight hours
        ln = 24-ld  # number of nightime hours
        rdist = 0.2  # max distance for event to qualify. In degrees
        N = 50

#        rperc = self.randrqb(N, day, ehourall.shape[0])
        rperc = 3.0
#        rperc = 1.97435897

        print('Calculating Rq values')

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

#        plt.xlabel('R')
#        plt.ylabel('Event Counts')
#        plt.hist(rq[nn != 0.00001], 50)
#        plt.show()

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

#        ttt.since_last_call('Total')
        print('Completed!')

#        print('New total number of events:', ehour.size)
#
#        plt.xlabel('Hours')
#        plt.ylabel('Event Counts')
#        plt.hist(ehourall, 24)
#        plt.show()
#
#        plt.xlabel('Hours')
#        plt.ylabel('Event Counts')
#        plt.hist(ehour, 24)
#        plt.show()

        return newevents.tolist()

    def randrq(self, nmax, nstep, nrange, day):
        """
        Calculate random Rq values.

        Parameters
        ----------
        nmax : int
            DESCRIPTION.
        nstep : int
            DESCRIPTION.
        nrange : list
            DESCRIPTION.
        day : tuple
            DESCRIPTION.

        Returns
        -------
        rperc : list
            Percentiles

        """
        rperc = [1.97435897, 1.64253394, 1.46153846, 1.41025641, 1.35737179,
                 1.3234714, 1.28444936, 1.26923077]
#        rperc[0] = 2.5

        print('Calculating random Rq values for calibration')
        rperc = []
        nd = 0
        ld = day[1]-day[0]
        ln = 24-ld

        nrange = [10]
        for N in nrange:
            print(str(N)+' of '+str(nmax), True)
            tmp = np.random.rand(1000000, nstep)
            tmp *= 24

            tmp = np.logical_and(tmp >= day[0], tmp <= day[1])

            nd += tmp.sum(1)
            nn = N-nd
            rq = (nd*ln)/(nn*ld)
            rperc.append(np.percentile(rq, 99))

        return rperc

    def randrqb(self, N1, day, num):
        """
        Calculate random Rq values.

        Parameters
        ----------
        N1 : TYPE
            DESCRIPTION.
        day : tuple
            DESCRIPTION.
        num : int
            DESCRIPTION.

        Returns
        -------
        rperc : list
            Percentiles

        """
        print('Calculating random Rq values for calibration')
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

    #            print(str(N))

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
            plt.xlim(0, 24)
            plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
            plt.grid(True)
            plt.show()

        return rperc


def import_for_plots(ifile, dind='R'):
    """
    Import data to plot.

    Parameters
    ----------
    ifile : str
        Input file name.
    dind : str, optional
        Distance indicator. The default is 'R'.

    Returns
    -------
    None.

    """
    iseis = iodefs.ImportSeisan()
    iseis.settings(ifile)

    dat = iseis.outdata['Seis']
    datd = {}

    for event in dat:
        if '1' not in event:
            continue
        if event['1'].distance_indicator not in dind:
            continue

        for rectype in event:
            if rectype not in datd:
                datd[rectype] = []
            datd[rectype].append(event[rectype])

            if rectype in ('1', 'E'):
                tmp = vars(event[rectype])
                for j in tmp:
                    newkey = rectype+'_'+j
                    if newkey not in datd:
                        datd[newkey] = []
                    datd[newkey].append(tmp[j])
# Custom
                    if 'type_of_magnitude' in j:
                        newkey = '1_M'+tmp[j]
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp[j.split('_of_')[1]])

#                        time = tmp['hour']+tmp['minutes']/60.+tmp['seconds']/3600.
#                        newkey = '1_M'+tmp[j]+'_time'
#                        if newkey not in datd:
#                            datd[newkey] = []
#                        datd[newkey].append(time)

                        newkey = '1_M'+tmp[j]+'_year'
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp['year'])

            if rectype == '4':
                for i in event[rectype]:
                    tmp = vars(i)
                    for j in tmp:
                        newkey = rectype+'_'+j
                        if newkey not in datd:
                            datd[newkey] = []
                        datd[newkey].append(tmp[j])
    return datd


#def gearth_plot(ifile):
#    """Plot with google earth background."""
#    datd = import_for_plots(ifile, 'R')
#
#    extent = [16, 33, -22, -35]
##    extent = [29, 29.05, -26, -26.05]
#    request = cimgt.GoogleTiles(style='satellite')
#
#    plt.figure(figsize=(9, 7))
#    ax = plt.axes(projection=ccrs.PlateCarree())
#    ax.set_extent(extent, crs=ccrs.PlateCarree())
#
#    # https://wiki.openstreetmap.org/wiki/Zoom_levels
#    ax.add_image(request, 6)
##    ax.add_image(request, 14)
#
#    ax.gridlines(draw_labels=True)
#
#    filt1 = np.equal(datd['1_longitude'], None)
#    filt2 = np.equal(datd['1_latitude'], None)
#    filt = np.logical_or(filt1, filt2)
#    filt1 = np.isnan(datd['1_longitude'])
#    filt2 = np.isnan(datd['1_latitude'])
#    filtb = np.logical_or(filt1, filt2)
#    filt = np.logical_or(filt, filtb)
#
#    lat1 = np.array(datd['1_latitude'])[~filt].astype(float)
#    lon1 = np.array(datd['1_longitude'])[~filt].astype(float)
#
#    X = np.transpose([lon1, lat1])
#    db = DBSCAN(eps=0.05, min_samples=10).fit(X)
#
#    filt = (db.labels_ != -1)
#    plt.scatter(lon1[filt], lat1[filt], s=1, c=db.labels_[filt],
#                cmap=plt.cm.jet)
#    plt.colorbar()
#
#    plt.show()
#
#def test():
#    """ test for google earth stuff """
#    import matplotlib as mpl
#    glookup = np.array([156412, 78206, 39103, 19551, 9776, 4888, 2444, 1222,
#                        610.984, 305.492, 152.746, 76.373, 38.187, 19.093,
#                        9.547, 4.773, 2.387, 1.193, 0.596, 0.298])
#
#
##    extent = [16, 33, -22, -35]
#    extent = [227800, 540000, 7110000, 6950000]
#    extent = [227800, 540000, 6950000, 7110000]
##    extent = [0, 100, 100, 0]
#
##    extent = [29, 29.05, -26, -26.05]
#    request = cimgt.GoogleTiles(style='satellite')
##    request = cimgt.GoogleTiles(style='street')
##    custproj = ccrs.PlateCarree()
#    custproj = ccrs.epsg(32736)
##    custproj = ccrs.epsg(3857)
#
#    fig = plt.figure(figsize=(9, 7))
#    ax = plt.axes(projection=custproj)
#    ax.set_extent(extent, crs=custproj)
#
#    xrange = extent[1]-extent[0]
#    yrange = abs(extent[2]-extent[3])
#    width = xrange/(fig.get_figwidth()*fig.get_dpi())
#    height = yrange/(fig.get_figheight()*fig.get_dpi())
#    mindim = min(width, height)
#
#    pchk1 = np.abs(glookup-mindim)
#    pchk2 = np.min(pchk1)
#    pchk = np.nonzero(pchk1 == pchk2)
#    pchk = int(pchk[0][0])
#
#    # https://wiki.openstreetmap.org/wiki/Zoom_levels
#    ax.add_image(request, pchk)
##    ax.add_image(request, 9)
#
#    plt.xticks([227800, 300000, 400000, 500000, 540000])
#    plt.yticks([6950000, 7000000, 7050000, 7100000, 7110000])
#    plt.grid(True)
#
#    plt.plot(400000, 7050000, 'o')
#    plt.show()
#
##    imgs = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage)]
##
##    image = imgs[0].get_array()[::-1]
##
##    plt.imshow(image)
##    plt.show()
#    lims2 = ax.get_images()[0]
#
#    lims2.set_alpha(0)
#
#    plt.show()


def main():
    """Routine for testing."""
    ifile = r'C:\Work\Programming\pygmi\data\seismology\pygmi.out'

#    gearth_plot(ifile)
#    test()

    quarry = Quarry()

    dat = iodefs.ImportSeisan()
    dat.settings(ifile)

    quarry.indata = dat.outdata
    quarry.settings()


# search for events closer than a certain distance (0.2 deg)
# order by highest number of events to lowest
# eliminate nodes with only daytime, then one nighttime etc until
#    a horizontal distibution is achieved.


if __name__ == "__main__":
    main()
