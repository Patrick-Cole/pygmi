# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:        rose.py (part of PyGMI)
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
""" This is a quick start routine to start the GUI form of PyGMI """
import numpy as np
from osgeo import ogr
import matplotlib.pyplot as plt
import matplotlib.collections as mc


def main():
    """ Main program """
    ifile = r'C:\Work\Programming\pygmi\data\2329AC_lin_wgs84sutm35.shp'
    nbins = 8

    shapef = ogr.Open(ifile)
    lyr = shapef.GetLayer()
    if lyr.GetGeomType() is not ogr.wkbLineString:
        return
        # self.parent.showprocesslog('You need polygons in that shape file')

    fangle = []
    fcnt = []
    flen = []
    allcrds = []

    for i in range(lyr.GetFeatureCount()):
        poly = lyr.GetNextFeature()
        geom = poly.GetGeometryRef()
        pnts = np.array(geom.GetPoints()).tolist()
        allcrds.append(pnts)
        pnts = np.transpose(pnts)
        xtmp = pnts[0, 1:]-pnts[0, :-1]
        ytmp = pnts[1, 1:]-pnts[1, :-1]
#        ntmp = xtmp + 1j*ytmp
        ftmp = np.arctan2(xtmp, ytmp)
#        ftmp = -1*np.angle(ntmp)+np.pi/2
        ftmp[ftmp < 0] += 2*np.pi
        ftmp[ftmp > np.pi] -= np.pi
        ltmp = np.sqrt(xtmp**2+ytmp**2)

        fangle += [np.sum(ftmp*ltmp)/ltmp.sum()]
        fcnt += ftmp.tolist()
        flen += ltmp.tolist()

    fangle = np.array(fangle)
    fcnt = np.array(fcnt)
    flen = np.array(flen)
    bwidth = np.pi/nbins
    bcols = plt.cm.Set1(np.arange(nbins)/nbins)
    np.random.shuffle(bcols)

    # Draw rose diagram base on one angle per linear feature

    radii, theta = np.histogram(fangle, bins=np.arange(0, np.pi+bwidth,
                                                       bwidth))
    ax = plt.subplot(221, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.yaxis.set_ticklabels([])
    xtheta = theta[:-1]  # +(theta[1]-theta[0])/2
    bcols2 = bcols[(xtheta/bwidth).astype(int)]
    ax.bar(xtheta, radii, width=bwidth, color=bcols2)
    ax.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

    ax = plt.subplot(222)
    bcols2 = bcols[(fangle/bwidth).astype(int)]
    lc = mc.LineCollection(allcrds, color=bcols2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.axis('equal')

    # Draw rose diagram base on one angle per linear segment, normalized

    radii, theta = histogram(fcnt, y=flen, xmin=0., xmax=np.pi, bins=nbins)
    ax = plt.subplot(223, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.yaxis.set_ticklabels([])
    xtheta = theta[:-1]  # +(theta[1]-theta[0])/2
#    bcols = plt.cm.jet(xtheta/(2*np.pi))
    bcols2 = bcols[(xtheta/bwidth).astype(int)]
    ax.bar(xtheta, radii, width=bwidth, color=bcols2)
    ax.bar(xtheta+np.pi, radii, width=bwidth, color=bcols2)

    ax = plt.subplot(224)
#    bcols = plt.cm.jet(fcnt/2*np.pi)
    bcols2 = bcols[(fcnt/bwidth).astype(int)]
    lc = mc.LineCollection(allcrds, color=bcols2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    #ax.axis('equal')

    plt.tight_layout()
    plt.show()


def histogram(x, y=None, xmin=None, xmax=None, bins=10):
    """ histogram """

    radii = np.zeros(bins)
    theta = np.zeros(bins+1)

    if y is None:
        y = np.ones_like(x)
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()

    x = np.array(x)
    y = np.array(y)
    theta[-1] = xmax

    xrange = xmax-xmin
    xbin = xrange/bins
    x2 = x/xbin
    x2 = x2.astype(int)

    for i in range(bins):
        radii[i] = y[x2 == i].sum()
        theta[i] = i*xbin

    return radii, theta

if __name__ == "__main__":
    main()
