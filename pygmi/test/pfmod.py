# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:        pfmod.py (part of PyGMI)
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
""" These are pfmod tests. Run this file from within this directory to do the
tests """

import numpy as np
from pygmi.pfmod.grvmag3d import quick_model
from pygmi.pfmod.grvmag3d import calc_field
import matplotlib.pyplot as plt


def main():
    """ Main test function """
    print('Testing modelling of gravity and potential field data')

    # First initialise the modelling object

    # now, we add the model
    gfile = open('Grav2dc_grav.txt')
    tmp = gfile.read()
    tmp2 = tmp.splitlines()

    numx = get_int(tmp2, 2, 2)
    cnrs = get_int(tmp2, 6, 4)
    dens = get_float(tmp2, 6, 8) + 2.67
    strike = get_float(tmp2, 7, 3)

    x = []
    z = []
    for i in range(cnrs):
        x.append(get_float(tmp2, 9+i, 0))
        z.append(get_float(tmp2, 9+i, 1))

    xpos = []
    fcalc = []
    for i in range(numx):
        xpos.append(get_float(tmp2, 11+cnrs+i, 0))
        fcalc.append(get_float(tmp2, 11+cnrs+i, 1))

    x = np.array(x)
    z = -np.array(z)
    xpos = np.array(xpos)
    fcalc = np.array(fcalc)

    mfile = open('Mag2dc_mag.txt')
    tmp = mfile.read()
    tmp2 = tmp.splitlines()

#    intensity = get_float(tmp2, 4, 2)
    finc = get_float(tmp2, 4, 5)
    fdec = get_float(tmp2, 4, 8)
    mht = get_float(tmp2, 7, 5)
    susc = get_float(tmp2, 12, 7)

    fcalc2 = []
    for i in range(numx):
        fcalc2.append(get_float(tmp2, 17+cnrs+i, 1))

    fcalc2 = np.array(fcalc2)

    # for testing purposes the cube being modelled should have dxy = d_z to
    # keep things simple
    dxy = xpos[1]-xpos[0]
    d_z = dxy
    ypos = np.arange(-strike, strike, dxy)
    numy = ypos.size
    zpos = np.arange(z.min(), 0, d_z)
    numz = zpos.size
    tlx = xpos.min()
    tly = ypos.max()
    tlz = zpos.max()

    lmod = quick_model(['Generic'], numx, numy, numz, dxy, d_z, tlx, tly, tlz,
                       0, 0, finc, fdec, [susc], [dens])

    for i in np.arange(x.min(), x.max(), dxy):
        for j in np.arange(0, 2*strike, dxy):
            for k in np.arange(1, numz):
                i2 = int(i/dxy)
                j2 = int(j/dxy)
                k2 = int(k)
                lmod.lith_index[i2, j2, k2] = 1

    # finally we calculate the new fields
    calc_field(lmod)
    gdata = lmod.griddata['Calculated Gravity'].data[20].copy()

    # Change to 100 meters and calculate mag
    lmod.mht = mht
    calc_field(lmod, altcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[20]

    # Display results
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Distance (m)')
    ax1.plot(xpos+dxy/2, gdata, 'r')
    ax1.plot(xpos, fcalc, 'r.')
    ax1.set_ylabel('mGal')

    ax2 = ax1.twinx()
    ax2.plot(xpos+dxy/2, mdata, 'b')
    ax2.plot(xpos, fcalc2, 'b.')
    ax2.set_ylabel('nT')
    plt.show()


def get_int(tmp, row, word):
    """ gets an int from a list of strings. First row or word is zero """
    return int(tmp[row].split()[word])


def get_float(tmp, row, word):
    """ gets an int from a list of strings. First row or word is zero """
    return float(tmp[row].split()[word])


if __name__ == "__main__":
    main()
