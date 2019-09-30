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

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..//..')))
from pygmi.pfmod.grvmag3d import quick_model
from pygmi.pfmod.grvmag3d import calc_field


def test(doplt=False):
    """
    Main test function

    This test function compares the calculations performed by PyGMI to
    calculations performed by a external software - GM-SYS

    A series of graphs are produced. If the test is successful, points and
    lines on the graphs will coincide.
    """
    print('Testing modelling of gravity and potential field data')

    ifile = 'data/block'
#    ifile = 'data/dyke'
    samplescale = 1.  # Horizontal scale of the samples.
    power = 1.  # This parameter changes thinkness of voxels with depth.
    print('File:', ifile)
    print('Sample Scale:', samplescale)
    print('Power:', power)

    with open(ifile+'.ecs') as fnr:
        tmp = fnr.read()

    scale = float(tmp.split('Scale=')[1].split('\n')[0])

    with open(ifile+'.blk') as fnr:
        tmp = fnr.read()

    tmp = tmp.splitlines()[6]
    tmp = tmp.split()
    dens = float(tmp[0]) + 2.67
    minc = float(tmp[1])
    mdec = float(tmp[2])
    susc = float(tmp[3])*4*np.pi
    mstrength = float(tmp[4])*1000.
    strikep = float(tmp[5])*scale
    striken = float(tmp[6])*scale

    with open(ifile+'.mag') as fnr:
        tmp = fnr.read()

    tmp = tmp.splitlines()[2]
    tmp = tmp.split()
    finc = float(tmp[0])
    fdec = float(tmp[1])
    hintn = float(tmp[2])

    mag = np.loadtxt(ifile+'.mag', skiprows=3)
    grv = np.loadtxt(ifile+'.grv', skiprows=2)
    body = np.loadtxt(ifile+'.sur', skiprows=7)

    x = body[:, 0] * scale
    z = -body[:, 1] * scale
    xpos = mag[:, 0] * scale

    m2dc = mag[:, 3]
    g2dc = grv[:, 3]

    mht = -mag[:, 1][0] * scale

    # for testing purposes the cube being modelled should have dxy = d_z to
    # keep things simple
    dxy = (xpos[1]-xpos[0])*samplescale
    d_z = dxy
    ypos = np.arange(striken, strikep, dxy)
    xpos2 = np.arange(np.min(xpos)-dxy/2, np.max(xpos)+dxy/2, dxy)
    numx = xpos2.size
    numy = ypos.size
    numz = int(abs(min(z)/d_z))
    tlx = np.min(xpos2)
    tly = np.max(ypos)
    tlz = 0

    print('Hintn (nT):', hintn)
    print('Finc:', finc)
    print('Fdec:', fdec)
    print('Susc:', susc)
    print('Density (g/cm3):', dens)
    print('Remanent Magnetisation (A/m):', mstrength)
    print('Minc:', minc)
    print('Mdec:', mdec)
    print('+Strike:', strikep)
    print('-Strike:', striken)
    print('mag height', mht)

    # quick model initialises a model with all the variables we have defined.
    print('')

    lmod = quick_model(numx, numy, numz, dxy, d_z,
                       tlx, tly, tlz, mht, 0, finc, fdec,
                       ['Generic'], [susc], [dens],
                       [minc], [mdec], [mstrength], hintn)

    # Create the actual model. It is a 3 dimensional vector with '1' where the
    # body lies

    pixels = []
    xmin = xpos2[0]
    for i in range(len(x)):
        pixels.append(((x[i]-xmin)/dxy, -z[i]/d_z))

    img = PIL.Image.new("RGB", (numx, numz), "black")
    draw = PIL.ImageDraw.Draw(img)
    draw.polygon(pixels, outline="rgb(1, 1, 1)", fill="rgb(1, 1, 1)")
    img = np.array(img)[:, :, 0]

    for i in img:
        if np.nonzero(i > 0)[0].size > 0:
            i[np.nonzero(i > 0)[0][-1]] = 0

    tline = []
    bline = []
    i2 = 0
    i = -1
    while i2 < (img.shape[0]-1):
        i += 1
        i1 = int(i**power)
        i2 = int((i+1)**power)
        if i2 > img.shape[0]:
            i2 = img.shape[0]-1

        imid = (i2-i1)//2+i1
        img[i1:i2] = img[imid]
        tline.append(i2)
        bline.append(i2-1)

    tline = np.array(tline)
    bline.pop(0)
    bline.append(tline.max())
    bline = np.array(bline)

    for j in np.arange(0, (strikep-striken), dxy):
        j2 = int(j/dxy)
        lmod.lith_index[:, j2, :] = img[:, :].T

    # Calculate the gravity
    calc_field(lmod)
    gdata = lmod.griddata['Calculated Gravity'].data[numy//2].copy()

    # Change to observation height to 100 meters and calculate magnetics
#    lmod.lith_index_old *= -1
#    lmod.mht = mht
    calc_field(lmod, magcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[numy//2]

    if doplt:
        # Display results
        fig, axes = plt.subplots(2, 1)
        fig.set_figheight(8)
        fig.set_figwidth(6)

        ax1 = axes[0]
        ax1.set_xlim([xpos2[0], xpos2[-1]])
        ax1.set_xlabel('Distance (m)')
        ax1.plot(xpos2+dxy/2, gdata, 'r.', label='Voxel')
        ax1.plot(xpos, g2dc, 'r', label='GM-SYS')
        ax1.set_ylabel('mGal')
        ax1.legend(loc='upper left', shadow=True, title='Gravity Calculation')

        ax2 = ax1.twinx()
        ax2.plot(xpos2+dxy/2, mdata, 'b.', label='Voxel')
        ax2.plot(xpos, m2dc, 'b', label='GM-SYS')
        ax2.set_ylabel('nT')
        ax2.legend(loc='upper right', shadow=True,
                   title='Magnetic Calculation')

        ax3 = axes[1]
        ax3.set_xlim([xpos[0], xpos[-1]])
        ax3.set_ylim([min(z)-d_z, 0])
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Depth (m)')

        mod = lmod.lith_index[:, numy//2].T.tolist()

        for i, row in enumerate(mod):
            for j, col in enumerate(row):
                if col < 1:
                    continue
                y2 = np.array([-i*d_z, -(i+1)*d_z])
                x2 = np.array([j*dxy, j*dxy])+xmin
                ax3.plot(x2, y2, 'c', linewidth=0.5)
                y2 = np.array([-i*d_z, -(i+1)*d_z])
                x2 = np.array([(j+1)*dxy, (j+1)*dxy])+xmin
                ax3.plot(x2, y2, 'c', linewidth=0.5)

                if i in bline:
                    y2 = np.array([-(i+1)*d_z, -(i+1)*d_z])
                    x2 = np.array([(j+1)*dxy, j*dxy])+xmin
                    ax3.plot(x2, y2, 'c', linewidth=0.5)

                if i in tline:
                    y2 = np.array([-i*d_z, -i*d_z])
                    x2 = np.array([j*dxy, (j+1)*dxy])+xmin
                    ax3.plot(x2, y2, 'c', linewidth=0.5)

        ax3.plot(x, z, 'k')
        plt.show()


if __name__ == "__main__":
    test(True)
