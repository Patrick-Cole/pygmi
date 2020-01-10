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
"""
These are pfmod tests. Run this file from within this directory to do the
tests.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..//..')))
from pygmi.pfmod.grvmag3d import quick_model
from pygmi.pfmod.grvmag3d import calc_field


def main():
    """
    Main test function

    This test function compares the calculations performed by PyGMI to
    calculations performed by a external software - GM-SYS

    A series of graphs are produced. If the test is successful, points and
    lines on the graphs will coincide.

    Returns
    -------
    None.

    """
    print('Testing modelling of gravity and potential field data')

    ifile = 'testdata/block'
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

    # Calculate magnetics
    calc_field(lmod, magcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[numy//2]

    # Display results
    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(6)

    # Plot Gravity Data
    ax1 = axes[0]
    ax1.set_xlim([xpos2[0], xpos2[-1]])
    ax1.set_xlabel('Distance (m)')
    ax1.plot(xpos2+dxy/2, gdata, 'r.', label='Voxel')
    ax1.plot(xpos, g2dc, 'r', label='GM-SYS')
    ax1.set_ylabel('mGal')
    ax1.legend(loc='upper left', shadow=True, title='Gravity Calculation')

    # Plot Magnetic Data
    ax2 = ax1.twinx()
    ax2.plot(xpos2+dxy/2, mdata, 'b.', label='Voxel')
    ax2.plot(xpos, m2dc, 'b', label='GM-SYS')
    ax2.set_ylabel('nT')
    ax2.legend(loc='upper right', shadow=True,
               title='Magnetic Calculation')

    # Plot the model
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


def test():
    """
    Test function using pytest.
    """
    dens = 2.8
    minc = 35.
    mdec = 80.
    susc = 0.01
    mstrength = 0.199
    finc = -63.
    fdec = -17.
    hintn = 30000.
    mht = 100.

    dxy = 50.
    d_z = 50.
    numx = 101
    numy = 40
    numz = 10
    tlx = 0
    tly = 0
    tlz = 0

    # quick model initialises a model with all the variables we have defined.
    print('')

    lmod = quick_model(numx, numy, numz, dxy, d_z,
                       tlx, tly, tlz, mht, 0, finc, fdec,
                       ['Generic'], [susc], [dens],
                       [minc], [mdec], [mstrength], hintn)

    # Create the actual model. It is a 3 dimensional vector with '1' where the
    # body lies
    lmod.lith_index[45:56, :, 1:] = 1

    # Calculate the gravity
    calc_field(lmod)
    gdata = lmod.griddata['Calculated Gravity'].data[numy//2].copy()

    calc_field(lmod, magcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[numy//2]

    mdata2 = np.array([-0.59099298, -0.62607842, -0.66391005, -0.70475333,
                       -0.74890404, -0.79669235, -0.84848730, -0.90470211,
                       -0.96580018, -1.03230202, -1.10479332, -1.18393415,
                       -1.27046966, -1.36524250, -1.46920710, -1.58344638,
                       -1.70919097, -1.84784162, -2.00099515, -2.17047451,
                       -2.35836358, -2.56704722, -2.79925736, -3.05812548,
                       -3.34724190, -3.67072174, -4.03327659, -4.44028952,
                       -4.89788861, -5.41300981, -5.99343274, -6.64776082,
                       -7.38529595, -8.21572134, -9.14844163, -10.19131102,
                       -11.34825914, -12.61489175, -13.97027311, -15.36131584,
                       -16.67268151, -17.66937768, -17.89812177, -16.58457686,
                       -12.80280563, -6.37991268, 1.38769507, 8.90468193,
                       15.46123688, 21.02432034, 25.75580485, 29.77137963,
                       33.02828531, 35.22299733, 35.69645555, 33.61201465,
                       28.86655641, 22.81082481, 17.08060706, 12.41076526,
                       8.83790363, 6.16820512, 4.18961534, 2.72758571,
                       1.64937720, 0.85643899, 0.27598829, -0.14580105,
                       -0.44887903, -0.66301928, -0.81050071, -0.90805538,
                       -0.96829001, -1.00072896, -1.01258339, -1.0093216,
                       -0.99509395, -0.97305091, -0.94558212, -0.91449691,
                       -0.88116101, -0.84660059, -0.81158152, -0.77667008,
                       -0.74227948, -0.70870570, -0.67615496, -0.64476507,
                       -0.61462182, -0.58577167, -0.55823159, -0.53199663,
                       -0.50704581, -0.48334664, -0.46085858, -0.43953574,
                       -0.41932891, -0.40018710, -0.38205873, -0.36489246,
                       -0.34863788])

    gdata2 = np.array([0.00696455, 0.00737615, 0.00782039, 0.00830052,
                       0.00882018, 0.00938348, 0.00999503, 0.01066000,
                       0.01138428, 0.01217452, 0.01303826, 0.01398411,
                       0.01502186, 0.01616274, 0.01741963, 0.01880734,
                       0.02034298, 0.02204636, 0.02394049, 0.02605218,
                       0.02841283, 0.03105928, 0.03403497, 0.03739135,
                       0.04118956, 0.04550266, 0.05041831, 0.05604228,
                       0.06250285, 0.06995649, 0.07859522, 0.08865620,
                       0.10043434, 0.11429901, 0.13071653, 0.15028062,
                       0.17375428, 0.20212873, 0.23670811, 0.27923620,
                       0.33209503, 0.39863634, 0.48376606, 0.59490701,
                       0.74145625, 0.91124932, 1.05097028, 1.14813237,
                       1.21145852, 1.24728095, 1.25888510, 1.24728095,
                       1.21145852, 1.14813237, 1.05097028, 0.91124932,
                       0.74145625, 0.59490701, 0.48376606, 0.39863634,
                       0.33209503, 0.27923620, 0.23670811, 0.20212873,
                       0.17375428, 0.15028062, 0.13071653, 0.11429901,
                       0.10043434, 0.08865620, 0.07859522, 0.06995649,
                       0.06250285, 0.05604228, 0.05041831, 0.04550266,
                       0.04118956, 0.03739135, 0.03403497, 0.03105928,
                       0.02841283, 0.02605218, 0.02394049, 0.02204636,
                       0.02034298, 0.01880734, 0.01741963, 0.01616274,
                       0.01502186, 0.01398411, 0.01303826, 0.01217452,
                       0.01138428, 0.01066000, 0.00999503, 0.00938348,
                       0.00882018, 0.00830052, 0.00782039, 0.00737615,
                       0.00696455])

    np.testing.assert_array_almost_equal(gdata, gdata2)
    np.testing.assert_array_almost_equal(mdata, mdata2)


if __name__ == "__main__":
    main()
#    test()
