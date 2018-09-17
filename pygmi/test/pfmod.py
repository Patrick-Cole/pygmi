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

import pdb
import numpy as np
import matplotlib.pyplot as plt
from pygmi.pfmod.grvmag3d import quick_model
from pygmi.pfmod.grvmag3d import calc_field
import pygmi.misc as ptimer
import PIL


def test(doplt=False):
    """
    Main test function

    This test function compares the calculations performed by PyGMI to
    calculations performed by a external software - namely mag2dc and grav2dc
    by G.R.J Cooper.

    A series of graphs are produced. If the test is successful, points and
    lines on the graphs will coincide.
    """
    print('Testing modelling of gravity and potential field data')

    ifile = 'block'
    ifile = 'dyke'
    samplescale = 1
    power = 2
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

    if power != 1:
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
            print(i1, i2)

    for j in np.arange(0, (strikep-striken), dxy):
        j2 = int(j/dxy)
        lmod.lith_index[:, j2, :] = img[:, :].T

#    pdb.set_trace()

    # Calculate the gravity
    calc_field(lmod)
    gdata = lmod.griddata['Calculated Gravity'].data[numy//2].copy()

    # Change to observation height to 100 meters and calculate magnetics
    lmod.lith_index_old *= -1
#    lmod.mht = mht
    calc_field(lmod, magcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[numy//2]

###############################################################################

    y0 = 0
    z0 = -100
    x1 = x.min()
    x2 = x.max()
    y1 = striken
    y2 = strikep
    z1 = z.max()
    z2 = z.min()
    fi = finc
    fd = fdec
    mi = minc
    md = mdec
    theta = 90
    h = hintn
    m = mstrength
    k = susc

    # Note: technically mbox is NED, so when rotated by 90 (theta)
    # this becomes ESD. Therefore, new y axis is negaitive on top, and
    # positive at bottom. This is what is below. The code below tests against
    # axis orientation in PyGMI.

    tt1 = []
    for y0 in ypos:
        t1 = []
        for x0 in xpos:
            t1.append(mbox(x0, y0, z0, x1, y1, -z1, x2, y2, mi, md, fi, fd, m,
                           theta, h, k))
        tt1.append(t1)

    tt2 = []
    for y0 in ypos:
        t2 = []
        for x0 in xpos:
            t2.append(mbox(x0, y0, z0, x1, y1, -z2, x2, y2, mi, md, fi, fd, m,
                           theta, h, k))
        tt2.append(t2)

    t1 = np.array(t1)
    t2 = np.array(t2)
    t = t1-t2

    ttall = np.array(tt1)-np.array(tt2)


    plt.subplot(211)
    plt.imshow(ttall)
    plt.subplot(212)
    plt.imshow(lmod.griddata['Calculated Magnetics'].data)
    plt.show()

#    pdb.set_trace()


###############################################################################


    if doplt:
        # Display results
        fig, axes = plt.subplots(2, 1)
        fig.set_figheight(8)
        fig.set_figwidth(6)

        ax1 = axes[0]
#        _, ax1 = plt.subplots(2,1)
        ax1.set_xlim([xpos2[0], xpos2[-1]])
        ax1.set_xlabel('Distance (m)')
        ax1.plot(xpos2+dxy/2, gdata, 'r.', label='Voxel')
        ax1.plot(xpos, g2dc, 'r', label='GM-SYS')
        ax1.set_ylabel('mGal')
        ax1.legend(loc='upper left', shadow=True, title='Gravity Calculation')

        ax2 = ax1.twinx()
        ax2.plot(xpos2+dxy/2, mdata, 'b.', label='Voxel')
        ax2.plot(xpos, m2dc, 'b', label='GM-SYS')
#        ax2.plot(xpos, t, '+', label='mbox')
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
                print(i)
#                y2 = np.array([-i*d_z, -i*d_z, -(i+1)*d_z, -(i+1)*d_z, -i*d_z])
#                x2 = np.array([j*dxy, (j+1)*dxy, (j+1)*dxy, j*dxy, j*dxy])+xmin
                y2 = np.array([-i*d_z, -(i+1)*d_z])
                x2 = np.array([j*dxy, j*dxy])+xmin
                ax3.plot(x2, y2, 'c', linewidth=0.5)
                y2 = np.array([-i*d_z, -(i+1)*d_z])
                x2 = np.array([(j+1)*dxy, (j+1)*dxy])+xmin
                ax3.plot(x2, y2, 'c', linewidth=0.5)

                if i==3 or i==8 or i==9:
                    y2 = np.array([-(i+1)*d_z, -(i+1)*d_z])
                    x2 = np.array([(j+1)*dxy, j*dxy])+xmin
                    ax3.plot(x2, y2, 'c', linewidth=0.5)

                if i==1 or i==4 or i==9:
                    y2 = np.array([-i*d_z, -i*d_z])
                    x2 = np.array([j*dxy, (j+1)*dxy])+xmin
                    ax3.plot(x2, y2, 'c', linewidth=0.5)


        ax3.plot(x,z, 'k')
        plt.show()


#        gdata2 = np.interp(xpos, xpos2+dxy/2, gdata)
#        mdata2 = np.interp(xpos, xpos2+dxy/2, mdata)

        if samplescale == 1:
            print('Gravity Error:', np.mean(np.abs(gdata-g2dc)))
            print(' Max Difference:', (np.abs(gdata-g2dc)).max())
            print('Magnetic Error:', np.mean(np.abs(mdata-m2dc)))
            print(' Max Difference:', (np.abs(mdata-m2dc)).max())
            print('Mbox Error:', np.mean(np.abs(t-m2dc)))
            print(' Max Difference:', (np.abs(t-m2dc)).max())

#    pdb.set_trace()


def test_old(doplt=False):
    """
    Main test function

    This test function compares the calculations performed by PyGMI to
    calculations performed by a external software - namely mag2dc and grav2dc
    by G.R.J Cooper.

    A series of graphs are produced. If the test is successful, points and
    lines on the graphs will coincide.
    """
    print('Testing modelling of gravity and potential field data')

    # First initialise variables
    x = []
    z = []
    xpos = []
    g2dc = []
    m2dc = []

    # start to load in gravity data
    gfile = open('Grav2dc_grav.txt')
    tmp = gfile.read()
    tmp2 = tmp.splitlines()

    numx = get_int(tmp2, 2, 2)
    cnrs = get_int(tmp2, 6, 4)
    dens = get_float(tmp2, 6, 8) + 2.67
    strike = get_float(tmp2, 7, 3)

    for i in range(cnrs):
        x.append(get_float(tmp2, 9+i, 0))
        z.append(get_float(tmp2, 9+i, 1))

    for i in range(numx):
        xpos.append(get_float(tmp2, 11+cnrs+i, 0))
        g2dc.append(get_float(tmp2, 11+cnrs+i, 1))

    gfile.close()

    # Convert to numpy and correct orientation of depths.
    x = np.array(x)
    z = -np.array(z)
    xpos = np.array(xpos)
    g2dc = np.array(g2dc)

    # Start to load in magnetic data
    mfile = open('Mag2dc_mag.txt')
#    mfile = open('Mag-1SI.txt')
    tmp = mfile.read()
    tmp2 = tmp.splitlines()

    hintn = get_float(tmp2, 4, 2)
    finc = get_float(tmp2, 4, 5)
    fdec = get_float(tmp2, 4, 8)
    mht = get_float(tmp2, 7, 5)
    susc = get_float(tmp2, 12, 7)
#    minc = -finc+1 # get_float(tmp2, 14, 9)
#    mdec = fdec+180 # get_float(tmp2, 14, 12)
    minc = get_float(tmp2, 14, 9)
    mdec = get_float(tmp2, 14, 12)
#    mstrength = get_float(tmp2, 14, 5)/(100)
    mstrength = get_float(tmp2, 14, 5)/(400*np.pi)
#    mstrength = 0
#    mstrength = hintn*susc/(400*np.pi)  # nT/mu0 = T*10-9/4pi*10-7



    print('k:', susc)

    for i in range(numx):
        m2dc.append(get_float(tmp2, 18+cnrs+i, 1))

    mfile.close()

    # Convert to numpy
    m2dc = np.array(m2dc)

    # for testing purposes the cube being modelled should have dxy = d_z to
    # keep things simple
    dxy = (xpos[1]-xpos[0])
    d_z = 50
    ypos = np.arange(-strike, strike, dxy)
    zpos = np.arange(z.min(), 0, d_z)
    xpos2 = np.arange(np.min(xpos), np.max(xpos), dxy)
    numy = ypos.size
    numz = zpos.size
    tlx = np.min(xpos)
    tly = np.max(ypos)
    tlz = np.max(zpos)

#    print('Remanent Parameters')
#    print(mstrength*400*np.pi, minc, mdec)
###############################################################################

    y0 = 0
    z0 = -100
    x1 = x.min()
    x2 = x.max()
    y1 = -strike
    y2 = strike
    z1 = z.max()
    z2 = z.min()
    fi = finc
    fd = fdec
    mi = minc
    md = mdec
    theta = 90
    h = hintn
    m = mstrength
    k = susc

    t1 = []
    for x0 in xpos:
        t1.append(mbox(x0, y0, z0, x1, y1, -z1, x2, y2, mi, md, fi, fd, m,
                       theta, h, k))

    t2 = []
    for x0 in xpos:
        t2.append(mbox(x0, y0, z0, x1, y1, -z2, x2, y2, mi, md, fi, fd, m,
                       theta, h, k))

    t1 = np.array(t1)
    t2 = np.array(t2)
    t = t1-t2

###############################################################################
    # quick model initialises a model with all the variables we have defined.
#    ttt = ptimer.PTime()
    lmod = quick_model(xpos2.size, numy, numz, dxy, d_z,
                       tlx, tly, tlz, 0, 0, finc, fdec,
                       ['Generic'], [susc], [dens],
                       [minc], [mdec], [mstrength], hintn)
#    ttt.since_last_call('quick model')

    # Create the actual model. It is a 3 dimensional vector with '1' where the
    # body lies
    for i in np.arange(np.min(x), np.max(x), dxy):
        for j in np.arange(0, 2*strike, dxy):
            for k in np.arange(abs(z).min()/d_z, abs(z).max()/d_z):
                i2 = int(i/dxy)
                j2 = int(j/dxy)
                k2 = int(k)
                lmod.lith_index[i2, j2, k2] = 1

#    ttt.since_last_call('model create')

    # Calculate the gravity
    calc_field(lmod)
    gdata = lmod.griddata['Calculated Gravity'].data[numy//2].copy()
#    ttt.since_last_call('gravity calculation')

    # Change to observation height to 100 meters and calculate magnetics
    lmod.lith_index_old *= -1
    lmod.mht = mht
    calc_field(lmod, magcalc=True)

    mdata = lmod.griddata['Calculated Magnetics'].data[numy//2]

#    ttt.since_last_call('magnetic calculation')

    if doplt:
        # Display results
        _, ax1 = plt.subplots()
        ax1.set_xlabel('Distance (m)')
        ax1.plot(xpos2+dxy/2, gdata, 'r', label='PyGMI')
        ax1.plot(xpos, g2dc, 'r.', label='Grav2DC')
        ax1.set_ylabel('mGal')
        ax1.legend(loc='upper left', shadow=True)

        ax2 = ax1.twinx()
        ax2.plot(xpos2+dxy/2, mdata, 'b.', label='PyGMI')
#        ax2.plot(xpos, m2dc, 'b', label='Mag2DC')
        ax2.plot(xpos, t, '.', label='mbox')
        ax2.set_ylabel('nT')
        ax2.legend(loc='upper right', shadow=True)
        plt.show()

#    print(mdata[:-1]-m2dc[2::2])
#    np.testing.assert_almost_equal(gdata[:-1], g2dc[2::2], 1)
#    np.testing.assert_almost_equal(mdata[:-1], m2dc[2::2], 1)


def get_int(tmp, row, word):
    """
    Gets an int from a list of strings.

    Parameters
    ----------
    tmp : list
        list of strings
    row : int
        row in list to extract. First row is 0
    word : int
        word to extract from row. First word is 0

    Returns
    -------
     output : int
         return an integer from the row.
    """
    return int(tmp[row].split()[word])


def get_float(tmp, row, word):
    """
    Gets a float from a list of strings.

    Parameters
    ----------
    tmp : list
        list of strings
    row : int
        row in list to extract. First row is 0
    word : int
        word to extract from row. First word is 0

    Returns
    -------
     output : float
         return a float from the row.
    """
    return float(tmp[row].split()[word])


def dircos(inc, dec, azim):
    """ dircos """

    I = np.deg2rad(inc)
    D = np.deg2rad(dec-azim)

    a = np.cos(I)*np.cos(D)
    b = np.cos(I)*np.sin(D)
    g = np.sin(I)

    return a, b, g


def mbox(x0, y0, z0, x1, y1, z1, x2, y2, mi, md, fi, fd, m, theta, h, k):
    """

    Subroutine MBOX computes the total field anomaly of an infinitely
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

    Output paramters:
        Total field anomaly t, in nT.
    """

    cm = 1.e-7  # constant for SI
    t2nt = 1.e9  # telsa to nT

    h = h / (400*np.pi)  #/(1+k)  # *10-9/mu0*10-7

    ma, mb, mc = dircos(mi, md, theta)
    fa, fb, fc = dircos(fi, fd, theta)

    mr = m*np.array([ma, mb, mc])
    mi = k*h*np.array([fa, fb, fc])
    m3 = mr+mi

    if np.max(np.abs(m3)) < np.finfo(float).eps:
        m3 = np.array([0., 0., 0.])
        mt = 0.
    else:
        mt = np.sqrt(m3 @ m3)
        m3 /= mt

    ma, mb, mc = m3

    fm1 = ma*fb + mb*fa
    fm2 = ma*fc + mc*fa
    fm3 = mb*fc + mc*fb
    fm4 = ma*fa
    fm5 = mb*fb
    fm6 = mc*fc
    alpha = [x1-x0, x2-x0]
    beta = [y1-y0, y2-y0]
    h = z1-z0
    t = 0.
    hsq = h**2

    for i in [0, 1]:
        alphasq = alpha[i]**2
        for j in [0, 1]:
            sign = 1.
            if i != j:
                sign = -1.
            r0sq = alphasq+beta[j]**2+hsq
            r0 = np.sqrt(r0sq)
            r0h = r0*h
            alphabeta = alpha[i]*beta[j]
            arg1 = (r0-alpha[i])/(r0+alpha[i])
            arg2 = (r0-beta[j])/(r0+beta[j])
            arg3 = alphasq+r0h+hsq
            arg4 = r0sq+r0h-alphasq
            tlog = fm3*np.log(arg1)/2.+fm2*np.log(arg2)/2. - fm1*np.log(r0+h)
            tatan = (-fm4*np.arctan2(alphabeta, arg3) -
                     fm5*np.arctan2(alphabeta, arg4) +
                     fm6*np.arctan2(alphabeta, r0h))

            t = t+sign*(tlog+tatan)
    t = t*mt*cm*t2nt

    return t


if __name__ == "__main__":
    test(True)
