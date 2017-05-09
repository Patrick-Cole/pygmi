# -*- coding: utf-8 -*-
"""
Anaglyph

"""

import pdb
from time import sleep
import scipy.interpolate as si
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import OpenGL.GLUT as GLUT
import OpenGL.GL as GL
import OpenGL.GLU as GLU

animationAngle = 0.0
frameRate = 20
stereoMode = "NONE"
lightColors = {"white": (1.0, 1.0, 1.0, 1.0),
               "red": (1.0, 0.0, 0.0, 1.0),
               "green": (0.0, 1.0, 0.0, 1.0),
               "blue": (0.0, 0.0, 1.0, 1.0)}

lightPosition = (5.0, 5.0, 20.0, 1.0)


class StereoCamera(object):
    """ Stereo Camera """
    def __init__(self):
        self.lookAtLeft = None
        self.lookAtRight = None
        self.frustumLeft = None
        self.frustumRight = None

    def update(self):
        """ Update """
        w = 518.4  # Physical display dimensions in mm. (Width)
        h = 324.0  # (Height)

        # Distance in the scene from the camera to the display plane.
        Z = 1000.0
        A = 65.0  # Camera inter-axial separation (eye separation).
        # Distance in the scene from the camera to the near plane.
        Near = 800.0
        Far = 1200.0  # Distance in the scene from the camera to the far plane.

        # Calculations for Left eye/camera Frustum
        L_l = -(Near * ((w/2.0 - A/2.0) / Z))  # left clipping pane
        L_r = (Near * ((w/2.0 + A/2.0) / Z))  # right clipping pane
        L_b = -(Near * ((h/2.0)/Z))  # bottom clipping pane
        L_t = (Near * ((h/2.0)/Z))  # top clipping pane

        # Calculations for Right eye/camera Frustum
        R_l = -(Near * ((w/2.0 + A/2.0) / Z))  # left clipping pane
        R_r = (Near * ((w/2.0 - A/2.0) / Z))  # right clipping pane
        R_b = - (Near * ((h/2.0)/Z))  # bottom clipping pane
        R_t = (Near * ((h/2.0)/Z))  # top clipping pane

        # Lookat points for left eye/camera
        self.lookAtLeft = (-A/2, 0, 0, -A/2, 0, -Z, 0, 1, 0)
        # Lookat points for right eye/camera
        self.lookAtRight = (A/2, 0, 0, A/2, 0, -Z, 0, 1, 0)
        # Parameters for glFrustum (Left)
        self.frustumLeft = (L_l, L_r, L_b, L_t, Near, Far)
        # Parameters for glFrustum (Right)
        self.frustumRight = (R_l, R_r, R_b, R_t, Near, Far)

# test program - when stereoCamera.py is run, will print values in lists
# (collection) for lookAt points and Frustum parameters


def animationStep():  # Setting abimation for centre cube under rotation
    """Update animated parameters."""
    global animationAngle
    global frameRate
    animationAngle += 2
    while animationAngle > 360:
        animationAngle -= 360
    sleep(1 / float(frameRate))
    GLUT.glutPostRedisplay()


def setLightColor(scol):
    """Set light color to 'white', 'red', 'green' or 'blue'."""
    if scol in lightColors:
        col = lightColors[scol]
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, col)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, col)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, col)


def render(side):
    """Render scene in either GLU_BACK_LEFT or GLU_BACK_RIGHT buffer"""
    boxSize = 50  # size of cube height, width and depth = 50mm
    separate = 100  # separation for array of cubes 100mm
    GL.glViewport(0, 0, GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH),
                  GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))

    if side == GL.GL_BACK_LEFT:  # render left frustum and lookAt points
        f = sC.frustumLeft
        l = sC.lookAtLeft
    else:  # render right side view
        f = sC.frustumRight
        l = sC.lookAtRight
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    # collect parameters from stereoCamera.py for frustum
    GL.glFrustum(f[0], f[1], f[2], f[3], f[4], f[5])
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    # collect lookAt parameters from stereoCamera.py
    GLU.gluLookAt(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8])


# draw array of cubes at varying positions across screen
    GL.glPushMatrix()
    GL.glTranslatef(-separate, 0.0, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    # draws centre cube - 100mm depth difference
    GL.glTranslatef(0.0, 0.0, -900)
    GL.glRotatef(animationAngle, 0.2, 0.7, 0.3)  # rotates the cube
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate, 0.0, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(-separate, separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(0.0, separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate, separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(-separate, -separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(0.0, -separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate, -separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate*2, separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate*2, 0.0, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(separate*2, -separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(-separate*2, separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(-separate*2, 0.0, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()

    GL.glPushMatrix()
    GL.glTranslatef(-separate*2, -separate, -1000)
    GLUT.glutSolidCube(boxSize)
    GL.glPopMatrix()


def display():
    """
    Glut display function.
    display relevant view - SHUTTER (true stereo, quad buffered),
    ANAGLYPH, or NONE (Monoscopic)
    """
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    GL.glDrawBuffer(GL.GL_BACK_LEFT)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    setLightColor("red")
    render(GL.GL_BACK_LEFT)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    GL.glColorMask(False, True, False, False)
    setLightColor("blue")
    render(GL.GL_BACK_RIGHT)
    GL.glColorMask(True, True, True, True)
    GLUT.glutSwapBuffers()


def init():  # OpenGL functions setting light, colour, texture etc
    """ Glut init function."""
    GL.glClearColor(0, 0, 0, 0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glShadeModel(GL.GL_SMOOTH)
    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 0)
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [4, 4, 4, 1])
    lA = 0.8
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [lA, lA, lA, 1])
    lD = 1
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [lD, lD, lD, 1])
    lS = 1
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [lS, lS, lS, 1])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT, [0.2, 0.2, 0.2, 1])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [0.5, 0.5, 0.5, 1])
    GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 50)
    sC.update()


def main_parallel():
    """ parallel
    SHUTTER | ANAGLYPH | NONE """
    global sC
    global stereoMode

    sC = StereoCamera()
    GLUT.glutInit([b'anaglyph.py', b"ANAGLYPH"])
    stereoMode = "ANAGLYPH"
    GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGB |
                             GLUT.GLUT_DEPTH)
    GLUT.glutInitWindowSize(800, 600)
    GLUT.glutInitWindowPosition(100, 100)
    GLUT.glutCreateWindow(b'anaglyph.py')
    init()
    GLUT.glutDisplayFunc(display)

    GLUT.glutIdleFunc(animationStep)
    GLUT.glutMainLoop()
    pdb.set_trace()


def sunshade(data, azim=-np.pi/4., elev=np.pi/4., alpha=1, cell=100,
             cmap=cm.terrain):
    """
    data: input MxN data to be imaged
    alpha: how much incident light is reflected (0 to 1)
    phi: azimuth
    theta: sun elevation
    cell: between 1 and 100 - controls sunshade detail.
    """
    mask = np.ma.getmaskarray(data)

    sunshader = currentshader(data, cell, elev, azim, alpha)
    snorm = norm2(sunshader)
    pnorm = np.uint8(norm2(histcomp(data))*255)
    # pnorm = uint8(norm2(data)*255)

    colormap = cmap(pnorm)
    colormap[:, :, 0] = colormap[:, :, 0]*snorm
    colormap[:, :, 1] = colormap[:, :, 1]*snorm
    colormap[:, :, 2] = colormap[:, :, 2]*snorm
    colormap[:, :, 3] = np.logical_not(mask)

    return colormap


def norm2(dat, datmin=None, datmax=None):
    """ Normalise vector """

    if datmin is None:
        datmin = np.min(dat)
    if datmax is None:
        datmax = np.max(dat)
    return (dat-datmin)/(datmax-datmin)


def currentshader(data, cell, theta, phi, alpha):
    """
    Blinn shader
        alpha: how much incident light is reflected
        n: how compact the bright patch is
        phi: azimuth
        theta: sun elevation (also called g in code below)
    """

    cdy = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    cdx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])

    dzdx = ndimage.convolve(data, cdx)  # Use convolve: matrix filtering
    dzdy = ndimage.convolve(data, cdy)  # 'valid' gets reduced array

    dzdx = dzdx/8.
    dzdy = dzdy/8.

    pinit = dzdx
    qinit = dzdy

# Update cell
    p = pinit/cell
    q = qinit/cell
    sqrt_1p2q2 = np.sqrt(1+p**2+q**2)

# Update angle
    cosg2 = np.cos(theta/2)
    p0 = -np.cos(phi)*np.tan(theta)
    q0 = -np.sin(phi)*np.tan(theta)
    sqrttmp = (1+np.sqrt(1+p0**2+q0**2))
    p1 = p0 / sqrttmp
    q1 = q0 / sqrttmp

    n = 2.0

    cosi = ((1+p0*p+q0*q)/(sqrt_1p2q2*np.sqrt(1+p0**2+q0**2)))
    coss = ((1+p1*p+q1*q)/(sqrt_1p2q2*np.sqrt(1+p1**2+q1**2)))
    Ps = coss**n
    R = ((1-alpha)+alpha*Ps)*cosi/cosg2
    return R


def histcomp(img, nbr_bins=256, perc=5.):
    """ Histogram Compaction """
    tmp = img.compressed()

    imhist, bins = np.histogram(tmp, nbr_bins)

    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / float(cdf[-1])  # normalize

    perc = perc/100.

    sindx = np.arange(nbr_bins)[cdf > perc][0]
    eindx = np.arange(nbr_bins)[cdf < (1-perc)][-1]+1
    svalue = bins[sindx]
    evalue = bins[eindx]

    scnt = perc*(nbr_bins-1)
    if scnt > sindx:
        scnt = sindx

    ecnt = perc*(nbr_bins-1)
    if ecnt > ((nbr_bins-1)-eindx):
        ecnt = (nbr_bins-1)-eindx

    img2 = np.empty_like(img, dtype=np.float32)
    np.copyto(img2, img)

    filt = np.ma.less(img2, svalue)
    img2[filt] = svalue

    filt = np.ma.greater(img2, evalue)
    img2[filt] = evalue

    return img2


def anaglyph(red, blue, atype='dubois'):
    """ color Aanaglyph """

    if atype == 'dubois':
        mat = np.array([[0.437, 0.449, 0.164, -0.011, -0.032, -0.007],
                        [-0.062, -0.062, -0.024, 0.377, 0.761, 0.009],
                        [-0.048, -0.050, -0.017, -0.026, -0.093, 1.234]])
    elif atype == 'color':
        mat = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    elif atype == 'true':
        mat = np.array([[0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114]])
    elif atype == 'gray':
        mat = np.array([[0.299, 0.587, 0.114, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114],
                        [0.0, 0.0, 0.0, 0.299, 0.587, 0.114]])

    newshape = (red.shape[0]*red.shape[1], 3)
    data1 = red[:, :, :3].copy()
    data2 = blue[:, :, :3].copy()
    data1.shape = newshape
    data2.shape = newshape
    mask = red[:, :, 3]
    data = np.transpose(np.hstack((data1, data2)))

    rgb = mat @ data
    rgb[rgb < 0] = 0
    rgb[rgb > 1] = 1

    rgb = np.vstack((rgb, mask.flatten()))
    rgb = rgb.T
    rgb.shape = red.shape

#    red1 = RL*0.4154 + GL*0.4710 + BL*0.1669
#    red2 = (-RR*0.0109 - GR*0.0364 - BR*0.006)
#    green1 = -RL*0.0458 - GL*0.0484 - BL*0.0257
#    green2 = RR*0.3756 + GR*0.7333 + BR*0.0111
#    blue1 = -RL*0.0547 - GL*0.0615 + BL*0.0128
#    blue2 = (-RR*0.0651 - GR*0.1287 + BR*1.2971)

    return rgb


def offsets(data, sdata, scale=2.0):
    """ offsets """

    scale = data.ptp() / scale

    odata = (data - data.min())/scale
    odata = odata.astype(int)

    rows, cols = data.shape

    for icol in range(cols):
        for jrow in range(rows):
            offset = icol - odata[jrow, icol]
            if np.ma.is_masked(offset):
                continue
            if offset < 0:
                offset = 0
            sdata[jrow, offset] = sdata[jrow, icol]

    return sdata


def load_data():
    """ load data """

    z = np.loadtxt('mag.asc', skiprows=6)
    z = np.ma.masked_equal(z, 1e+20)

    y, x = np.indices(z.shape)

    dxy = 125
    dxy = 125

    x *= dxy
    y *= dxy

    x = x[20:100, 25:150]
    y = y[20:100, 25:150]
    z = z[20:100, 25:150]

#    z.set_fill_value(np.nan)
#    z = z.filled()
#    z = np.ma.masked_invalid(z)

    return x, y, z


def rot_and_clean(x, y, z, rotang=5, rtype='red', doshade=False):
    """ rotates and cleans rotated data for 2d view """

    cmap = cm.viridis
    cmap = cm.jet
    if rtype == 'red':
        rotang = -1. * abs(rotang)
    else:
        rotang = -1. * abs(rotang)
        z = z[:, ::-1]

    x = x-x.min()
    y = y-y.min()
    z = z-np.median(z)
    xa = [x.min(), x.max()]
    ya = [y.min(), y.max()]
    za = [0, 0]

    a = np.deg2rad(rotang)
    m = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]

    t = np.transpose(np.array([xa, ya, za]))
    xa1, _, _ = np.transpose(np.dot(t, m))

    t = np.transpose(np.array([x, y, z]))
    x1, y1, z1 = np.transpose(np.dot(t, m))

    zmap = z1.copy()
    zi = np.ma.filled(z)

    for j, xi in enumerate(x1):
        xmask = np.ones_like(xi, dtype=bool)
        xmax = xi.min()-1.
        for i, xtmp in enumerate(xi):
            if xtmp > xmax:
                if xmask[i-1] == False and i > 0:
                    zi[j][i-1] = np.interp([xmax], [xi[i-1], xi[i]],
                                           [zi[j][i-1], zi[j][i]])[0]
                    xi[i-1] = xmax
                    xmask[i-1] = True

                xmax = xtmp
            else:
                xmask[i] = False

        xmask[xi < xa1[0]] = False
        xmask[xi > xa1[1]] = False

        x2 = np.linspace(xa1.min(), xa1.max(), x1.shape[1])
        zmap[j] = np.interp(x2, xi[xmask], zi[j][xmask])

    if rtype != 'red':
        zmap = zmap[:, ::-1]
    zmap = np.ma.masked_equal(zmap, 1e+20)
    z3 = zmap.copy()
    zmask = zmap.mask.copy()

    if not doshade:
        zmin = np.median(z) - 2 * np.std(z)
        zmax = np.median(z) + 2 * np.std(z)

        tmp = norm2(zmap, zmin, zmax)
        zmap = cmap(tmp)
        zmap[:, :, 3] = np.logical_not(zmask)
    else:
        alpha = 0
        cell = 100
        azim = np.deg2rad(45)
        elev = np.deg2rad(45)
        zmap = sunshade(z3, azim=azim, elev=elev, alpha=alpha, cell=cell,
                        cmap=cmap)

    return zmap, z3


def main_sunshade():
    """ main """

    x, y, data = load_data()

    doffset = 15.
    alpha = 1.
    cell = 5

    deg = np.deg2rad(doffset)
    azim = np.deg2rad(0)
    elev = np.deg2rad(45)

    sdata1 = sunshade(data, azim=azim, elev=elev, alpha=alpha, cell=cell)
    tdata = sunshade(data, azim=azim-deg, elev=elev-deg, alpha=alpha,
                     cell=cell)

#    tdata = offsets(data, sdata1, scale)

    adata = anaglyph(sdata1, tdata)

    plt.imshow(adata)
    plt.show()

    pdb.set_trace()


def main_gsc():
    """
    http://www.animesh.me/2011/05/rendering-3d-anaglyph-in-opengl.html

    red is left eye
    blue is right eye

    plane of convergence is the plane of the screen
    for deep objects (valleys), red image is more to the left
    for high objects, red image is more to the right

    """

    scale = 30.0

    x, y, data = load_data()

    alpha = 1.
    cell = 5
    azim = np.deg2rad(0)
    elev = np.deg2rad(45)
    sdata = sunshade(data, azim=azim, elev=elev, alpha=alpha, cell=cell)

    blue = sdata.copy()
    red = sdata.copy()

#    red[:, :, :3] *= 0

#    scale = data.ptp() / scale

    odata = (data - np.min(data))/scale
    odata = odata.astype(int)
    tmp = np.zeros_like(odata)

    rows, cols = data.shape

    for icol in range(cols):
        for jrow in range(rows):
            offset = icol + odata[jrow, icol]
            if np.ma.is_masked(offset):
                continue
            try:
                if odata[jrow, icol] > tmp[jrow, offset]:
                    red[jrow, icol:offset+1] = sdata[jrow, icol]
                    tmp[jrow, offset] = odata[jrow, icol]
            except IndexError:
                pass
#            if offset < icol:
#                red[jrow, offset:icol] = sdata[jrow, icol]
#            else:
#                red[jrow, icol:offset+1] = sdata[jrow, icol]

    adata = anaglyph(red, blue)

    plt.subplot(121)
    plt.imshow(red)
    plt.subplot(122)
    plt.imshow(blue)
    plt.show()

    plt.imshow(adata)
    plt.show()


def main_rotate():
    """ main rotate """

    scale = 7
    rotang = 10  # 0.5 to 10 degrees

    x, y, z = load_data()
    z *= scale

    red, zr = rot_and_clean(x, y, z, rotang, 'red', True)
    blue, zb = rot_and_clean(x, y, z, rotang, 'blue', True)

#    plt.plot(zr[50], 'r')
#    plt.plot(zb[50], 'b')
#    plt.show()

    plt.subplot(211)
    plt.imshow(red, origin='lower')
    plt.subplot(212)
    plt.imshow(blue, origin='lower')
    plt.show()

    adata = anaglyph(red, blue, 'dubois')

    plt.imshow(adata, origin='lower', interpolation='spline36')
#    plt.contour(zr, colors='r', origin='lower')
#    plt.contour(zb, colors='c', origin='lower')
    plt.show()

#    x, y = np.indices(zr.shape)
#    ax = plt.gca()
#    ax.set_axis_bgcolor('black')
#    plt.contour(zr, colors='r', origin='lower')
#    plt.contour(zb, colors='c', origin='lower')
#    plt.show()

    pdb.set_trace()


if __name__ == "__main__":
    main_rotate()
