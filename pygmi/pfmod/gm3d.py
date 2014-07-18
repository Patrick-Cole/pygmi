""" Grvmag3d """
# pylint: disable=C0103, E1101

import numpy as np
from scipy.linalg import norm
import pygmi.pfmod.grvmagc_33_x64 as grvmagc
# import pygmi.ptimer as ptimer

flimit = 64*np.spacing(1)    # floating point accuracy


def grvmag():
    """
# Algorithm for simultaneous computation of gravity and magnetic fields is
# based on the formulation published in GEOPHYSICS v. 66,521-526,2001.
# by Bijendra Singh and D. Guptasarma,
# The source code, gm3dnew.m, an m-file written in MATLAB
# The program requires the file angle.m and a model file. The model file
# provides the model parameters and also specifies whether only the magnetic,
# gravity, or both anomalies are to be computed. A model file, trapezod.m, for
# the trapezohedron model from Coggon (1976), is provided. The variables
# defining the model parameters are explained in the comments in this file.
# The function program angle.m computes the angle between two planes to get
# the solid angle subtended by a polygonal facet at the origin, using the
# scheme given by Guptasarma and Singh (1999).
# As written, gm3dnew.m computes the cardinal components of the magnetic
# (Hx, Hy and Hz), and the gravity (Gx, Gy and Gz) fields over a rectangular
# array of stations along one or more NS profiles with uniformly spaced
# stations, on uniformly spaced profiles. The correct total magnetic field
# anomaly (Dt), and the usual approximation (Dta), obtained as the
# projection of the anomalous field along the direction of the ambient earths
# field, are also computed. Comments within the source code lines
# (written after the # sign) would facilitate reading the code for modifying
# it or rewriting in some other programming language.

 # Script for simultaneous computation of G-M fields from a 3D polyhedron.
 # With all distances in metres, model density in g/cm3,
 # ambient magnetic induction and remanant magnetization in nT,
 # and the magnetic susceptibility in SI,
 # it gives gravity fields in milligals and magnetic fields in nT.

# There are some misprints in the paper, which may lead to the incorrect
# results.
#
#The offending line is printed as
#
#  W=W+2angle(p1,p2,p3,Un(f))
#
# which should read as W=W+angle(p1,p2,p3,Un(f))
#
#The other serious misprints are:
#
#  The clearing of the record of integration was printed within the
#  loop over faces which meant that housekeeping to avoid double
#  calculations of I was not operational.
#
# There were some less serious misprints:
#
#c) The line
#
#    if Edge(Eno,6)==1 was printed with = instead of ~=.
#
#d) The calculation of Htot referred to H as a 2 dimensional matrix
#    while it is earlier defined as a vector.
#
"""
# -----------------MAIN PROGRAM--------------------------------------

# Trapezohedron model from Coggon, J. H.,1976, Magnetic and gravity anomalies
# of polyhedra:Geoexploration, 14, 93-105
# The model has 24 faces and 26 corners. It is centered at the origin of a
# right handed system with z downwards. The sections of the model along the
# xy, yz and zx planes are identical octagons. The corners of the octagon in
# the xy plane are at:
# (100,0,0) (75,75,0) (0,100,0) (-75,75,0)
# (-100,0,0) (-75,-75,0) (0,-100,0) (75,-75,0).
# The corners in a plane making 45 deg with the +ve x and +ve y directions are
# at
# (0,0,-100) (60,60,-60) (75,75,0) (60,60,60)
# (0,0,100) (-60,-60,60) (-75,-75,0) (-60,-60,-60)

#    ttt = ptimer.PTime()

    Hintn = 50000    # total intensity of ambient magnetic field, nT
    Hincl = 50       # inclination, degrees, downward from horizontal
    Decl = 0        # Declination, degrees, clockwise from North

    Susc = 0.1            # susceptibility in SI units

    Mstrength = 0          # Remanent magnetic induction, nT
    Mincl = 0              # corresponding inclination
    Mdecl = 0              # and declination

    # Corner is an array of x,y,z coordinates in metres, one in each row
    # in a RHS with x North, y East and z down. Corners can be given in
    # any order.

    Ncor = 8
    Corner = np.array([[-10, -10, 1],
                       [-10, 10, 1],
                       [10, 10, 1],
                       [10, -10, 1],
                       [-10, -10, 10],
                       [-10, 10, 10],
                       [10, 10, 10],
                       [10, -10, 10]])

    Corner = np.array(Corner, dtype=float)

    dens = 10  	  # density of model gm/cm3
    fht = 0     	 # height of observation plane above origin
    # Add fht to depth of all corners

    Corner[:, 2] = Corner[:, 2] + fht

    # In each row of Face, the first no is the number of corners forming a
    # face, the following are row numbers of the Corner array with
    # coordinates of the corners which form the face , seen in ccw order
    # from outside the object. The faces may have any orientation and may
    # be given in any order, but all faces must be included.

    Nf = 6
    Face = np.array([[4, 1, 2, 3, 4],
                     [4, 5, 8, 7, 6],
                     [4, 2, 6, 7, 3],
                     [4, 5, 1, 4, 8],
                     [4, 4, 3, 7, 8],
                     [4, 5, 6, 2, 1]])

    s_end = -50
    stn_spcng = 5
    n_end = 50
    w_end = -20
    prof_spcng = 10
    e_end = 20

    Face = np.array(Face)
    Face[:, 1:] -= 1  # convert indices to python form.

#    Nedges = sum(Face(1:Nf,1))
    Nedges = 4*Nf
    Edge = np.zeros([Nedges, 8])
    # get edge lengths
    for f in range(Nf):
        indx = Face[f, 1:Face[f, 0]+1].tolist() + [Face[f, 1]]
        for t in range(Face[f, 0]):
            # edgeno = sum(Face(1:f-1,1))+t
            edgeno = f*4+t
            ends = indx[t:t+2]
            p1 = Corner[ends[0]]
            p2 = Corner[ends[1]]
            V = p2-p1
            L = norm(V)
            Edge[edgeno, 0:3] = V
            Edge[edgeno, 3] = L
            Edge[edgeno, 6:8] = ends

    Un = np.zeros([Nf, 3])
    # get face normals
    for t in range(Nf):
        ss = np.zeros([1, 3])
        for t1 in range(1, Face[t, 0]-1):
            v1 = Corner[Face[t, t1+2], :]-Corner[Face[t, 1], :]
            v2 = Corner[Face[t, t1+1], :]-Corner[Face[t, 1], :]
            ss = ss+np.cross(v2, v1)
        Un[t, :] = ss/norm(ss)

    Un = Un.tolist()
    # Define the survey grid
    X, Y = np.meshgrid(range(s_end, n_end+stn_spcng, stn_spcng),
                       range(w_end, e_end+prof_spcng, prof_spcng))
    X = X.astype(float)
    Y = Y.astype(float)
    npro, nstn = X.shape
    # Initialise

    Gc = 6.6732e-3            # Universal gravitational constant
    Gx = np.zeros(X.shape)
    Gy = Gx.copy()
    Gz = Gx.copy()

    Hin = Hincl*np.pi/180
    Dec = Decl * np.pi/180
    cx = np.cos(Hin)*np.cos(Dec)
    cy = np.cos(Hin)*np.sin(Dec)
    cz = np.sin(Hin)
    Uh = np.array([cx, cy, cz])
    H = Hintn*Uh               # The ambient magnetic field
    Ind_magn = Susc*H/(4*np.pi)   # Induced magnetization
    Min = Mincl*np.pi/180
    Mdec = Mdecl*np.pi/180
    mcx = np.cos(Min)*np.cos(Mdec)
    mcy = np.cos(Min)*np.sin(Mdec)
    mcz = np.sin(Min)
    Um = np.array([mcx, mcy, mcz])
    Rem_magn = Mstrength*Um     # Remanent magnetization
    Net_magn = Rem_magn+Ind_magn  # Net magnetization
    Pd = np.transpose(np.dot(Un, Net_magn.T))   # Pole densities
    Hx = np.zeros(X.shape)
    Hy = Hx.copy()
    Hz = Hx.copy()

    # For each observation point do the following.
    # For each face find the solid angle.
    # For each side find p,q,r and add p,q,r of sides to get P,Q,R for the
    # face.
    # If calmag find hx,hy,hz.
    # If calgrv find gx,gy,gz.
    # Add the components from all the faces to get Hx,Hy,Hz and Gx,Gy,Gz.

#    for pr in range(npro):
#        for st in range(nstn):
#            opt = np.array([X[pr, st], Y[pr, st], 0])
# ##            cor = Corner-opt
#            for t in range(Ncor):
#                cor[t] = Corner[t]-opt          # shift origin
#            Edge[:, 4:6] = 0  # clear record of integration
#            for f in range(Nf):
#                nsides = Face[f, 0]
#                cors = Face[f, 1:nsides+1]
#                indx = list(range(nsides))+[0, 1]
#                crs = cor[cors]
#                fsign = np.sign(np.dot(Un[f], crs[0]))
#                # face is seen from the inside?
#    #       find solid angle subtended by face f at opt
#                dp1 = np.dot(crs[indx[0]], Un[f])
#     # perp distance of origin from plane of face
#                if abs(dp1) <= flimit:
#                    Omega = 0.0
#                else:
#                    W = 0
#                    for t in range(nsides):
#                        p1 = crs[indx[t]]
#                        p2 = crs[indx[t+1]]
#                        p3 = crs[indx[t+2]]
#                        W = W + angle(p1, p2, p3, Un, f)
#                    W = W-(nsides-2)*np.pi
#                    Omega = -fsign*W
#    #       Integrate over each side if not done and save result
#                PQR = np.array([0, 0, 0])
#                for t in range(nsides):
#                    p1 = crs[indx[t]]
#                    p2 = crs[indx[t+1]]
#                    Eno = np.sum(Face[0:f, 0])+t   # Edge no
#                    V = Edge[Eno, 0:3]
#                    if Edge[Eno, 5] == 1:       # already done
#                        I = Edge[Eno, 4]
#                    if Edge[Eno, 5] != 1:       # not already done
#                        if np.dot(p1, p2)/(norm(p1)*norm(p2)) == 1:
#    # origin, p1 and  p2 are on a straight line
#                            if norm(p1) > norm(p2):  # and p1 further than p2
#                                psave = p1
#                                p1 = p2
#                                p2 = psave
#                        L = Edge[Eno, 3]
#                        r1 = norm(p1)
#                        r2 = norm(p2)
#                        I = (1/L)*np.log((r1+r2+L)/(r1+r2-L))  # mod. formula
#    #              Save edge integrals and mark as done
#                        s = np.nonzero(
#                            np.logical_and(Edge[:, 6] == Edge[Eno, 7],
#                                           Edge[:, 7] == Edge[Eno, 6]))
#                        Edge[Eno, 4] = I
#                        Edge[s, 4] = I
#                        Edge[Eno, 5] = 1
#                        Edge[s, 5] = 1
#                    pqr = I*V
#                    PQR = PQR+pqr
#    #        From Omega, l, m, n PQR get components of field due to face f
#                l = Un[f, 0]
#                m = Un[f, 1]
#                n = Un[f, 2]
#                p = PQR[0]
#                q = PQR[1]
#                r = PQR[2]
#                gmtf1 = (l*Omega+n*q-m*r)
#                gmtf2 = (m*Omega+l*r-n*p)
#                gmtf3 = (n*Omega+m*p-l*q)
#
#                if calmag == 1:
#                    hx = Pd[f]*gmtf1
#                    Hx[pr, st] = Hx[pr, st]+hx
#                    hy = Pd[f]*gmtf2
#                    Hy[pr, st] = Hy[pr, st]+hy
#                    hz = Pd[f]*gmtf3
#                    Hz[pr, st] = Hz[pr, st]+hz
#
#                if calgrv == 1:
#                    gx = -dens*Gc*dp1 * gmtf1
#                    Gx[pr, st] = Gx[pr, st]+gx
#                    gy = -dens*Gc*dp1 * gmtf2
#                    Gy[pr, st] = Gy[pr, st]+gy
#                    gz = -dens*Gc*dp1 * gmtf3
#                    Gz[pr, st] = Gz[pr, st]+gz

    Face = Face[:, 1:].tolist()
#    Corner = Corner.tolist()
    gval = []
    mval = []
    for depth in range(0, 1000, 100):
        cor = (Corner + [0., 0., depth]).tolist()
        grvmagc.gm3d(npro, nstn, X, Y, Edge, cor,
                     Face, Gx, Gy, Gz, Hx, Hy, Hz, Pd, Un)
        gval.append(Gz)
        mval.append(Hz)
        # ttt.since_last_call()

    # Plotting is set up for the test case with one profile
    #   [Hx' Hy' Hz']
    Gx *= dens*Gc
    Gy *= dens*Gc
    Gz *= dens*Gc
    Htot = np.sqrt((Hx+H[0])**2 + (Hy+H[1])**2 + (Hz+H[2])**2)
    Dt = Htot-Hintn              	        # Change in total field
    Dta = Hx*cx + Hy*cy + Hz*cz    # Approximate change in total field


# def angle(p1, p2, p3, Un, f):
#    """Angle.m finds the angle between planes O-p1-p2 and O-p2-p3 where p1
#    p2 p3 are coordinates of three points taken in ccw order as seen from the
#    origin O.
#    This is used by gm3d for finding the solid angle subtended by a
#    polygon at the origin. Un is the unit outward normal vector to the
#    polygon.
#    """
#
#    anout = np.dot(p1, Un[f])
#
#    if abs(anout) <= flimit:
#        return 0
#
#    if anout > flimit:    # face seen from inside, interchange p1 and p3
#        p1, p3 = p3, p1
#
#    n1 = np.cross(p2, p1)
#    n2 = np.cross(p2, p3)
#
#    pn1 = norm(n1)
#    pn2 = norm(n2)
#    if (pn1 <= flimit) or (pn2 <= flimit):
#        ang = np.nan
#    else:
#        n1 = n1/pn1
#        n2 = n2/pn2
#        r = np.dot(n1, n2)
#        ang = np.arccos(r)
#        perp = np.sign(np.dot(p3, n1))
#        if perp < (-flimit):        # points p1,p2,p3 in cw order
#            ang = (2*np.pi-ang)
#
#    return ang


def main():
    """ Main """
    grvmag()

if __name__ == '__main__':
    main()
