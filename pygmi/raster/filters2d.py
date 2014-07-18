# -----------------------------------------------------------------------------
# Name:        filters2D.py (part of PyGMI)
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
""" Filters 2D """

# pylint: disable=E1101, C0103
import numpy as np
import math
# These filter definitions have been translated from the matlab function
# 'fspecial'. All credit must go to Mathworks


def filters2d(ftype, tmp, *sigma):
    """ Filters 2D """
    if ftype == 'average':  # create moving average filter window
        hhh = np.ones(tmp)/np.prod(tmp)
        return hhh

    if ftype == 'disc':  # create circular averaging filter window
        rad = tmp
        crad = math.ceil(rad-0.5)
        [yyy, xxx] = np.mgrid[-crad:crad+1, -crad:crad+1]
        maxxy = np.maximum(abs(xxx), abs(yyy))
        minxy = np.minimum(abs(xxx), abs(yyy))

# 0j is needed to force a complex sqrt

        m_1 = ((rad**2 < (maxxy+0.5)**2+(minxy-0.5)**2)*(minxy-0.5) +
               (rad**2 >= (maxxy+0.5)**2+(minxy-0.5)**2) *
               np.sqrt(rad**2-(maxxy+0.5)**2+0j)).real

        m_2 = ((rad**2 > (maxxy-0.5)**2+(minxy+0.5)**2)*(minxy+0.5) +
               (rad**2 <= (maxxy-0.5)**2+(minxy+0.5)**2) *
               np.sqrt(rad**2-(maxxy-0.5)**2+0j)).real

        sgrid = ((rad**2 * (0.5*(np.arcsin(m_2/rad)-np.arcsin(m_1/rad)) +
                            0.25*(np.sin(2*np.arcsin(m_2/rad)) -
                                  np.sin(2*np.arcsin(m_1/rad)))) -
                 (maxxy-0.5) * (m_2-m_1) +
                 (m_1-minxy+0.5)) *
                 ((((rad**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) &
                    (rad**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) |
                   ((minxy == 0) & (maxxy-0.5 < rad) & (maxxy+0.5 >= rad)))))

        sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < rad**2)

        sgrid[crad, crad] = np.minimum(np.pi*rad**2, np.pi/2)

        if ((crad > 0) and (rad > crad-0.5) and
                (rad**2 < (crad-0.5)**2 + 0.25)):
            m_1 = np.sqrt(rad**2-(crad-0.5)**2)
            m_1n = m_1/rad
            sg0 = 2.0*(rad**2*(0.5*np.arcsin(m_1n) +
                       0.25*np.sin(2*np.arcsin(m_1n)))-m_1*(crad-0.5))
            sgrid[2*crad, crad] = sg0
            sgrid[crad, 2*crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0
            sgrid[2*crad-1, crad] = sgrid[2*crad-1, crad] - sg0
            sgrid[crad, 2*crad-1] = sgrid[crad, 2*crad-1] - sg0
            sgrid[crad, 1] = sgrid[crad, 1] - sg0
            sgrid[1, crad] = sgrid[1, crad] - sg0

        sgrid[crad, crad] = np.minimum(sgrid[crad, crad], 1)
        hhh = sgrid/sgrid.sum()

        return hhh

    if ftype == 'gaussian':  # create a Gaussian low pass filter window
        siz = [(tmp[0]-1)/2, (tmp[1]-1)/2]
        std = sigma[0]
        [yyy, xxx] = np.mgrid[-siz[1]:siz[1]+1, -siz[0]:siz[0]+1]
        arg = -(xxx*xxx+yyy*yyy)/(2*std*std)

        hhh = np.exp(arg)
        hhh[hhh < np.finfo(float).eps*hhh.max()] = 0

        sumh = hhh.sum()
        if sumh != 0:
            hhh = hhh/sumh

        return hhh
