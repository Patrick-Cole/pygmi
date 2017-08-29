"""
Tests
"""
# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import pandas as pd

def main():
    """ main """

#    x	line	Resistivity	y	z

    aaa = np.loadtxt('2D_INV_Lines_Pygmi.csv', skiprows=1, delimiter=',')

    aaa = aaa.T

    line = aaa[0]
    x = aaa[0]
    y = aaa[3]
    z = aaa[4]
    res = aaa[1]

    fn = res.copy()

    fn[fn < 150] = 1
    fn[np.logical_and(fn > 150, fn <= 850)] = 2
    fn[fn > 850] = 3

    out = np.array([x.flatten(), y.flatten(), z.flatten(), fn.flatten()])
    out = out.T

    np.savetxt('hope3.csv', out, delimiter=',')

    dx = 1000.
    dx = 100.
    dy = 100.
    dz = 100.

    xrange = np.arange(x.min(), x.max()+dx, dx)
    yrange = np.arange(y.min(), y.max()+dy, dy)
    zrange = np.arange(z.min(), z.max()+dz, dz)

    xx, yy, zz = np.meshgrid(xrange, yrange, zrange)

    xi = np.transpose([xx.flatten(), yy.flatten(), zz.flatten()])
    fn = si.griddata(np.transpose([x, y, z]), res, xi, method='nearest')

#    fn.shape = xx.shape

    fn[fn < 150] = 1
    fn[np.logical_and(fn > 150, fn <= 850)] = 2
    fn[fn > 850] = 2

    out = np.array([xx.flatten(), yy.flatten(), zz.flatten(), fn])
    out = out.T

    np.savetxt('hope2.csv', out, delimiter=',')

#    hist = np.histogram(res,1000)
#    plt.loglog(np.diff(hist[1])/2+hist[1][:-1], hist[0])
#    plt.show()


#  x	line	Resistivity	y(utm)	x (utm)	z	Z1


def main2():
    """ main2 """
    ifile = '2D_INV_Lines_Pygmi.csv'

    pdin = pd.read_csv(ifile)
    pdin['Lithology'] = 'Metasedimentary Rocks'
    pdin.loc[pdin['Resistivity'] < 150, 'Lithology'] = 'Water Filled Metasediments'
    pdin.loc[pdin['Resistivity'] > 850, 'Lithology'] = 'Karoo Sediments'

# Example of more complex: (pdin['Resistivity']<150) & (pdin['Resistivity']>50)

    pdin.to_csv('hope1.csv', index=False, header=False,
                columns=['X(utm)', 'Y(utm)', 'Z1', 'Lithology'])


if __name__ == "__main__":
    main2()
