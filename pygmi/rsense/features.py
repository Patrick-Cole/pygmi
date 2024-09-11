# -----------------------------------------------------------------------------
# Name:        features.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2021 Council for Geoscience
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
"""List of features for hyperspectral interpretation."""

feature = {}
feature['f900'] = [776, 1050, 850, 910]
feature['f1300'] = [1260, 1420]
feature['f1480'] = [1440, 1520]
feature['f1550'] = [1510, 1610]
feature['f1760'] = [1730, 1790]
feature['f1800'] = [1740, 1820]
feature['f2080'] = [2060, 2100]
feature['f2160'] = [2138, 2179]
feature['f2200p'] = [2180, 2245]
feature['f2200'] = [2120, 2245]
feature['f2250'] = [2230, 2280]
feature['f2290'] = [2270, 2330]
feature['f2320'] = [2295, 2345]
feature['f2330'] = [2300, 2349]  # [2265, 2349]
feature['f2350'] = [2310, 2370]
feature['f2390'] = [2375, 2435]

ratio = {}
ratio['NDVI'] = '(R860-R687)/(R860+R687)'
ratio['dryveg'] = '(R2006+R2153)/(R2081+R2100)'
ratio['albedo'] = 'R1650'
ratio['rkgrpcrys'] = '((R2138+R2173)/R2156)/((R2156+R2190)/R2173)'

ratio['r1100D'] = '(R921+R1650)/(R1020+R1231)'   # Ferrous iron
ratio['r1550D'] = '(R1518+R1572)/(R1532+R1545)'  # Epidote
ratio['r1750D'] = '(R1720+R1790)/(R1740+R1760)'  # Gypsum
ratio['r1950D'] = '(R1900+R1980)/(R1930+R1960)'   # Water feature
ratio['r2160D2190'] = '(R2136+R2188)/(R2153+R2171)'  # Kaolin from non kaolin
ratio['r2200D'] = '(R2120+R2245)/(R2175+R2220)'
ratio['r2250D'] = '(R2227+R2275)/(R2241+R2259)'  # Chlorite epidote biotite
ratio['r2320D'] = '(R2295+R2345)/(R2305+R2322)'
ratio['r2330D'] = '(R2300+R2349)/(R2316+R2333)'  # (R2265+R2349)/(R2316+R2333)'  # MgOH and CO3
ratio['r2350De'] = '(R2326+R2376)/(R2343+R2359)'
ratio['r2380D'] = '(R2365+R2415)/(R2381+R2390)'  # Amphibole, talc

product = {}
product['filter'] = ['NDVI < .25', 'dryveg < 1.015', 'albedo > 1000']
product['pyrophyllite'] = ['f2160', 'f2160 > f2200p',
                           'f2080 > 0.01', 'f2200 > 0.01']
product['white mica'] = ['f2200', 'r2350De > 1.02', 'r2160D2190 < 1.005',
                         'r2200D > 1.01']
# product['white micanew'] = ['f2200', 'f2350 > 0.1', 'f2160 < f2200',
#                             'f2200 > .01']
product['smectite'] = ['f2200', 'r2350De < 1.0', 'r2160D2190 < 1.005',
                       'r2200D > 1.01']
product['kaolin-2200'] = ['f2200', 'r2160D2190 > 1.005']
product['kaolin-2200new'] = ['f2200', 'f2160 < f2200p',
                             'f2200 > 0.01', 'r2160D2190 > 1.005']
# product['kaolin-2160'] = ['f2160', 'r2160D2190 > 1.005']
product['chlorite-2250'] = ['f2250', 'r2330D > 1.0', 'r2250D > 1.005',
                            'r1550D < 1.01']
product['chlorite-2330'] = ['f2330', 'r2330D > 1.0', 'r2250D > 1.005',
                            'r1550D < 1.01']
product['epidote'] = ['f1550', 'r1550D > 1.01', 'r2330D > 1.0', 'r2250D > 1.0']
product['amphibole'] = ['f2390', 'r2380D > 1.02', 'r2320D > 1.01',
                        'r2160D2190 < 1.005', 'r2200D < 1.1', 'r2250D < 1.01']
product['ferrous iron'] = ['r1100D']
product['kaolin group crystallinity'] = ['rkgrpcrys']
product['ferric iron'] = ['f900']
product['carbonate'] = ['f2320', 'r2250D < 1.009', 'r2380D < 1.0',
                        'r2200D < 1.0']

# product['amphibole1, talc'] = ['r2380D', 'r2330D > 1.01', 'r2160D2190 < 1.005']
# product['amphibole'] = ['r2380D', 'r2330D > 1.01', 'r2160D2190 < 1.005',
#                         'r2200D < 1.01']
# product['amphibole'] = ['r2380D', 'r2320D > 1.01', 'r2160D2190 < 1.005',
#                         'r2200D < 1.01', 'r2250D < 1.01']
# product['pyrophyllite'] = ['f2160', 'r2160D2190 < 1.005']
# product['chlorite, epidote'] = ['r2250D', 'r2330D > 1.06']

# product['gypsum'] = ['r1750D', 'r1750D > 1.0'] #, 'r1950D > 1.001']
# product['gypsum'] = ['f1760']
# product['alunite'] = ['f1480', 'r1750D > 1.1']
# product['carbonate'] = ['r2330D']
# product['carbonate_w'] = ['f2330', 'r2330D > 1.1']


if __name__ == "__main__":
    from pygmi.rsense.hyperspec import _testfn
    _testfn()
