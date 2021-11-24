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
"""
List of features for hyperspectral interpretation.
"""

feature = {}
feature['f900'] = [776, 1050, 850, 910]
feature['f1300'] = [1260, 1420]
feature['f1480'] = [1440, 1520]
feature['f1550'] = [1510, 1610]
feature['f1760'] = [1730, 1790]
feature['f1800'] = [1740, 1820]
feature['f2080'] = [2000, 2150]
feature['f2160'] = [2138, 2188]
feature['f2200'] = [2120, 2245]
feature['f2250'] = [2230, 2280]
feature['f2290'] = [2270, 2330]
feature['f2320'] = [2295, 2345]
feature['f2330'] = [2220, 2337]
feature['f2350'] = [2310, 2370]
feature['f2390'] = [2375, 2435]

ratio = {}
ratio['NDVI'] = '(R860-R687)/(R860+R687)'
ratio['dryveg'] = '(R2006+R2153)/(R2081+R2100)'
ratio['albedo'] = 'R1650'

ratio['r2350De'] = '(R2326+R2376)/(R2343+R2359)'
ratio['r2160D2190'] = '(R2136+R2188)/(R2153+R2171)'  # Kaolin from non kaolin
ratio['r2250D'] = '(R2227+R2275)/(R2241+R2259)'  # Chlorite epidote biotite
ratio['r2380D'] = '(R2365+R2415)/(R2381+R2390)'  # Amphibole, talc
ratio['r2330D'] = '(R2265+R2349)/(R2316+R2333)'  # MgOH and CO3
ratio['r1100D'] = '(R921+R1650)/(R1020+R1231)'   # Ferrous iron
ratio['r1750D'] = '(R1720+R1790)/(R1740+R1760)'  # Gypsum
ratio['r1950D'] = '(R1900+R1980)/(R1930+R1960)'   # Water feature
ratio['r1550D'] = '(R1518+R1572)/(R1532+R1545)'  # Epidote
ratio['r2200D'] = '(R2120+R2245)/(R2175+R2220)'

product = {}
product['filter'] = ['NDVI < .25', 'dryveg < 1.015', 'albedo > 1000']
product['white mica'] = ['f2200', 'r2350De > 1.02', 'r2160D2190 < 1.005', 'r2200D > 1.01']
product['smectite'] = ['f2200', 'r2350De < 1.0', 'r2160D2190 < 1.005', 'r2200D > 1.01']
product['kaolin-2200'] = ['f2200', 'r2160D2190 > 1.005']
product['kaolin-2160'] = ['f2160', 'r2160D2190 > 1.005']
# product['pyrophyllite'] = ['f2160', 'r2160D2190 < 1.005']
#product['chlorite, epidote'] = ['r2250D', 'r2330D > 1.06']
product['chlorite'] = ['f2250', 'r2330D > 1.0', 'r2250D > 1.005']
product['epidote'] = ['f1550', 'r1550D > 1.01', 'r2330D > 1.0', 'r2250D > 1.0']
# product['amphibole1, talc'] = ['r2380D', 'r2330D > 1.01', 'r2160D2190 < 1.005']
# product['amphibole2, talc'] = ['f2390', 'r2380D > 1.02','r2330D > 1.01', 'r2160D2190 < 1.005', 'r2250D < 1.0']
product['amphibole'] = ['r2380D', 'r2330D > 1.01', 'r2160D2190 < 1.005', 'r2200D < 1.01']

product['ferrous iron'] = ['r1100D']
product['ferric iron'] = ['f900']
#product['gypsum'] = ['r1750D', 'r1750D > 1.0'] #, 'r1950D > 1.001']
#product['gypsum'] = ['f1760']
#product['alunite'] = ['f1480', 'r1750D > 1.1']
# product['carbonate'] = ['r2330D']
# product['carbonate_w'] = ['f2330', 'r2330D > 1.1']
product['carbonate'] = ['f2320', 'r2250D < 1.009', 'r2380D < 1.0','r2200D < 1.0']
