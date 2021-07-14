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
feature[900] = [776, 1050, 850, 910]
feature[1300] = [1260, 1420]
feature[1480] = [1440, 1520]
feature[1760] = [1730, 1790]
feature[1800] = [1740, 1820]
feature[2080] = [2000, 2150]
feature[2200] = [2120, 2245]
feature[2290] = [2270, 2330]
feature[2320] = [2295, 2345]
feature[2330] = [2220, 2337]
feature[2390] = [2375, 2435]

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

product = {}
product['filter'] = ['NDVI < .25', 'dryveg < 1.015', 'albedo > 1000']
product['mica'] = [2200, 'r2350De > 1.02', 'r2160D2190 < 1.005']
product['smectite'] = [2200, 'r2350De < 1.02', 'r2160D2190 < 1.005']
product['kaolin'] = [2200, 'r2160D2190 > 1.005']
product['chlorite, epidote'] = ['r2250D', 'r2330D > 1.06']
product['amphibole, talc'] = ['r2380D', 'r2330D > 1.01', 'r2160D2190 < 1.005']
#product['amphibole2390, talc'] = [2390]
product['ferrous iron'] = ['r1100D']
product['ferric iron'] = [900]
product['gypsum'] = ['r1750D > 1.0'] #, 'r1950D > 1.001']
product['alunite'] = [1480, 'r1750D > 1.1']
product['carbonate'] = ['r2330D']
product['carbonate_w'] = [2330, 'r2330D > 1.1']
