# -----------------------------------------------------------------------------
# Name:        setup.py (part of PyGMI)
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
"""Setup script for packaging PyGMI."""

from setuptools import setup, find_packages
from pygmi import __version__ as PVER

BASE_URL = 'https://github.com/Patrick-Cole/pygmi'
HOMEPAGE = 'http://patrick-cole.github.io/pygmi/'

setup(name='pygmi',
      version=PVER,

      description='Python Geoscience Modelling and Interpretation',
      long_description=open('README.rst').read(),

      url=HOMEPAGE,
      download_url=BASE_URL+'/archive/pygmi-' + PVER + '.tar.gz',

      author='Patrick Cole',
      author_email='pcole@geoscience.org.za',

      license='GNU General Public License v3 (GPLv3)',

      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Win32 (MS Windows)',
          'Environment :: X11 Applications :: Qt',
          'Intended Audience :: Education',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],

      keywords='Geophysics Magnetic Gravity Modelling Interpretation',

      packages=(find_packages(exclude=['docs'])),

      install_requires=['fiona',
                        'gdal',
                        'geopandas',
                        'llvmlite',
                        'matplotlib',
                        'mtpy',
                        'numba',
                        'numexpr',
                        'numpy',
                        'pandas',
                        'pillow',
                        'pymatsolver',
                        'pyopengl',
                        'PyQt5',
                        'pytest',
                        'scikit-image',
                        'scikit-learn',
                        'scipy',
                        'segyio',
                        'shapely',
                        'SimPEG',
                        'setuptools'],

      package_data={'pygmi': ['raster/*.cof', 'helpdocs/*.html',
                              'helpdocs/*.png', 'images/*.png',
                              'images/*.emf', 'images/*.ico']},

      entry_points={'gui_scripts': ['pygmi = pygmi:main']},  # test this, might need to be pygmi.main:main

      zip_safe=False)
