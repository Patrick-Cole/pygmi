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
""" setup script for packaging PyGMI """

from distutils.core import setup

PVER = '2.2.1'

setup(name='pygmi',
      version=PVER,
      description='Python Geophysical Modelling and Interpretation',
      author='Patrick Cole',
      author_email='pcole@geoscience.org.za',
      url='https://github.com/Patrick-Cole/pygmi',
      download_url='https://github.com/Patrick-Cole/pygmi/archive/v' + PVER +
      '.tar.gz',
      license='GNU General Public License v3 (GPLv3)',
      platforms=['Windows', 'Linux'],
      long_description=open('README.md').read(),
      packages=['pygmi', 'pygmi.raster', 'pygmi.clust', 'pygmi.pfmod',
                'pygmi.test', 'pygmi.vector'],
      package_data={'pygmi': ['pfmod/*.pyd', 'pfmod/*.pyx', 'raster/*.cof',
                              'images/*.png', 'images/*.emf', 'images/*.ico']},
      requires={"numpy": [">=1.8.1"],
                "scipy": [">=0.13.3"],
                "matplotlib": [">=1.3.1"],
                "PyQt4": [">=4.9.6"],
                "GDAL": [">=1.11.1"],
                "numexpr": [">=2.3.1"]},
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Win32 (MS Windows)',
          'Environment :: X11 Applications :: Qt',
          'Intended Audience :: Education',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: Microsoft :: Windows :: Windows 7',
          'Operating System :: Microsoft :: Windows :: Windows Vista',
          'Operating System :: Microsoft :: Windows :: Windows XP',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ])
