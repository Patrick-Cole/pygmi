# -----------------------------------------------------------------------------
# Name:        datatypes.py (part of PyGMI)
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
""" Class for data types """

from ..raster.datatypes import Data


class Clust(Data):
    """ Class Clust """
    def __init__(self):
        Data.__init__(self)

        self.input_type = []
        self.no_clusters = 0
        self.center = []
        self.center_std = []
        self.memdat = []
        self.vrc = None
        self.nce = None
        self.xbi = None
        self.obj_fcn = None

#        self.proc_history = []
#        self.type = "cluster"
#        self.algorithm = 0
#        self.initialization = 0
#        self.init_mod = 0
#        self.init_constrains = 0
#        self.runs = 0
#        self.max_iterations = 0
#        self.denormalize = 0
#        self.term_threshold = 0
#        self.shape_constrain = 0
#        self.zonal = 0
#        self.alpha = 0
#        self.xxx = np.array([])
#        self.yyy = np.array([])
#        self.obj_fcn = 0
#        self.denorm_center = 0
#        self.denorm_center_stdup = 0
#        self.denorm_center_stdlow = 0
#        self.iterations = 0
#        self.fuzziness_exp = 0
