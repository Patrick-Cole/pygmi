# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:18:46 2019

@author: pcole
"""

import os
import numpy as np
import glob
import osgeo
import matplotlib.pyplot as plt
from mtpy.core.mt import MT
from mtpy.core.z import Z, Tipper
from mtpy.utils.calculator import get_period_list
from mtpy.analysis.geometry import dimensionality, strike_angle
from mtpy.imaging.plotstrike import PlotStrike
from mtpy.imaging import penetration_depth1d as pd1d
from mtpy.imaging import penetration_depth2d as pen2d
from mtpy.imaging import penetration_depth3d as pen3d
import mtpy.modeling.occam1d as occam1d

import mtpy.processing.birrp as MTbp

def core_edi(datadir):
    """ read edi """
    # Define the path to your edi file

    savepath = datadir
    allfiles = glob.glob(datadir+'\s*.edi')
    for edi_file in allfiles:

        mt_obj = MT(edi_file)
        pt_obj = mt_obj.plot_mt_response(plot_num=1, plot_tipper='yri',
                                         plot_pt='y')

#    # To see the latitude and longitude
#    print(mt_obj.lat, mt_obj.lon)
#
#    # for example, to see the frequency values represented in the impedance
#    # tensor:
#    print(mt_obj.Z.freq)
#
#    # or to see the impedance tensor (first 4 elements)
#    print(mt_obj.Z.z[:4])
#
#    # or the resistivity or phase (first 4 values)
#    print(mt_obj.Z.resistivity[:4])
#    print(mt_obj.Z.phase[:4])

    # To plot the edi file & save to file:
    # plot_num 1 = yx and xy; 2 = all 4 components,
    #          3 = off diagonal + determinant
    # plot pt (phase tensor) 'y' or 'n'
#    pt_obj = mt_obj.plot_mt_response(plot_num=1, plot_tipper='yri',
#                                     plot_pt='y')


#    plt.loglog(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 0, 0], 'b.') # xx
#    plt.loglog(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 1, 1], 'r.') # xy
#    plt.tight_layout()
#    plt.show()
#
#
#    plt.errorbar(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 0, 1],
#                 yerr=mt_obj.Z.resistivity_err[:, 0, 1],
#                 ls=' ', marker='.', mfc='b', mec='b', ecolor='b', label=r'$\rho_{xy}$')
#    plt.errorbar(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 1, 0],
#                 yerr=mt_obj.Z.resistivity_err[:, 1, 0],
#                 ls=' ', marker='.', mfc='r', mec = 'r', ecolor='r', label=r'$\rho_{yx}$')
#    plt.legend()
##    plt.loglog(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 1, 0], 'r.') # yx
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.tight_layout()
#    plt.show()
#
#
#    plt.errorbar(1/mt_obj.Z.freq, mt_obj.Z.phase[:, 0, 1],
#                 yerr=mt_obj.Z.phase_err[:, 0, 1],
#                 ls=' ', marker='.', mfc='b', mec='b', ecolor='b', label=r'$\rho_{xy}$')
#    plt.errorbar(1/mt_obj.Z.freq, mt_obj.Z.phase[:, 1, 0],
#                 yerr=mt_obj.Z.phase_err[:, 1, 0],
#                 ls=' ', marker='.', mfc='r', mec = 'r', ecolor='r', label=r'$\rho_{yx}$')
#    plt.legend()
##    plt.loglog(1/mt_obj.Z.freq, mt_obj.Z.resistivity[:, 1, 0], 'r.') # yx
##    plt.yscale('log')
#
#    plt.xscale('log')
#    plt.tight_layout()
#    plt.show()
#
#
##    pt_obj.save_plot(os.path.join(savepath,"Synth00.png"), fig_dpi=400)
#
#    # First, define a frequency array:
#    # Every second frequency:
#    new_freq_list = mt_obj.Z.freq[::2]
#
#    # OR 5 periods per decade from 10^-4 to 10^3 seconds
#    # new_freq_list = 1./get_period_list(1e-4,1e3,5)
#
#    # Create new Z and Tipper objects containing interpolated data
#    new_Z_obj, new_Tipper_obj = mt_obj.interpolate(new_freq_list)
#
#    # Write a new edi file using the new data
#    # write longitudes as 'LONG' not ‘LON’
#    # 'dd' writes as decimal degrees (any other input will write as
#    # degrees:minutes:seconds
#    mt_obj.write_mt_file(save_dir=savepath,
#                         fn_basename='Synth00_5ppd',
#                         file_type='edi',
#                         new_Z_obj=new_Z_obj,
#                         new_Tipper_obj=new_Tipper_obj,
#                         longitude_format='LONG',
#                         latlon_format='dd')


def analysis_edi(datadir):
    """ analysis """
    # Define the path to your edi file
    edi_file = datadir+r"edifiles2\15125A.edi"
    savepath = datadir
    edi_path = datadir+'edifiles2'

    # Create an MT object
    mt_obj = MT(edi_file)

    # look at the skew values as a histogram
    plt.hist(mt_obj.pt.beta, bins=50)
    plt.xlabel('Skew angle (degree)')
    plt.ylabel('Number of values')

    plt.show()

    # Have a look at the dimensionality
    dim = dimensionality(z_object=mt_obj.Z, skew_threshold=5,
                         eccentricity_threshold=0.1)

    print(dim)

    # calculate strike
    strike = strike_angle(z_object=mt_obj.Z, skew_threshold=5,
                          eccentricity_threshold=0.1)

    # display the median strike angle for this station
    # two values because of 90 degree ambiguity in strike
    strikemedian = np.nanmedian(strike, axis=0)

    print(strikemedian)

    # Use dimensionality to mask a file

    mask = dim < 3
    # Apply masking. The new arrays z_array, z_err_array, and freq will
    # exclude values where mask is False (i.e. the 3D parts)
    new_Z_obj = Z(z_array=mt_obj.Z.z[mask],
                  z_err_array=mt_obj.Z.z_err[mask],
                  freq=mt_obj.Z.freq[mask])

    new_Tipper_obj = Tipper(tipper_array=mt_obj.Tipper.tipper[mask],
                            tipper_err_array=mt_obj.Tipper.tipper_err[mask],
                            freq=mt_obj.Tipper.freq[mask])

    # Write a new edi file
    mt_obj.write_mt_file(save_dir=savepath,
                         fn_basename='Synth00_mask3d',
                         file_type='edi',
                         new_Z_obj=new_Z_obj,
                         new_Tipper_obj=new_Tipper_obj,
                         longitude_format='LONG',
                         latlon_format='dd')

    # Plot strike
    # Get full path to all files with the extension '.edi' in edi_path
    edi_list = [os.path.join(edi_path, ff) for ff in os.listdir(edi_path)
                if ff.endswith('.edi')]

    # make a plot (try also plot_type = 1 to plot by decade)
    strikeplot = PlotStrike(fn_list=edi_list,
                            plot_type=2,
                            plot_tipper='y')
    # save to file
    # strikeplot.save_plot(savepath,
    #                      file_format='.png',
    #                      fig_dpi=400)

    strike = strikemedian[0] # 0 index chosen based on geological information
    mt_obj.Z.rotate(strike)
    mt_obj.Tipper.rotate(strike)

    # check the rotation angle
    print(mt_obj.Z.rotation_angle)
    # Write a new edi file (as before)
    mt_obj.write_mt_file(save_dir=savepath,
                         fn_basename='Synth00_rotate%1i' % strike,
                         file_type='edi',
                         longitude_format='LONG',
                         latlon_format='dd')


def image_edi(datadir):
    """ image """

    edi_file = datadir+r"edifiles2\15125A.edi"
    save_file = datadir+'penetration_depth1d.png'
    edi_path = datadir+'edifiles'
    savepath = datadir

    pd1d.plot_edi_file(edi_file, savefile=save_file, fig_dpi=400)

    # Choose indices of periods to plot
    period_index_list = [0, 1, 10, 20, 30, 40, 50, 59]
    # Plot profiles for different modes ('det','zxy' or 'zyx')
    pen2d.plot2Dprofile(edi_path, period_index_list,
                        'det', marker='o',
                        tick_params={'rotation': 'vertical'})

    # Create plot for period index number 10 for determinant
    pen3d.plot_latlon_depth_profile(edi_path, 10, 'det', showfig=True,
                                    savefig=True, savepath=savepath,
                                    fontsize=11, fig_dpi=400,
                                    file_format='png')


def modelling(datadir):
    """ modelling """
    edi_file = os.path.join(datadir, r"synth02.edi")
    save_path = os.path.join(datadir, 'TM')

    allfiles = glob.glob(save_path+'\\*.*')
    for i in allfiles:
        os.remove(i)
    try:
        os.remove(save_path+'\\Model1D')
        os.remove(save_path+'\\OccamStartup1D')
    except:
        pass

    d1 = occam1d.Data()
    d1.write_data_file(edi_file=edi_file,
                       mode='TE',
                       save_path=save_path,
                       res_err='data',
                       phase_err='data',
                       res_errorfloor=4.,
                       phase_errorfloor=2.,
                       remove_outofquadrant=True
                       )

    m1 = occam1d.Model(target_depth=40000,
                       n_layers=100,
                       bottom_layer=100000,
                       z1_layer=10
                       )
    m1.write_model_file(save_path=d1.save_path)

    #--> make a startup file
    s1 = occam1d.Startup(data_fn=d1.data_fn,
                         model_fn= m1.model_fn,
                         max_iter=200,
                         target_rms=1.0)

    s1.write_startup_file()

    #--> run occam1d from python
    occam_path = r"C:\Work\Programming\pygmi\pygmi\bin\occam1d.exe"
    occam1d.Run(s1.startup_fn, occam_path, mode='TM')

    #--plot the L2 curve
#    l2 = occam1d.PlotL2(d1.save_path, m1.model_fn,
#                        fig_dpi=100,
#                        fig_size=(2,2)
#                        )

    #--> see that iteration 7 is the optimum model to plot

    iterfn = os.path.join(save_path, 'TM_005.iter')
    respfn = os.path.join(save_path, 'TM_005.resp')

    p1 = occam1d.Plot1DResponse(data_te_fn = d1.data_fn,
                                model_fn = m1.model_fn,
                                iter_te_fn = iterfn,
                                resp_te_fn = respfn,
#                                depth_limits=(0,20),
                                fig_dpi=100,
                                fig_size=(2,2)
                                )


    oc1m = occam1d.Model(model_fn=m1.model_fn)

    allfiles = glob.glob(save_path+'\*.iter')

    roughness = []
    rms = []
    for iterfn in allfiles:
        oc1m.read_iter_file(iterfn)
        roughness.append(float(oc1m.itdict['Roughness Value']))
        rms.append(float(oc1m.itdict['Misfit Value']))

    roughness.pop(0)
    rms.pop(0)

    plt.plot(roughness, rms)
    plt.xlabel('Roughness')
    plt.ylabel('RMS')
    plt.show()
    plt.plot(rms)
    plt.xlabel('Iteration')
    plt.ylabel('RMS')
    plt.show()


    iterfn = os.path.join(save_path, 'TM_005.iter')
    respfn = os.path.join(save_path, 'TM_005.resp')

    oc1m.read_iter_file(iterfn)

    oc1d = occam1d.Data(data_fn=d1.data_fn)
    oc1d.read_resp_file(respfn)


    depths = []
    res = []

    for i, val in enumerate(oc1m.model_res[:, 1]):
        if i == 0:
            continue
        if i > 1:
            depths.append(-oc1m.model_depth[i-1])
            res.append(val)

        depths.append(-oc1m.model_depth[i])
        res.append(val)

    plt.plot(res, depths)

    plt.xlabel('Res')
    plt.ylabel('Depth')

    plt.show()

    plt.plot(1/oc1d.freq, oc1d.data['resxy'][0], 'bs')
    plt.plot(1/oc1d.freq, oc1d.data['resxy'][2], 'r')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    plt.plot(1/oc1d.freq, oc1d.data['phasexy'][0], 'bs')
    plt.plot(1/oc1d.freq, oc1d.data['phasexy'][2], 'r')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    breakpoint()

#    p1.plot()

def birrp():
    """ birrp """
    birrp_exe = r"C:\Work\Programming\pygmi\pygmi\bin\birrp.exe"
    savefn = r"C:\Work\Programming\pygmi\data\MT\biirp\test"

#    MTbp.runbirrp2in2out_simple(birrp_exe, stationname, ts_dir,
#                                coherence_th, None, None,
#                                starttime, endtime)

    fn_dtype = np.dtype([('fn', 'S100'),
                         ('nread', np.int),
                         ('nskip', np.int),
                         ('comp', 'S2'),
                         ('calibration_fn', 'S100'),
                         ('rr', np.bool),
                         ('rr_num', np.int),
                         ('start_dt', 'S19'),
                         ('end_dt', 'S19')])

    bs = MTbp.ScriptFile()


    tmp0 = np.zeros(1, dtype=bs._fn_dtype)
    tmp0['fn'] = r'C:\Work\Programming\pygmi\data\MT\mtd_files\BP02_1day_20130513_0_microvoltpermeter.ex'
    tmp0['nread'] = 1
    tmp0['nskip'] = 0
    tmp0['calibration_fn'] = r''
    tmp0['rr'] = False
    tmp0['rr_num'] = 0
    tmp0['start_dt'] = r''  # start date ?
    tmp0['end_dt'] = r''  # end date ?
    tmp0['comp'] = 'ex'

    tmp1 = np.zeros(1, dtype=bs._fn_dtype)
    tmp1['nread'] = 1
    tmp1['nskip'] = 0
    tmp1['calibration_fn'] = r''
    tmp1['rr'] = False
    tmp1['rr_num'] = 0
    tmp1['start_dt'] = r''  # start date ?
    tmp1['end_dt'] = r''  # end date ?
    tmp1['comp'] = 'ey'
    tmp1['fn'] = r'C:\Work\Programming\pygmi\data\MT\mtd_files\BP02_1day_20130513_0_microvoltpermeter.ey'

    bs.fn_arr = np.array([tmp0, tmp1])
    bs.script_fn = r'C:\Work\Programming\pygmi\data\MT\mtd_files\hope.txt'
    bs._validate_fn_arr()

    bs.ninp = 5

    bs.write_script_file()


    breakpoint()



def main():
    """ main """
    datadir = r'C:\Python\mtpy\mtpy-1.0\data\\'
    datadir = r'C:\Work\Programming\pygmi\data\MT'
#    core_edi(datadir)
#    analysis_edi(datadir)
#    image_edi(datadir)
#    modelling(datadir)
    birrp()


if __name__ == "__main__":
    main()

    print('Finished!')
