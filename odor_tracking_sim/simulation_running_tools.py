

import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import cPickle as pickle
import sys


import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.odor_models as odor_models
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.borrowed_puff_models as puff_models

def setup_wind_field(wind_angle,wind_data_file,dt,release_delay,wind_dt=None,wind_speed=0.5):
    if not(wind_data_file==None):
        wind_dct = utility.process_wind_data(wind_data_file,release_delay,wind_dt=wind_dt)
        wind_param = {
        'speed':wind_dct['wind_speed'],
        'angle':wind_dct['wind_angle'],
        'evolving': True,
        'wind_dt': wind_dct['wind_dt'],
        'dt': dt
        }
    else:
        wind_angle=wind_angle*scipy.pi/180.0
        wind_param = {
            'speed': wind_speed,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
    wind_field = wind_models.WindField(param=wind_param)
    return wind_field

def setup_traps(number_sources = 6,radius_sources = 1000.0,strength_sources = 10.0,
trap_radius = 5.):
    if number_sources>1:
        #Standard geometry: 6 traps around the center
        location_list, strength_list = utility.create_circle_of_sources(
                number_sources,
                radius_sources,
                strength_sources
                )
    else:
        #Toy example with just one trap
        strength_list = [strength_sources]
        location_list = [(radius_sources*scipy.cos(scipy.pi/3),radius_sources*scipy.sin(scipy.pi/3))]

#   Set up the trap object (separated from odor object 3/8)

    trap_param = {
            'source_locations' : location_list,
            'source_strengths' : strength_list,
            'epsilon'          : 0.01,
            'trap_radius'      : trap_radius,
            'source_radius'    : radius_sources
    }

    traps = trap_models.TrapModel(trap_param)
    return traps

def setup_odor_field(wind_field,traps,plot_scale,puff_mol_amount=None,
    puffs=False,horizontal_diffusion=1.5):
    if traps.num_traps>1:
        #Standard geometry: 6 traps around the center
        plot_size = plot_scale*traps.param['source_radius']
        xlim = (-plot_size, plot_size)
        ylim = (-plot_size, plot_size)
    else:
        #Toy example with just one trap
        xlim = (-0.25*traps.param['source_radius'],0.25*traps.param['source_radius'])
        ylim = (-0.1*traps.param['source_radius'],0.75*traps.param['source_radius'])

    if not(puffs):
        '''This is Will's odor implementation'''
        odor_param = {
                'wind_field'       : wind_field,
                'diffusion_coeff'  :  0.25,
                'source_locations' : traps.param['source_locations'],
                'source_strengths' : traps.param['source_strengths'],
                'epsilon'          : 0.01,
                'trap_radius'      : traps.param['trap_radius']
                }
        odor_field = odor_models.FakeDiffusionOdorField(traps,param=odor_param)
        odor_plot_param = {
                'xlim' : xlim,
                'ylim' : ylim,
                'xnum' : 500,
                'ynum' : 500,
                'cmap' : 'binary',
                'fignums' : (1,2)
                }
        plumes=None
    else:
        '''This is the odor puff implementation borrowed/adapted from here: https://github.com/InsectRobotics/pompy'''
        '''Get the odor puff stuff ready'''
        sim_region = puff_models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
        #source_pos = traps.param['source_locations'][4]
        source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']]).T
        '''*****************'''
        plumes = puff_models.PlumeModel(sim_region, source_pos, wind_field,
                                        puff_release_rate=1.,model_z_disp=False,
                                        centre_rel_diff_scale=horizontal_diffusion,
                                        puff_init_rad=1.,puff_spread_rate=0.05)
    #Concentration generator object
        grid_size = 1000
        odor_field = puff_models.ConcentrationArrayGenerator(sim_region, 0.1, grid_size,
                                                           grid_size, puff_mol_amount,kernel_rad_mult=5)
    #Pompy version---------------------------
        odor_plot_param = {
            'xlim' : xlim,
            'ylim' : ylim,
            'cmap' : plt.cm.YlGnBu}
    return odor_plot_param,odor_field,plumes

def setup_swarm(swarm_size,wind_field,beta,kappa,start_type, upper_prob,release_delay=0.,
    heading_data = None, wind_slippage = (0.,0.),upper_threshold=0.02,schmitt_trigger=True,
    heading_mean=None,track_plume_bouts=False):
    if wind_field.evolving:
        wind_angle_0 = wind_field.angle[0]
    else:
        wind_angle_0 = wind_field.angle
    release_times = scipy.random.exponential(beta,(swarm_size,)) + release_delay*60.
    if kappa==0:
        dist = scipy.stats.uniform(0.,2*scipy.pi)
    else:
        if heading_mean is None:
            heading_mean = wind_angle_0
        dist= scipy.stats.vonmises(loc=heading_mean,kappa=kappa)
    swarm_param = {
        #    'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
            'swarm_size'          : swarm_size,
            'heading_data'        : heading_data,
            'initial_heading_dist': dist,
            'initial_heading'     : scipy.random.vonmises(heading_mean,kappa,(swarm_size,)),
            'x_start_position'    : scipy.zeros((swarm_size,)),
            'y_start_position'    : scipy.zeros((swarm_size,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((swarm_size,), 0.7),
            #'flight_speed'        : scipy.random.uniform(0.3,1.0,(swarm_size,)),
            #'release_time'        : scipy.full((swarm_size,), 0.0),
            'release_time'        : release_times,
            'release_time_constant': beta,
            'release_delay'       : release_delay*60,
            'cast_interval'       : [1.0, 10.0],
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': upper_threshold
                },
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': upper_prob,  # detection probability/sec of exposure
                },
            'schmitt_trigger':schmitt_trigger
            }
    # print(swarm_param['initial_heading_dist'])
    # print(swarm_param['initial_heading_dist'].mean())
    swarm = swarm_models.BasicSwarmOfFlies(wind_field,param=swarm_param,
    start_type=start_type,track_plume_bouts=track_plume_bouts)
    # time.sleep(10)
    return swarm

def initial_plot(odor_field,plot_param,flies,release_delay,swarm=None,fignum=1,plumes=None):
    #Initial odor plots
    plot_dict = {}
    if isinstance(odor_field,odor_models.FakeDiffusionOdorField):
        #Will's version
        image = odor_field.plot(0.,plot_param=plot_param)
    else:
        #Pompy version
        conc_array = (odor_field.generate_single_array(plumes.puff_array).T[::-1])
        image=odor_field.plot(conc_array,plot_param)
    plot_dict.update({'image':image})

    #Sub-dictionary for color codes for the fly modes
    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3

    color_dict = {Mode_StartMode : 'blue',
    Mode_FlyUpWind : 'red',
    Mode_CastForOdor : 'orange',
    Mode_Trapped :   'black'}

    plot_dict.update({'color_dict':color_dict})

    #Initial fly plots
    plt.ion()
    print(fignum)
    fig = plt.figure(fignum)
    plot_dict.update({'fig':fig})
    ax = plt.subplot(111)


    #Put the time in the corner
    (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
    text = str(int(release_delay))+' min 0 sec'
    timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')
    plot_dict.update({'timer':timer})

    plt.figure(fignum)
    plot_dict.update({'fignum':fignum})
    if flies:
        fly_colors = [color_dict[mode] for mode in swarm.mode]
        fly_dots = plt.scatter(swarm.x_position, swarm.y_position,color=fly_colors,alpha=0.5)
        plot_dict.update({'fly_dots':fly_dots})

    return plot_dict
