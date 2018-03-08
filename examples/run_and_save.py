'''This does what run_simulation.py does with specified output name and parameters.'''

import time
import scipy
import matplotlib.pyplot as plt
import matplotlib
import cPickle as pickle

import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.odor_models as odor_models
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.borrowed_puff_models as puff_models

def run_sim(file_name,wind_angle,release_time_constant,
kappa=0.,t_stop=15000.0,display_speed=1,
wind_slippage = (0.,0.),swarm_size=10000,start_type='fh',upper_prob=0.002,
heading_data=None,wind_data_file=None,dt=0.25,wind=True,flies=True):
    output_file = file_name+'.pkl'
    if not(wind_data_file==None):
        wind_angle,wind_speed,wind_dt = utility.process_wind_data(wind_data_file)
        wind_param = {
        'speed':wind_speed,
        'angle':wind_angle,
        'evolving': True,
        'wind_dt': wind_dt,
        'dt': dt
        }
    else:
        wind_angle=wind_angle*scipy.pi/180.0
        wind_param = {
            'speed': 0.5,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
    wind_field = wind_models.WindField(param=wind_param)

    # Create circular odor field, set source locations and strengths
    number_sources = 6
    radius_sources = 1000.0
    strength_sources = 10.0
    location_list, strength_list = utility.create_circle_of_sources(
            number_sources,
            radius_sources,
            strength_sources
            )
    trap_radius = 5.
    '''This is Will's odor implementation'''
#    odor_param = {
#            'wind_field'       : wind_field,
#            'diffusion_coeff'  :  0.25,
#            'source_locations' : location_list,
#            'source_strengths' : strength_list,
#            'epsilon'          : 0.01,
#            'trap_radius'      : trap_radius
#            }
#    odor_field = odor_models.FakeDiffusionOdorField(odor_param)

    plot_scale = 2.0
    plot_size = plot_scale*radius_sources


    '''This is the odor puff implementation borrowed/adapted from here: https://github.com/InsectRobotics/pompy'''
    '''Get the odor puff stuff ready'''
    sim_region = puff_models.Rectangle(-plot_size, -plot_size, plot_size, plot_size)
    source_pos = location_list[3]
    '''*****************'''
    plumes = puff_models.PlumeModel(sim_region, source_pos, wind_field,
                                    puff_release_rate=0.1,model_z_disp=False,
                                    centre_rel_diff_scale=1.5,puff_init_rad=1.,puff_spread_rate=5.)
    #Concentration generator object
    grid_size = 1000
    conc_array_gen = puff_models.ConcentrationArrayGenerator(sim_region, 0.1, grid_size,
                                                       grid_size, 10.,trap_radius=trap_radius,sources = [plumes.source_pos])
    #Concentration object for t=0
    conc_array = (
        conc_array_gen.generate_single_array(plumes.puff_array).T[::-1])

    '''Get swarm ready'''
    if flies:
        swarm_size = swarm_size
        beta = release_time_constant #time constant for release time distribution
        if wind_field.evolving:
            wind_angle_0 = wind_angle[0]
        else:
            wind_angle_0 = wind_angle
        swarm_param = {
            #    'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
                'swarm_size'          : swarm_size,
                'heading_data'        : heading_data,
                'initial_heading_dist': scipy.stats.vonmises(loc=wind_angle_0,kappa=kappa),
                'initial_heading'     : scipy.random.vonmises(wind_angle_0,kappa,(swarm_size,)),
                'x_start_position'    : scipy.zeros((swarm_size,)),
                'y_start_position'    : scipy.zeros((swarm_size,)),
                'heading_error_std'   : scipy.radians(10.0),
                'flight_speed'        : scipy.full((swarm_size,), 0.7),
                #'flight_speed'        : scipy.random.uniform(0.3,1.0,(swarm_size,)),
                #'release_time'        : scipy.full((swarm_size,), 0.0),
                'release_time'        : scipy.random.exponential(beta,(swarm_size,)),
                'release_time_constant': beta,
                'cast_interval'       : [60.0, 1000.0],
                'wind_slippage'       : wind_slippage,
                'odor_thresholds'     : {
                    'lower': 0.002,
                    'upper': 0.02
                    },
                'odor_probabilities'  : {
                    'lower': 0.9,    # detection probability/sec of exposure
                    'upper': upper_prob,  # detection probability/sec of exposure
                    }
                }
        swarm = swarm_models.BasicSwarmOfFlies(wind_field,param=swarm_param,start_type=start_type)
    # Setup live plot
    fignum = 1
    '''Plot the initial odor field'''
    #Will's version

#    plot_param = {
#            'xlim' : (-plot_size, plot_size),
#            'ylim' : (-plot_size, plot_size),
#            'xnum' : 500,
#            'ynum' : 500,
#            'cmap' : 'binary',
#            'fignums' : (1,2),
            #'threshold': 0.001,
#            }

    #odor_field.plot(0.,plot_param=plot_param)

    #Pompy version
    plot_param = {
                'xlim' : (-plot_size, plot_size),
                'ylim' : (-plot_size, plot_size),
                'cmap' : plt.cm.YlGnBu}
    image=conc_array_gen.plot(conc_array,plot_param)
    odor_field = conc_array_gen
    #end Pompy version stuff

    plt.ion()
    fig = plt.figure(fignum)
    ax = plt.subplot(111)

    plt.figure(fignum)
    if flies:
        fly_dots, = plt.plot(swarm.x_position, swarm.y_position,'.r')

    fig.canvas.flush_events()
    plt.pause(0.001)
    t = 0.0
    dt = dt
    dt_plot = 10.0*display_speed
    t_plot_last = 0.0

    while t<t_stop:
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        if flies:
            swarm.update(t,dt,wind_field,odor_field,plumes=plumes)
        #Update the plumes, then odor concentration field
        plumes.update(t, dt)
        plumes.any_yet()
        conc_array = (
            conc_array_gen.generate_single_array(plumes.puff_array).T[::-1])
        t+= dt
        #print(max(scipy.unique(conc_array)))
        #print(sum(sum(conc_array)))


        # Update live display
        if t_plot_last + dt_plot < t:

            plt.figure(fignum)
            if flies:
                fly_dots.set_xdata([swarm.x_position])
                fly_dots.set_ydata([swarm.y_position])

            trap_list = []
            for trap_num, trap_loc in enumerate(odor_field.param['source_locations']):
                mask_trap = swarm.trap_num == trap_num
                trap_cnt = mask_trap.sum()
                trap_list.append(trap_cnt)
            total_cnt = sum(trap_list)
            plt.title('{0}/{1}: {2}'.format(total_cnt,swarm.size,trap_list))

            if total_cnt > 0:
                frac_list = [float(x)/float(total_cnt) for x in trap_list]
            else:
                frac_list = [0 for x in trap_list]
            frac_list = ['{0:1.2f}'.format(x) for x in frac_list]
            #plt.title('{0}/{1}: {2} {3}'.format(total_cnt,swarm.size,trap_list,frac_list))

            #conc_array_gen.plot(conc_array,plot_param)
             #puff version odor field
        #    vmin = conc_array.min()
        #    vmax=conc_array.max()
        #    n = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        #    image.set_norm(n)
            conc_array=conc_array/conc_array.max()
            image.set_data(conc_array)
            #image.set_norm(plt.colors.Normalize())
#            odor_field.plot(t,plot_param=plot_param) #Will's version odor field
            fig.canvas.flush_events()
            #plt.cla()
            t_plot_last = t
            plt.figure(10);plt.clf()
            plt.hist(scipy.unique(conc_array))
            #plt.title(str((vmin,vmax)))
            #plt.pause(5)

            #time.sleep(0.05)

        #end = time.time()
    # Write swarm to file
    with open(output_file, 'w') as f:
        pickle.dump(swarm,f)

heading_data = {'angles':(scipy.pi/180)*scipy.array([0.,90.,180.,270.]),
                'counts':scipy.array([[1724,514,1905,4666],[55,72,194,192]])
                }
wind_data_file = '10_26_wind_vectors.csv'
run_sim('puffs_take1',45.,50.,t_stop=10000.,
swarm_size =10,start_type='fh',wind_slippage=(1.,1.),kappa=0.,upper_prob=0.002,
display_speed=1.,heading_data=heading_data,wind_data_file=wind_data_file)


raw_input('Done?')
