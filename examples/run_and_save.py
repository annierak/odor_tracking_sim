'''This does what run_simulation.py does with specified output name and parameters.'''

import time
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle

import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.odor_models as odor_models
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.utility as utility

def run_sim(file_name,wind_angle,release_time_constant,kappa=2,t_stop=15000.0,display_speed=1,swarm_size=10000):
    output_file = file_name+'.pkl'
    wind_angle=wind_angle*scipy.pi/180.0
    wind_param = {
            'speed': 0.5,
            'angle': wind_angle,
            }
    wind_field = wind_models.ConstantWindField(param=wind_param)

    # Create circular odor field, set source locations and strengths
    number_sources = 6
    radius_sources = 1000.0
    strength_sources = 10.0
    location_list, strength_list = utility.create_circle_of_sources(
            number_sources,
            radius_sources,
            strength_sources
            )
    odor_param = {
            'wind_field'       : wind_field,
            'diffusion_coeff'  :  0.25,
            'source_locations' : location_list,
            'source_strengths' : strength_list,
            'epsilon'          : 0.01,
            'trap_radius'      : 5.0
            }
    odor_field = odor_models.FakeDiffusionOdorField(odor_param)
    swarm_size = swarm_size
    beta = release_time_constant #time constant for release time distribution
    swarm_param = {
        #    'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
            'initial_heading_dist': scipy.stats.vonmises(loc=wind_angle,kappa=kappa),
            'initial_heading'     : scipy.random.vonmises(wind_angle,kappa,(swarm_size,)),
            'x_start_position'    : scipy.zeros((swarm_size,)),
            'y_start_position'    : scipy.zeros((swarm_size,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((swarm_size,), 0.7),
            #'flight_speed'        : scipy.random.uniform(0.3,1.0,(swarm_size,)),
            #'release_time'        : scipy.full((swarm_size,), 0.0),
            'release_time'        : scipy.random.exponential(beta,(swarm_size,)),
            'release_time_constant': beta,
            'cast_interval'       : [60.0, 1000.0],
            'wind_slippage'       : 0.0,
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': 0.02
                },
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': 0.002,  # detection probability/sec of exposure
                }
            }
    swarm = swarm_models.BasicSwarmOfFlies(param=swarm_param,start_type='rw')
    # Setup live plot
    fignum = 1
    plot_scale = 2.0
    plot_size = plot_scale*radius_sources
    plot_param = {
            'xlim' : (-plot_size, plot_size),
            'ylim' : (-plot_size, plot_size),
            'xnum' : 500,
            'ynum' : 500,
            'cmap' : 'binary',
            'fignums' : (1,2),
            #'threshold': 0.001,
            }
    odor_field.plot(plot_param=plot_param)
    #plt.show()

    plt.ion()
    fig = plt.figure(fignum)
    ax = plt.subplot(111)

    plt.figure(fignum)
    fly_dots, = plt.plot(swarm.x_position, swarm.y_position,'.r')

    fig.canvas.flush_events()
    plt.pause(0.001)
    t = 0.0
    dt = 0.25
    dt_plot = 10.0*display_speed
    t_plot_last = 0.0

    while t<t_stop:

        print('t: {0:1.2f}'.format(t))
        swarm.update(t,dt,wind_field,odor_field)
        t+= dt

        # Update live display
        if t_plot_last + dt_plot < t:

            plt.figure(fignum)
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

            fig.canvas.flush_events()
            t_plot_last = t

            #time.sleep(0.05)


    # Write swarm to file
    with open(output_file, 'w') as f:
        pickle.dump(swarm,f)

run_sim('brownian',25.,50.,t_stop=25000.,display_speed=1,swarm_size = 1000)
