'''This does what run_simulation.py does with specified output name and parameters.'''

import time
import scipy
import matplotlib.pyplot as plt
import matplotlib
import cPickle as pickle
import sys

import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.odor_models as odor_models
import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.borrowed_puff_models as puff_models



def setup_wind_field(wind_angle,wind_data_file,dt):
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
    return wind_field

def setup_traps(number_sources = 6,radius_sources = 400.0,strength_sources = 10.0,
trap_radius = 5.):
    location_list, strength_list = utility.create_circle_of_sources(
            number_sources,
            radius_sources,
            strength_sources
            )

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

def setup_odor_field(wind_field,traps,plot_scale,puff_mol_amount,puffs=False):
    plot_size = plot_scale*traps.param['source_radius']
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
                'xlim' : (-plot_size, plot_size),
                'ylim' : (-plot_size, plot_size),
                'xnum' : 500,
                'ynum' : 500,
                'cmap' : 'binary',
                'fignums' : (1,2)
                }
        plumes=None
    else:
        '''This is the odor puff implementation borrowed/adapted from here: https://github.com/InsectRobotics/pompy'''
        '''Get the odor puff stuff ready'''
        sim_region = puff_models.Rectangle(-plot_size, -plot_size, plot_size, plot_size)
        source_pos = traps.param['source_locations'][4]
        #source_pos = traps.param['source_locations']
        '''*****************'''
        plumes = puff_models.PlumeModel(sim_region, source_pos, wind_field,
                                        puff_release_rate=0.1,model_z_disp=False,
                                        centre_rel_diff_scale=1.5,puff_init_rad=1.,puff_spread_rate=0.05)
    #Concentration generator object
        grid_size = 1000
        odor_field = puff_models.ConcentrationArrayGenerator(sim_region, 0.1, grid_size,
                                                           grid_size, puff_mol_amount,kernel_rad_mult=5)
    #Pompy version---------------------------
        odor_plot_param = {
            'xlim' : (-plot_size, plot_size),
            'ylim' : (-plot_size, plot_size),
            'cmap' : plt.cm.YlGnBu}
    return odor_plot_param,odor_field,plumes

def setup_swarm(swarm_size,wind_field,beta,kappa,start_type, upper_prob,release_delay=0.,
    heading_data = None, wind_slippage = (0.,0.)):
    if wind_field.evolving:
        wind_angle_0 = wind_field.angle[0]
    else:
        wind_angle_0 = wind_field.angle
    release_times = scipy.random.exponential(beta,(swarm_size,)) + release_delay
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
            'release_time'        : release_times,
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
    return swarm

def initial_plot(odor_field,plot_param,flies,swarm=None,fignum=1,plumes=None):
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

    #Initial fly plots
    plt.ion()
    fig = plt.figure(fignum)
    plot_dict.update({'fig':fig})
    ax = plt.subplot(111)

    plt.figure(fignum)
    plot_dict.update({'fignum':fignum})
    if flies:
        fly_dots, = plt.plot(swarm.x_position, swarm.y_position,'.r')
        plot_dict.update({'fly_dots':fly_dots})

    return plot_dict

def run_sim(file_name,wind_angle,release_time_constant,
kappa=0.,t_stop=15000.0,display_speed=1,
wind_slippage = (0.,0.),swarm_size=10000,start_type='fh',upper_prob=0.002,
heading_data=None,wind_data_file=None,dt=0.25,wind=True,flies=True,puffs=False,plot_scale = 2.0,release_delay=0.):
    if puffs:
        lower_prob = 0.05
        upper_prob = 0.05
        puff_mol_amount = 100.

    output_file = file_name+'.pkl'
    #Create wind field
    wind_field=setup_wind_field(wind_angle,wind_data_file,dt)
    # Create circular trap setup, set source locations and strengths
    traps = setup_traps()


    odor_plot_param,odor_field,plumes = setup_odor_field(wind_field,traps,plot_scale,puff_mol_amount,puffs=puffs)

    '''Get swarm ready'''
    if flies:
        swarm = setup_swarm(swarm_size,wind_field,
            release_time_constant, kappa, start_type, upper_prob,release_delay=release_delay,
            heading_data=heading_data,wind_slippage=wind_slippage)
    else:
        swarm=None

    # Setup live plot
    '''Plot the initial odor field'''
    plot_dict = initial_plot(odor_field,odor_plot_param,flies,swarm=swarm,fignum = 1,plumes=plumes)
    fig = plot_dict['fig']
    fig.canvas.flush_events()
    plt.pause(0.001)
    t = 0.0
    dt = dt
    dt_plot = 10.0*display_speed
    t_plot_last = 0.0

    while t<t_stop:
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        if flies & (t>=release_delay):
            try:
                swarm.update(t,dt,wind_field,odor_field,traps,plumes=plumes)
            except(IndexError):
                print('Out of wind data')
                sys.exit()
        #Update the plumes
        if plumes is not None:
            plumes.update(t, dt)
            plumes.report()
        t+= dt


        # Update live display
        if t_plot_last + dt_plot < t:

            '''First, plot the flies and display fraction of flies caught'''
            plt.figure(plot_dict['fignum'])
            if flies:
                fly_dots = plot_dict['fly_dots']
                fly_dots.set_xdata([swarm.x_position])
                fly_dots.set_ydata([swarm.y_position])

                trap_list = []
                for trap_num, trap_loc in enumerate(traps.param['source_locations']):
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
            '''Next, plot the odor concentration field'''
            image = plot_dict['image']
            if not(puffs): #Will's version odor field
                odor_mesh=odor_field.value_to_mesh(t,odor_plot_param)
                image.set_data(odor_mesh)
                #plt.figure(10);plt.clf();plt.hist(scipy.unique(odor_mesh),bins=100)
            else: #Pompy version
                conc_array = (
                odor_field.generate_single_array(plumes.puff_array).T[::-1])
                #conc_array=conc_array/conc_array.max()
                image.set_data(conc_array)
                #image.set_data(scipy.log(conc_array))
                #plt.figure(10);plt.clf();plt.hist(scipy.unique(conc_array),bins=100);plt.xlim(0,20)

#            plt.pause(0.5)


        #    vmin = conc_array.min()
        #    vmax=conc_array.max()
        #    n = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        #    image.set_norm(n)
        #



            fig.canvas.flush_events()
            t_plot_last = t

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
swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
display_speed=0.1,heading_data=None,wind_data_file=None,puffs=True,flies=True,release_delay=600.)


raw_input('Done?')
