'''This does what run_simulation.py does with specified output name and parameters.'''

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

def setup_odor_field(wind_field,traps,plot_scale,puff_mol_amount=None,
    puffs=False,horizontal_diffusion=1.5):
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
            'xlim' : (-plot_size, plot_size),
            'ylim' : (-plot_size, plot_size),
            'cmap' : plt.cm.YlGnBu}
    return odor_plot_param,odor_field,plumes

def setup_swarm(swarm_size,wind_field,beta,kappa,start_type, upper_prob,release_delay=0.,
    heading_data = None, wind_slippage = (0.,0.),upper_threshold=0.02):
    if wind_field.evolving:
        wind_angle_0 = wind_field.angle[0]
    else:
        wind_angle_0 = wind_field.angle
    release_times = scipy.random.exponential(beta,(swarm_size,)) + release_delay*60.
    if kappa==0:
        dist = scipy.stats.uniform(0.,2*scipy.pi)
    else:
        dist= scipy.stats.vonmises(loc=wind_angle_0,kappa=kappa)
    swarm_param = {
        #    'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
            'swarm_size'          : swarm_size,
            'heading_data'        : heading_data,
            'initial_heading_dist': dist,
            'initial_heading'     : scipy.random.vonmises(wind_angle_0,kappa,(swarm_size,)),
            'x_start_position'    : scipy.zeros((swarm_size,)),
            'y_start_position'    : scipy.zeros((swarm_size,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((swarm_size,), 0.7),
            #'flight_speed'        : scipy.random.uniform(0.3,1.0,(swarm_size,)),
            #'release_time'        : scipy.full((swarm_size,), 0.0),
            'release_time'        : release_times,
            'release_time_constant': beta,
            'release_delay'       : release_delay*60,
            'cast_interval'       : [60.0, 1000.0],
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': upper_threshold
                },
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': upper_prob,  # detection probability/sec of exposure
                }
            }
    # print(swarm_param['initial_heading_dist'])
    # print(swarm_param['initial_heading_dist'].mean())
    swarm = swarm_models.BasicSwarmOfFlies(wind_field,param=swarm_param,start_type=start_type)
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

    #Initial fly plots
    plt.ion()
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
        fly_dots, = plt.plot(swarm.x_position, swarm.y_position,'.r')
        plot_dict.update({'fly_dots':fly_dots})

    return plot_dict

'''***************************MAIN SIMULATION FUNCTION*****************************************'''

def run_sim(file_name,wind_angle,release_time_constant,
kappa=0.,t_stop=15000.0,display_speed=1,
wind_slippage = (0.,0.),swarm_size=10000,start_type='fh',upper_prob=0.002,
heading_data=None,wind_data_file=None,dt=0.25,wind=True,flies=True,puffs=False,
plot_scale = 2.0,release_delay=0.,wind_dt=None,video_name=None,wind_speed=0.5,
puff_horizontal_diffusion=1.,upper_threshold=0.02):
    if puffs:
        lower_prob = 0.05
        upper_prob = 0.05
        puff_mol_amount = 100.
    else:
        puff_mol_amount=None

    output_file = file_name+'.pkl'
    #Create wind field,
    #Release delay is in minutes
    wind_field=setup_wind_field(wind_angle,wind_data_file,dt,release_delay,wind_dt=wind_dt,wind_speed=wind_speed)

    # if wind_field.param['negative_time']>0:
    #     release_delay=wind_field.param['negative_time']

    # Create circular trap setup, set source locations and strengths
    traps = setup_traps()


    odor_plot_param,odor_field,plumes = setup_odor_field(wind_field,traps,
    plot_scale,puff_mol_amount=puff_mol_amount,puffs=puffs,horizontal_diffusion=puff_horizontal_diffusion)

    '''Get swarm ready'''
    if flies:
        swarm = setup_swarm(swarm_size,wind_field,
            release_time_constant, kappa, start_type, upper_prob,release_delay=release_delay,
            heading_data=heading_data,wind_slippage=wind_slippage,upper_threshold=upper_threshold)
    else:
        swarm=None

    # Setup live plot
    '''Plot the initial odor field'''
    plot_dict = initial_plot(odor_field,odor_plot_param,flies,release_delay,swarm=swarm,fignum = 1,plumes=plumes)
    fig = plot_dict['fig']
    fig.set_size_inches(8,8,True)
    fig.canvas.flush_events()
    ax = fig.add_subplot(111)
    plt.pause(0.001)
    t = 0.0
    dt = dt
    dt_plot = display_speed
    t_plot_last = 0.0

    '''Set up video tools'''
    if video_name is not None:
        FFMpegWriter = animate.writers['ffmpeg']
        metadata = {'title':video_name,
                }
        fps = 24.
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        writer.setup(fig, video_name+'.mp4', 500)

    '''Begin simulation loop'''
    while t<t_stop:
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        if flies & (t>=release_delay*60):
            try:
                swarm.update(t,dt,wind_field,odor_field,traps,plumes=plumes)
            except(IndexError):
                print('Out of wind data')

                sys.exit()
        #Update the plumes
        if plumes is not None:
            plumes.update(t, dt)
            plumes.report()
        #Update time display
        timer = plot_dict['timer']
        if t<release_delay*60.:
            text ='-{0} min {1} sec'.format(int(scipy.floor(abs(t/60.-release_delay))),int(scipy.floor(abs(t-release_delay*60)%60.)))
        else:
            text ='{0} min {1} sec'.format(int(scipy.floor(t/60.-release_delay)),int(scipy.floor(t%60.)))
        timer.set_text(text)
        t+= dt


        # Update live display
        if t_plot_last + dt_plot <= t:

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
                #image.set_data(conc_array)
                image.set_data(scipy.log(conc_array))
            if video_name is not None:
                writer.grab_frame()

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
    if video_name is not None:
        writer.finish()
    with open(output_file, 'w') as f:
        pickle.dump((swarm,wind_field),f)

    plt.clf()

heading_data = {'angles':(scipy.pi/180)*scipy.array([0.,90.,180.,270.]),
                'counts':scipy.array([[1724,514,1905,4666],[55,72,194,192]])
                }
wind_data_file = '2017_10_26_wind_vectors_1_min_pre_60_min_post_release.csv'

run_sim('new_hderr_dist',45.,10.,t_stop=600.,
swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.008,
display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
release_delay=0.,wind_dt=5,video_name='new_hderr_dist')

# run_sim('flies319_101',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.008,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_101')

# run_sim('highe_prob',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.025    ,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='highe_prob')

# run_sim('test_322',45.,10.,t_stop=1000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.025    ,
# display_speed=10,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,wind_speed=0.75,video_name='test_322')

# run_sim('michael_combo',45.,100.,t_stop=4000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.025    ,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,wind_speed=0.75,video_name='michael_combo')

# run_sim('michael_combo_w_wind',45.,100.,t_stop=6000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.025    ,
# display_speed=2.5,heading_data=None,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,wind_speed=0.75,video_name='michael_combo_w_wind')

# run_sim('michael_combo_w_wind_slower',45.,100.,t_stop=4500.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.025    ,
# display_speed=1.25,heading_data=None,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=15.,wind_dt=5,wind_speed=0.75,video_name='michael_combo_w_wind_slower')


# run_sim('lower_thres',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='lower_thres',upper_threshold=0.002)

# run_sim('lower_thres+higher_prob',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.1,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='lower_thres+higher_prob',upper_threshold=0.002)


#
# run_sim('flies319_102',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_102')
#
# run_sim('flies319_103',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='cvrw',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_103')
#
# run_sim('flies319_104',225.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=heading_data,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_104')
# #
# run_sim('flies319_105',45.,10.,t_stop=4000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=heading_data,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=10.,wind_dt=5,video_name='flies319_105')
#
# run_sim('flies319_106',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,video_name='flies319_106')
#
# run_sim('flies319_107',45.,100.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_107')
#
# run_sim('flies319_108',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_108')
#
# # run_sim('flies319_109',45.,10.,t_stop=7800.,
# # swarm_size =1000,start_type='cvrw',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# # display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# # release_delay=10.,wind_dt=5,video_name='flies319_109')
# #
# # run_sim('flies315_110',45.,10.,t_stop=7800.,
# # swarm_size =1000,start_type='cvrw',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# # display_speed=2.5,heading_data=None,wind_data_file=wind_data_file,puffs=True,flies=True,
# # release_delay=10.,wind_dt=5,video_name='flies315_110')
#
# run_sim('flies319_111',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_111',wind_speed=1.0)
#
# run_sim('flies319_112',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_112',wind_speed=1.5)

# run_sim('pompy_width0001',45.,10.,t_stop=3500.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,puff_horizontal_diffusion=0.001,video_name='pompy_width0001')
#
# run_sim('pompy_width001',45.,10.,t_stop=3500.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,puff_horizontal_diffusion=0.01,video_name='pompy_width001')
#
# run_sim('pompy_width01',45.,10.,t_stop=3500.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,puff_horizontal_diffusion=0.1,video_name='pompy_width01')
#
# run_sim('pompy_width1',45.,10.,t_stop=3500.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=True,flies=True,
# release_delay=25.,wind_dt=5,puff_horizontal_diffusion=1.,video_name='pompy_width1')





#raw_input('Done?')
