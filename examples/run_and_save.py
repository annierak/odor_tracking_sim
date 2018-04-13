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
import odor_tracking_sim.simulation_running_tools as srt

def run_sim(file_name,wind_angle,release_time_constant,
kappa=0.,t_stop=15000.0,display_speed=1,
wind_slippage = (0.,0.),swarm_size=10000,start_type='fh',upper_prob=0.002,
heading_data=None,wind_data_file=None,dt=0.25,wind=True,flies=True,puffs=False,
plot_scale = 2.0,release_delay=0.,wind_dt=None,video_name=None,wind_speed=0.5,
puff_horizontal_diffusion=1.,upper_threshold=0.02,schmitt_trigger=True,number_sources=6,
heading_mean=None,track_plume_bouts=False):
    if puffs:
        lower_prob = 0.05
        upper_prob = 0.05
        puff_mol_amount = 100.
    else:
        puff_mol_amount=None

    output_file = file_name+'.pkl'
    #Create wind field,
    #Release delay is in minutes
    wind_field=srt.setup_wind_field(wind_angle,wind_data_file,dt,
    release_delay,wind_dt=wind_dt,wind_speed=wind_speed)

    # if wind_field.param['negative_time']>0:
    #     release_delay=wind_field.param['negative_time']

    # Create circular trap setup, set source locations and strengths
    if number_sources>1:
        radius_sources = 1000.0
    else:
        radius_sources = 400.0
    traps = srt.setup_traps(number_sources=number_sources,radius_sources=radius_sources)


    odor_plot_param,odor_field,plumes = srt.setup_odor_field(wind_field,traps,
    plot_scale,puff_mol_amount=puff_mol_amount,puffs=puffs,horizontal_diffusion=puff_horizontal_diffusion)

    '''Get swarm ready'''
    if flies:
        swarm = srt.setup_swarm(swarm_size,wind_field,traps,
            release_time_constant, kappa, start_type, upper_prob,release_delay=release_delay,
            heading_data=heading_data,wind_slippage=wind_slippage,upper_threshold=upper_threshold,
            schmitt_trigger=schmitt_trigger,heading_mean=heading_mean,track_plume_bouts=track_plume_bouts)
    else:
        swarm=None

    # Setup live plot
    '''Plot the initial odor field'''
    plot_dict = srt.initial_plot(odor_field,odor_plot_param,
    flies,release_delay,swarm=swarm,fignum = 1,plumes=plumes)
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
            image = plot_dict['image']
            xmin,xmax,ymin,ymax=image.get_extent()
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            '''First, plot the flies and display fraction of flies caught'''
            plt.figure(plot_dict['fignum'])
            if flies:
                fly_dots = plot_dict['fly_dots']
                fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])
                # fly_dots.set_xdata([swarm.x_position])
                # fly_dots.set_ydata([swarm.y_position])

                color_dict = plot_dict['color_dict']
                fly_colors = [color_dict[mode] for mode in swarm.mode]
                fly_dots.set_color(fly_colors)

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
                plt.show()


            #plt.title('{0}/{1}: {2} {3}'.format(total_cnt,swarm.size,trap_list,frac_list))
            '''Next, plot the odor concentration field'''
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

        #    time.sleep(0.5)

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

# run_sim('lp_filter_trial_2',45.,10.,t_stop=3000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,schmitt_trigger=False,video_name='lp_filter_trial_2')

# run_sim('casting_inspection',45.,10.,t_stop=3000.,
# swarm_size =100,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,schmitt_trigger=False,video_name='casting_inspection')

# run_sim('flies319_101',45.,10.,t_stop=7800.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.008,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,video_name='flies319_101')

# run_sim('small_scale_test',225.,10.,t_stop=1000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.008,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=1,video_name='small_scale_test')

# run_sim('lp_filter_small_scale_3',225.,10.,t_stop=1000.,
# swarm_size =100,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=1,schmitt_trigger=False,
# video_name='lp_filter_small_scale_3',heading_mean=115.)

# run_sim('lp_filter_small_scale_4',225.,10.,t_stop=1000.,
# swarm_size =10000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=1,schmitt_trigger=False,
# video_name='lp_filter_small_scale_4',heading_mean=115.)

# run_sim('shorter_casting_small_scale_4',225.,10.,t_stop=1000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=1.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=1,schmitt_trigger=False,
# video_name='shorter_casting_small_scale_4',heading_mean=115.)

# run_sim('shorter_casting_large_scale',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=1.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,
# video_name='shorter_casting_large_scale')

# run_sim('color_changing_1',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=1.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,
# video_name='color_changing_1')

# run_sim('color_changing_wind_data_longer',45.,10.,t_stop=8000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=2.5,heading_data=heading_data,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=20.,wind_dt=5,number_sources=6,schmitt_trigger=False,
# video_name='color_changing_wind_data_longer')

# run_sim('side_slip_smaller_casting',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,1.),kappa=0.,upper_prob=1.,
# display_speed=1.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,
# video_name='side_slip_smaller_casting')

# run_sim('plume_bout_tracking_test',45.,10.,t_stop=3000.,
# swarm_size =3000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,track_plume_bouts=True,
# video_name='plume_bout_tracking_test')

# run_sim('plume_bout_tracking_test_2',45.,10.,t_stop=3000.,
# swarm_size =3000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,track_plume_bouts=True,
# video_name='plume_bout_tracking_test_2')

# run_sim('plume_bout_tracking_debugging',225.,10.,t_stop=3000.,
# swarm_size =15,start_type='fh',wind_slippage=(0.,0.),kappa=2.,upper_prob=1.,
# display_speed=1.25,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=1,schmitt_trigger=False,track_plume_bouts=True,
# video_name='plume_bout_tracking_debugging',heading_mean=115.)

# run_sim('plume_bout_tracking_take_2',45.,10.,t_stop=3000.,
# swarm_size =3000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=False,track_plume_bouts=True,
# video_name='plume_bout_tracking_take_2')

# run_sim('angle_arrival_test',45.,10.,t_stop=2000.,
# swarm_size =3000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
# release_delay=0.,wind_dt=5,number_sources=6,schmitt_trigger=True,track_plume_bouts=True,
# video_name='angle_arrival_test')

# run_sim('angle_arrival_puffs',45.,10.,t_stop=2000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
# display_speed=2.5,heading_data=None,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=20.,wind_dt=5,number_sources=6,schmitt_trigger=False,track_plume_bouts=True,
# video_name='angle_arrival_puffs')

run_sim('test_8_traps',60.,10.,t_stop=2000.,
swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=1.,
display_speed=2.5,heading_data=None,wind_data_file=None,puffs=False,flies=True,
release_delay=0.,wind_dt=5,number_sources=8,schmitt_trigger=False,track_plume_bouts=True,
video_name='test_8_traps')

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

# run_sim('lw_filter_dyn_wind',45.,10.,t_stop=4000.,
# swarm_size =1000,start_type='fh',wind_slippage=(0.,0.),kappa=0.,upper_prob=0.002,
# display_speed=2.5,heading_data=heading_data,wind_data_file=wind_data_file,puffs=True,flies=True,
# release_delay=10.,wind_dt=5,schmitt_trigger=False,video_name='lw_filter_dyn_wind')

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
