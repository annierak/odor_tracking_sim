

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

def run_small_sim(file_name,wind_angle,release_time_constant,
t_stop=15000.0,display_speed=1, wind_slippage = (0.,0.),
swarm_size=100,upper_prob=0.002,dt=0.25,wind_dt=None,pompy_wind_model=False,
wind_speed=0.5,wind_data_file=None,video=True,release_delay=5.,
upper_threshold=0.001,schmitt_trigger=False,track_plume_bouts=False,
puff_horizontal_diffusion=1.,plot_scale = 2.0,puff_mol_amount=1.):
    output_file = file_name+'.pkl'
    #Create wind field
    if pompy_wind_model:
    else:
        wind_field=srt.setup_wind_field(wind_angle,
        wind_data_file,dt,0.,wind_dt=wind_dt,wind_speed=wind_speed)
    #--- Setup odor arena
    xlim = (-15., 15.)
    ylim = (0., 40.)

    # xlim = (0., 200.)
    # ylim = (0., 200.)


    trap_param = {
            'source_locations' : [(7.5,25.),],
            'source_strengths' : [1.,],
            'epsilon'          : 0.01,
            'trap_radius'      : 2.,
            'source_radius'    : 0.
    }

    traps = trap_models.TrapModel(trap_param)

    odor_plot_param,odor_field,plumes = srt.setup_odor_field(wind_field,traps,
        plot_scale,puff_mol_amount=puff_mol_amount,puffs=True,
        xlim=xlim,ylim=ylim)

    #Setup fly swarm
    swarm_param = {
            'swarm_size'          : swarm_size,
            'heading_data'        : None,
            'initial_heading_dist': scipy.radians(90.),
            'initial_heading'     : scipy.random.uniform(scipy.radians(80.),scipy.radians(100.),swarm_size),
            'x_start_position'    : scipy.random.uniform(-2.5,7.5,swarm_size),
            'y_start_position'    : 5.*scipy.ones((swarm_size,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((swarm_size,), 0.5),
            'release_time'        : scipy.random.exponential(release_time_constant,(swarm_size,)),
            'release_time_constant': release_time_constant,
            'release_delay'       : 0.,
            'cast_interval'       : [5, 10],
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': upper_threshold
                },
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': upper_prob,  # detection probability/sec of exposure
                },
            'schmitt_trigger':schmitt_trigger,
            'dt_plot': display_speed,
            't_stop':t_stop
            }

    swarm = swarm_models.BasicSwarmOfFlies(wind_field,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=track_plume_bouts,track_arena_exits=True)
    plot_dict = srt.initial_plot(odor_field,wind_field,odor_plot_param,True,0.,
    swarm=swarm,fignum = 1,plumes=plumes)
    fig = plot_dict['fig']
    fig.set_size_inches(8,8,True)
    fig.canvas.flush_events()
    ax = fig.add_subplot(111)
    plt.pause(0.001)
    t = 0.0
    dt_plot = display_speed*dt
    t_plot_last = 0.0


    '''Set up video tools'''
    if video:
        FFMpegWriter = animate.writers['ffmpeg']
        metadata = {'title':file_name,}
        fps = 24.
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        writer.setup(fig, file_name+'.mp4', 500)
    '''Begin simulation loop'''
    while t<t_stop:
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        swarm.update(t,dt,wind_field,odor_field,traps,xlim=xlim,ylim=ylim,
        plumes=plumes)
        plumes.update(t, dt)
        #Update time display
        timer = plot_dict['timer']
        text ='{0} min {1} sec'.format(int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        t+= dt
        # Update live display
        if t_plot_last + dt_plot <= t:
            image = plot_dict['image']
            xmin,xmax,ymin,ymax=image.get_extent()
            ax = plot_dict['ax']
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            '''plot the flies'''
            plt.figure(plot_dict['fignum'])
            fly_dots = plot_dict['fly_dots']
            fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

            color_dict = plot_dict['color_dict']
            fly_colors = [color_dict[mode] for mode in swarm.mode]
            fly_dots.set_color(fly_colors)

            wind_arrow = plot_dict['wind_arrow']

            arrow_magn = (xmax-xmin)/20
            x_wind,y_wind = wind_field.value(t,0,0)
            wind_arrow.set_positions((xmin+(xmax-xmin)/2,ymax-0.2*(ymax-ymin)),
            (xmin+(xmax-xmin)/2+arrow_magn*x_wind,
            ymax-0.2*(ymax-ymin)+arrow_magn*y_wind))



            trap_list = []
            for trap_num, trap_loc in enumerate(traps.param['source_locations']):
                mask_trap = swarm.trap_num == trap_num
                trap_cnt = mask_trap.sum()
                trap_list.append(trap_cnt)
            total_cnt = sum(trap_list)
            plt.title('{0}/{1}: {2}'.format(total_cnt,swarm.size,trap_list))

            '''plot the odor concentration field'''
            conc_array = (
            odor_field.generate_single_array(plumes.puff_array).T[::-1])
            image.set_data(conc_array)

            if video:
                writer.grab_frame()
            fig.canvas.flush_events()
            t_plot_last = t
    if video:
        writer.finish()
    with open(output_file, 'w') as f:
        pickle.dump((swarm,wind_field),f)

    plt.clf()

wind_data_file = '2017_10_26_wind_vectors_1_min_pre_60_min_post_release.csv'


run_small_sim('small_scale_sim',270.,0.1,
t_stop=180.,display_speed=1, wind_slippage = (0.,0.),
swarm_size=100,dt=0.25,wind_dt=5,wind_data_file=wind_data_file,
wind_speed=0.5,upper_threshold=0.001,schmitt_trigger=False)
