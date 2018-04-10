

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

def run_tunnel_sim(file_name,release_time_constant,
t_stop=15000.0,display_speed=50, wind_slippage = (0.,0.),
swarm_size=10000,upper_prob=0.002,dt=0.01,
video_name=None,wind_speed=0.5,
upper_threshold=0.001,schmitt_trigger=True,track_plume_bouts=False):
    output_file = file_name+'.pkl'
    #Create wind field
    wind_field=srt.setup_wind_field(270.,None,dt,0.,wind_dt=None,wind_speed=wind_speed)
    #--- Setup odor arena
    xlim = (0., 0.3)
    ylim = (0., 1.5)
    odor_param = {
            'wind_field'       : wind_field,
            'diffusion_coeff'  :  0.00001,
            'source_locations' : [(0.15,1.45)],
            'source_strengths' : [1.],
            'epsilon'          : 2.,
            'trap_radius'      : 0.
            }
    odor_field = odor_models.FakeDiffusionOdorField([],param=odor_param)
    odor_plot_param = {
            'xlim' : xlim,
            'ylim' : ylim,
            'xnum' : 500,
            'ynum' : 500,
            'cmap' : 'binary',
            'fignums' : (1,2)
            }
    #single trap
    trap_param = {
            'source_locations' : odor_param['source_locations'],
            'source_strengths' : odor_param['source_strengths'],
            'epsilon'          : 0.01,
            'trap_radius'      : 0.05,
            'source_radius'    : 0.
    }

    traps = trap_models.TrapModel(trap_param)
    #Setup fly swarm
    swarm_param = {
            'swarm_size'          : swarm_size,
            'heading_data'        : None,
            'initial_heading_dist': scipy.radians(90.),
            'initial_heading'     : scipy.random.uniform(scipy.radians(80.),scipy.radians(100.),swarm_size),
            'x_start_position'    : scipy.random.uniform(0.075,0.225,swarm_size),
            'y_start_position'    : scipy.zeros((swarm_size,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((swarm_size,), 0.5),
            'release_time'        : scipy.random.exponential(release_time_constant,(swarm_size,)),
            'release_time_constant': release_time_constant,
            'release_delay'       : 0.,
            'cast_interval'       : [.05, .10],
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

    swarm = swarm_models.BasicSwarmOfFlies(wind_field,param=swarm_param,
        start_type='fh',track_plume_bouts=track_plume_bouts,track_arena_exits=True)
    plot_dict = srt.initial_plot(odor_field,odor_plot_param,True,0.,
    swarm=swarm,fignum = 1,plumes=None)
    fig = plot_dict['fig']
    fig.set_size_inches(8,12,True)
    fig.canvas.flush_events()
    ax = fig.add_subplot(111)
    plt.pause(0.001)
    t = 0.0
    dt_plot = display_speed*dt
    t_plot_last = 0.0


    '''Set up video tools'''
    if video_name is not None:
        FFMpegWriter = animate.writers['ffmpeg']
        metadata = {'title':video_name,}
        fps = 24.
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        writer.setup(fig, video_name+'.mp4', 500)
    '''Begin simulation loop'''
    while t<t_stop:
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        swarm.update(t,dt,wind_field,odor_field,traps,xlim=xlim,ylim=ylim)
        #Update time display
        timer = plot_dict['timer']
        text ='{0} min {1} sec'.format(int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        t+= dt
        # Update live display
        if t_plot_last + dt_plot <= t:
            image = plot_dict['image']
            xmin,xmax,ymin,ymax=image.get_extent()
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            '''plot the flies'''
            plt.figure(plot_dict['fignum'])
            fly_dots = plot_dict['fly_dots']
            fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

            color_dict = plot_dict['color_dict']
            fly_colors = [color_dict[mode] for mode in swarm.mode]
            fly_dots.set_color(fly_colors)

            '''plot the odor concentration field'''
            odor_mesh=odor_field.value_to_mesh(t,odor_plot_param)
            image.set_data(odor_mesh)
            plt.show()

            if video_name is not None:
                writer.grab_frame()
            fig.canvas.flush_events()
            t_plot_last = t
    if video_name is not None:
        writer.finish()
    with open(output_file, 'w') as f:
        pickle.dump((swarm,wind_field),f)

    plt.clf()

run_tunnel_sim('tunnel_sim_test',0.1,
t_stop=10.,display_speed=2, wind_slippage = (0.,0.),
swarm_size=100,upper_prob=0.002,dt=0.01,
video_name='tunnel_sim_test',wind_speed=0.5,
upper_threshold=0.001,schmitt_trigger=False,track_plume_bouts=False)
