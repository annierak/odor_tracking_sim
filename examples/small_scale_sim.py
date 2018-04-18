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
puff_horizontal_diffusion=1.,upper_threshold=0.02,schmitt_trigger=True):
