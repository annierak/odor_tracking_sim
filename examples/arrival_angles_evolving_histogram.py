import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility
import time
import numpy as np

f = sys.argv[1]
input_file = f+'.pkl'

with open(input_file,'r') as f:
    (swarm,wind_field,video_info) = pickle.load(f)

trap_num_list = swarm.list_all_traps()

plt.ion()
fig = plt.figure(1)


#Put the time in the middle
text = '0 min 0 sec'
timer= plt.figtext(0.5,0.5,text,color='r',horizontalalignment='center')

ax1 = plt.subplot2grid((3,4),(1,3),polar=True)
ax2 = plt.subplot2grid((3,4),(0,2),polar=True)
ax3 = plt.subplot2grid((3,4),(0,1),polar=True)
ax4 = plt.subplot2grid((3,4),(1,0),polar=True)
ax5 = plt.subplot2grid((3,4),(2,1),polar=True)
ax6 = plt.subplot2grid((3,4),(2,2),polar=True)
trap_axes = [ax1,ax2,ax3,ax4,ax5,ax6]

num_bins = 20
# fig.canvas.flush_events()
# plt.pause(0.001)

"""This is the initial plot"""
bar_containers = []

for trap in trap_num_list:
    ax = trap_axes[trap]
    ax.set_yticks([])
    ax.set_title('trap:{0}'.format(trap))
    arrival_angles = swarm.get_angle_trapped(trap,[0,0])
    # arrival_angles = np.random.randn(50)
    (counts,bin_edges) = np.histogram(arrival_angles,bins=num_bins,range=(0,2*scipy.pi))
    w = bin_edges[1:]-bin_edges[:-1]
    bar_container = ax.bar(bin_edges[:-1],counts,width=w)
    bar_containers.append(bar_container)
    ax.set_xlabel('Arrival angle')
    ax.set_ylim(0,1)
    '''Now travel through time and update the histogram'''
    dt_plot = swarm.dt_plot
    hist_dt_plot = 15.
    t_stop = swarm.t_stop
    window_length = 60.

plt.draw()

t = 0.
while t<t_stop:
    print(t)
    time_window= [t,t+window_length]
    peak_counts = scipy.zeros(len(trap_axes))
    for trap in trap_num_list:
        ax = trap_axes[trap]
        bar_container = bar_containers[trap]
        arrival_angles = swarm.get_angle_trapped(trap,time_window)
        # arrival_angles = np.random.randn(50)
        (counts,bin_edges) = np.histogram(arrival_angles,bins=num_bins,range=(0,2*scipy.pi))
        bars = bar_container.patches
        for index,bar in enumerate(bars):
            # print(counts[index])
            bar.set_height(counts[index])
        if len(arrival_angles)>0:
            peak_counts[trap]=max(counts)
            # print(counts)
            # print('some arrivals')
            time.sleep(0.01)
        else:
            peak_counts[trap]=1
    top = max(peak_counts)
    for i,num in enumerate(trap_num_list):
        ax = trap_axes[num]
        ax.set_ylim(0,top)
    plt.draw()
    plt.pause(0.001)
    t+=hist_dt_plot








plt.show()
