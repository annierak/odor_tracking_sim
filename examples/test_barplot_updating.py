import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
import math
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility

f = sys.argv[1]
input_file = f+'.pkl'

with open(input_file,'r') as f:
    (swarm,wind_field,video_info) = pickle.load(f)


plt.ion()
fig = plt.figure(1)


text = '0 min 0 sec'
timer= plt.figtext(0.5,0.5,text,color='r',horizontalalignment='center')


# fig.canvas.flush_events()
ax1 = plt.subplot2grid((3,4),(1,3),polar=True)
ax2 = plt.subplot2grid((3,4),(0,2),polar=True)
ax3 = plt.subplot2grid((3,4),(0,1),polar=True)
ax4 = plt.subplot2grid((3,4),(1,0),polar=True)
ax5 = plt.subplot2grid((3,4),(2,1),polar=True)
ax6 = plt.subplot2grid((3,4),(2,2),polar=True)
axes = [ax1,ax2,ax3,ax4,ax5,ax6]
bar_containers = []

num_bins = 20


for index,ax in enumerate(axes):
    ax.set_title('trap:{0}'.format(index))
    # data = np.random.randn(500)
    # myHist, myBinEdges = np.histogram(data)
    arrival_angles = swarm.get_angle_trapped(index,[0,0])
    (counts,bin_edges) = np.histogram(arrival_angles,bins=num_bins,range=(0,2*scipy.pi))
    wid = bin_edges[1:] - bin_edges[:-1]
    bar_container = ax.bar(bin_edges[:-1], counts, width=wid)
    bar_containers.append(bar_container)
# plt.plot(range(10),range(10))
# plt.show()

# time.sleep(5)



dt = 1
t_stop = swarm.t_stop
t= 0
window_length = 60.
while t<t_stop:
    print(t)
    time_window= [t,t+window_length]
    for index,ax in enumerate(axes):
        # data = np.random.randn(500)
        # counts, myBinEdges = np.histogram(data)
        arrival_angles = swarm.get_angle_trapped(index,time_window)
        (counts,bin_edges) = np.histogram(arrival_angles,bins=num_bins,range=(0,2*scipy.pi))
        bar_container = bar_containers[index]
        bars = bar_container.patches
        for index,bar in enumerate(bars):
            bar.set_height(counts[index])
    plt.draw()
    plt.pause(0.1)
    t +=dt
