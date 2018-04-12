import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility

f = sys.argv[1]
input_file = f+'.pkl'

with open(input_file,'r') as f:
    (swarm,wind_field) = pickle.load(f)

trap_num_list = swarm.list_all_traps()

plt.figure(2)
ax1 = plt.subplot2grid((3,4),(1,3),polar=True)
ax2 = plt.subplot2grid((3,4),(0,2),polar=True)
ax3 = plt.subplot2grid((3,4),(0,1),polar=True)
ax4 = plt.subplot2grid((3,4),(1,0),polar=True)
ax5 = plt.subplot2grid((3,4),(2,1),polar=True)
ax6 = plt.subplot2grid((3,4),(2,2),polar=True)
trap_axes = [ax1,ax2,ax3,ax4,ax5,ax6]

num_bins = 20

peak_counts = scipy.zeros(len(trap_axes))

for i in trap_num_list:
    ax = trap_axes[i]
    ax.set_yticks([])
    ax.set_title('trap:{0}'.format(i))
    print('here')
    try:
        arrival_angles = swarm.get_angle_trapped(i)
    except(IndexError):
        continue
    (n, bins, patches) = ax.hist(arrival_angles,num_bins,range=(0,2*scipy.pi))
    peak_counts[i]=max(n)
    ax.set_xlabel('Arrival angle')


top = max(peak_counts)
trap_counts = swarm.get_trap_counts()
for i,num in enumerate(trap_num_list):
    ax = trap_axes[num]
    ax.set_ylim(0,top)
    xmin,xmax = ax.get_xlim()
    ax.text(xmin+(xmax-xmin)/2,top/2,str(trap_counts[i]),color='maroon',size =20)

plt.show()
