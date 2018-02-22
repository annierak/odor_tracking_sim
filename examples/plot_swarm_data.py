import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle

input_file = 'swarm_data.pkl'

with open(input_file,'r') as f:
    swarm = pickle.load(f)


num_bins = 20

trap_num_list = swarm.get_trap_nums()

plt.figure(1)
t = swarm.get_time_trapped()
plt.hist(t,num_bins)

plt.xlabel('(s)')
plt.ylabel('count')
plt.title('time trapped (all traps)')

plt.figure(2)
ax1 = plt.subplot2grid((4,3),(1,2))
ax2 = plt.subplot2grid((4,3),(0,1))
ax3 = plt.subplot2grid((4,3),(1,0))
ax4 = plt.subplot2grid((4,3),(2,0))
ax5 = plt.subplot2grid((4,3),(3,1))
ax6 = plt.subplot2grid((4,3),(2,2))
trap_axes = [ax1,ax2,ax3,ax4,ax5,ax6]


peak_counts = scipy.zeros(len(trap_num_list))
for i,num in enumerate(trap_num_list):
    t = swarm.get_time_trapped(num)
    ax = trap_axes[i]
    (n, bins, patches) = ax.hist(t,num_bins)
    peak_counts[i]=max(n)
    ax.set_xlabel('(s)')
    ax.set_ylabel('trap:{0}'.format(num))
top = max(peak_counts)
for i in range(len(trap_num_list)):
    ax = trap_axes[i]
    ax.set_ylim(0,top)
    #plt.title('time trapped, trap_num = {0}'.format(num))

plt.show()
