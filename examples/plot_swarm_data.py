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
ax1 = plt.subplot2grid((3,4),(1,3))
ax2 = plt.subplot2grid((3,4),(0,2))
ax3 = plt.subplot2grid((3,4),(0,1))
ax4 = plt.subplot2grid((3,4),(1,0))
ax5 = plt.subplot2grid((3,4),(2,1))
ax6 = plt.subplot2grid((3,4),(2,2))
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

trap_counts = swarm.get_trap_counts()
ax7 = plt.subplot2grid((3,4),(1,1),colspan=2,polar=True)
borders = -scipy.pi/(len(trap_num_list))+2*scipy.pi/(len(trap_num_list))*scipy.linspace(0,len(trap_num_list)-1,len(trap_num_list))
print borders/scipy.pi
ax7.bar(borders,trap_counts,align='edge',width=2*scipy.pi/(len(trap_num_list)))

plt.show()
