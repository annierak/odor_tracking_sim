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
trap_counts = swarm.get_trap_counts()
for i in range(len(trap_num_list)):
    ax = trap_axes[i]
    ax.set_ylim(0,top)
    xmin,xmax = ax.get_xlim()
    ax.text(xmin+(xmax-xmin)/2,top/2,str(trap_counts[i]))
    #plt.title('time trapped, trap_num = {0}'.format(num))


ax7 = plt.subplot2grid((3,4),(1,1),polar=True)
headings = swarm.param['initial_heading']
heading_dist = swarm.param['initial_heading_dist']
heading_mean = heading_dist.mean()
(n,bins,patches) = ax7.hist(headings,bins=100)
ax7.set_yticks([])
r = max(n)
ax7.set_ylim((0,r))
ax7.arrow(heading_mean,0,0,0.5*r, width = 0.015,edgecolor = 'red',
facecolor = 'red', lw = 2, zorder = 1,head_width=0.2,head_length=30)
#ax7.arrow(0,0,r*scipy.cos(heading_mean),r*scipy.sin(heading_mean),color='red',lw=4)

ax8 = plt.subplot2grid((3,4),(1,2))
angles = scipy.linspace(heading_mean-scipy.pi,heading_mean+scipy.pi,400)
ax8.plot(angles,heading_dist.pdf(angles))
ax8.set_xlabel('Variance = '+str(round(heading_dist.var(),3)))

plt.show()
