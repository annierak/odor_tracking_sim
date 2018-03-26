import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility

f = sys.argv[1]
last_spot = sys.argv[2]

input_file = f+'.pkl'

with open(input_file,'r') as f:
    (swarm,wind_field) = pickle.load(f)


num_bins = 40

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


peak_counts = scipy.zeros(len(trap_axes))
rasters = []
print(trap_num_list)
for i in trap_num_list:
    t = swarm.get_time_trapped(i)
    ax = trap_axes[i]

    (n, bins, patches) = ax.hist(t,num_bins,range=(0,7800))
    peak_counts[i]=max(n)
    t1 = swarm.get_time_trapped(i,straight_shots=True)
    ax.hist(t1,bins,color='red')
    utility.customaxis(ax)
    ax.set_xlabel('(s)',horizontalalignment='left',x=1.0)
    ax.set_ylabel('trap:{0}'.format(i))

    #This is the raster plot option:
    # r = ax.eventplot(t,colors=['green'])[0]
    # rasters.append(r)
    #Add on a plot for the flies that went straight into the trap
print(len(rasters))
top = max(peak_counts)
trap_counts = swarm.get_trap_counts()
for i,num in enumerate(trap_num_list):
    ax = trap_axes[num]
    ax.set_ylim(0,top)
    # rasters[i].set_lineoffset(top-1 )
    # rasters[i].set_linelength(3)
    xmin,xmax = ax.get_xlim()
    ax.text(xmin+(xmax-xmin)/2,top/2,str(trap_counts[i]),color='maroon',size =20)
    #plt.title('time trapped, trap_num = {0}'.format(num))


ax7 = plt.subplot2grid((3,4),(1,1),polar=True)
headings = swarm.param['initial_heading']
heading_dist = swarm.param['initial_heading_dist']
heading_mean = heading_dist.mean()
print(heading_mean)
(n,bins,patches) = ax7.hist(headings,bins=100)
ax7.set_yticks([])
r = max(n)
ax7.set_ylim((0,r))
ax7.set_xlabel('Initial heading distr')
if not(heading_dist.dist.name=='uniform'):
    ax7.annotate("", xy=(heading_mean,r*0.75), xytext=(0, 0), arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1.",edgecolor = 'red',
     facecolor = 'red',linewidth=5.))
#ax7.arrow(0,0,r*scipy.cos(heading_mean),r*scipy.sin(heading_mean),color='red',lw=4)


ax8 = plt.subplot2grid((3,4),(1,2))

if last_spot == 'pdf':
    #The below will plot the pdf of the heading distribution
    angles = scipy.linspace(heading_mean-scipy.pi,heading_mean+scipy.pi,400)
    ax8.plot(angles,heading_dist.pdf(angles))
    try:
        last_bit = '(Kappa = '+str(round(heading_dist.kwds['kappa'],4))+') '
    except KeyError:
        last_bit = ''
    ax8.set_xlabel('Initial heading distr. ''(Variance = '+str(round(heading_dist.var(),3))+
    last_bit+')')
    ax8.set_ylim(0,0.6)
    ax8.set_xlim(heading_mean-scipy.pi,heading_mean+scipy.pi)
elif last_spot == 'dep':
    #The below will plot the time course of the departures from the release point
    release_times = swarm.param['release_time'] - swarm.param['release_delay']
    ax8.hist(release_times,bins=100)
    ax8.set_xlabel('Release Time Course (Time Constant= '+str(round(swarm.param['release_time_constant'],3))+')      (s)')
    ax8.set_xlim((0,1000))
    #ax8.set_xlabel('(s)',horizontalalignment='left',x=1.0)
utility.customaxis(ax8)

#Plot the wind direction
if wind_field.evolving:
    wind_angle_0 = wind_field.angle[0]
else:
    wind_angle_0 = wind_field.angle


ax9 = plt.subplot2grid((3,4),(0,3),polar=True)
ax9.set_yticks([])
ax9.set_ylim((0,1))
# ax9.arrow(0,0,wind_angle_0,0.5,  edgecolor = 'teal',
# facecolor = 'blue' , alpha = 0.5, width = 0.015,
#                  lw = 2, zorder = 5)
ax9.annotate("", xy=(wind_angle_0,0.75), xytext=(0, 0), arrowprops=dict(arrowstyle="->,head_width=0.5,head_length=1.",edgecolor = 'teal',
 facecolor = 'blue',linewidth=5.))
ax9.set_xlabel('(Initial) Wind Direction')

plt.show()
