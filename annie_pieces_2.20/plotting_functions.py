
def plot_probability_heading_dist(phi,heading_mean='Default',
                               r=1.,n=6,kappa=2.,epsilon=0.,num_flies=100,fig_num=1,distr='von Mises'):

 #generate the randomly drawn headings
 if heading_mean=='Default':
     heading_mean=phi

 inputs = np.linspace(0,2*pi,num_flies)

 cdf_output = vm_cdf_diff(inputs,heading_mean,kappa)

 #set up a color scaling scheme for the distribution
 darkest_rgb = np.array([0.,128.,128.])/255.
 scalars = cdf_output/max(cdf_output) #scalars is a sign of how dark it is (higher prob)
 white = np.array([1.,1.,1.])
 colors = np.zeros((num_flies,3))
 for j in range(num_flies):
     colors[j,:] = white+scalars[j]*(darkest_rgb-white)

 #call plot_probability_uniform and add on to this figure
 plot_probability_unif(phi,r,n,epsilon,num_flies,fig_num)
 fig=plt.figure(fig_num)
 ax_list = fig.axes


 for i in range(n):
     ax = ax_list[i]
     dist_fill(np.linspace(0,2*pi,num_flies),0.25*np.ones(num_flies),colors,ax)



def plot_probability_unif(phi,r=1.,n=6,epsilon=0.,num_flies=100,fig_num=1):
 heading_vec = np.linspace(0,2*pi,num_flies)
 distance_vec = compute_dist_vec(heading_vec,r,n,phi,epsilon)
 prob_vec = compute_prob_vec(distance_vec)
 traps_vec=where_caught_vec(heading_vec,n,phi)
 trap_angles = np.arange(0,(n-1)*2*pi/n,2*pi/n)

 plt.figure(fig_num,figsize=(15,11))
 ax1 = plt.subplot2grid((6,2),(0,0))
 ax2 = plt.subplot2grid((6,2),(1,0))
 ax3 = plt.subplot2grid((6,2),(2,0))
 ax4 = plt.subplot2grid((6,2),(3,0))
 ax5 = plt.subplot2grid((6,2),(4,0))
 ax6 = plt.subplot2grid((6,2),(5,0))
 left_aspect = 0.2*(abs((max(heading_vec)-min(heading_vec))/(max(prob_vec)-min(prob_vec))))

 trap_axes = [ax1,ax2,ax3,ax4,ax5,ax6]

 ax7 = plt.subplot2grid((6,2),(0,1),rowspan=6)

 for i in range(len(trap_axes)):
     ax = trap_axes[i]
     ax.set_xticks(np.arange(0,2*pi,pi/4))
     ax.set_xticklabels(('0','$\\pi/4$','$\\pi/2$','$3\\pi/4$','$\\pi$','$5\\pi/4$','$3\\pi/2$','$7\\pi/4$'))
     ax.set_xlim(0,2*pi)
     ax.set_ylim(0,1)
     ax.set_aspect(left_aspect)
     ax.set_ylabel('Trap '+str(i+1),rotation=0)
     probs = [prob_vec[j] for j in range(len(traps_vec)) if traps_vec[j]==trap_angles[i]]
     headings =  [heading_vec[k] for k in range(len(traps_vec)) if traps_vec[k]==trap_angles[i]]
     if len(probs)>0:
         ax.plot(headings,probs,'o',color='red',markersize=6)


 trap_xs = np.array(list(map(lambda x:r*math.cos(x),trap_angles)))
 trap_ys = np.array(list(map(lambda x:r*math.sin(x),trap_angles)))
 ax7.plot(trap_xs,trap_ys,'o')
 counts = get_trap_counts(prob_vec,traps_vec,r,n,phi,epsilon)
 normalized_counts = 0.3*counts/max(counts) #So that the largest has radius 0.3
 for p in [
     patches.Circle((trap_xs[i], trap_ys[i]), normalized_counts[i]) for i in range(n)]:
     ax7.add_patch(p)
 for i in range(n):
     ax7.text(trap_xs[i], trap_ys[i],str(round(counts[i],2)),color='k')
 ax7.set_xlim(-r*1.5,r*1.5)
 ax7.set_ylim(-r*1.5,r*1.5)
 ax7.set_aspect(1.0/ax7.get_data_ratio())
 ax7.set_axis_off()
 ax7.arrow(0,0,0.5*math.cos(phi),0.5*math.sin(phi),head_width=0.05, head_length=0.1,fc='k')
