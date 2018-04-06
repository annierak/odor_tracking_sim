#For inputted pickle, displays the surging error distribution
#AND the distribution of times spent in the plumes (plume bout lengths)

import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import sys
import odor_tracking_sim.utility as utility
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

f = sys.argv[1]

input_file = f+'.pkl'

with open(input_file,'r') as f:
    (swarm,wind_field) = pickle.load(f)

#Surging error distribution
plt.figure()
plt.subplot(121)
distf = swarm.param['surging_error_dist']
plt.plot(swarm.param['surging_error_std']*scipy.linspace(-2*scipy.pi,2*scipy.pi,181),scipy.radians(22.5)*distf.pdf(scipy.linspace(-2*scipy.pi,2*scipy.pi,181)))
plt.xlim(-scipy.pi,scipy.pi)
plt.title('Surging Error Distribution')
l= plt.ylabel(r"$P(\theta)$");l.set_rotation(0)
plt.xlabel('Surging Error '+r"($\theta$)")
#Plume bout length distribution
plt.subplot(122)
plume_bout_lengths = swarm.plume_bout_lengths[(swarm.plume_bout_lengths>0)]
#convert timesteps to mins
plume_bout_lengths = plume_bout_lengths*swarm.dt/60.
plt.hist(plume_bout_lengths,bins=40)
plt.title('Time Spent in Plume')
plt.ylabel('Frequency')
plt.xlabel('Time (min)')
plt.show()
