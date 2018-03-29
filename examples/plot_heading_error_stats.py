#This bit creates a new figure with the fly's heading error distribution.

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

plt.figure()
distf = swarm.param['heading_error_dist']
plt.plot(scipy.radians(22.5)*scipy.linspace(-2*scipy.pi,2*scipy.pi,181),scipy.radians(22.5)*distf.pdf(scipy.linspace(-2*scipy.pi,2*scipy.pi,181)))
plt.xlim(-scipy.pi,scipy.pi)
plt.show()
