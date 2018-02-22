import numpy as np
import math
pi = math.pi
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import vonmises

#theta_i = heading
#n = number of traps
#phi_0 = wind angle
#epsilon = plume_width
#r = radius from release to trap

def dist_to_trap(theta_i,r,n,phi_0,epsilon,redraw=False):
#For a given initial heading, wind angle, trap count, plume width angle, and trap radius, return the distance to the trap
#at which the fly will intersect the plume.
#If redraw=True, then return a k-tuple of distances with one entry for each of k plumes intersected
    phi = compute_phi(theta_i,phi_0,epsilon)
    theta_T = where_caught(theta_i,n,phi)
    if theta_i == theta_T:
        return 0
    elif np.isnan(theta_T):
            return np.nan
    else:
        #print 'theta_i = '+str(theta_i)
        #print 'theta_T = '+str(theta_T)
        #print 'theta_i-theta_T = '+str(theta_i-theta_T)
        #print 'phi-theta_i = '+str(phi-theta_i)
        #print 'theta_i-theta_T = '+str(math.sin(theta_i-theta_T))
        #print 'sin(phi-theta_i)= '+str(math.sin(phi-theta_i))
        d = r*((math.sin(theta_i-theta_T))/(math.sin(phi-theta_i)))
        return  d

def compute_naive_bin(theta_i,n):
    #Returns the bin of the traps the heading is between (0-indexed)
    trap_interval =  (theta_i - (theta_i%(2*(math.pi)/n)))/(2*(math.pi)/n)
    return trap_interval % 6
#Tested

def round_down_angle(theta_i,n):
    #Returns the lower boundary of theta_i in units of 2pi/n
    angle_in_units = (theta_i)/(2*(math.pi)/n)
    lower_bound = math.floor(angle_in_units)*(2*pi/n)
    return lower_bound
def round_up_angle(theta_i,n):
    #Returns the upper boundary of theta_i in units of 2pi/n
    angle_in_units = (theta_i)/(2*(math.pi)/n)
    upper_bound = math.ceil(angle_in_units)*(2*pi/n)
    return upper_bound
def pure_round_angle(theta_i,n):
    #Same as above but rounds both ways, traditional round
    angle_in_units = (theta_i)/(2*(math.pi)/n)
    rounded = round(angle_in_units)*(2*pi/n)
    return rounded
def smaller_diff(a,b):
    #Returns the smaller angle between two angles
    diff = b -a
    return (diff + pi) % (2*pi) - pi
def is_between(angle_1,angle_2,test_angle):
    #True/false as to whether test_angle lies on the shorter arc between angle_1 and angle_2
    if np.sign(smaller_diff(angle_1,angle_2)) !=np.sign(smaller_diff(angle_1,test_angle)):
        return False
    elif smaller_diff(angle_1,angle_2)<=0:
        return smaller_diff(angle_1,angle_2)<smaller_diff(angle_1,test_angle)
    elif smaller_diff(angle_1,angle_2)>0:
        return smaller_diff(angle_1,angle_2)>smaller_diff(angle_1,test_angle)
    else:
        return 'Error'

def where_caught(theta_i,n,phi,redraw=False):
#Finds the angle of the trap the fly ends up at for given heading theta_i, trap count, and wind angle phi
#If redraw=True, return a tuple of the angles the fly could end up at
    if compute_naive_bin(theta_i,n)==compute_naive_bin(phi+pi,n) or phi==theta_i:
    #First, the case where the heading is in the compartment where nothing is detected or exactly in the wind angle
        return np.nan
    elif not(redraw):
        #There are 3 relevant traps, which we'll call T_A, T_B, T_C.
        #T_C is 2pi/n clockwise of T_A
        T_C = round_down_angle(phi+pi,n)
        T_A = round_up_angle(phi+pi,n)
        T_B = pure_round_angle(phi,n)
        if theta_i in [T_C,T_A,T_B]:
            destination = theta_i
        else:
            if phi %(2*pi) < T_B %(2*pi):
                first_trans = phi
                second_trans = T_B
            else:
                first_trans = T_B
                second_trans = phi
            #Figure out which of the three fly will go to
            if is_between(T_A,first_trans,theta_i):
                destination = T_A
            elif is_between(first_trans,second_trans,theta_i):
                destination = T_B
            else:
                 destination = T_C
        return pure_round_angle(destination%(2*pi),n)
    else:

def compute_phi(theta_i,phi_0,epsilon):
#Determines for a given initial heading, plume width, and wind angle, phi, the angle between the plume intersection
#the trap, and east (clockwise)
    if (theta_i-phi_0)%(2*math.pi)<=(math.pi):
        phi = phi_0-epsilon
    else :
        phi = phi_0+epsilon
    return phi
#Tested for one value of phi_0--check a few otherss

def compute_dist_vec(heading_vec,r,n,phi_0,epsilon):
    #Applies dist_to_trap to a vector of headings, with specified r,n,phi_0,epsilon
    distance_vec = np.array(list(map(lambda x: dist_to_trap(x,r,n,phi_0,epsilon),heading_vec)))
    return distance_vec

def compute_prob_vec(distance_vec):
    #Applies decreasing probability of recapture as a function of distance
    #function given by p(x)=e^(-x)
    return np.array(list(map(lambda x: math.exp(-x),distance_vec)))

def where_caught_vec(theta_i_vec,n,phi):
    #Vectorizes where_caught and converts "None" to NaN
    return np.array(list(map(lambda x:where_caught(x,n,phi),theta_i_vec)))

def get_trap_counts(prob_vec,traps_vec,r,n,phi,epsilon):
    #Returns a vector n long with the fraction of initial trajectories ending up at each trap
    #This can be reconfigured to stop duplicating computations if it's useful
    trap_counts = np.zeros(n)
    for trap in np.unique(traps_vec):
        if np.isnan(trap):
            pass
        else:
            probs = [prob_vec[i] for i in range(len(traps_vec)) if traps_vec[i]==trap]
            trap_counts[int(trap/(2*pi/n))]=np.sum(probs)
    trap_counts = trap_counts/len(prob_vec)
    return trap_counts


def vm_cdf_diff(inputs,heading_mean,kappa):
    heading_cdf= lambda theta : vonmises.cdf(theta,loc=heading_mean,kappa=kappa)
    scale = inputs[1]-inputs[0]
    inputs_shifted = inputs-scale
    outputs = np.zeros(len(inputs))
    for i in range(len(inputs)):
        outputs[i] = heading_cdf(inputs[i]) - heading_cdf(inputs_shifted[i])
    return outputs
