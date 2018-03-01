import scipy
import math
import matplotlib.pyplot as plt
import cPickle as pickle

with open('kappa2_beta50_departure_upperprob0002.pkl','r') as f:
    swarm = pickle.load(f)
print(len(swarm.x_position))
