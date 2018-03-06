import scipy
import scipy.stats
import math
import pandas as pd


def rotate_vecs(x,y,angle):
    xrot = x*scipy.cos(angle) - y*scipy.sin(angle)
    yrot = x*scipy.sin(angle) + y*scipy.cos(angle)
    return xrot, yrot


def shift_and_rotate(p, shift, angle):
    p_vec = scipy.array(p)
    shift_vec = scipy.array(shift)
    rot_mat = rotation_matrix(angle)
    if len(p_vec.shape) > 1:
        shift_vec = scipy.reshape(shift_vec,(2,1))
    return scipy.dot(rot_mat, p_vec - shift_vec)


def rotation_matrix(angle):
    A = scipy.array([
        [scipy.cos(angle), -scipy.sin(angle)],
        [scipy.sin(angle),  scipy.cos(angle)]
        ])
    return A


def create_circle_of_sources(number,radius,strength):
    location_list = []
    for i in range(number):
        angle = i*(2.0*scipy.pi)/number
        x = radius*scipy.cos(angle)
        y = radius*scipy.sin(angle)
        location_list.append((x,y))
    strength_list = [strength for x in location_list]
    return location_list, strength_list


def create_grid_of_sources(x_num, y_num, x_range, y_range,  strength):
    x_vals = scipy.linspace(x_range[0], x_range[1], x_num)
    y_vals = scipy.linspace(y_range[0], y_range[1], y_num)
    location_list = [(x,y) for x in x_vals for y in y_vals]
    strength_list = [strength for x in location_list]
    return location_list, strength_list


def distance(p,q):
    return scipy.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)


def unit_vector(x,y):
    v_mag = scipy.sqrt(x**2 + y**2)
    if type(v_mag) == scipy.ndarray:
        mask = v_mag > 0
        x_unit = scipy.zeros(x.shape)
        y_unit = scipy.zeros(y.shape)
        x_unit[mask] = x/v_mag
        y_unit[mask] = y/v_mag
    else:
        if (v_mag > 0):
            x_unit = x/v_mag
            y_unit = y/v_mag
        else:
            x_unit = 0.0
            y_unit = 0.0
    return x_unit, y_unit


def logistic(x,x0,k):
    return 1.0/(1.0 + scipy.exp(-k*(x-x0)))

def par_perp(u,v):
    #Returns the components of u parallel to and perpendicular to v, as cartesian vectors.
    par = (scipy.inner(u,v))/(scipy.inner(v,v))*v
    perp = u - par
    return par,perp
def test_function(x):
    return x


def fit_von_mises(heading_data):
    #Returns tuple (mean, kappa) of the von mises fit to inputted data.
    #Structure of input: heading data is dict with
    #key1: 'angles' : 1xn array of angles
    #key2: 'counts' : mxn array of angle counts
    angles = heading_data['angles']
    counts = heading_data['counts']
    #Create a weighted histogram where each row gets weight inversely proportional to
    counts[1,:] = 5*counts[1,:]
    counts = scipy.sum(counts,0)
    #Fit the histogram to a von mises
    draws = tuple(scipy.repeat(angles[i],counts[i]) for i in range(len(angles)))
    headings = scipy.concatenate(draws)
    #import matplotlib.pyplot as plt
    #plt.subplot(111,polar=True);plt.hist(headings);plt.show()
    (kappa_est,mu_est,scale) = scipy.stats.vonmises.fit(headings,fscale=1)
    #raw_input('Done?')
    return mu_est, kappa_est

wind_data_file = '10_26_wind_vectors.csv'

def process_wind_data(wind_data_file):
    #Takes in a csv file and outputs wind_angle,wind_speed,wind_dt
    wind_df = pd.read_csv('/home/annie/work/programming/odor_tracking_sim/data_files/'+wind_data_file)
    cols = list(wind_df.columns.values)
    secs,degs,mph = tuple(wind_df[col].as_matrix() for col in cols)
    #Convert min to seconds
    times = 60.*secs
    wind_dt = times[1]-times[0]
    #Convert degrees to radians and switch to going vs coming
    wind_angle = (scipy.radians(degs)+scipy.pi)%(2*scipy.pi)
    #Convert mph to meters/sec
    wind_speed = mph*(1/3600.)*1609.34
    return wind_angle,wind_speed,wind_dt

wind_angle,wind_speed,wind_dt = process_wind_data(wind_data_file)
#print wind_angle, wind_speed,wind_dt
#import matplotlib.pyplot as plt
#plt.subplot(111,polar=True)
#plt.hist(wind_angle);plt.show()
#raw_input('Done?')

# Testing/development
# --------------------------------------------------------------------
if __name__ == '__main__':

    pass
