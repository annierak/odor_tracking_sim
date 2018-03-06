from __future__ import print_function
import scipy
import scipy.stats
import time

from utility import unit_vector
from utility import rotate_vecs
from utility import distance
from utility import par_perp
from utility import fit_von_mises



class BasicSwarmOfFlies(object):

    """
    New vectorized (faster) fly model.

    """

    DefaultSize = 500
    DefaultParam = {
            'initial_heading_dist': scipy.stats.uniform(0,2*scipy.pi), #continuous_distribution object
            'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(DefaultSize,))),
            'x_start_position'    : scipy.zeros((DefaultSize,)),
            'y_start_position'    : scipy.zeros((DefaultSize,)),
            'heading_error_std'   : scipy.radians(10.0),
            'flight_speed'        : scipy.full((DefaultSize,), 0.7),
            'release_time'        : scipy.full((DefaultSize,), 0.0),
            'release_time_constant': None,
            'cast_interval'       : [60.0, 1000.0],
            'wind_slippage'       : (0.0,0.0), #(// to fly's path, perp to fly's path)
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': 0.004
                },
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': 0.002,  # detection probability/sec of exposure
                }
            }

    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


    def __init__(self,wind_field,param={},start_type='fh'): #default start type is fixed heading
        self.param = dict(self.DefaultParam)
        self.param.update(param)
        self.check_param()
        self.x_position = self.param['x_start_position']
        self.y_position = self.param['y_start_position']

        if(not(self.param['heading_data']==None)):
            heading_data = self.param['heading_data']
            (mean,kappa) = fit_von_mises(heading_data)
            self.param['initial_heading_dist'] = scipy.stats.vonmises(loc=mean,kappa=kappa)
            self.param['initial_heading'] = scipy.random.vonmises(mean,kappa,(self.param['swarm_size'],))
        self.x_velocity = self.param['flight_speed']*scipy.cos(self.param['initial_heading'])
        self.y_velocity = self.param['flight_speed']*scipy.sin(self.param['initial_heading'])
        self.mode = scipy.full((self.size,), self.Mode_StartMode, dtype=int)
        self.heading_error = scipy.zeros((self.size,))
        self.t_last_cast = scipy.zeros((self.size,))

        self.increments_until_turn = scipy.ones((self.size,)) #This is for the Levy walk option.
        #self.uniform_directions_pool = scipy.radians(scipy.random.uniform(0.0,360.0,(100,)))#Again for the Levy walk option, to save time drawing.
        #self.increments_pool = scipy.stats.lognorm.rvs(0.25,size=100,scale=
        #(300/3.0)/0.25*
        #scipy.exp(0))

        cast_interval = self.param['cast_interval']
        self.dt_next_cast = scipy.random.uniform(cast_interval[0], cast_interval[0], (self.size,))
        self.cast_sign = scipy.random.choice([-1,1],(self.size,))

        self.parallel_coeff,self.perp_coeff = self.param['wind_slippage']
        self.par_wind,self.perp_wind = self.get_par_perp_comps(0.,wind_field)
        #^This is the set of 2 x time arrays of the components of each fly's velocity par/perp to wind
        self.ever_tracked = scipy.full((self.size,), False, dtype=bool) #Bool that keeps track if the fly ever plume tracked (false=never tracked)
        self.trap_num = scipy.full((self.size,),-1, dtype=int)
        self.in_trap = scipy.full((self.size,), False, dtype=bool)
        self.x_trap_loc = scipy.zeros((self.size,))
        self.y_trap_loc = scipy.zeros((self.size,))
        self.t_in_trap = scipy.full((self.size,),scipy.inf)
        self.start_type = start_type #Either 'fh' (fixed heading) or 'rw' (random walk)
        if start_type=='rw':
            self.rw_dist = scipy.stats.lognorm(0.25,scale=1)
        else:
            self.rw_dist = None

    def check_param(self):
        """
        Check parameters - mostly just that shape of ndarrays match
        """
        if scipy.ndim(self.param['initial_heading'].shape) > 1:
            raise(ValueError, 'initial_heading must have ndim=1')

        equal_shape_list = ['x_start_position','y_start_position','flight_speed','release_time']
        for item in equal_shape_list:
            if self.param[item].shape != self.param['initial_heading'].shape:
                raise(ValueError, '{0}.shape must equal initial_heading.shape'.format(item))


    @property
    def size(self):
        return self.param['initial_heading'].shape[0]


    def update(self, t, dt, wind_field, odor_field):
        """
        Update fly swarm one time step.
        """

        # Get masks for selecting fly based on mode
        mask_release = t > self.param['release_time']
        mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
        mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
        mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)

        #Keep track of which flies have never tracked
        self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode

        # Get odor value and wind vectors at current position and time
        odor = odor_field.value(t,self.x_position,self.y_position)
        x_wind, y_wind = wind_field.value(t,self.x_position, self.y_position)
        x_wind_unit, y_wind_unit = unit_vector(x_wind, y_wind)
        wind_uvecs = {'x': x_wind_unit,'y': y_wind_unit}

        # Update state for flies detectoring odor plumes
        masks = {'startmode': mask_startmode, 'castfor': mask_castfor}
        self.update_for_odor_detection(dt, odor, wind_uvecs, masks)

        # Update state for files losing odor plume or already casting.
        masks = {'flyupwd': mask_flyupwd, 'castfor': mask_castfor}
        self.update_for_odor_loss(t, dt, odor, wind_uvecs, masks)

        # Udate state for flies in traps
        self.update_for_in_trap(t, odor_field)

        # Update position based on mode and current velocities
        mask_trapped = self.mode == self.Mode_Trapped

        self.update_positions(mask_release,mask_trapped,mask_startmode,dt)

        #Original: apply wind slippage to all flies
        #mask_move = mask_release & (~mask_trapped)
        #ws = self.param['wind_slippage'][0]
        #self.x_position[mask_move] += dt*ws*x_wind[mask_move]
        #self.y_position[mask_move] += dt*ws*y_wind[mask_move]

        #Michael's idea 2/27/18: apply wind slippage according to c_1*(component
        #parallel to fly's velocity) + c2*(component pe rp to fly's velocity)

        self.update_par_perp_comps(t,wind_field,mask_release,mask_startmode)
        #par/perp comps for flys not {released and in fly mode} are set to 0.
        c1 = self.parallel_coeff
        c2 = self.perp_coeff
        self.x_position += dt*(c1*self.par_wind[0,:]+c2*self.perp_wind[0,:])
        self.y_position += dt*(c1*self.par_wind[1,:]+c2*self.perp_wind[1,:])


    def update_for_odor_detection(self, dt, odor, wind_uvecs, masks):
        """
         Update simulation for odor detection
         * Find flies in StartMode and CastForOdor modes where the odor value >= upper threshold.
         * Test if they detect odor (roll dice and compare with dection probabilty).
         * If they do detect odor change their  mode to FlyUpWind.
         * set x and y velocities to upwind at speed
        """
        x_wind_unit = wind_uvecs['x']
        y_wind_unit = wind_uvecs['y']
        mask_startmode = masks['startmode']
        mask_castfor = masks['castfor']

        mask_gt_upper = odor >= self.param['odor_thresholds']['upper']
        mask_candidates = mask_gt_upper & (mask_startmode | mask_castfor)
        dice_roll = scipy.full((self.size,),scipy.inf)
        dice_roll[mask_candidates] = scipy.rand(mask_candidates.sum())

        # Convert probabilty/sec to probabilty for time step interval dt
        odor_probability_upper = 1.0 - (1.0 - self.param['odor_probabilities']['upper'])**dt
        mask_change = dice_roll < odor_probability_upper
        self.mode[mask_change] = self.Mode_FlyUpWind

        # Compute new heading error for flies which change mode
        heading_error_std = self.param['heading_error_std']
        self.heading_error[mask_change] = heading_error_std*scipy.randn(mask_change.sum())

        # Set x and y velocities for the flies which just changed to FlyUpWind.
        x_unit_change, y_unit_change = rotate_vecs(
                x_wind_unit[mask_change],
                y_wind_unit[mask_change],
                self.heading_error[mask_change]
                )
        speed = self.param['flight_speed'][mask_change]
        self.x_velocity[mask_change] = -speed*x_unit_change
        self.y_velocity[mask_change] = -speed*y_unit_change

    def update_for_odor_loss(self, t, dt, odor, wind_uvecs, masks):
        """
         Update simulation for flies which lose odor or have lost odor and are
         casting.
         * Find flies in FlyUpWind mode where the odor value <= lower threshold.
         * Test if they lose odor (roll dice and compare with probabilty).
         * If they lose odor change mode to CastForOdor.
         * Update velocties for flies in CastForOdor mode.
        """

        x_wind_unit = wind_uvecs['x']
        y_wind_unit = wind_uvecs['y']
        mask_flyupwd = masks['flyupwd']
        mask_castfor = masks['castfor']

        mask_lt_lower = odor <= self.param['odor_thresholds']['lower']
        mask_candidates = mask_lt_lower & mask_flyupwd
        dice_roll = scipy.full((self.size,),scipy.inf)
        dice_roll[mask_candidates] = scipy.rand(mask_candidates.sum())

        # Convert probabilty/sec to probabilty for time step interval dt
        odor_probability_lower = 1.0 - (1.0 - self.param['odor_probabilities']['lower'])**dt
        mask_change = dice_roll < odor_probability_lower
        self.mode[mask_change] = self.Mode_CastForOdor

        # Lump together flies changing to CastForOdor mode with casting flies which are
        # changing direction (e.g. time to make cast direction change)
        mask_change |= mask_castfor & (t > (self.t_last_cast + self.dt_next_cast))

        # Computer new heading errors for flies which change mode
        self.heading_error[mask_change] = self.param['heading_error_std']*scipy.randn(mask_change.sum())

        # Set new cast intervals and directions for flies chaning to CastForOdor or starting a new cast
        cast_interval = self.param['cast_interval']
        self.dt_next_cast[mask_change] = scipy.random.uniform(
                cast_interval[0],
                cast_interval[0],
                (mask_change.sum(),)
                )
        self.t_last_cast[mask_change] = t
        self.cast_sign[mask_change] = scipy.random.choice([-1,1],(mask_change.sum(),))

        # Set x and y velocities for new CastForOdor flies
        x_unit_change, y_unit_change = rotate_vecs(
                x_wind_unit[mask_change],
               -y_wind_unit[mask_change],
                self.heading_error[mask_change]
                )
        speed = self.param['flight_speed'][mask_change]
        self.x_velocity[mask_change] = self.cast_sign[mask_change]*speed*x_unit_change
        self.y_velocity[mask_change] = self.cast_sign[mask_change]*speed*y_unit_change


    def update_for_in_trap(self, t, odor_field):
        """
         Update simulation for flies in traps.
         * If flies are in traps. If so record trap info and time.
        """
        for trap_num, trap_loc in enumerate(odor_field.param['source_locations']):
            dist_vals = distance((self.x_position, self.y_position),trap_loc)
            mask_trapped = dist_vals < odor_field.param['trap_radius']
            self.mode[mask_trapped] = self.Mode_Trapped
            self.trap_num[mask_trapped] = trap_num
            self.x_trap_loc[mask_trapped] = trap_loc[0]
            self.y_trap_loc[mask_trapped] = trap_loc[1]
            self.x_velocity[mask_trapped] = 0.0
            self.y_velocity[mask_trapped] = 0.0

            # Get time stamp for newly trapped flies
            mask_newly_trapped = mask_trapped & (self.t_in_trap == scipy.inf)
            self.t_in_trap[mask_newly_trapped] = t


    def get_time_trapped(self,trap_num=None,straight_shots=False):
        #adjusted this function to isolate flies that went straight to traps
        mask_trapped = self.mode == self.Mode_Trapped
        if straight_shots:
            mask_trapped = mask_trapped & scipy.logical_not(self.ever_tracked)
        if trap_num is None:
            return self.t_in_trap[mask_trapped]
        else:
            mask_trapped_in_num = mask_trapped & (self.trap_num == trap_num)
            return self.t_in_trap[mask_trapped_in_num]

    def get_trap_nums(self):
        mask_trap_num_set = self.trap_num != -1
        trap_num_array = scipy.unique(self.trap_num[mask_trap_num_set])
        trap_num_array.sort()
        return list(trap_num_array)

    def get_trap_counts(self):
        mask_trap_num_set = self.trap_num != -1
        (trap_num_array,trap_counts)=scipy.unique(self.trap_num[mask_trap_num_set],return_counts = True)
        return trap_counts

    def update_positions(self,mask_release,mask_trapped,mask_startmode,dt):
        if self.start_type=='fh' or sum(mask_startmode)<1.:
            mask_move = mask_release & (~mask_trapped)
            self.x_position[mask_move] += dt*self.x_velocity[mask_move]
            self.y_position[mask_move] += dt*self.y_velocity[mask_move]
        elif self.start_type=='rw':
            #The flies who are not in start_mode move the same way
            mask_move = mask_release & (~mask_trapped) & (~mask_startmode)
            self.x_position[mask_move] += dt*self.x_velocity[mask_move]
            self.y_position[mask_move] += dt*self.y_velocity[mask_move]
            '''Option 1: path lengths are chosen from a heavy-tailed distribution'''
            '''For those in startmode, the x step and y step of each fly is chosen from lognormal (heavy-tailed) distribution
            right now the distribution is manually set up so that moving faster than peak velocity (1.8 m/s) happens with close to
            0 probability: the distribution is lognormal with sigma = 0.25 and mu = 0, and then *(1.8/2.0)*timestep'''
            sigma = 0.25
            mu = 0
            scaling_factor = (1.8/2.0)*dt #So that 1.8 m/s is the fastest it ever flies
            draws = sum(mask_startmode)
            self.x_position[mask_startmode] += scaling_factor*scipy.random.choice([1,-1],
                size=draws)*scipy.stats.lognorm.rvs(sigma,size=draws,scale=scipy.exp(mu))
            self.y_position[mask_startmode] += scaling_factor*scipy.random.choice([1,-1],
                size=draws)*scipy.stats.lognorm.rvs(sigma,size=draws,scale=scipy.exp(mu))
        elif self.start_type=='cvrw':
            start = time.time()
            #All flies update position according to velocity, including start_mode flies
            mask_move = mask_release & (~mask_trapped)
            self.x_position[mask_move] += dt*self.x_velocity[mask_move]
            self.y_position[mask_move] += dt*self.y_velocity[mask_move]

            '''Option 2: Durations of a given direction are chosen from a heavy-tailed distribution
            Draw from same kind of distribution as above, but round up for discrete time steps.
            Distribution is lognormal with sigma = 0.5 and mu = 0, and then *(300/3.0)/timestep,
            which makes the max occuring duration around 300 s= 5 min.'''
            sigma = 0.5
            mu = 0.
        #Every fly

        #Every startmode fly has one time step less left in current direction
            self.increments_until_turn[mask_startmode&mask_move] -=1
            #print(self.increments_until_turn[mask_startmode&mask_move])
            #Flies whose time is up get assigned a new direction --> x and y velocity
            mask_redraw = mask_move&mask_startmode & (self.increments_until_turn == 0)

            cp = time.time()
            draws = sum(mask_redraw)
            if draws>0:
                #directions = scipy.random.choice(self.uniform_directions_pool,draws)
                sines, cosines = scipy.zeros(draws),scipy.zeros(draws)
                for x in xrange(draws):
                    direction = scipy.stats.uniform.rvs(0.0,2*scipy.pi)
                    sines[x],cosines[x] = scipy.sin(direction),scipy.cos(direction)
                #cp1 = time.time();print('cp1 :'+str(cp1-cp))
                #self.x_velocity[mask_redraw]=self.param['flight_speed'][0]*scipy.cos(directions)
                self.x_velocity[mask_redraw]=self.param['flight_speed'][0]*cosines
                #cp2 = time.time();print(cp2-cp1)
                self.y_velocity[mask_redraw]=self.param['flight_speed'][0]*sines
                #self.y_velocity[mask_redraw]=self.param['flight_speed'][0]*scipy.sin(directions)
                #cp3 = time.time();print(cp3-cp2)
                #and get assigned a new interval count until they change direction again
                self.increments_until_turn[mask_redraw] = scipy.floor(#scipy.random.choice(self.increments_pool,sum(mask_redraw))
                    scipy.stats.lognorm.rvs(sigma,size=draws,scale=
                    (300/3.0)/dt*
                    scipy.exp(mu)))
                #cp4 = time.time(); print(cp4-cp3)
                #print(cp4-cp)


    def get_par_perp_comps(self,t,wind_field):
        x_wind, y_wind = wind_field.value(t,self.x_position, self.y_position)
        wind = scipy.array([x_wind,y_wind])
        velocity = scipy.array([self.x_velocity,self.y_velocity])
        par_vec = scipy.zeros(scipy.shape(velocity))
        perp_vec = scipy.zeros(scipy.shape(velocity))
        for i in range(scipy.size(velocity,1)):
            u,v = velocity[:,i],wind[:,i]
            par,perp = par_perp(v,u)
            par_vec[:,i],perp_vec[:,i] = par,perp
        return par_vec,perp_vec
    def update_par_perp_comps(self,t,wind_field,mask_release,mask_startmode):
        #Check if the wind field has changed since last time-step, if so, re-compute get_par_perp_comps
        if wind_field.evolving:
            self.par_wind,self.perp_wind = self.get_par_perp_comps(t,wind_field)
        #Set the flys who have been release and who are not in start_mode to zero par and zero perp
        self.par_wind[:,mask_release&~mask_startmode] = 0.
        self.perp_wind[:,mask_release&~mask_startmode] = 0.
