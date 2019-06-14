from __future__ import print_function
import scipy
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import time
from odor_models import FakeDiffusionOdorField
from pompy import models

from utility import unit_vector
from utility import rotate_vecs
from utility import distance
from utility import par_perp
from utility import fit_von_mises
from utility import cartesian_to_polar
from utility import speed_sigmoid_func
import utils_find_1st as utf1st



class BasicSwarmOfFlies(object):

    """
    New vectorized (faster) fly model.

    """

    DefaultSize = 500

    DefaultParam = {
            'dt'                  : 0.25,
            'initial_heading_dist': scipy.stats.uniform(0,2*scipy.pi), #continuous_distribution object
            'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(DefaultSize,))),
            'x_start_position'    : scipy.zeros((DefaultSize,)),
            'y_start_position'    : scipy.zeros((DefaultSize,)),
            'surging_error_dist'  : scipy.stats.laplace(loc=0.,scale=1.),
            'surging_error_std'   : scipy.radians(5.),
            'flight_speed'        : scipy.full((DefaultSize,), 0.7),
            'release_time'        : scipy.full((DefaultSize,), 0.0),
            'release_time_constant': None,
            'cast_interval'       : [1.0, 10.0],
            'wind_slippage'       : (0.0,0.0), #(// to fly's path, perp to fly's path)
            'odor_thresholds'     : {
                'lower': 0.002,
                'upper': 0.004
                },
            'cast_timeout':20,
            'odor_probabilities'  : {
                'lower': 0.9,    # detection probability/sec of exposure
                'upper': 0.002,  # detection probability/sec of exposure
                },
            'schmitt_trigger' : True,
            'low_pass_filter_length':3,
            'reset_distribution': scipy.stats.uniform(0,2*scipy.pi),
            'pure_advection': False,
            'airspeed_saturation': False
            }

    #If the fly is doing pure_advection, it is carried by the wind whenever it
    #is in startmode. This is implemented by assigning  wind_slippage to 0,0),
    #AND ignoring the x, y velocities when updating position when in startmode,
    #instead just updating position using the wind velocities. (see .update_positions)

    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


    def __init__(self,wind_field,traps,param={},start_type='fh',
    track_plume_bouts=False, #track each fly's plume interaction history
    track_arena_exits=False #track each fly leaving the grid
        ):

    #default start type is fixed heading

        '''Basic parameters'''

        self.param = dict(self.DefaultParam)
        self.param.update(param)
        self.dt = self.param['dt']
        self.dt_plot=self.param['dt_plot']
        self.t_stop = self.param['t_stop']
        self.x_position = np.copy(self.param['x_start_position'])
        self.y_position = np.copy(self.param['y_start_position'])
        self.distance_to_origin = scipy.zeros((self.size))
        self.num_traps = traps.num_traps

        '''Open space parameters and variables'''
        if(not(self.param['heading_data']==None)):
            ##If heading data field is provided to init, any mu/kappa information will be OVERRIDDEN
            (mean,kappa) = fit_von_mises(self.param['heading_data'])
            self.param['initial_heading_dist'] = scipy.stats.vonmises(loc=mean,kappa=kappa)
            self.param['initial_heading'] = scipy.random.vonmises(mean,kappa,(self.param['swarm_size'],))
            self.check_param()
        self.x_velocity = self.param['flight_speed']*scipy.cos(self.param['initial_heading'])
        self.y_velocity = self.param['flight_speed']*scipy.sin(self.param['initial_heading'])
        self.mode = scipy.full((self.size,), self.Mode_StartMode, dtype=int)
        if self.param['pure_advection']:
            self.param['wind_slippage'] = (0,0)
        self.parallel_coeff,self.perp_coeff = self.param['wind_slippage']
        self.par_wind,self.perp_wind = self.get_par_perp_comps(0.,wind_field,
            scipy.full((self.size,),True,dtype=bool))
        #^This is the set of 2 x time arrays of the components of each fly's velocity par/perp to wind
        self.start_type = start_type #Either 'fh' (fixed heading) or 'rw' (random walk)
        if start_type=='rw':
            self.rw_dist = scipy.stats.lognorm(0.25,scale=1)
        else:
            self.rw_dist = None


        '''Arena tracking variables'''
        self.track_arena_exits=track_arena_exits
        if self.track_arena_exits:
            self.still_in_arena = scipy.full(scipy.shape(self.x_position),True,dtype=bool)

        '''Plume tracking related parameters and variables'''
        self.lp_filter_duration = int(self.param['low_pass_filter_length']/self.dt) # in multiples of dt
        self.track_plume_bouts = track_plume_bouts
        self.surging_error = scipy.zeros((self.size,))
        self.t_last_cast = scipy.zeros((self.size,))
        #Parameters relating to cast timeout.
        self.time_began_casting = scipy.full(self.size,scipy.inf)
        self.reset_distribution = self.param['reset_distribution']
        self.reset_pool = self.reset_distribution.rvs(2000)
        self.cast_timeout = self.param['cast_timeout']
        self.reset_pool_counter = 0
        #for the case of the low pass filter, a vector that tracks time
        #since plume update_for_odor_loss
        if not(self.param['schmitt_trigger']):
            self.surging_plumeless_count = scipy.zeros((self.size))
        #for the case of plume bout tracking (time spent in plume tracking),
        #a vector that tracks how long each fly has been in the plume if it's
        #in the plume currently
        if self.track_plume_bouts:
            self.timesteps_since_plume_entry = scipy.full(self.size,scipy.nan)
        #also, a matrix that tracks plume bout lengths for each fly,
        #estimated rows is 100
        self.plume_bout_lengths = scipy.zeros((100,self.size))
        self.plume_bout_lengths_row = 0
        self.increments_until_turn = scipy.ones((self.size,)) #This is for the Levy walk option.
        cast_interval = self.param['cast_interval']
        self.dt_next_cast = scipy.random.uniform(cast_interval[0], cast_interval[1], (self.size,))
        self.cast_sign = scipy.random.choice([-1,1],(self.size,))
        self.ever_tracked = scipy.full((self.size,), False, dtype=bool) #Bool that keeps track if the fly ever plume tracked (false=never tracked)

        '''Trapped fly parameters and variables'''
        self.trap_num = scipy.full((self.size,),-1, dtype=int)
        self.in_trap = scipy.full((self.size,), False, dtype=bool)
        self.x_trap_loc = scipy.zeros((self.size,))
        self.y_trap_loc = scipy.zeros((self.size,))
        self.t_in_trap = scipy.full((self.size,),scipy.inf)
        self.angle_in_trap = scipy.full(self.size,scipy.inf)



    def check_param(self):
        """
        Check parameters - mostly just that shape of ndarrays match
        """
        if scipy.ndim(self.param['initial_heading'].shape) > 1:
            raise(ValueError, 'initial_heading must have ndim=1')

        equal_shape_list = ['x_start_position','y_start_position','flight_speed','release_time']
        for item in equal_shape_list:
            if self.param[item].shape != self.param['initial_heading'].shape:
                print(item)
                print(self.param[item].shape,self.param['initial_heading'].shape)
                raise(ValueError, '{0}.shape must equal initial_heading.shape'.format(item))


    @property
    def size(self):
        return self.param['initial_heading'].shape[0]


    def update(self, t, dt, wind_field, odor_field,traps,plumes=None,
        xlim=None,ylim=None,pre_stored=False):
        """
        Update fly swarm one time step.
        """

        ''' (0) Grab current time to track the updating time'''
        last = time.time()


        ''' (1) Categorize flies for update type and report category counts'''
        mask_release = t > self.param['release_time']
        mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
        mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
        mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
        self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
        mask_trapped = self.mode == self.Mode_Trapped
        mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

        print(str(sum(mask_castfor))+' flies are casting')
        print(str(sum(mask_flyupwd))+' flies are surging')
        #Keep track of which flies have never tracked
        print('time categorizing flies: '+str(time.time()-last))
        last = time.time()


        ''' (2) Get odor and wind info for each fly that is flying '''
        ''' (a) odor info'''
        odor = scipy.full(self.size,scipy.nan)
        mask_odor_relevant = mask_release & (~mask_trapped)
        if plumes is not None:
            puff_array = plumes.puffs
        if isinstance(odor_field,FakeDiffusionOdorField):
            odor[mask_odor_relevant] = odor_field.value(
                t,self.x_position[mask_odor_relevant],self.y_position[mask_odor_relevant])
        elif isinstance(odor_field,models.SuttonModelPlume) or \
            isinstance(odor_field,models.GaussianFitPlume) or \
            isinstance(odor_field,models.OnlinePlume) or \
            isinstance(odor_field,models.AdjustedGaussianFitPlume) or \
            isinstance(odor_field,models.LogisticProbPlume):
            odor[mask_odor_relevant] = odor_field.value(
                self.x_position[mask_odor_relevant],self.y_position[mask_odor_relevant])


        elif pre_stored:
            odor[mask_odor_relevant] = odor_field.value(
                t,self.x_position[mask_odor_relevant],self.y_position[mask_odor_relevant])
        else:
            odor[mask_odor_relevant]= odor_field.calc_conc_list(
                puff_array, self.x_position[mask_odor_relevant],\
                    self.y_position[mask_odor_relevant], z=0)
        print('time obtaining odor info: '+str(time.time()-last))

        ''' (b) wind info'''
        last = time.time()
        x_wind, y_wind = wind_field.value(t,self.x_position, self.y_position)
        x_wind_unit, y_wind_unit = unit_vector(x_wind, y_wind)
        wind_uvecs = {'x': x_wind_unit,'y': y_wind_unit}
        print('time obtaining wind info: '+str(time.time()-last))
        last = time.time()


        '''(3) Update the fly velocities according to mode and open space behavior paradigm'''
        # Update state for flies in traps
        self.update_for_in_trap(t, traps)

        #----Update the masks-----
        mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
        mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
        mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
        self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
        mask_trapped = self.mode == self.Mode_Trapped
        mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

        if not(self.start_type=='cvrw' or self.start_type=='rw'):
            #The random walk mode excludes odor detection
            masks = {'startmode': mask_startmode, 'flyupwd': mask_flyupwd,
            'castfor': mask_castfor}
            #Before anything else, adjust the velocities of flies already in
            #cast or surge mode based on the new wind direction this time step
            self.update_for_changing_wind(masks,wind_uvecs)

            #----Update the masks-----
            mask_release = t > self.param['release_time']
            mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
            mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
            mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
            self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
            mask_trapped = self.mode == self.Mode_Trapped
            mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

            # Update state for flies detecting odor plumes
            self.update_for_odor_detection(dt, odor,odor_field, wind_uvecs, masks)

            #----Update the masks-----
            mask_release = t > self.param['release_time']
            mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
            mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
            mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
            self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
            mask_trapped = self.mode == self.Mode_Trapped
            mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

            print('time updating for odor detection: '+str(time.time()-last))
            last = time.time()
            # Update state for files losing odor plume or already casting.
            self.update_for_odor_loss(t, dt, odor, odor_field, wind_uvecs, masks)

            #----Update the masks-----
            mask_release = t > self.param['release_time']
            mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
            mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
            mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
            self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
            mask_trapped = self.mode == self.Mode_Trapped
            mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

            print('time updating for odor loss: '+str(time.time()-last))
            last = time.time()
            #Update state for flies giving up on casting.
            self.reset_pool_counter=self.update_for_reset(t,masks,self.reset_pool_counter)

            #----Update the masks-----
            mask_release = t > self.param['release_time']
            mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
            mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
            mask_castfor = mask_release & (self.mode == self.Mode_CastForOdor)
            self.ever_tracked = self.ever_tracked | (mask_release & (self.mode == self.Mode_FlyUpWind)) #this is true if the fly has previously tracked or has been released and is now in upwind mode
            mask_trapped = self.mode == self.Mode_Trapped
            mask_reset_startmode = self.ever_tracked & (self.mode == self.Mode_StartMode)

        # At this point, add one timestep to all entries of
        # timesteps_since_plume_entry that are not nan
        if self.track_plume_bouts:
            add_inds = scipy.logical_not(scipy.isnan(self.timesteps_since_plume_entry))
            self.timesteps_since_plume_entry[add_inds]+=1
            # print('timesteps_since_plume_entry:'+str(self.timesteps_since_plume_entry))
        # Update position based on mode and current velocities

        ''' (4) Update the positions with the new velocities'''
        ''' (a) Update the positions with general direction excluding par/perp slip component'''
        self.update_positions(mask_release,mask_trapped,mask_startmode,x_wind,y_wind,dt)

        #check for flies that have left the arena
        if self.track_arena_exits:
            inside_x_bounds = (xlim[0]<=self.x_position) & (self.x_position<=xlim[1])
            inside_y_bounds = (ylim[0]<=self.y_position) & (self.y_position<=ylim[1])
            inside_bounds = inside_x_bounds & inside_y_bounds
            self.still_in_arena = self.still_in_arena & inside_bounds
            print(sum(self.still_in_arena))



        #Michael's idea 2/27/18: apply wind slippage according to c_1*(component
        #parallel to fly's velocity) + c2*(component pe rp to fly's velocity)
        ''' (b) Update the positions with the perp slip component'''
        self.update_par_perp_comps(t,wind_field,mask_release,mask_startmode,
            mask_trapped, mask_reset_startmode)
        #par/perp comps for flies not {released and in fly mode} are set to 0.
        c1 = self.parallel_coeff
        c2 = self.perp_coeff

        self.x_position[mask_startmode] += dt*(c1*self.par_wind[0,mask_startmode]+c2*self.perp_wind[0,mask_startmode])
        self.y_position[mask_startmode] += dt*(c1*self.par_wind[1,mask_startmode]+c2*self.perp_wind[1,mask_startmode])




    def update_for_changing_wind(self,masks,wind_uvecs):
        '''This function updates the velocities of flies in cast or surge mode
        with the current wind info '''

        x_wind_unit = wind_uvecs['x']
        y_wind_unit = wind_uvecs['y']
        mask_flyupwd = masks['flyupwd']
        mask_castfor = masks['castfor']
        mask_change = (mask_flyupwd) | (mask_castfor)

        #Select new heading errors for the casting and surging flies
        surging_error_std = self.param['surging_error_std']
        distf = self.param['surging_error_dist'] #this variable is a pdf
        self.surging_error[mask_change] = surging_error_std*distf.rvs(size=mask_change.sum())

        # Set x and y velocities for the surging flies according to current wind
        x_unit_change, y_unit_change = rotate_vecs(
                x_wind_unit[mask_flyupwd],
                y_wind_unit[mask_flyupwd],
                self.surging_error[mask_flyupwd]
                )
        speed = self.param['flight_speed'][mask_flyupwd]
        self.x_velocity[mask_flyupwd] = -speed*x_unit_change
        self.y_velocity[mask_flyupwd] = -speed*y_unit_change

        # Set x and y velocities for the casting flies according to current wind,
        #keeping in mind their pre-determined cast signs (left/right)
        x_unit_change, y_unit_change = rotate_vecs(
                y_wind_unit[mask_castfor],
               -x_wind_unit[mask_castfor],
                self.surging_error[mask_castfor]
                )
        speed = self.param['flight_speed'][mask_castfor]

        #Actually update velocities
        self.x_velocity[mask_castfor] = self.cast_sign[mask_castfor]*speed*x_unit_change
        self.y_velocity[mask_castfor] = self.cast_sign[mask_castfor]*speed*y_unit_change

    def update_for_odor_detection(self, dt, odor, odor_field, wind_uvecs, masks):
        """
         Update simulation for odor detection
         * Find flies in StartMode and CastForOdor modes where the odor value >= upper threshold.
         * Test if they detect odor (roll dice and compare with detection probabilty).
         * If they do detect odor change their  mode to FlyUpWind.
         * set x and y velocities to upwind at speed
        """
        x_wind_unit = wind_uvecs['x']
        y_wind_unit = wind_uvecs['y']
        mask_startmode = masks['startmode']
        mask_castfor = masks['castfor']

        if self.param['schmitt_trigger']:
        #Case where mask_change (to surging) is determined by Schmitt trigger
            mask_gt_upper = odor >= self.param['odor_thresholds']['upper']
            mask_candidates = mask_gt_upper & (mask_startmode | mask_castfor)
            dice_roll = scipy.full((self.size,),scipy.inf)
            dice_roll[mask_candidates] = scipy.rand(mask_candidates.sum())
            # Convert probabilty/sec to probabilty for time step interval dt
            odor_probability_upper = 1.0 - (1.0 - self.param['odor_probabilities']['upper'])**dt
            mask_change = dice_roll < odor_probability_upper
        else:
        #Case where mask_change (to surging) is determined by low-pass filter
            if not(isinstance(odor_field,models.SuttonModelPlume) or \
                isinstance(odor_field,models.GaussianFitPlume)):
                #This is where the odor is deterministically accessed, as for pompy plumes
                mask_gt_upper = odor >= self.param['odor_thresholds']['upper']
            else:
                #For the GaussianFitPlume, the odor value is interpreted as
                #the probability the odor is greater than the threshold
                mask_gt_upper = np.random.binomial(1,odor,size=np.shape(odor)).astype(bool)

            mask_change = mask_gt_upper & (mask_startmode | mask_castfor)
        #In both cases the mask_change flies are assigned to Mode_FlyUpWind
        self.mode[mask_change] = self.Mode_FlyUpWind
        if self.track_plume_bouts:
        #These flies have their entries in timesteps_since_plume_entry changed from nan to 0
            self.timesteps_since_plume_entry[mask_change] = 0

        # Compute new heading error for flies which change mode according to Laplace dist (Floris paper)
        surging_error_std = self.param['surging_error_std']
        distf = self.param['surging_error_dist'] #this variable is a pdf
        self.surging_error[mask_change] = surging_error_std*distf.rvs(size=mask_change.sum())
        # Set x and y velocities for the flies which just changed to FlyUpWind.
        '''This is the insertion of heading error for surging flies'''
        x_unit_change, y_unit_change = rotate_vecs(
                x_wind_unit[mask_change],
                y_wind_unit[mask_change],
                self.surging_error[mask_change]
                )
        speed = self.param['flight_speed'][mask_change]

        self.x_velocity[mask_change] = -speed*x_unit_change
        self.y_velocity[mask_change] = -speed*y_unit_change

    def update_for_odor_loss(self, t, dt, odor, odor_field, wind_uvecs, masks):
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

        if self.param['schmitt_trigger']:

            mask_lt_lower = odor <= self.param['odor_thresholds']['lower']
            mask_candidates = mask_lt_lower & mask_flyupwd
            dice_roll = scipy.full((self.size,),scipy.inf)
            dice_roll[mask_candidates] = scipy.rand(mask_candidates.sum())

            # Convert probabilty/sec to probabilty for time step interval dt
            odor_probability_lower = 1.0 - (1.0 - self.param['odor_probabilities']['lower'])**dt
            mask_change = dice_roll < odor_probability_lower
        else:
            if not(isinstance(odor_field,models.SuttonModelPlume) or \
                isinstance(odor_field,models.GaussianFitPlume)):

                #Find the indices of the flies below the (single) threshold
                mask_blw_thres = odor <= self.param['odor_thresholds']['upper']
            else:
                #For the GaussianFitPlume, the odor value is interpreted as
                #the probability the odor is greater than the threshold--
                #so it's under the threshold with probability (1 - odor value)
                mask_blw_thres = np.random.binomial(1,1.-odor,size=np.shape(odor)).astype(bool)


            #Filter by the flies who are surging
            mask_candidates = mask_blw_thres & mask_flyupwd
            #If the fly's counter is at 0, or at 1, add 1 to the counter and do nothing (preserve mode)
            mask_categ1 = mask_candidates & (self.surging_plumeless_count<self.lp_filter_duration)
            self.surging_plumeless_count[mask_categ1] +=1
            self.mode[mask_categ1] = self.Mode_FlyUpWind
            #If the fly's counter is at 2, assign to casting mode, and reset counter to 0
            mask_change = mask_candidates & (self.surging_plumeless_count==self.lp_filter_duration)
            self.surging_plumeless_count[mask_change] =0

        #In both cases set mask_change to cast for odor mode
        self.mode[mask_change] = self.Mode_CastForOdor
        #Grab the time these flies starting casting.
        self.time_began_casting[mask_change] = t
        #Then drop these flies' plume bout durations into plume_bout_lengths
        if self.track_plume_bouts & sum(mask_change)>0:
            #(a) grab the saved index j of the first empty row in plume_bout_lengths
            j = self.plume_bout_lengths_row
            if j>scipy.shape(self.plume_bout_lengths)[0]-1:
                self.plume_bout_lengths = scipy.append(
                self.plume_bout_lengths,scipy.zeros((100,self.size)),axis=0)
                print('appending')
            #(b) fill plume_bout_lengths row j
            self.plume_bout_lengths[j,mask_change] = self.timesteps_since_plume_entry[mask_change]
            #move up row counter
            self.plume_bout_lengths_row+=1
            #Reassign these indices of timesteps_since_plume_entry to nan
            self.timesteps_since_plume_entry[mask_change]=scipy.nan

        # Lump together flies changing to CastForOdor mode with casting flies which are
        # changing direction (e.g. time to make cast direction change)
        mask_change |= mask_castfor & (t > (self.t_last_cast + self.dt_next_cast))

        # Compute new heading errors for flies which change mode (to casting)
        self.surging_error[mask_change] = self.param['surging_error_std']*scipy.randn(mask_change.sum())

        # Set new cast intervals and directions for flies changing to CastForOdor or starting a new cast
        cast_interval = self.param['cast_interval']
        self.dt_next_cast[mask_change] = scipy.random.uniform(
                cast_interval[0],
                cast_interval[1],
                (mask_change.sum(),)
                )
        self.t_last_cast[mask_change] = t
        self.cast_sign[mask_change] = scipy.random.choice([-1,1],(mask_change.sum(),))

        '''This is the insertion of heading error for casting flies'''
        # Set x and y velocities for new CastForOdor flies
        x_unit_change, y_unit_change = rotate_vecs(
                y_wind_unit[mask_change],
               -x_wind_unit[mask_change],
                self.surging_error[mask_change]
                )
        speed = self.param['flight_speed'][mask_change]
        self.x_velocity[mask_change] = self.cast_sign[mask_change]*speed*x_unit_change
        self.y_velocity[mask_change] = self.cast_sign[mask_change]*speed*y_unit_change

    def update_for_reset(self,t,masks,reset_pool_counter):
        '''This is for the flies who give up on casting'''
        mask_castfor = masks['castfor']
        reset_mask = mask_castfor & ((t-self.time_began_casting)> self.cast_timeout)
        self.mode[reset_mask] = self.Mode_StartMode
        for index in range(len(reset_mask)):
            if reset_mask[index]:
                try:
                    angle = self.reset_pool[reset_pool_counter]
                except(IndexError):
                    self.reset_pool = scipy.append(
                            self.reset_pool,self.reset_distribution.rvs(2000))
                    angle = self.reset_pool[reset_pool_counter]
                self.x_velocity[index] = self.param['flight_speed'][0]*scipy.cos(angle)
                self.y_velocity[index] = self.param['flight_speed'][0]*scipy.sin(angle)
                reset_pool_counter +=1
        return reset_pool_counter


    def update_for_in_trap(self, t, traps): #******
        """
         Update simulation for flies in traps.
         * If flies are in traps. If so record trap info and time.
        """
        sources = traps.param['source_locations'] #Of format [(0,0),]
        for trap_num, trap_loc in enumerate(sources):
            dist_vals = distance((self.x_position, self.y_position),trap_loc)
            mask_trapped = dist_vals < traps.param['trap_radius']
            self.mode[mask_trapped] = self.Mode_Trapped
            self.trap_num[mask_trapped] = trap_num

            self.x_trap_loc[mask_trapped] = trap_loc[0]
            self.y_trap_loc[mask_trapped] = trap_loc[1]

            # Get time stamp for newly trapped flies
            mask_newly_trapped = mask_trapped & (self.t_in_trap == scipy.inf)
            self.t_in_trap[mask_newly_trapped] = t

            #Get arrival angle for newly trapped flies
            vfunc = scipy.vectorize(cartesian_to_polar)
            xvels,yvels = self.x_velocity[mask_newly_trapped],self.y_velocity[mask_newly_trapped]
            if scipy.size(xvels)>0:
                _,thetas = vfunc(xvels,yvels)
                thetas = (thetas+scipy.pi)%(2*scipy.pi)
                self.angle_in_trap[mask_newly_trapped] = thetas

            #Stop the flies trapped
            self.x_velocity[mask_trapped] = 0.0
            self.y_velocity[mask_trapped] = 0.0



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

    def get_angle_trapped(self,trap_num,time_window):
        mask_trapped = self.mode == self.Mode_Trapped
        mask_trapped_in_num = mask_trapped & (self.trap_num == trap_num)
        if not(time_window==[]):
        #This case returns angle trapped of those trapped in a given time window
            time_bool = (self.t_in_trap>=time_window[0]) & (self.t_in_trap<=time_window[1])
            mask_trapped_in_num = mask_trapped_in_num & time_bool
        return self.angle_in_trap[mask_trapped_in_num]


    def get_trap_nums(self):
        mask_trap_num_set = self.trap_num != -1
        trap_num_array = scipy.unique(self.trap_num[mask_trap_num_set])
        trap_num_array.sort()
        return list(trap_num_array)

    def list_all_traps(self):
        return(range(self.num_traps))

    def get_trap_counts(self):
        mask_trap_num_set = self.trap_num != -1
        (trap_num_array,trap_counts)=scipy.unique(
        self.trap_num[mask_trap_num_set],return_counts = True)
        all_trap_counts = scipy.zeros(self.num_traps)
        all_trap_counts[trap_num_array] = trap_counts
        return all_trap_counts

    def update_positions(self,mask_release,mask_trapped,mask_startmode,x_wind,y_wind,dt):

        if self.start_type=='fh' or sum(mask_startmode)<1.:
            mask_move = mask_release & (~mask_trapped)
            print('number moving: '+str(scipy.sum(mask_move)))
            if self.param['pure_advection']:
                self.x_position[mask_startmode] += dt*x_wind[mask_startmode]
                self.y_position[mask_startmode] += dt*y_wind[mask_startmode]
                self.x_position[mask_move& (~mask_startmode)] += dt*self.x_velocity[mask_move& (~mask_startmode)]
                self.y_position[mask_move& (~mask_startmode)] += dt*self.y_velocity[mask_move& (~mask_startmode)]

            elif self.param['airspeed_saturation']:
                wind_par = self.par_wind[:,mask_move&mask_startmode]
                wind_par_mags = np.sqrt(np.sum(wind_par*wind_par,axis=0))
                wind_par_signs = np.sign(np.sum(wind_par*
                    (np.vstack((self.x_velocity[mask_move&mask_startmode],self.y_velocity[
                    mask_move&mask_startmode])))
                    ,axis=0))
                signed_wind_par_mags = wind_par_mags*wind_par_signs
                flight_speed = self.param['flight_speed'][0]

                # #Old version: linear/flat/linear function from intended speed to effective speed
                # adjusted_mag = flight_speed*np.ones_like(signed_wind_par_mags)
                # adjusted_mag[signed_wind_par_mags<-0.8] = signed_wind_par_mags[signed_wind_par_mags<-0.8]+2.4
                # adjusted_mag[signed_wind_par_mags>4.] = signed_wind_par_mags[signed_wind_par_mags>4.]-2.4
                # adjusted_mag[(signed_wind_par_mags<4.)&(signed_wind_par_mags>-0.8)] = 1.6
                #
                # plt.figure()
                # plt.plot(signed_wind_par_mags,adjusted_mag,'o')

                # New version: a sigmoidal smoothing of the above
                adjusted_mag = speed_sigmoid_func(signed_wind_par_mags)
                # plt.plot(signed_wind_par_mags,adjusted_mag,'o')

                adjusted_mag = adjusted_mag/flight_speed #normalization to unit mag

                # plt.show()
                # raw_input()


                self.x_position[mask_move&mask_startmode] += dt*adjusted_mag*self.x_velocity[mask_move&mask_startmode]
                self.y_position[mask_move&mask_startmode] += dt*adjusted_mag*self.y_velocity[mask_move&mask_startmode]

                self.x_position[mask_move&(~mask_startmode)] += dt*self.x_velocity[mask_move&(~mask_startmode)]
                self.y_position[mask_move&(~mask_startmode)] += dt*self.y_velocity[mask_move&(~mask_startmode)]

            else:
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
                #and get assigned a fnew interval count until they change direction again
                self.increments_until_turn[mask_redraw] = scipy.floor(#scipy.random.choice(self.increments_pool,sum(mask_redraw))
                    scipy.stats.lognorm.rvs(sigma,size=draws,scale=
                    (300/3.0)/dt*
                    scipy.exp(mu)))
                #cp4 = time.time(); print(cp4-cp3)
                #print(cp4-cp)
        #Once positions are updated, update the variable that keeps track of distance to origin
        self.distance_to_origin = scipy.sqrt(self.x_position**2+self.y_position**2)


    def get_par_perp_comps(self,t,wind_field,mask):
        x_wind, y_wind = wind_field.value(
            t,self.x_position[mask], self.y_position[mask])
        wind = scipy.array([x_wind,y_wind])
        velocity = scipy.array([
            self.x_velocity[mask],self.y_velocity[mask]])
        par_vec = scipy.zeros(scipy.shape(velocity))
        perp_vec = scipy.zeros(scipy.shape(velocity))
        for i in range(scipy.size(velocity,1)):
            u,v = velocity[:,i],wind[:,i]
            par,perp = par_perp(v,u)
            par_vec[:,i],perp_vec[:,i] = par,perp
        return par_vec,perp_vec

    def update_par_perp_comps(self,t,wind_field,mask_release,mask_startmode,mask_trapped,mask_reset_startmode):
            #Check if the wind field has changed since last time-step, if so, re-compute get_par_perp_comps

            if wind_field.evolving:
                mask_not_stuck = scipy.logical_not(mask_trapped)
                self.par_wind[:,mask_not_stuck],self.perp_wind[:,mask_not_stuck] = \
                    self.get_par_perp_comps(t,wind_field,mask_not_stuck)

            #Set the flys who have been released and who are not in start_mode to zero par and zero perp
            self.par_wind[:,mask_release&~mask_startmode] = 0.
            self.perp_wind[:,mask_release&~mask_startmode] = 0.

            #For the flies that just returned to startmode after casting, restore par/perp wind components
            self.par_wind[:,mask_reset_startmode],self.perp_wind[:,mask_reset_startmode] = \
                self.get_par_perp_comps(t,wind_field,mask_reset_startmode)

            #Set the flys who have not been released to zero par and zero perp
            # self.par_wind[:,~mask_release] = 0.
            # self.perp_wind[:,~mask_release] = 0.




class ReducedSwarmOfFlies(object):

    """
    These flies, designed to interact with the LogisticProbPlume object
    skip the cast and surge process and just head straight to the source
    or miss the plume depending on their draw with the logistic probability function.

    They still have all the capacities for distributed release, and all the
    open space navigation parameters.

    Only designed to work with non-changing, straight wind.

    (has no surging error)
    """

    DefaultSize = 500

    DefaultParam = {
            'dt'                  : 0.25,
            'initial_heading_dist': scipy.stats.uniform(0,2*scipy.pi), #continuous_distribution object
            'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(DefaultSize,))),
            'x_start_position'    : scipy.zeros((DefaultSize,)),
            'y_start_position'    : scipy.zeros((DefaultSize,)),
            'flight_speed'        : scipy.full((DefaultSize,), 0.7),
            'release_time'        : scipy.full((DefaultSize,), 0.0),
            'release_time_constant': None,
            'wind_slippage'       : (0.0,0.0), #(// to fly's path, perp to fly's path)
            'pure_advection': False,
            'airspeed_saturation': False
            }

    Mode_StartMode = 0
    Mode_FlyUpWind = 1
    Mode_CastForOdor = 2
    Mode_Trapped = 3


    def __init__(self,wind_field,traps,param={},start_type='fh'):

    #default start type is fixed heading

        '''Basic parameters'''

        self.param = dict(self.DefaultParam)
        self.param.update(param)
        self.dt = self.param['dt']
        self.dt_plot=self.param['dt_plot']
        self.t_stop = self.param['t_stop']
        self.x_position = np.copy(self.param['x_start_position'])
        self.y_position = np.copy(self.param['y_start_position'])
        self.distance_to_origin = scipy.zeros((self.size))
        self.num_traps = traps.num_traps

        '''Open space parameters and variables'''
        if(not(self.param['heading_data']==None)):
            ##If heading data field is provided to init, any mu/kappa information will be OVERRIDDEN
            (mean,kappa) = fit_von_mises(self.param['heading_data'])
            self.param['initial_heading_dist'] = scipy.stats.vonmises(loc=mean,kappa=kappa)
            self.param['initial_heading'] = scipy.random.vonmises(mean,kappa,(self.param['swarm_size'],))
            self.check_param()
        self.x_velocity = self.param['flight_speed']*scipy.cos(self.param['initial_heading'])
        self.y_velocity = self.param['flight_speed']*scipy.sin(self.param['initial_heading'])
        self.mode = scipy.full((self.size,), self.Mode_StartMode, dtype=int)
        if self.param['pure_advection']:
            self.param['wind_slippage'] = (0,0)
        self.parallel_coeff,self.perp_coeff = self.param['wind_slippage']
        self.par_wind,self.perp_wind = self.get_par_perp_comps(0.,wind_field,
            scipy.full((self.size,),True,dtype=bool))
        #^This is the set of 2 x time arrays of the components of each fly's velocity par/perp to wind
        self.start_type = start_type #Either 'fh' (fixed heading) or 'rw' (random walk)
        if start_type=='rw':
            self.rw_dist = scipy.stats.lognorm(0.25,scale=1)
        else:
            self.rw_dist = None

        '''Trapped fly parameters and variables'''
        self.trap_num = scipy.full((self.size,),-1, dtype=int)
        self.in_trap = scipy.full((self.size,), False, dtype=bool)
        self.x_trap_loc = scipy.zeros((self.size,))
        self.y_trap_loc = scipy.zeros((self.size,))
        self.t_in_trap = scipy.full((self.size,),scipy.inf)
        self.angle_in_trap = scipy.full(self.size,scipy.inf)

        self.mask_in_odor_band_last_step= np.zeros((self.size),dtype=bool) #variable used to make refractory period for odor detection

    def check_param(self):
        """
        Check parameters - mostly just that shape of ndarrays match
        """
        if scipy.ndim(self.param['initial_heading'].shape) > 1:
            raise(ValueError, 'initial_heading must have ndim=1')

        equal_shape_list = ['x_start_position','y_start_position','flight_speed','release_time']
        for item in equal_shape_list:
            if self.param[item].shape != self.param['initial_heading'].shape:
                print(item)
                print(self.param[item].shape,self.param['initial_heading'].shape)
                raise(ValueError, '{0}.shape must equal initial_heading.shape'.format(item))


    @property
    def size(self):
        return self.param['initial_heading'].shape[0]


    def update(self, t, dt, wind_field, odor_field,traps,plumes=None,
        xlim=None,ylim=None,pre_stored=False):
        """
        Update fly swarm one time step.
        """

        ''' (1) Categorize flies for update type and report category counts'''
        mask_release = t > self.param['release_time']
        mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
        mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
        mask_trapped = self.mode == self.Mode_Trapped

        last = time.time()

        ''' (2) Get odor and wind info for each fly that is flying '''
        ''' Note, for this version of the swarm model, the odor is really just
            the probability value given by the LogisticProbPlume object.
            (the name is a holdover)'''
        odor = scipy.full(self.size,scipy.nan)
        mask_odor_relevant = mask_release & (~mask_trapped)
        odor[mask_odor_relevant] = odor_field.value(
            self.x_position[mask_odor_relevant],self.y_position[mask_odor_relevant])
        print('time obtaining odor info: '+str(time.time()-last))

        ''' (b) wind info'''
        x_wind, y_wind = wind_field.value(t,self.x_position, self.y_position)
        x_wind_unit, y_wind_unit = unit_vector(x_wind, y_wind)
        wind_uvecs = {'x': x_wind_unit,'y': y_wind_unit}

        '''(3) Update the fly velocities according to mode and open space behavior paradigm'''
        # Update state for flies in traps
        self.update_for_in_trap(t, traps)

        #----Update the masks-----
        mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
        mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
        mask_trapped = self.mode == self.Mode_Trapped

        if not(self.start_type=='cvrw' or self.start_type=='rw'):
            #The random walk mode excludes odor detection
            masks = {'startmode': mask_startmode, 'flyupwd': mask_flyupwd}

            # Update state for flies detecting odor plumes
            self.update_for_odor_detection(dt, odor,odor_field, wind_uvecs, masks)

            #----Update the masks-----
            mask_release = t > self.param['release_time']
            mask_startmode = mask_release & (self.mode == self.Mode_StartMode)
            mask_flyupwd = mask_release & (self.mode == self.Mode_FlyUpWind)
            mask_trapped = self.mode == self.Mode_Trapped


        # Update position based on mode and current velocities

        ''' (4) Update the positions with the new velocities'''
        ''' (a) Update the positions with general direction excluding par/perp slip component'''
        self.update_positions(mask_release,mask_trapped,mask_startmode,x_wind,y_wind,dt)

        #Michael's idea 2/27/18: apply wind slippage according to c_1*(component
        #parallel to fly's velocity) + c2*(component pe rp to fly's velocity)
        ''' (b) Update the positions with the perp slip component'''
        self.update_par_perp_comps(t,wind_field,mask_release,mask_startmode,
            mask_trapped)
        #par/perp comps for flies not {released and in fly mode} are set to 0.
        c1 = self.parallel_coeff
        c2 = self.perp_coeff

        self.x_position[mask_startmode] += dt*(c1*self.par_wind[0,mask_startmode]+c2*self.perp_wind[0,mask_startmode])
        self.y_position[mask_startmode] += dt*(c1*self.par_wind[1,mask_startmode]+c2*self.perp_wind[1,mask_startmode])



    def update_for_odor_detection(self, dt, odor, odor_field, wind_uvecs, masks):
        #Worth noting that the shape of the vector odor by the time it's passed
        #into this function by the update() function has been masked to exclude
        #pre-release flies and trapped flies.
        """
         Update simulation for odor detection
         * For flies in StartMode, compute their LogisticProbPlume value.
         * Test if they detect odor (Bernoulli draw).
         * If they do detect odor change their  mode to FlyUpWind.
         * set x and y velocities to upwind at speed
        """
        x_wind_unit = wind_uvecs['x']
        y_wind_unit = wind_uvecs['y']
        mask_startmode = masks['startmode']

        mask_gt_upper = np.random.binomial(1,odor,size=np.shape(odor)).astype(bool)

        #we also want to know which flies are in the band but didn't "get their card drawn"
        #(the flies that we want to stop from detecting in the next couple frames to
        # avoid double drawing problems):
        mask_in_odor_band = odor>0.01
        print('----------'+str(np.sum(mask_in_odor_band))+'----------')

        already_drawn = (mask_in_odor_band & self.mask_in_odor_band_last_step)

        mask_change = mask_gt_upper & (mask_startmode) & (~already_drawn)

        self.mode[mask_change] = self.Mode_FlyUpWind

        # Set x and y velocities for the flies which just changed to FlyUpWind.
        speed = self.param['flight_speed'][mask_change]

        self.x_velocity[mask_change] = -speed*x_wind_unit[mask_change]
        self.y_velocity[mask_change] = -speed*y_wind_unit[mask_change]


        self.mask_in_odor_band_last_step = mask_in_odor_band

    def update_for_in_trap(self, t, traps): #******
        """
         Update simulation for flies in traps.
         * If flies are in traps. If so record trap info and time.
        """
        sources = traps.param['source_locations'] #Of format [(0,0),]
        for trap_num, trap_loc in enumerate(sources):
            dist_vals = distance((self.x_position, self.y_position),trap_loc)
            mask_trapped = dist_vals < traps.param['trap_radius']
            self.mode[mask_trapped] = self.Mode_Trapped
            self.trap_num[mask_trapped] = trap_num

            self.x_trap_loc[mask_trapped] = trap_loc[0]
            self.y_trap_loc[mask_trapped] = trap_loc[1]

            # Get time stamp for newly trapped flies
            mask_newly_trapped = mask_trapped & (self.t_in_trap == scipy.inf)
            self.t_in_trap[mask_newly_trapped] = t

            #Get arrival angle for newly trapped flies
            vfunc = scipy.vectorize(cartesian_to_polar)
            xvels,yvels = self.x_velocity[mask_newly_trapped],self.y_velocity[mask_newly_trapped]
            if scipy.size(xvels)>0:
                _,thetas = vfunc(xvels,yvels)
                thetas = (thetas+scipy.pi)%(2*scipy.pi)
                self.angle_in_trap[mask_newly_trapped] = thetas

            #Stop the flies trapped
            self.x_velocity[mask_trapped] = 0.0
            self.y_velocity[mask_trapped] = 0.0



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

    def get_angle_trapped(self,trap_num,time_window):
        mask_trapped = self.mode == self.Mode_Trapped
        mask_trapped_in_num = mask_trapped & (self.trap_num == trap_num)
        if not(time_window==[]):
        #This case returns angle trapped of those trapped in a given time window
            time_bool = (self.t_in_trap>=time_window[0]) & (self.t_in_trap<=time_window[1])
            mask_trapped_in_num = mask_trapped_in_num & time_bool
        return self.angle_in_trap[mask_trapped_in_num]


    def get_trap_nums(self):
        mask_trap_num_set = self.trap_num != -1
        trap_num_array = scipy.unique(self.trap_num[mask_trap_num_set])
        trap_num_array.sort()
        return list(trap_num_array)

    def list_all_traps(self):
        return(range(self.num_traps))

    def get_trap_counts(self):
        mask_trap_num_set = self.trap_num != -1
        (trap_num_array,trap_counts)=scipy.unique(
        self.trap_num[mask_trap_num_set],return_counts = True)
        all_trap_counts = scipy.zeros(self.num_traps)
        all_trap_counts[trap_num_array] = trap_counts
        return all_trap_counts

    def update_positions(self,mask_release,mask_trapped,mask_startmode,x_wind,y_wind,dt):

        if self.start_type=='fh' or sum(mask_startmode)<1.:
            mask_move = mask_release & (~mask_trapped)
            print('number moving: '+str(scipy.sum(mask_move)))
            if self.param['pure_advection']:
                self.x_position[mask_startmode] += dt*x_wind[mask_startmode]
                self.y_position[mask_startmode] += dt*y_wind[mask_startmode]
                self.x_position[mask_move& (~mask_startmode)] += dt*self.x_velocity[mask_move& (~mask_startmode)]
                self.y_position[mask_move& (~mask_startmode)] += dt*self.y_velocity[mask_move& (~mask_startmode)]

            elif self.param['airspeed_saturation']:
                wind_par = self.par_wind[:,mask_move&mask_startmode]
                wind_par_mags = np.sqrt(np.sum(wind_par*wind_par,axis=0))
                wind_par_signs = np.sign(np.sum(wind_par*
                    (np.vstack((self.x_velocity[mask_move&mask_startmode],self.y_velocity[
                    mask_move&mask_startmode])))
                    ,axis=0))
                signed_wind_par_mags = wind_par_mags*wind_par_signs
                flight_speed = self.param['flight_speed'][0]


                # New version: a sigmoidal smoothing of the above
                adjusted_mag = speed_sigmoid_func(signed_wind_par_mags)
                # plt.plot(signed_wind_par_mags,adjusted_mag,'o')

                adjusted_mag = adjusted_mag/flight_speed #normalization to unit mag

                self.x_position[mask_move&mask_startmode] += dt*adjusted_mag*self.x_velocity[mask_move&mask_startmode]
                self.y_position[mask_move&mask_startmode] += dt*adjusted_mag*self.y_velocity[mask_move&mask_startmode]

                self.x_position[mask_move&(~mask_startmode)] += dt*self.x_velocity[mask_move&(~mask_startmode)]
                self.y_position[mask_move&(~mask_startmode)] += dt*self.y_velocity[mask_move&(~mask_startmode)]

            else:
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
                #and get assigned a fnew interval count until they change direction again
                self.increments_until_turn[mask_redraw] = scipy.floor(#scipy.random.choice(self.increments_pool,sum(mask_redraw))
                    scipy.stats.lognorm.rvs(sigma,size=draws,scale=
                    (300/3.0)/dt*
                    scipy.exp(mu)))
                #cp4 = time.time(); print(cp4-cp3)
                #print(cp4-cp)
        #Once positions are updated, update the variable that keeps track of distance to origin
        self.distance_to_origin = scipy.sqrt(self.x_position**2+self.y_position**2)


    def get_par_perp_comps(self,t,wind_field,mask):
        x_wind, y_wind = wind_field.value(
            t,self.x_position[mask], self.y_position[mask])
        wind = scipy.array([x_wind,y_wind])
        velocity = scipy.array([
            self.x_velocity[mask],self.y_velocity[mask]])
        par_vec = scipy.zeros(scipy.shape(velocity))
        perp_vec = scipy.zeros(scipy.shape(velocity))
        for i in range(scipy.size(velocity,1)):
            u,v = velocity[:,i],wind[:,i]
            par,perp = par_perp(v,u)
            par_vec[:,i],perp_vec[:,i] = par,perp
        return par_vec,perp_vec

    def update_par_perp_comps(self,t,wind_field,mask_release,mask_startmode,mask_trapped):
        #Check if the wind field has changed since last time-step, if so, re-compute get_par_perp_comps

        if wind_field.evolving:
            mask_not_stuck = scipy.logical_not(mask_trapped)
            self.par_wind[:,mask_not_stuck],self.perp_wind[:,mask_not_stuck] = \
                self.get_par_perp_comps(t,wind_field,mask_not_stuck)

        #Set the flies who have been released and who are not in start_mode to zero par and zero perp
        self.par_wind[:,mask_release&~mask_startmode] = 0.
        self.perp_wind[:,mask_release&~mask_startmode] = 0.

        #Set the flies who have not been released to zero par and zero perp
        # self.par_wind[:,~mask_release] = 0.
        # self.perp_wind[:,~mask_release] = 0.
