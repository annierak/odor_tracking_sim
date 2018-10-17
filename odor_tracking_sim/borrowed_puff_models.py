from __future__ import division
import scipy
import scipy.interpolate as interp
import math
import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import wind_models
import odor_models
import time

class Rectangle(object):

    """
    Axis-aligned rectangle defined by two points (x_min, y_min) and
    (x_max, y_max) with it required that x_max > x_min and y_max > y_min.
    """

    def __init__(self, x_min, y_min, x_max, y_max):
        """
        Parameters
        ----------
        x_min : float
            x-coordinate of bottom-left corner of rectangle.
        y_min : float
            x-coordinate of bottom-right corner of rectangle.
        x_max : float
            x-coordinate of top-right corner of rectangle.
        y_max : float
            y-coordinate of top-right corner of rectangle.
        """
        try:
            if float(x_min) >= float(x_max):
                raise InvalidRectangleCoordinateError('Rectangle x_min must \
                                                       be < x_max.')
            if float(y_min) >= float(y_max):
                raise InvalidRectangleCoordinateError('Rectangle y_min must \
                                                       be < y_max.')
        except ValueError as e:
            raise InvalidRectangleCoordinateError(
                'Rectangle coordinates must be numeric ({0}).'.format(e))
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def w(self):
        """Width of rectangle (i.e. distance covered on x-axis)."""
        return self.x_max - self.x_min

    @property
    def h(self):
        """Height of rectangle (i.e. distance covered on y-axis)."""
        return self.y_max - self.y_min

    def as_tuple(self):
        """Tuple representation of Rectangle (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def contains(self, x, y):
        """
        Tests whether the supplied position, an (x,y) pair, is contained
        within the region defined by this Rectangle object and returns
        True if so and False if not.
        """
        return (x >= self.x_min and x <= self.x_max and
                y >= self.y_min and y <= self.y_max)

class Puff(object):
    """
    Lightweight container for the properties of a single odour puff.

    Implemented with slots to improve memory management when a large number
    of puffs are being simulated. Squared radius stored rather than radius
    as puff growth model and concentration distribution models both use
    squared radius hence this minimises needless exponent operations.
    """

    __slots__ = ["x", "y", "z", "r_sq"]

    def __init__(self, x, y, z, r_sq):
        self.x = x
        self.y = y
        self.z = z

        self.r_sq = r_sq

    def __iter__(self):
        for field_name in self.__slots__:
            yield getattr(self, field_name)



class PlumeModel(object):

    """
    Puff-based odour plume dispersion model from Farrell et. al. (2002).

    The odour plume is modelled as a series of odour puffs which are released
    from a fixed source position. The odour puffs are dispersed by a modelled
    2D wind velocity field plus a white noise process model of mid-scale
    puff mass diffusion relative to the plume centre line. The puffs also
    spread in size over time to model fine-scale diffusive processes.
    """

    def __init__(self, sim_region, source_pos, wind_model, model_z_disp=True,
                 centre_rel_diff_scale=1.5, puff_init_rad=0.03,
                 puff_spread_rate=0.001, puff_release_rate=10,
                 init_num_puffs=10, max_num_puffs=25000, prng=scipy.random):
        """
        Parameters
        ----------
        sim_region : Rectangle
            2D rectangular region of space over which the simulation is
            conducted. This should be the same simulation region as defined
            for the wind model.
        source_pos : float sequence
            (x,y,z) coordinates of the fixed source position within the
            simulation region from which puffs are released. If a length 2
            sequence is passed, the z coordinate will be set a default of 0.
            (dimensionality: length)
        ***AKR 3/12 I adjusted this to be a list of positions --> multiple sources.
            Specifically, source pos is an 3xm array, coordinates by sources
        wind_model : WindModel
            Dynamic model of the large scale wind velocity field in the
            simulation region.
            ***AKR changed this to work with Will's wind model.***
        model_z_disp : boolean
            Whether to model dispersion of puffs from plume centre-line in
            z direction. If set True then the puffs will be modelled as
            dispersing in the vertical direction by a random walk process
            (the wind model is limited to 2D hence the vertical wind speed
            is assumed to be zero), if set False the puff z-coordinates will
            not be updated from their initial value of 0.
        centre_rel_diff_scale : float or float sequence
            Scaling for the stochastic process used to model the centre-line
            relative diffusive transport of puffs. Either a single float
            value of isotropic diffusion in all directions, or one of a pair
            of values specifying different scales for the x and y directions
            respectively if model_z_disp=False or a triplet of values
            specifying different scales for x, y and z scales respectively if
            model_z_disp=True.
            (dimensionality: length/time^0.5)
        puff_init_rad: float
            Initial radius of the puffs.
            (dimensionality: length)
        puff_spread_rate : float
            Constant which determines the rate at which the odour puffs
            increase in size as time progresses.
            (dimensionality: length^2/time)
        puff_release_rate : float
            Mean rate at which new puffs are released into the plume. Puff
            release is modelled as a stochastic Poisson process, with each
            puff released assumed to be independent and the mean release rate
            fixed.
            (dimensionality: count/time)
        init_num_puffs : integer
            Initial number of puffs to release at the beginning of the
            simulation.
        max_num_puffs : integer
            Maximum number of puffs to permit to be in existence
            simultaneously within model, used to limit memory and processing
            requirements of model. This parameter needs to be set carefully
            in relation to the puff release rate and simulation region size
            as if too small it will lead to a breaks in puff release when the
            number of puffs remaining in the simulation region reaches the
            limit.
        prng : RandomState
            Pseudo-random number generator to use in generating input noise
            for puff centre-line relative dispersion random walk and puff
            release Poisson processes. If no value is set (default) the
            numpy.random global generator is used however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        """
        self.sim_region = sim_region
        self.wind_model = wind_model
        self.source_pos = source_pos
        self.unique_sources = len(source_pos[0,:])
        self.prng = prng
        self.model_z_disp = model_z_disp
        self._vel_dim = 3 if model_z_disp else 2
        if (model_z_disp and hasattr(centre_rel_diff_scale, '__len__') and
                len(centre_rel_diff_scale) == 2):
            raise InvalidCentreRelDiffScaleError('When model_z_disp=True, \
                                                  len(centre_rel_diff_scale) \
                                                  must be 1 or 3')
        self.centre_rel_diff_scale = centre_rel_diff_scale
        for i in range(self.unique_sources):
            if not sim_region.contains(source_pos[0,i], source_pos[1,i]):
                raise InvalidSourcePositionError('Specified source (x,y) \
                                              position must be within \
                                              simulation region.')
        # default to zero height source
        self.source_z = 0.0
        if len(source_pos[:,0]) == 3:
            source_pos[2,:] = source_z
        self.puff_init_rad = puff_init_rad
        self._new_puff_params = (source_pos[0,:], source_pos[1,:], self.source_z,
                                 puff_init_rad**2)
        self.puff_spread_rate = puff_spread_rate
        self.puff_release_rate = puff_release_rate
        self.max_num_puffs = max_num_puffs
        # initialise puff list with specified number of new puffs
        #print((source_pos[0,j],source_pos[1,j],source_z,puff_init_rad**2)
        #    for j in range(self.unique_sources))
        self.puffs = list(scipy.ndarray.flatten(scipy.array(
        [[Puff(source_pos[0,j],source_pos[1,j],self.source_z,puff_init_rad**2) for j in range(self.unique_sources)]
                      for i in range(init_num_puffs)])))
    def report(self):
        print('We have '+str(len(self.puffs))+' puffs going on.')

    def update(self, t, dt):
        """Perform time-step update of plume model with Euler integration."""
        # add more puffs (stochastically) if enough capacity
        if len(self.puffs) < self.max_num_puffs*self.unique_sources:
            # puff release modelled as Poisson process at fixed mean rate
            # with number to release clipped if it would otherwise exceed
            # the maximum allowed

            #****Draw separately for each trap
            for j in range(self.unique_sources):
                num_to_release = self.prng.poisson(self.puff_release_rate*dt)
                num_to_release = min(num_to_release,
                                 self.max_num_puffs - len(self.puffs))
                for i in range(num_to_release):
                    self.puffs.append(Puff(self.source_pos[0,j],
                    self.source_pos[1,j],self.source_z,self.puff_init_rad**2))
        # initialise empty list for puffs that have not left simulation area
        alive_puffs = []
        for puff in self.puffs:
            # interpolate wind velocity at Puff position from wind model grid
            # assuming zero wind speed in vertical direction if modelling
            # z direction dispersion
            wind_vel = scipy.zeros(self._vel_dim)
            wind_vel[:2] = self.wind_model.value(t,puff.x, puff.y)
            # approximate centre-line relative puff transport velocity
            # component as being a (Gaussian) white noise process scaled by
            # constants
            filament_diff_vel = (self.prng.normal(size=self._vel_dim) *
                                 self.centre_rel_diff_scale)
            vel = wind_vel + filament_diff_vel
            # update puff position using Euler integration
            puff.x += vel[0] * dt
            puff.y += vel[1] * dt
            if self.model_z_disp:
                puff.z += vel[2] * dt
            # update puff size using Euler integration with second puff
            # growth model described in paper
            puff.r_sq += self.puff_spread_rate * dt
            # only keep puff alive if it is still in the simulated region
            if self.sim_region.contains(puff.x, puff.y):
                alive_puffs.append(puff)
        # store alive puffs only
        self.puffs = alive_puffs

    @property
    def puff_array(self):
        """
        Returns a numpy array of the properties of the simulated puffs.

        Each row corresponds to one puff with the first column containing the
        puff position x-coordinate, the second the y-coordinate, the third
        the z-coordinate and the fourth the puff squared radius.
        """
        return scipy.array([tuple(puff) for puff in self.puffs])



class ConcentrationArrayGenerator(object):

    """
    Produces odour concentration field arrays from puff property arrays.

    Instances of this class can take single or multiple arrays of puff
    properties outputted from a PlumeModel and process them to produce an
    array of the concentration values across the a specified region using
    a Gaussian model for the individual puff concentration distributions.

    Compared to the ConcentrationValueCalculator class, this class should be
    more efficient for calculating large concentration field arrays for
    real-time graphical display of odour concentration fields for example
    at the expense of (very) slightly less accurate values due to the
    truncation of spatial extent of each puff.

    Notes
    -----
    The returned array values correspond to the *point* concentration
    measurements across a regular grid of sampling points - i.e. the
    equivalent to convolving the true continuous concentration distribution
    with a regular 2D grid of Dirac delta / impulse functions. An improvement
    in some ways would be to instead calculate the integral of the
    concentration distribution over the (square) region around each grid point
    however this would be extremely computationally costly and due to the lack
    of a closed form solution for the integral of a Gaussian also potentially
    difficult to implement without introducing other numerical errors. An
    integrated field can be approximated with this class by generating an
    array at a higher resolution than required and then filtering with a
    suitable kernel and down-sampling.

    This implementation estimates the concentration distribution puff kernels
    with sub-grid resolution, giving improved accuracy at the cost of
    increased computational cost versus using a precomputed radial field
    aligned with the grid to compute kernel values or using a library of
    precomputed kernels.

    For cases where the array region cover the whole simulation region the
    computational cost could also be reduced by increasing the size of the
    region the array corresponds to outside of the simulation region such that
    when adding the puff concentration kernels to the concentration field
    array, checks do not need to be made to restrict to the overlapping region
    for puffs near the edges of the simulation region which have a
    concentration distribution which extends beyond its extents.

    ***AKR I've adjusted this to query individual positions as well (borrowing
    functions from the ConcentrationValueCalculator object).
    """

    def __init__(self, array_xy_region, array_z, nx, ny, puff_mol_amount,
                 kernel_rad_mult=3):
        """
        Parameters
        ----------
        array_region : Rectangle
            Two-dimensional rectangular region defined in world coordinates
            over which to calculate the concentration field.
        array_z : float
            Height on the vertical z-axis at which to calculate the
            concentration field over.
        nx : integer
            Number of grid points to sample at across x-dimension.
        ny : integer
            Number of grid points to sample at across y-dimension.
        puff_mol_amount : float
            Molecular content of each puff (e.g. in moles or raw number of
            molecules). This is conserved as the puff is transported within
            the plume but the puff becomes increasingly diffuse as it's radius
            grows due to diffusion.
            (dimensionality:molecular amount)
        kernel_rad_mult : float
            Multiplier used to determine to within how many puff radii from
            the puff centre to truncate the concentration distribution
            kernel calculated to. The default value of 3 will truncate the
            Gaussian kernel at (or above) the point at which the concentration
            has dropped to 0.004 of the peak value at the puff centre.
        """
        self.array_xy_region = array_xy_region
        self.array_z = array_z
        self.nx = nx
        self.ny = ny
        self._dx = array_xy_region.w / nx  # calculate x grid point spacing
        self._dy = array_xy_region.h / ny  # calculate y grid point spacing
        # precompute constant used to scale Gaussian kernel amplitude
        self._ampl_const = puff_mol_amount / (8*scipy.pi**3)**0.5
        self.kernel_rad_mult = kernel_rad_mult
        self.param = {}
        self.puff_mol_amount = puff_mol_amount
        #self.param.update({'trap_radius':trap_radius}) #moved to trap object
        #self.param.update({'source_locations':sources})
    def _puff_conc_dist(self, x, y, z, px, py, pz, r_sq):
        #print(len(x),len(y),len(px),len(py))
        # calculate Gaussian puff concentration distribution
        return (
            self._ampl_const / r_sq**1.5 *
            scipy.exp(-((x - px)**2 + (y - py)**2 + (z - pz)**2) / (2 * r_sq))
        )

    def calc_conc_point(self, puff_array, x, y, z=0):
        """
        Calculate concentration at a single point.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : float
            x-coordinate of point.
        y : float
            y-coordinate of point.
        z : float
            z-coordinate of point.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~scipy.isnan(puff_array[:, 0]), :].T
        return self._puff_conc_dist(x, y, z, px, py, pz, r_sq).sum(-1)
    def calc_conc_list(self, puff_array, x, y, z=0):
        """
        Calculate concentrations across a 1D list of points in a xy-plane.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : (np) numpy-array-like of floats
            1D array of x-coordinates of points.
        y : (np) numpy-array-like of floats
            1D array of y-coordinates of points.
        z : float
            z-coordinate (height) of plane.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~scipy.isnan(puff_array[:, 0]), :].T
        na = scipy.newaxis
        return self._puff_conc_dist(x[:, na], y[:, na], z, px[na, :],
                                    py[na, :], pz[na, :], r_sq[na, :]).sum(-1)

    def puff_kernel(self, shift_x, shift_y, z_offset, r_sq, even_w, even_h):
        # kernel is truncated to min +/- kernel_rad_mult * effective puff
        # radius from centre i.e. Gaussian kernel with >= kernel_rad_mult *
        # standard deviation span
        # (effective puff radius is (r_sq - (z_offset/k_r_mult)**2)**0.5 to
        # account for the cross sections of puffs with centres out of the
        # array plane being 'smaller')
        # the truncation will introduce some errors - an improvement would
        # be to use some form of windowing e.g. Hann or Hamming window
        shape = (2*(r_sq*self.kernel_rad_mult**2 - z_offset**2)**0.5 /
                 scipy.array([self._dx, self._dy]))
        # depending on whether centre is on grid points or grid centres
        # kernel dimensions will need to be forced to odd/even respectively
        shape[0] = self.round_up_to_next_even_or_odd(shape[0], even_w)
        shape[1] = self.round_up_to_next_even_or_odd(shape[1], even_h)
        # generate x and y grids with required shape
        [x_grid, y_grid] = 0.5 + scipy.mgrid[-shape[0]/2:shape[0]/2,
                                          -shape[1]/2:shape[1]/2]
        # apply shifts to correct for offset of true centre from nearest
        # grid-point / centre
        x_grid = x_grid * self._dx + shift_x
        y_grid = y_grid * self._dy + shift_y
        # compute square radial field
        r_sq_grid = x_grid**2 + y_grid**2 + z_offset**2
        # output scaled Gaussian kernel
        return self._ampl_const / r_sq**1.5 * scipy.exp(-r_sq_grid / (2 * r_sq))

    @staticmethod
    def round_up_to_next_even_or_odd(value, to_even):
        # Returns value rounded up to first even number >= value if
        # to_even==True and to first odd number >= value if to_even==False.
        value = math.ceil(value)
        if to_even:
            if value % 2 == 1:
                value += 1
        else:
            if value % 2 == 0:
                value += 1
        return value

    def generate_single_array(self, puff_array):
        """
        Generates a single concentration field array from an array of puff
        properties.
        """
        # initialise concentration array
        conc_array = scipy.zeros((self.nx, self.ny))
        # loop through all the puffs
        for (puff_x, puff_y, puff_z, puff_r_sq) in puff_array:
            # to begin with check this a real puff and not a placeholder nan
            # entry as puff arrays may have been pre-allocated with nan
            # at a fixed size for efficiency and as the number of puffs
            # existing at any time interval is variable some entries in the
            # array will be unallocated, placeholder entries should be
            # contiguous (i.e. all entries after the first placeholder will
            # also be placeholders) therefore break out of loop completely
            # if one is encountered
            if scipy.isnan(puff_x):
                break
            # check puff centre is within region array is being calculated
            # over otherwise skip
            if not self.array_xy_region.contains(puff_x, puff_y):
                continue
            # finally check that puff z-coordinate is within
            # kernel_rad_mult*r_sq of array evaluation height otherwise skip
            puff_z_offset = (self.array_z - puff_z)
            if abs(puff_z_offset) / puff_r_sq**0.5 > self.kernel_rad_mult:
                continue
            # calculate (float) row index corresponding to puff x coord
            p = (puff_x - self.array_xy_region.x_min) / self._dx
            # calculate (float) column index corresponding to puff y coord
            q = (puff_y - self.array_xy_region.y_min) / self._dy
            # calculate nearest integer or half-integer row index to p
            u = math.floor(2 * p + 0.5) / 2
            # calculate nearest integer or half-integer row index to q
            v = math.floor(2 * q + 0.5) / 2
            # generate puff kernel array of appropriate scale and taking
            # into account true centre offset from nearest half-grid
            # points (u,v)
            kernel = self.puff_kernel((p - u) * self._dx, (q - v) * self._dy,
                                      puff_z_offset, puff_r_sq,
                                      u % 1 == 0, v % 1 == 0)
            # compute row and column slices for source kernel array and
            # destination concentration array taking in to the account
            # the possibility of the kernel being partly outside the
            # extents of the destination array
            (w, h) = kernel.shape
            r_rng_arr = slice(max(0, u - w / 2.),
                              max(min(u + w / 2., self.nx), 0))
            c_rng_arr = slice(max(0, v - h / 2.),
                              max(min(v + h / 2., self.ny), 0))
            r_rng_knl = slice(max(0, -u + w / 2.),
                              min(-u + w / 2. + self.nx, w))
            c_rng_knl = slice(max(0, -v + h / 2.),
                              min(-v + h / 2. + self.ny, h))
            # add puff kernel values to concentration field array
            conc_array[r_rng_arr, c_rng_arr] += kernel[r_rng_knl, c_rng_knl]
        return conc_array

    def generate_multiple_arrays(self, puff_arrays):
        """
        Generates multiple concentration field arrays from a sequence of
        arrays of puff properties.
        """
        conc_arrays = []
        for puff_array in puff_arrays:
            conc_arrays.append(self.generate_single_frame(puff_array))
        return conc_arrays
        '''
        Here is a copy of Will's odor plotting adpated for this concentration object.
        We can skip the mesh generation/odor value computation; this is already in the
        concentration object.
        '''
    def plot(self,conc_array,plot_param):
        xlim = plot_param['xlim']
        ylim = plot_param['ylim']
        cmap = plot_param['cmap']

        try:
            threshold = plot_param['threshold']
        except KeyError:
            threshold = None

        try:
            fignums = plot_param['fignums']
        except KeyError:
            fignums = (1,2)
#        vmax = max(scipy.unique(conc_array))
#        print('-----------                          vmax'+ str(vmax))
        plt.figure(fignums[0])

        vmin = 0.0
        # vmax = 5e4 #value used by Graham
        vmax = 20.
        # vmax= 20.*(self.puff_mol_amount/1000.) #empirically observed scaling
        #(for parameters used until 5/15) lol
    #    t = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        #conc_array=conc_array/conc_array.max()
        image=plt.imshow(conc_array, extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=cmap,vmin=vmin,vmax=vmax)
#            #plt.plot([x],[y],'ok')
#            s = scipy.linspace(0,2.0*scipy.pi,100)
#            cx = x + self.param['trap_radius']*scipy.cos(s)
#            cy = y + self.param['trap_radius']*scipy.sin(s)
#            plt.plot(cx,cy,'k')
        plt.plot([0],[0],'ob')
        plt.grid('on')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
#        plt.title('Odor Concentration')

        if threshold is not None:
            plt.figure(fignums[1])
            odor_thresh = odor_value >= threshold
            plt.imshow(odor_thresh, extent=(xlim[0],xlim[1],ylim[0],ylim[1]),cmap=cmap)
            for x,y in self.param['source_locations']:
                plt.plot([x],[y],'.k')

            plt.plot([0],[0],'ob')
            plt.grid('on')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title('Odor Concentration >= {0}'.format(threshold))
        return image
