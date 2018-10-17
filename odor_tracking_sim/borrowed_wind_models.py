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

class WindModel(object):

    """
    Wind velocity model to calculate advective transport of odour.

    A 2D approximation is used as described in the paper, with the wind
    velocities calculated over a regular 2D grid of points using a finite
    difference method. The boundary conditions at the edges of the simulated
    region are for both components of the velocity field constant mean values
    plus coloured noise. For each of the field components these are calculated
    for the four corners of the simulated region and then linearly
    interpolated over the edges.
    """

    def __init__(self, sim_region, nx=15, ny=15, u_av=1., v_av=0., Kx=2.,
                 Ky=2., noise_gain=3., noise_damp=0.1, noise_bandwidth=0.71,
                 noise_rand=scipy.random,wind_update_period=5):
        """
        Parameters
        ----------
        sim_region : Rectangle
            Two-dimensional rectangular region over which to model wind
            velocity field.
        nx : integer
            Number of grid points in x direction.
        ny : integer
            Number of grid points in y direction.
        u_av : float
            Mean x-component of wind velocity (u).
            (dimensionality: length/time)
        v_av : float
            Mean y-component of wind velocity (v).
            (dimensionality: length/time)
        Kx : float or array_like
            Diffusivity constant in x direction. Either a single scalar value
            across the whole simulated region or an array of size (nx, ny)
            defining different values for each grid point.
            (dimensionality: length^2/time)
        Ky : float or array_like
            Diffusivity constant in y direction. Either a single scalar value
            across the whole simulated region or an array of size (nx, ny)
            defining different values for each grid point.
            (dimensionality: length^2/time)
        noise_gain : float
            Iscipyut gain constant for boundary condition noise generation.
            (dimensionless)
        noise_damp : float
            Damping ratio for boundary condition noise generation.
            (dimensionless)
        noise_bandwidth : float
            Bandwidth for boundary condition noise generation.
            (dimensionality: angular measure/time)
        noise_rand : RandomState : float
            Pseudo-random number generator to use in generating iscipyut noise.
            Defaults to numpy.random global generator however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        wind_update_period : float
            Number of seconds per wind direction datum
        """
        self.evolving = True
        self.wind_update_period=wind_update_period
        self.u_av = u_av
        self.v_av = v_av

        # store grid parameters interally
        self._dx = abs(sim_region.w) / (nx-1)  # x grid point spacing
        self._dy = abs(sim_region.h) / (ny-1)  # y grid point spacing
        self.nx = nx
        self.ny = ny
        # precompute constant coefficients in PDE for efficiency
        self._Bx = Kx / (2.*self._dx**2)
        self._By = Ky / (2.*self._dy**2)
        self._C = 2. * (self._Bx + self._By)
        # initialise wind velocity field to mean values
        # +2s are to account for boundary grid points
        self._u = scipy.ones((nx+2, ny+2)) * u_av[0]
        self._v = scipy.ones((nx+2, ny+2)) * v_av[0]
        # create views on to field interiors (i.e. not including boundaries)
        # for notational ease - note this does NOT copy any data
        self._u_int = self._u[1:-1, 1:-1]
        self._v_int = self._v[1:-1, 1:-1]
        # set coloured noise generator for applying boundary condition
        # need to generate coloured noise samples at four corners of boundary
        # for both components of the wind velocity field so (2,8) state
        # vector (2 as state includes first derivative)
        self.noise_gen = ColouredNoiseGenerator(scipy.zeros((2, 8)), noise_damp,
                                                noise_bandwidth, noise_gain,
                                                noise_rand)
        # preassign array of corner means values
        self._corner_means = scipy.array([
        scipy.array([u_av[i], v_av[i]]).repeat(4) \
        for i in range(len(u_av))])

        #here is the way to get back the empirical wind velocity (at current time)
        self._empirical_velocity = (u_av[0],v_av[0])

        # precompute linear ramp arrays with size of boundary edges for
        # linear interpolation of corner values
        self._rx = scipy.linspace(0., 1., nx+2)
        self._ry = scipy.linspace(0., 1., ny+2)
        # set up cubic spline interpolator for calculating off-grid wind
        # velocity field values
        self._x_points = scipy.linspace(sim_region.x_min, sim_region.x_max, nx)
        self._y_points = scipy.linspace(sim_region.y_min, sim_region.y_max, ny)
        self._set_interpolators()

    def _set_interpolators(self):
        """ Set spline interpolators using current velocity fields."""
        self._interp_u = interp.RectBivariateSpline(self.x_points,
                                                    self.y_points,
                                                    self._u_int)
        self._interp_v = interp.RectBivariateSpline(self.x_points,
                                                    self.y_points,
                                                    self._v_int)

    @property
    def x_points(self):
        """1D array of the range of x-coordinates of simulated grid points."""
        return self._x_points

    @property
    def y_points(self):
        """1D array of the range of y-coordinates of simulated grid points."""
        return self._y_points

    @property
    def velocity_field(self):
        """Current calculated velocity field across simulated grid points."""
        return scipy.dstack((self._u_int, self._v_int))

    def velocity_at_pos(self, x, y):
        """
        Calculates the components of the velocity field at arbitrary point
        in the simulation region using a bivariate spline interpolation over
        the calculated grid point values.

        Parameters
        ----------
        x : float
            x-coordinate of the point to calculate the velocity at.
            (dimensionality: length)
        y : float
            y-coordinate of the point to calculate the velocity at.
            (dimensionality: length)

        Returns
        -------
        vel : array_like
            Velocity field (2D) values evaluated at specified point(s).
            (dimensionality: length/time)
        """
        return scipy.array([float(self._interp_u(x, y)),
                         float(self._interp_v(x, y))])
    def value(self,t,x,y):
        if type(x)==scipy.ndarray:
            wind = scipy.array([
            self.velocity_at_pos(x[i],y[i]) for i in range(len(x))
            ])
            return wind[:,0],wind[:,1]
        else:
            return self.velocity_at_pos(x,y)

    def update(self, t, dt):
        """
        Updates wind velocity field values using finite difference
        approximations for spatial derivatives and Euler integration for
        time-step update.

        Parameters
        ----------
        dt : float
            Simulation time-step.
            (dimensionality: time)
        """
        # update boundary values
        self._apply_boundary_conditions(t,dt)
        #update current empirical wind velocity
        corner_mean_index = int(scipy.floor(t/self.wind_update_period))
        self._empirical_velocity = self.u_av[corner_mean_index], self.v_av[corner_mean_index]
        # initialise wind speed derivative arrays
        du_dt = scipy.zeros((self.nx, self.ny))
        dv_dt = scipy.zeros((self.nx, self.ny))
        # approximate spatial first derivatives with centred finite difference
        # equations for both components of wind field
        du_dx, du_dy = self._centred_first_derivs(self._u)
        dv_dx, dv_dy = self._centred_first_derivs(self._v)
        # calculate centred first sums i.e. sf_x = f(x+dx,y)+f(x-dx,y) and
        # sf_y = f(x,y+dy)-f(x,y-dy) as first step in approximating spatial
        # second derivatives with second order finite difference equations
        #   d2f/dx2 ~ [f(x+dx,y)-2f(x,y)+f(x-dx,y)] / (dx*dx)
        #           = [sf_x-2f(x,y)] / (dx*dx)
        #   d2f/dy2 ~ [f(x,y+dy)-2f(x,y)+f(x,y-dy)] / (dy*dy)
        #           = [sf_y-2f(x,y)] / (dy*dy)
        # second finite differences are not computed in full as the common
        # f(x,y) term in both expressions can be extracted out to reduce
        # the number of +/- operations required
        su_x, su_y = self._centred_first_sums(self._u)
        sv_x, sv_y = self._centred_first_sums(self._v)
        # use finite difference method to approximate time derivatives across
        # simulation region interior from defining PDEs
        #     du/dt = -(u*du/dx + v*du/dy) + 0.5*Kx*d2u/dx2 + 0.5*Ky*d2u/dy2
        #     dv/dt = -(u*dv/dx + v*dv/dy) + 0.5*Kx*d2v/dx2 + 0.5*Ky*d2v/dy2
        du_dt = (-self._u_int * du_dx - self._v_int * du_dy +
                 self._Bx * su_x + self._By * su_y -
                 self._C * self._u_int)
        dv_dt = (-self._u_int * dv_dx - self._v_int * dv_dy +
                 self._Bx * sv_x + self._By * sv_y -
                 self._C * self._v_int)
        # perform update with Euler integration
        self._u_int += du_dt * dt
        self._v_int += dv_dt * dt
        # update spline interpolators
        self._set_interpolators()

    def _apply_boundary_conditions(self,t,dt):
        """Applies boundary conditions to wind velocity field."""
        # update coloured noise generator
        self.noise_gen.update(dt)
        # extract four corner values for each of u and v fields as component
        # mean plus current noise generator output
        corner_mean_index = int(scipy.floor(t/self.wind_update_period))
        added_noise = self.noise_gen.output
        (u_tl, u_tr, u_bl, u_br, v_tl, v_tr, v_bl, v_br) = \
            added_noise + self._corner_means[corner_mean_index,:]
        # linearly interpolate along edges
        self._u[:, 0] = u_tl + self._rx * (u_tr - u_tl)   # u top edge
        self._u[:, -1] = u_bl + self._rx * (u_br - u_bl)  # u bottom edge
        self._u[0, :] = u_tl + self._ry * (u_bl - u_tl)   # u left edge
        self._u[-1, :] = u_tr + self._ry * (u_br - u_tr)  # u right edge
        self._v[:, 0] = v_tl + self._rx * (v_tr - v_tl)   # v top edge
        self._v[:, -1] = v_bl + self._rx * (v_br - v_bl)  # v bottom edge
        self._v[0, :] = v_tl + self._ry * (v_bl - v_tl)   # v left edge
        self._v[-1, :] = v_tr + self._ry * (v_br - v_tr)  # v right edge

    def _centred_first_derivs(self, f):
        """Calculates centred first difference derivative approximations."""
        return ((f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * self._dx),
                (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * self._dy))

    def _centred_first_sums(self, f):
        """Calculates centred first sums."""
        return (f[2:, 1:-1] + f[0:-2, 1:-1]), (f[1:-1, 2:]+f[1:-1, 0:-2])


class ColouredNoiseGenerator(object):

    """
    Generates a coloured noise output via Euler integration of a state space
    system formulation.
    """

    def __init__(self, init_state, damping, bandwidth, gain,
                 prng=scipy.random):
        """
        Parameters
        ----------
        init_state : array_like
            The initial state of system, must be of shape (2,n) where n is
            the size of the noise vector to be produced. The first row
            sets the initial values and the second the initial first
            derivatives.
        damping : float
            Damping ratio for the system, affects system stability, values of
            <1 give an underdamped system, =1 a critically damped system and
            >1 an overdamped system.
            (dimensionless)
        bandwidth : float
            Bandwidth or equivalently undamped natural frequency of system,
            affects system reponsiveness to variations in (noise) iscipyut.
            (dimensionality = angular measure / time)
        gain : float
            Iscipyut gain of system, affects scaling of (noise) iscipyut.
            (dimensionless)
        prng : RandomState
            Pseudo-random number generator to use in generating iscipyut noise.
            Defaults to numpy.random global generator however a specific
            RandomState can be set if it is desired to have reproducible
            output.
        """
        # set up state space matrices
        self._A = scipy.array([[0., 1.],
                            [-bandwidth**2, -2. * damping * bandwidth]])
        self._B = scipy.array([[0.], [gain * bandwidth**2]])
        # initialise state
        self._x = init_state
        self.prng = prng

    @property
    def output(self):
        """Coloured noise output."""
        return self._x[0, :]

    def update(self, dt):
        """Updates state of noise generator."""
        # get normal random iscipyut
        u = self.prng.normal(size=(1, self._x.shape[1]))
        # calculate state time derivative with state space equation
        dx_dt = self._A.dot(self._x) + self._B * u
        # apply update with Euler integration
        self._x += dx_dt * dt
